
import numpy as np
import pandas as pd
import time

from imblearn.over_sampling import SMOTE
from enum import Enum
from dataclasses import dataclass
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from logger import Log


log = Log()


@dataclass
class Preds:
    class_prob: np.ndarray
    class_pred: np.ndarray
    reg_pred: np.ndarray
    expected_volume_pred: np.ndarray


class TuneMode(Enum):
    NONE = 0
    RANDOM_SEARCH = 1
    GRID_SEARCH = 2


def tune_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 19)

    best_threshold = 0.5
    best_f1 = -1

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def apply_smote_single_label(X_train, y_train, label_name, random_state=12, k_neighbors=5):
    X_train = X_train.copy()
    y_train = pd.Series(y_train).copy()

    log.h2(f"SMOTE REPORT FOR {label_name}")

    before_pos = int((y_train == 1).sum())
    before_neg = int((y_train == 0).sum())
    log.body(f"Before SMOTE: X shape={X_train.shape}, pos={before_pos}, neg={before_neg}")

    if before_pos < 2:
        log.warn("Skipping SMOTE: not enough minority samples.")
        return X_train, y_train

    X_train = X_train.fillna(0)

    k = min(k_neighbors, before_pos - 1)
    if k < 1:
        log.warn("Skipping SMOTE: k_neighbors would be invalid.")
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    after_pos = int((y_res == 1).sum())
    after_neg = int((y_res == 0).sum())
    log.body(f"After  SMOTE: X shape={X_res.shape}, pos={after_pos}, neg={after_neg}")
    log.body(f"Added samples: {len(X_res) - len(X_train)}")

    return X_res, y_res


class TwoHeadPipeline:
    def __init__(
        self,
        classifier=None,
        regressor=None,
        class_param_dist=None,
        reg_param_dist=None,
        class_default_params=None,
        reg_default_params=None,
        apply_smote=False,
        apply_scale_pos_weight=False,
        log_regression_target=False,
        positive_only_regression=False
    ):
        self.head1 = classifier
        self.head2 = regressor
        self.class_param_dist = class_param_dist
        self.reg_param_dist = reg_param_dist
        self.class_default_params = class_default_params or {}
        self.reg_default_params = reg_default_params or {}
        self.apply_smote = apply_smote
        self.apply_scale_pos_weight = apply_scale_pos_weight
        self.threshold = 0.5
        self.best_f1 = None
        self.log_regression_target = log_regression_target
        self.positive_only_regression = positive_only_regression
        self.is_fitted = False


    def train_classifier(self, splits, class_target_col, tune_mode):
        log.info("Training Classifier...")

        X_train_cls = splits.X_train
        y_train_cls = splits.y_class_train[class_target_col]

        neg_count = (y_train_cls == 0).sum()
        pos_count = (y_train_cls == 1).sum()

        if pos_count > 0:
            scale_pos_weight = neg_count / pos_count
        else:
            scale_pos_weight = 1.0

        if tune_mode == TuneMode.NONE:
            self.head1.set_params(**self.class_default_params)

        if self.apply_scale_pos_weight:
            self.head1.set_params(scale_pos_weight=scale_pos_weight)
            print(f"Applied scale_pos_weight={scale_pos_weight:.2f} to classifier.")

        if self.apply_smote:
            X_train_cls, y_train_cls = apply_smote_single_label(
                X_train_cls,
                y_train_cls,
                label_name=class_target_col,
                random_state=12,
                k_neighbors=5
            )

        start = time.perf_counter()

        if tune_mode == TuneMode.RANDOM_SEARCH:
            log.info("Beginning RandomizedSearch hyperparameter tuning for classifier...")
            search = RandomizedSearchCV(
                estimator=self.head1,
                param_distributions=self.class_param_dist,
                n_iter=10,
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
                random_state=12
            )
            search.fit(X_train_cls, y_train_cls)
            self.head1 = search.best_estimator_
            log.info(f"Best classifier params: {search.best_params_}")

        elif tune_mode == TuneMode.GRID_SEARCH:
            log.info("Beginning GridSearch hyperparameter tuning for classifier...")
            search = GridSearchCV(
                estimator=self.head1,
                param_grid=self.class_param_dist,
                scoring="roc_auc",
                cv=3,
                n_jobs=-1
            )
            search.fit(X_train_cls, y_train_cls)
            self.head1 = search.best_estimator_
            log.info(f"Best classifier params: {search.best_params_}")

        elif tune_mode == TuneMode.NONE:
            self.head1.fit(
                X_train_cls,
                y_train_cls,
                eval_set=[(splits.X_val, splits.y_class_val[class_target_col])],
                verbose=False
            )

        end = time.perf_counter()

        log.info("Classifier training complete.")
        log.info(f"Training time: {end - start:.2f} seconds")

        y_val_prob = self.head1.predict_proba(splits.X_val)[:, 1]
        self.threshold, self.best_f1 = tune_threshold(splits.y_class_val[class_target_col], y_val_prob)

        log.info(f"Chosen threshold: {self.threshold:.2f} (best F1 on validation: {self.best_f1:.4f})")


    def train_regressor(self, splits, class_target_col, reg_target_col, tune_mode):
        log.info("Training Regressor...")

        if self.positive_only_regression:
            train_mask = splits.y_class_train[class_target_col] == 1
            val_mask = splits.y_class_val[class_target_col] == 1

            X_train_reg = splits.X_train.loc[train_mask]
            y_train_reg = splits.y_reg_train[reg_target_col].loc[train_mask]
            X_val_reg = splits.X_val.loc[val_mask]
            y_val_reg = splits.y_reg_val[reg_target_col].loc[val_mask]
        else:
            X_train_reg = splits.X_train
            y_train_reg = splits.y_reg_train[reg_target_col]
            X_val_reg = splits.X_val
            y_val_reg = splits.y_reg_val[reg_target_col]

        if self.log_regression_target:
            y_train_reg = np.log1p(y_train_reg)
            y_val_reg = np.log1p(y_val_reg)

        if len(X_train_reg) == 0:
            log.warn(f"Skipping regressor for {reg_target_col}: no positive training rows.")
            self.head2 = None
            self.is_fitted = True
            return

        start = time.perf_counter()

        if tune_mode == TuneMode.RANDOM_SEARCH:
            log.info("Beginning RandomizedSearch hyperparameter tuning for regressor...")
            search = RandomizedSearchCV(
                estimator=self.head2,
                param_distributions=self.reg_param_dist,
                n_iter=20,
                scoring="neg_mean_squared_error",
                cv=3,
                n_jobs=-1,
                random_state=12
            )
            search.fit(X_train_reg, y_train_reg)
            self.head2 = search.best_estimator_
            log.info(f"Best regressor params: {search.best_params_}")

        elif tune_mode == TuneMode.GRID_SEARCH:
            log.info("Beginning GridSearch hyperparameter tuning for regressor...")
            search = GridSearchCV(
                estimator=self.head2,
                param_grid=self.reg_param_dist,
                scoring="neg_mean_squared_error",
                cv=3,
                n_jobs=-1
            )
            search.fit(X_train_reg, y_train_reg)
            self.head2 = search.best_estimator_
            log.info(f"Best regressor params: {search.best_params_}")

        elif tune_mode == TuneMode.NONE:
            self.head2.set_params(**self.reg_default_params)
            self.head2.fit(
                X_train_reg,
                y_train_reg,
                eval_set=[(X_val_reg, y_val_reg)],
                verbose=False
            )

        end = time.perf_counter()

        log.info("Regressor training complete.")
        log.info(f"Training time: {end - start:.2f} seconds")

        self.is_fitted = True


    def train(
        self,
        splits,
        class_target_col,
        reg_target_col,
        class_tune_mode=TuneMode.NONE,
        reg_tune_mode=TuneMode.NONE
    ):
        self.train_classifier(splits, class_target_col, class_tune_mode)
        self.train_regressor(splits, class_target_col, reg_target_col, reg_tune_mode)


    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")

        y_prob = self.head1.predict_proba(X)[:, 1]
        y_class = (y_prob >= self.threshold).astype(int)

        if self.head2 is None:
            y_reg = np.zeros(len(X))
        else:
            y_reg = self.head2.predict(X)
            if self.log_regression_target:
                y_reg = np.expm1(y_reg)

        y_expected = y_prob * y_reg

        return Preds(
            class_prob=y_prob,
            class_pred=y_class,
            reg_pred=y_reg,
            expected_volume_pred=y_expected
        )


    def predict_df(self, X, prefix):
        preds = self.predict(X)
        return pd.DataFrame({
            f"{prefix}_class_prob": preds.class_prob,
            f"{prefix}_class_pred": preds.class_pred,
            f"{prefix}_reg_pred": preds.reg_pred,
            f"{prefix}_expected_volume_pred": preds.expected_volume_pred,
        }, index=X.index)
