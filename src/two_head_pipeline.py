
import numpy as np
import pandas as pd
import time

from imblearn.over_sampling import SMOTE
from enum import Enum
from dataclasses import dataclass
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


@dataclass
class Preds:
    class_prob: np.ndarray
    class_pred: np.ndarray
    reg_pred: np.ndarray
    expected_volume_pred: np.ndarray


class ParamTuningMode(Enum):
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


def apply_smote_to_single_label(X_train, y_train, random_state=12, k_neighbors=5):
    X_train = X_train.copy()
    y_train = y_train.copy()

    X_train = X_train.fillna(0)

    minority_count = int((y_train == 1).sum())
    if minority_count < 2:
        print("Skipping SMOTE: not enough minority samples.")
        return X_train, y_train

    k = min(k_neighbors, minority_count - 1)
    if k < 1:
        print("Skipping SMOTE: not enough minority samples for neighbors.")
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    return X_resampled, y_resampled


class TwoHeadPipeline:
    def __init__(
        self,
        classifier=None,
        regressor=None,
        class_param_dist=None,
        reg_param_dist=None,
        apply_smote=False,
        log_regression_target=False,
        positive_only_regression=False
    ):
        self.head1 = classifier
        self.head2 = regressor
        self.class_param_dist = class_param_dist
        self.reg_param_dist = reg_param_dist
        self.apply_smote = apply_smote
        self.threshold = 0.5
        self.best_f1 = None
        self.log_regression_target = log_regression_target
        self.positive_only_regression = positive_only_regression
        self.is_fitted = False


    def train_classifier(self, splits, class_target_col, tune_mode):
        print("Training Classifier...")

        X_train_cls = splits.X_train
        y_train_cls = splits.y_class_train[class_target_col]

        start = time.perf_counter()

        if tune_mode == ParamTuningMode.RANDOM_SEARCH:
            print("Beginning RandomizedSearch hyperparameter tuning for classifier...")
            search = RandomizedSearchCV(
                estimator=self.head1,
                param_distributions=self.class_param_dist,
                n_iter=20,
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
                random_state=12
            )
            search.fit(X_train_cls, y_train_cls)
            self.head1 = search.best_estimator_
            print(f"Best classifier params: {search.best_params_}")

        elif tune_mode == ParamTuningMode.GRID_SEARCH:
            print("Beginning GridSearch hyperparameter tuning for classifier...")
            search = GridSearchCV(
                estimator=self.head1,
                param_grid=self.class_param_dist,
                scoring="roc_auc",
                cv=3,
                n_jobs=-1
            )
            search.fit(X_train_cls, y_train_cls)
            self.head1 = search.best_estimator_
            print(f"Best classifier params: {search.best_params_}")

        elif tune_mode == ParamTuningMode.NONE:
            if self.apply_smote:
                X_train_cls, y_train_cls = apply_smote_to_single_label(
                    X_train_cls,
                    y_train_cls,
                    random_state=12,
                    k_neighbors=5
                )

            self.head1.fit(
                X_train_cls,
                y_train_cls,
                eval_set=[(splits.X_val, splits.y_class_val[class_target_col])],
                verbose=False
            )

        end = time.perf_counter()

        print("Classifier training complete.")
        print(f"Training time: {end - start:.2f} seconds\n")

        y_val_prob = self.head1.predict_proba(splits.X_val)[:, 1]
        self.threshold, self.best_f1 = tune_threshold(splits.y_class_val[class_target_col], y_val_prob)

        print(f"Chosen threshold: {self.threshold:.2f} (best F1 on validation: {self.best_f1:.4f})")


    def train_regressor(self, splits, class_target_col, reg_target_col, tune_mode):
        print("Training Regressor...")

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

        start = time.perf_counter()

        if tune_mode == ParamTuningMode.RANDOM_SEARCH:
            print("Beginning RandomizedSearch hyperparameter tuning for regressor...")
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
            print(f"Best regressor params: {search.best_params_}")

        elif tune_mode == ParamTuningMode.GRID_SEARCH:
            print("Beginning GridSearch hyperparameter tuning for regressor...")
            search = GridSearchCV(
                estimator=self.head2,
                param_grid=self.reg_param_dist,
                scoring="neg_mean_squared_error",
                cv=3,
                n_jobs=-1
            )
            search.fit(X_train_reg, y_train_reg)
            self.head2 = search.best_estimator_
            print(f"Best regressor params: {search.best_params_}")

        elif tune_mode == ParamTuningMode.NONE:
            self.head2.fit(
                X_train_reg,
                y_train_reg,
                eval_set=[(X_val_reg, y_val_reg)],
                verbose=False
            )

        end = time.perf_counter()

        print("Regressor training complete.")
        print(f"Training time: {end - start:.2f} seconds\n")

        self.is_fitted = True


    def train(
        self,
        splits,
        class_target_col,
        reg_target_col,
        class_tune_mode=ParamTuningMode.NONE,
        reg_tune_mode=ParamTuningMode.NONE
    ):
        self.train_classifier(splits, class_target_col, class_tune_mode)
        self.train_regressor(splits, class_target_col, reg_target_col, reg_tune_mode)


    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")

        y_prob = self.head1.predict_proba(X)[:, 1]
        y_class = (y_prob >= self.threshold).astype(int)
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
