import numpy as np
import time

from dataclasses import dataclass
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV


@dataclass
class Preds:
    class_prob: np.ndarray
    class_pred: np.ndarray
    reg_pred: np.ndarray
    expected_volume_pred: np.ndarray


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


class TwoHeadDebrisModel:
    def __init__(self, classifier=None, regressor=None, tune_hyperparams=False, log_regression_target=False, positive_only_regression=False):
        self.head1 = classifier
        self.head2 = regressor
        self.tune_hyperparams = tune_hyperparams
        self.threshold = 0.5
        self.best_f1 = None
        self.log_regression_target = log_regression_target
        self.positive_only_regression = positive_only_regression
        self.is_fitted = False


    def train_classifier(self, X_train, y_class_train, X_val, y_class_val):
        print("Training Classifier...")

        start = time.perf_counter()

        if self.tune_hyperparams:
            print("Tuning hyperparameters for classifier...")
            search = RandomizedSearchCV(
                estimator=self.head1,
                param_distributions=self.head1.get_params(),
                n_iter=20,
                scoring="roc_auc",
                cv=3,
                verbose=1,
                n_jobs=-1,
                random_state=12
            )
            search.fit(X_train, y_class_train)
            self.head1 = search.best_estimator_
            print(f"Best classifier params: {search.best_params_}")

        else:
            self.head1.fit(
                X_train,
                y_class_train,
                eval_set=[(X_val, y_class_val)],
                verbose=False
            )

        end = time.perf_counter()

        print("Classifier training complete.")
        print(f"Training time: {end - start:.2f} seconds\n")

        y_val_prob = self.head1.predict_proba(X_val)[:, 1]
        self.threshold, self.best_f1 = tune_threshold(y_class_val, y_val_prob)

        print(f"Chosen threshold: {self.threshold:.2f} (best F1 on validation: {self.best_f1:.4f})")


    def train_regressor(self, X_train, y_class_train, y_reg_train, X_val, y_class_val, y_reg_val):
        print("Training Regressor...")

        if self.positive_only_regression:
            train_mask = y_class_train == 1
            val_mask = y_class_val == 1

            X_train_reg = X_train.loc[train_mask]
            y_train_reg = y_reg_train.loc[train_mask]
            X_val_reg = X_val.loc[val_mask]
            y_val_reg = y_reg_val.loc[val_mask]
        else:
            X_train_reg = X_train
            y_train_reg = y_reg_train
            X_val_reg = X_val
            y_val_reg = y_reg_val

        if self.log_regression_target:
            y_train_reg = np.log1p(y_train_reg)
            y_val_reg = np.log1p(y_val_reg)

        start = time.perf_counter()

        if self.tune_hyperparams:
            print("Tuning hyperparameters for regressor...")
            search = RandomizedSearchCV(
                estimator=self.head2,
                param_distributions=self.head2.get_params(),
                n_iter=20,
                scoring="neg_mean_squared_error",
                cv=3,
                verbose=1,
                n_jobs=-1,
                random_state=12
            )
            search.fit(X_train_reg, y_train_reg)
            self.head2 = search.best_estimator_
            print(f"Best regressor params: {search.best_params_}")

        else:
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


    def train(self, X_train, y_class_train, y_reg_train, X_val, y_class_val, y_reg_val):
        self.train_classifier(X_train, y_class_train, X_val, y_class_val)
        self.train_regressor(X_train, y_class_train, y_reg_train, X_val, y_class_val, y_reg_val)


    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")
        return self.head1.predict_proba(X)[:, 1]


    def predict_class(self, X):
        y_prob = self.predict_proba(X)
        return (y_prob >= self.threshold).astype(int)


    def predict_reg(self, X):
        if not self.is_fitted:
            raise ValueError("Regressor Model must be trained before prediction.")

        y_pred = self.head2.predict(X)

        if self.log_regression_target:
            y_pred = np.expm1(y_pred)

        return y_pred


    def predict_expected_volume(self, X):
        y_prob = self.predict_proba(X)
        y_reg = self.predict_reg(X)
        return y_prob * y_reg


    def predict(self, X):
        y_prob = self.predict_proba(X)
        y_class = (y_prob >= self.threshold).astype(int)
        y_reg = self.predict_reg(X)
        y_expected = y_prob * y_reg

        return Preds(
            class_prob=y_prob,
            class_pred=y_class,
            reg_pred=y_reg,
            expected_volume_pred=y_expected
        )
