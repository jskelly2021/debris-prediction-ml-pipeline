
import numpy as np
import time

from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tune_mode import TuneMode
from logger import Log


log = Log()


@dataclass
class RegressorTrainingResult:
    estimator: object
    training_time: float


def apply_positive_only_mask(splits, positive_only, class_target_col, reg_target_col):
    if positive_only:
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

    return X_train_reg, y_train_reg, X_val_reg, y_val_reg


def apply_log_transform(y_train_reg, y_val_reg):
    y_train_reg = np.log1p(y_train_reg)
    y_val_reg = np.log1p(y_val_reg)

    return y_train_reg, y_val_reg


def train_regressor(
    estimator,
    splits,
    param_dist,
    default_params,
    class_target_col,
    reg_target_col,
    positive_only,
    log_target,
    tune_mode
) -> RegressorTrainingResult:

    log.info("Training Regressor...")

    X_train_reg, y_train_reg, X_val_reg, y_val_reg = apply_positive_only_mask(
        splits=splits,
        positive_only=positive_only,
        class_target_col=class_target_col,
        reg_target_col=reg_target_col
    )

    if log_target:
        y_train_reg, y_val_reg = apply_log_transform(y_train_reg, y_val_reg)

    if len(X_train_reg) == 0:
        log.warn(f"Skipping regressor for {reg_target_col}: no positive training rows.")
        return None

    start = time.perf_counter()

    if tune_mode == TuneMode.RANDOM_SEARCH:
        log.info("Beginning RandomizedSearch hyperparameter tuning for regressor...")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=20,
            scoring="neg_mean_squared_error",
            cv=3,
            n_jobs=-1,
            random_state=12
        )
        search.fit(X_train_reg, y_train_reg)
        estimator = search.best_estimator_
        log.info(f"Best regressor params: {search.best_params_}")

    elif tune_mode == TuneMode.GRID_SEARCH:
        log.info("Beginning GridSearch hyperparameter tuning for regressor...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_dist,
            scoring="neg_mean_squared_error",
            cv=3,
            n_jobs=-1
        )
        search.fit(X_train_reg, y_train_reg)
        estimator = search.best_estimator_
        log.info(f"Best regressor params: {search.best_params_}")

    elif tune_mode == TuneMode.NONE:
        estimator.set_params(**default_params)
        estimator.fit(
            X_train_reg,
            y_train_reg,
            eval_set=[(X_val_reg, y_val_reg)],
            verbose=False
        )

    end = time.perf_counter()

    log.info("Regressor training complete.")
    log.info(f"Training time: {end - start:.2f} seconds")

    return RegressorTrainingResult(
        estimator=estimator,
        training_time=end - start
    )
