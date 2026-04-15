
import numpy as np
import time

from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tune_mode import TuneMode
from logger import Log
from split import Splits


log = Log()


@dataclass
class RegressorTrainingResult:
    """Store the result of regression-head training.

    Attributes:
        estimator: Fitted estimator, or None when skipped.
        training_time: Training duration in seconds.
    """

    estimator: object | None
    training_time: float


def apply_log_transform(y_train_reg, y_val_reg):
    y_train_reg = np.log1p(y_train_reg)
    y_val_reg = np.log1p(y_val_reg)

    return y_train_reg, y_val_reg


def train_regressor(
    estimator,
    splits: Splits,
    param_dist,
    default_params,
    reg_target_col,
    log_target,
    tune_mode
) -> RegressorTrainingResult:
    """Train a regression head for one target.

    Args:
        splits: Regression feature matrices and targets.
    """

    log.info("Training Regressor...")

    X_train_reg = splits.X_train_reg
    X_val_reg = splits.X_val_reg
    y_train_reg = splits.y_train_reg
    y_val_reg = splits.y_val_reg

    if len(X_train_reg) == 0:
        log.warn(f"Skipping regressor for {reg_target_col}: no positive training rows.")
        return RegressorTrainingResult(estimator=None, training_time=0.0)

    if log_target:
        if (y_train_reg < 0).any() or (y_val_reg < 0).any():
            raise ValueError(f"log_target_reg requires nonnegative regression targets for {reg_target_col}.")
        y_train_reg, y_val_reg = apply_log_transform(y_train_reg, y_val_reg)

    start = time.perf_counter()

    if tune_mode == TuneMode.RANDOM_SEARCH:
        log.info("Beginning RandomizedSearch hyperparameter tuning for regressor...")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=20,
            scoring="neg_mean_squared_error",
            cv=5,
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
            cv=5,
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
