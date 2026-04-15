
import numpy as np
import time

from dataclasses import dataclass
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from resampling import apply_smote_single_label
from tune_mode import TuneMode
from logger import Log
from split import Splits


log = Log()


@dataclass
class ClassifierTrainingResult:
    """Store the result of classification-head training.

    Attributes:
        estimator: Fitted classifier.
        best_threshold: Validation threshold selected by F1.
    """

    estimator: object
    training_time: float
    best_threshold: float
    best_f1: float


def tune_threshold(y_true, y_prob):
    """Select the classification threshold with best F1."""

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


def apply_class_imbalance_setting(estimator, y_train_cls):
    """Set a class imbalance parameter when the estimator supports one."""

    neg_count = (y_train_cls == 0).sum()
    pos_count = (y_train_cls == 1).sum()

    if pos_count > 0:
        scale_pos_weight = neg_count / pos_count
    else:
        scale_pos_weight = 1.0

    if estimator.__class__.__module__.startswith("xgboost"):
        estimator.set_params(scale_pos_weight=scale_pos_weight)
        log.info(f"Applied scale_pos_weight={scale_pos_weight:.2f} to classifier.")
    elif hasattr(estimator, "get_params") and "class_weight" in estimator.get_params():
        estimator.set_params(class_weight="balanced")
        log.info("Applied class_weight='balanced' to classifier.")
    else:
        log.info("No classifier imbalance parameter applied.")


def _fit_classifier_estimator(estimator, X_train, y_train, X_val, y_val):
    if estimator.__class__.__module__.startswith("xgboost"):
        estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        estimator.fit(X_train, y_train)


def train_classifier(
    estimator,
    splits: Splits,
    param_dist,
    default_params,
    class_target_col,
    apply_scale_pos_weight,
    tune_mode
) -> ClassifierTrainingResult:
    """Train a classification head for one target.

    Args:
        splits: Classification feature matrices and targets.
    """

    log.info("Training Classifier...")

    X_train_cls = splits.X_train_class
    y_train_cls = splits.y_train_class

    if tune_mode == TuneMode.NONE:
        estimator.set_params(**default_params)

    if apply_scale_pos_weight:
        apply_class_imbalance_setting(estimator, y_train_cls)

    start = time.perf_counter()

    if tune_mode == TuneMode.RANDOM_SEARCH:
        log.info("Beginning RandomizedSearch hyperparameter tuning for classifier...")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=10,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            random_state=12
        )
        search.fit(X_train_cls, y_train_cls)
        estimator = search.best_estimator_
        log.info(f"Best classifier params: {search.best_params_}")

    elif tune_mode == TuneMode.GRID_SEARCH:
        log.info("Beginning GridSearch hyperparameter tuning for classifier...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_dist,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1
        )
        search.fit(X_train_cls, y_train_cls)
        estimator = search.best_estimator_
        log.info(f"Best classifier params: {search.best_params_}")

    elif tune_mode == TuneMode.NONE:
        _fit_classifier_estimator(
            estimator,
            X_train_cls,
            y_train_cls,
            splits.X_val_class,
            splits.y_val_class
        )

    end = time.perf_counter()

    log.info("Classifier training complete.")
    log.info(f"Training time: {end - start:.2f} seconds")

    y_val_prob = estimator.predict_proba(splits.X_val_class)[:, 1]
    threshold, best_f1 = tune_threshold(splits.y_val_class, y_val_prob)

    log.info(f"Chosen threshold: {threshold:.2f} (best F1 on validation: {best_f1:.4f})")

    return ClassifierTrainingResult(
        estimator=estimator,
        training_time=end - start,
        best_threshold=threshold,
        best_f1=best_f1
    )
