
import numpy as np
import time

from dataclasses import dataclass
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from resampling import apply_smote_single_label
from tune_mode import TuneMode
from logger import Log


log = Log()


@dataclass
class ClassifierTrainingResult:
    estimator: object
    training_time: float
    best_threshold: float
    best_f1: float


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


def scale_pos_weight(estimator, y_train_cls):
    neg_count = (y_train_cls == 0).sum()
    pos_count = (y_train_cls == 1).sum()

    if pos_count > 0:
        scale_pos_weight = neg_count / pos_count
    else:
        scale_pos_weight = 1.0

    estimator.set_params(scale_pos_weight=scale_pos_weight)
    log.info(f"Applied scale_pos_weight={scale_pos_weight:.2f} to classifier.")


def train_classifier(
    estimator,
    splits,
    param_dist,
    default_params,
    class_target_col,
    apply_smote,
    apply_scale_pos_weight,
    tune_mode
) -> ClassifierTrainingResult:
    log.info("Training Classifier...")

    X_train_cls = splits.X_train
    y_train_cls = splits.y_class_train[class_target_col]

    if tune_mode == TuneMode.NONE:
        estimator.set_params(**default_params)

    if apply_scale_pos_weight:
        scale_pos_weight(estimator, y_train_cls)

    if apply_smote:
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
        estimator.fit(
            X_train_cls,
            y_train_cls,
            eval_set=[(splits.X_val, splits.y_class_val[class_target_col])],
            verbose=False
        )

    end = time.perf_counter()

    log.info("Classifier training complete.")
    log.info(f"Training time: {end - start:.2f} seconds")

    y_val_prob = estimator.predict_proba(splits.X_val)[:, 1]
    threshold, best_f1 = tune_threshold(splits.y_class_val[class_target_col], y_val_prob)

    log.info(f"Chosen threshold: {threshold:.2f} (best F1 on validation: {best_f1:.4f})")

    return ClassifierTrainingResult(
        estimator=estimator,
        training_time=end - start,
        best_threshold=threshold,
        best_f1=best_f1
    )
