
import numpy as np

from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Store binary classification metric values."""

    positive_rate: float
    n_samples: int
    n_positive: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self):
        """Return metrics as a plain dictionary."""

        return {
            "positive_rate": self.positive_rate,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


@dataclass
class RegressionMetrics:
    """Store regression metric values."""

    n_samples: int
    rmse: float
    mae: float
    r2: float

    def to_dict(self):
        """Return metrics as a plain dictionary."""

        return {
            "n_samples": self.n_samples,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
        }


def compute_classification_metrics(y_true, y_pred, y_prob):
    """Compute binary classification metrics from predictions."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    return ClassificationMetrics(
        positive_rate=float(np.mean(y_true)),
        n_samples=int(len(y_true)),
        n_positive=int(np.sum(y_true == 1)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
    )


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics from predicted values."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return RegressionMetrics(
            n_samples=0,
            rmse=np.nan,
            mae=np.nan,
            r2=np.nan,
        )

    return RegressionMetrics(
        n_samples=int(len(y_true)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    )
