
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
    positive_rate: float
    n_samples: int
    n_positive: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def print(self, label):
        print(f"=== {label} Classification Metrics ===")
        print(f"Positive Rate: {self.positive_rate}")
        print(f"N Samples    : {self.n_samples}")
        print(f"N Positive   : {self.n_positive}")
        print(f"Accuracy     : {self.accuracy}")
        print(f"Precision    : {self.precision}")
        print(f"Recall       : {self.recall}")
        print(f"F1 Score     : {self.f1}")
        print(f"ROC AUC      : {self.roc_auc}")
        print()


@dataclass
class RegressionMetrics:
    n_samples: int
    rmse: float
    mae: float
    r2: float

    def print(self, label):
        print(f"=== {label} Regression Metrics ===")
        print(f"N Samples: {self.n_samples}")
        print(f"RMSE     : {self.rmse}")
        print(f"MAE      : {self.mae}")
        print(f"R²       : {self.r2}")
        print()


def compute_classification_metrics(y_true, y_pred, y_prob):
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return RegressionMetrics(
        n_samples=int(len(y_true)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    )
