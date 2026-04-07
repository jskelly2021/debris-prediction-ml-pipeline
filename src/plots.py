from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from logger import Log


log = Log()


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _safe_feature_importances(model, feature_names):
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({
        "feature": list(feature_names),
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return df.reset_index(drop=True)


def print_top_features(model, feature_names, label, top_n=10):
    fi = _safe_feature_importances(model, feature_names)

    if fi.empty:
        log.warn(f"No feature importances available for {label}.")
        return

    log.h2(f"Top {top_n} features for {label}")
    print(fi.head(top_n).to_string(index=False))
    print()


def save_feature_importance_plot(model, feature_names, title, out_path, top_n=15):
    fi = _safe_feature_importances(model, feature_names)

    if fi.empty:
        log.warn(f"Skipping feature importance plot for {title}: no importances available.")
        return

    fi = fi.head(top_n)
    fi = fi.iloc[::-1]  # reverse so largest appears at top in barh

    plt.figure(figsize=(10, 6))
    plt.barh(fi["feature"], fi["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_plot(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_plot(y_true, y_prob, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_precision_recall_curve_plot(y_true, y_prob, title, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_actual_vs_predicted_plot(y_true, y_pred, title, out_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    max_val = max(float(np.max(y_true)), float(np.max(y_pred))) if len(y_true) else 1.0

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_residuals_vs_predicted_plot(y_true, y_pred, title, out_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_residual_histogram_plot(y_true, y_pred, title, out_path, bins=30):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=bins)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_multilabel_dashboards(model, output_dir, top_n_features=15):
    """
    Saves all useful plots for each label/head using the model's current test splits.

    Expected model interface:
      - model.models[label_name]["pipeline"]
      - model.splits[label_name]
      - split has X_test_class, X_test_reg, y_test_class, y_test_reg
      - pipeline.predict_df(X, prefix=label_name)
      - pipeline.head1 / pipeline.head2
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    if not getattr(model, "is_fitted", False):
        raise ValueError("Model must be fitted before saving plots.")

    for label_name, info in model.models.items():
        log.h1(f"Saving plots for {label_name}")

        pipeline = info["pipeline"]
        split = model.splits[label_name]

        label_dir = output_dir / label_name
        _ensure_dir(label_dir)

        feature_names = split.X_train_class.columns

        # -------- Classification plots --------
        class_pred_df = pipeline.predict_df(split.X_test_class, prefix=label_name)

        y_true_class = split.y_test_class
        y_pred_class = class_pred_df[f"{label_name}_class_pred"]
        y_prob_class = class_pred_df[f"{label_name}_class_prob"]

        save_confusion_matrix_plot(
            y_true=y_true_class,
            y_pred=y_pred_class,
            title=f"{label_name} Confusion Matrix",
            out_path=label_dir / "confusion_matrix.png",
        )

        save_roc_curve_plot(
            y_true=y_true_class,
            y_prob=y_prob_class,
            title=f"{label_name} ROC Curve",
            out_path=label_dir / "roc_curve.png",
        )

        save_precision_recall_curve_plot(
            y_true=y_true_class,
            y_prob=y_prob_class,
            title=f"{label_name} Precision-Recall Curve",
            out_path=label_dir / "precision_recall_curve.png",
        )

        save_feature_importance_plot(
            model=pipeline.head1,
            feature_names=feature_names,
            title=f"{label_name} Classifier Feature Importance",
            out_path=label_dir / "classifier_feature_importance.png",
            top_n=top_n_features,
        )

        # -------- Regression plots --------
        # Use regression split explicitly so this stays correct if class/reg features diverge later.
        reg_pred_df = pipeline.predict_df(split.X_test_reg, prefix=label_name)

        y_true_reg = split.y_test_reg
        y_pred_reg = reg_pred_df[f"{label_name}_expected_volume_pred"]

        save_actual_vs_predicted_plot(
            y_true=y_true_reg,
            y_pred=y_pred_reg,
            title=f"{label_name} Actual vs Expected Volume Predicted",
            out_path=label_dir / "actual_vs_predicted.png",
        )

        save_residuals_vs_predicted_plot(
            y_true=y_true_reg,
            y_pred=y_pred_reg,
            title=f"{label_name} Residuals vs Predicted",
            out_path=label_dir / "residuals_vs_predicted.png",
        )

        save_residual_histogram_plot(
            y_true=y_true_reg,
            y_pred=y_pred_reg,
            title=f"{label_name} Residual Histogram",
            out_path=label_dir / "residual_histogram.png",
        )

        if pipeline.head2 is not None:
            save_feature_importance_plot(
                model=pipeline.head2,
                feature_names=feature_names,
                title=f"{label_name} Regressor Feature Importance",
                out_path=label_dir / "regressor_feature_importance.png",
                top_n=top_n_features,
            )

    log.info(f"Saved plots to: {output_dir}")
