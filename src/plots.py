from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# General utilities
# ============================================================

def ensure_output_dir(output_dir):
    if output_dir is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_label(label):
    return str(label).lower().replace(" ", "_").replace("/", "_")


def save_figure(fig, output_path):
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot helpers for dashboard axes
# ============================================================

def plot_confusion_matrix_ax(y_true, y_pred, ax, title):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        normalize="true",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)


def plot_precision_recall_ax(y_true, y_prob, ax, title):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} (AUC={pr_auc:.3f})")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_expected_vs_actual_ax(y_true, y_pred, ax, title):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ax.scatter(y_true, y_pred, alpha=0.35)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    if lo == hi:
        hi = lo + 1.0

    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2)
    ax.set_xlabel("Actual Volume")
    ax.set_ylabel("Predicted Volume")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def plot_feature_importance_ax(model, feature_names, ax, title, top_n=15):
    importances = np.asarray(model.feature_importances_)
    indices = np.argsort(importances)[::-1][: min(top_n, len(feature_names))]

    ax.bar(range(len(indices)), importances[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    ax.set_title(title)


def print_top_features(model, feature_names, label, top_n=5):
    importances = np.asarray(model.feature_importances_)
    indices = np.argsort(importances)[::-1][: min(top_n, len(feature_names))]

    print(f"Top {len(indices)} features for {label}:")
    for i in indices:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")
    print()


# ============================================================
# Dashboard
# ============================================================

def save_dashboard(
    preds,
    y_class_test,
    y_reg_test,
    X_train,
    model,
    debris_type,
    output_dir,
):
    output_dir = ensure_output_dir(output_dir)
    if output_dir is None:
        return

    y_class_test = np.asarray(y_class_test)
    y_reg_test = np.asarray(y_reg_test)

    y_prob = np.asarray(preds.class_prob)
    y_class_pred = np.asarray(preds.class_pred)
    y_reg_pred = np.asarray(preds.reg_pred)
    y_expected_pred = np.asarray(preds.expected_volume_pred)

    positive_mask = (y_class_test == 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_confusion_matrix_ax(
        y_class_test,
        y_class_pred,
        axes[0],
        f"{debris_type.upper()} Confusion Matrix",
    )

    plot_precision_recall_ax(
        y_class_test,
        y_prob,
        axes[1],
        f"{debris_type.upper()} PR Curve",
    )

    plot_expected_vs_actual_ax(
        y_reg_test,
        y_expected_pred,
        axes[2],
        f"{debris_type.upper()} Expected Volume",
    )

    if positive_mask.sum() > 0:
        plot_expected_vs_actual_ax(
            y_reg_test[positive_mask],
            y_reg_pred[positive_mask],
            axes[3],
            f"{debris_type.upper()} Positive-Only Volume",
        )
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "No positive test samples", ha="center", va="center")

    plot_feature_importance_ax(
        model.head1,
        X_train.columns,
        axes[4],
        f"{debris_type.upper()} Classifier Importance",
        top_n=15,
    )

    plot_feature_importance_ax(
        model.head2,
        X_train.columns,
        axes[5],
        f"{debris_type.upper()} Regressor Importance",
        top_n=15,
    )

    fig.suptitle(f"{debris_type.upper()} Model Dashboard", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    dashboard_path = output_dir / f"{safe_label(debris_type)}_dashboard.png"
    save_figure(fig, dashboard_path)


# ============================================================
# Main public entry point
# ============================================================

def show_metrics(
    preds,
    y_class_test,
    y_reg_test,
    X_train,
    model,
    debris_type,
    output_dir=None,
    save_dashboard_plot=True,
):
    y_class_test = np.asarray(y_class_test)
    y_reg_test = np.asarray(y_reg_test)

    y_prob = np.asarray(preds.class_prob)
    y_class_pred = np.asarray(preds.class_pred)
    y_reg_pred = np.asarray(preds.reg_pred)
    y_expected_pred = np.asarray(preds.expected_volume_pred)

    positive_mask = (y_class_test == 1)

    # ----- Classification -----
    class_metrics = compute_classification_metrics(y_class_test, y_class_pred, y_prob)
    class_metrics.print(f"{debris_type.upper()}")
    print_top_features(model.head1, X_train.columns, f"{debris_type.upper()} classifier")

    # ----- Overall expected-volume regression -----
    overall_reg_metrics = compute_regression_metrics(y_reg_test, y_expected_pred)
    overall_reg_metrics.print(f"{debris_type.upper()} Expected Volume")

    # ----- Positive-only regression -----
    if positive_mask.sum() > 0:
        pos_reg_metrics = compute_regression_metrics(
            y_reg_test[positive_mask],
            y_reg_pred[positive_mask],
        )
        pos_reg_metrics.print(f"{debris_type.upper()} Positive-Only Volume")
        print_top_features(model.head2, X_train.columns, f"{debris_type.upper()} regressor")
    else:
        print(f"No positive {debris_type} cases in test set; skipping positive-only regression metrics.\n")

    # ----- Dashboard -----
    if save_dashboard_plot:
        save_dashboard(
            preds=preds,
            y_class_test=y_class_test,
            y_reg_test=y_reg_test,
            X_train=X_train,
            model=model,
            debris_type=debris_type,
            output_dir=output_dir,
        )
