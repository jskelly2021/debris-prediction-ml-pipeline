import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
)


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

    if len(y_true) == 0 or len(y_pred) == 0:
        ax.set_title(title)
        ax.set_xlabel("Actual Volume")
        ax.set_ylabel("Predicted Volume")
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.grid(True, linestyle="--", alpha=0.4)
        return

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


def save_label_dashboard(
    pred_df,
    splits,
    pipeline,
    class_target_col,
    reg_target_col,
    label_name,
    output_dir,
):
    output_dir = ensure_output_dir(output_dir)
    if output_dir is None:
        return

    y_class_test = np.asarray(splits.y_class_test[class_target_col])
    y_reg_test = np.asarray(splits.y_reg_test[reg_target_col])

    y_prob = np.asarray(pred_df[f"{label_name}_class_prob"])
    y_class_pred = np.asarray(pred_df[f"{label_name}_class_pred"])
    y_reg_pred = np.asarray(pred_df[f"{label_name}_reg_pred"])
    y_expected_pred = np.asarray(pred_df[f"{label_name}_expected_volume_pred"])

    positive_mask = (y_class_test == 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_confusion_matrix_ax(
        y_class_test,
        y_class_pred,
        axes[0],
        f"{label_name.upper()} Confusion Matrix"
    )

    plot_precision_recall_ax(
        y_class_test,
        y_prob,
        axes[1],
        f"{label_name.upper()} PR Curve"
    )

    plot_expected_vs_actual_ax(
        y_reg_test,
        y_expected_pred,
        axes[2],
        f"{label_name.upper()} Expected Volume"
    )

    if positive_mask.sum() > 0:
        plot_expected_vs_actual_ax(
            y_reg_test[positive_mask],
            y_reg_pred[positive_mask],
            axes[3],
            f"{label_name.upper()} Positive-Only Volume",
        )
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "No positive test samples", ha="center", va="center")

    plot_feature_importance_ax(
        pipeline.head1,
        splits.X_train.columns,
        axes[4],
        f"{label_name.upper()} Classifier Importance",
        top_n=15,
    )

    if pipeline.head2 is not None:
        plot_feature_importance_ax(
            pipeline.head2,
            splits.X_train.columns,
            axes[5],
            f"{label_name.upper()} Regressor Importance",
            top_n=15,
        )
    else:
        axes[5].axis("off")
        axes[5].text(0.5, 0.5, "No regressor trained", ha="center", va="center")

    fig.suptitle(f"{label_name.upper()} Model Dashboard", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    dashboard_path = output_dir / f"{safe_label(label_name)}_dashboard.png"
    save_figure(fig, dashboard_path)


def save_multilabel_dashboards(multilabel_model, splits, output_dir):
    pred_df = multilabel_model.predict(splits.X_test)

    for label_name, info in multilabel_model.models.items():
        save_label_dashboard(
            pred_df=pred_df,
            splits=splits,
            pipeline=info["pipeline"],
            class_target_col=info["class_col"],
            reg_target_col=info["reg_col"],
            label_name=label_name,
            output_dir=output_dir,
        )
