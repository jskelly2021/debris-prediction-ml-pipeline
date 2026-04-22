from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_FILENAMES = {
    "f1_by_experiment": "f1_by_experiment.png",
    "roc_auc_by_experiment": "roc_auc_by_experiment.png",
    "r2_by_experiment": "r2_by_experiment.png",
    "rmse_by_experiment": "rmse_by_experiment.png",
    "f1_vs_r2_scatter": "f1_vs_r2_scatter.png",
}


def save_comparison_figures(
    combined_results: pd.DataFrame,
    figures_dir: Path,
    experiment_order: list[str],
) -> dict[str, Path]:
    """Generate and save all packet comparison figures."""

    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = {
        "f1_by_experiment": figures_dir / FIGURE_FILENAMES["f1_by_experiment"],
        "roc_auc_by_experiment": figures_dir / FIGURE_FILENAMES["roc_auc_by_experiment"],
        "r2_by_experiment": figures_dir / FIGURE_FILENAMES["r2_by_experiment"],
        "rmse_by_experiment": figures_dir / FIGURE_FILENAMES["rmse_by_experiment"],
        "f1_vs_r2_scatter": figures_dir / FIGURE_FILENAMES["f1_vs_r2_scatter"],
    }

    save_grouped_metric_plot(
        combined_results=combined_results,
        metric="f1",
        title="F1 by Experiment",
        ylabel="F1",
        out_path=figure_paths["f1_by_experiment"],
        experiment_order=experiment_order,
    )
    save_grouped_metric_plot(
        combined_results=combined_results,
        metric="roc_auc",
        title="ROC AUC by Experiment",
        ylabel="ROC AUC",
        out_path=figure_paths["roc_auc_by_experiment"],
        experiment_order=experiment_order,
    )
    save_grouped_metric_plot(
        combined_results=combined_results,
        metric="r2",
        title="R2 by Experiment",
        ylabel="R2",
        out_path=figure_paths["r2_by_experiment"],
        experiment_order=experiment_order,
    )
    save_grouped_metric_plot(
        combined_results=combined_results,
        metric="rmse",
        title="RMSE by Experiment",
        ylabel="RMSE",
        out_path=figure_paths["rmse_by_experiment"],
        experiment_order=experiment_order,
    )
    save_f1_vs_r2_scatter(
        combined_results=combined_results,
        out_path=figure_paths["f1_vs_r2_scatter"],
        experiment_order=experiment_order,
    )

    return figure_paths


def save_grouped_metric_plot(
    combined_results: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    experiment_order: list[str],
) -> None:
    """Save a grouped bar chart for one metric."""

    ordered = _ordered_results(combined_results, experiment_order)
    if ordered.empty:
        _save_empty_figure(out_path=out_path, title=title, message="No data available")
        return

    labels = list(ordered["label"].drop_duplicates())
    experiments = [name for name in experiment_order if name in set(ordered["experiment_name"])]

    pivot = (
        ordered.pivot(index="experiment_name", columns="label", values=metric)
        .reindex(index=experiments, columns=labels)
    )

    fig, ax = plt.subplots(figsize=(max(7, len(experiments) * 1.2), 5))
    x_positions = np.arange(len(pivot.index))
    width = 0.8 / max(len(labels), 1)

    for index, label in enumerate(labels):
        offset = (index - (len(labels) - 1) / 2) * width
        values = pivot[label].to_numpy()
        ax.bar(x_positions + offset, values, width=width, label=label)

    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.legend(title="Label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_f1_vs_r2_scatter(
    combined_results: pd.DataFrame,
    out_path: Path,
    experiment_order: list[str],
) -> None:
    """Save the F1 vs R2 scatter plot."""

    ordered = _ordered_results(combined_results, experiment_order)
    scatter_df = ordered.dropna(subset=["f1", "r2"]).copy()
    if scatter_df.empty:
        _save_empty_figure(out_path=out_path, title="F1 vs R2", message="No data available")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(scatter_df["f1"], scatter_df["r2"])

    for row in scatter_df.itertuples(index=False):
        ax.annotate(
            f"{row.experiment_name}:{row.label}",
            (row.f1, row.r2),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_title("F1 vs R2")
    ax.set_xlabel("F1")
    ax.set_ylabel("R2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _ordered_results(combined_results: pd.DataFrame, experiment_order: list[str]) -> pd.DataFrame:
    """Return results sorted by manifest order then source order."""

    if combined_results.empty:
        return combined_results.copy()

    ordered = combined_results.copy()
    order_lookup = {name: index for index, name in enumerate(experiment_order)}
    ordered["_experiment_order"] = ordered["experiment_name"].map(order_lookup).fillna(len(order_lookup))
    ordered["_row_order"] = range(len(ordered))
    ordered = ordered.sort_values(["_experiment_order", "_row_order"]).drop(
        columns=["_experiment_order", "_row_order"]
    )
    return ordered


def _save_empty_figure(out_path: Path, title: str, message: str) -> None:
    """Save a placeholder figure when there is no plottable data."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
