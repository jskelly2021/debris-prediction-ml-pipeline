import pandas as pd


CLASSIFICATION_SORT_COLUMNS = ["f1", "roc_auc", "precision", "_experiment_order"]
CLASSIFICATION_ASCENDING = [False, False, False, True]
REGRESSION_SORT_COLUMNS = ["r2", "rmse", "mae", "_experiment_order"]
REGRESSION_ASCENDING = [False, True, True, True]
BEST_CLASSIFICATION_COLUMNS = [
    "label",
    "experiment_name",
    "run_id",
    "f1",
    "roc_auc",
    "precision",
]
BEST_REGRESSION_COLUMNS = [
    "label",
    "experiment_name",
    "run_id",
    "r2",
    "rmse",
    "mae",
    "nrmse",
    "cov",
    "percent_error_mean",
]


def prepare_results_frame(
    combined_results: pd.DataFrame,
    experiment_order: list[str],
) -> pd.DataFrame:
    """Attach deterministic ordering helpers to combined results."""

    frame = combined_results.copy()
    order_lookup = {name: index for index, name in enumerate(experiment_order)}
    frame["_experiment_order"] = frame["experiment_name"].map(order_lookup).fillna(len(order_lookup))
    frame["_label_order"] = range(len(frame))
    return frame


def build_summary_by_label(
    combined_results: pd.DataFrame,
    experiment_order: list[str],
) -> pd.DataFrame:
    """Build one summary row per label."""

    if combined_results.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "n_runs",
                "best_f1",
                "best_roc_auc",
                "best_r2",
                "lowest_rmse",
                "lowest_nrmse",
                "lowest_percent_error_mean",
                "best_f1_experiment",
                "best_r2_experiment",
            ]
        )

    label_order = _label_order_frame(combined_results)
    best_f1_experiments = build_best_classification_runs(
        combined_results=combined_results,
        experiment_order=experiment_order,
    )[["label", "experiment_name"]].rename(columns={"experiment_name": "best_f1_experiment"})
    best_r2_experiments = build_best_regression_runs(
        combined_results=combined_results,
        experiment_order=experiment_order,
    )[["label", "experiment_name"]].rename(columns={"experiment_name": "best_r2_experiment"})

    summary = (
        combined_results.groupby("label", sort=False)
        .agg(
            n_runs=("experiment_name", "count"),
            best_f1=("f1", "max"),
            best_roc_auc=("roc_auc", "max"),
            best_r2=("r2", "max"),
            lowest_rmse=("rmse", "min"),
            lowest_nrmse=("nrmse", "min"),
            lowest_percent_error_mean=("percent_error_mean", "min"),
        )
        .reset_index()
    )

    summary = summary.merge(best_f1_experiments, on="label", how="left")
    summary = summary.merge(best_r2_experiments, on="label", how="left")
    summary = summary.merge(label_order, on="label", how="left")
    summary = summary.sort_values("_label_position").drop(columns="_label_position")
    return summary


def build_summary_by_experiment(
    combined_results: pd.DataFrame,
    experiment_order: list[str],
) -> pd.DataFrame:
    """Build one summary row per experiment."""

    if combined_results.empty:
        return pd.DataFrame(
            columns=[
                "experiment_name",
                "run_id",
                "n_labels",
                "mean_f1",
                "mean_roc_auc",
                "mean_r2",
                "mean_rmse",
                "mean_nrmse",
                "mean_cov",
                "mean_percent_error_mean",
                "classifier_model",
                "regressor_model",
                "n_features",
            ]
        )

    summary = (
        combined_results.groupby("experiment_name", sort=False)
        .agg(
            run_id=("run_id", "first"),
            n_labels=("label", "nunique"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_r2=("r2", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_nrmse=("nrmse", "mean"),
            mean_cov=("cov", "mean"),
            mean_percent_error_mean=("percent_error_mean", "mean"),
            classifier_model=("classifier_model", "first"),
            regressor_model=("regressor_model", "first"),
            n_features=("n_features", "first"),
        )
        .reset_index()
    )

    order_lookup = {name: index for index, name in enumerate(experiment_order)}
    summary["_experiment_order"] = summary["experiment_name"].map(order_lookup).fillna(len(order_lookup))
    summary = summary.sort_values("_experiment_order").drop(columns="_experiment_order")
    return summary


def build_summary_by_experiment_label(
    combined_results: pd.DataFrame,
    experiment_order: list[str],
) -> pd.DataFrame:
    """Build one summary row per experiment and label."""

    columns = [
        "experiment_name",
        "run_id",
        "label",
        "f1",
        "roc_auc",
        "r2",
        "rmse",
        "nrmse",
        "cov",
        "percent_error_mean",
        "classifier_model",
        "regressor_model",
        "n_features",
    ]
    if combined_results.empty:
        return pd.DataFrame(columns=columns)

    label_order = _label_order_frame(combined_results)
    summary = combined_results[columns].copy()

    order_lookup = {name: index for index, name in enumerate(experiment_order)}
    summary["_experiment_order"] = summary["experiment_name"].map(order_lookup).fillna(len(order_lookup))
    summary = summary.merge(label_order, on="label", how="left")
    summary = summary.sort_values(
        ["_experiment_order", "_label_position"],
        kind="mergesort",
    ).drop(columns=["_experiment_order", "_label_position"])
    return summary.reset_index(drop=True)


def build_best_classification_runs(
    combined_results: pd.DataFrame,
    experiment_order: list[str],
) -> pd.DataFrame:
    """Select the best classification row per label."""

    if combined_results.empty:
        return pd.DataFrame(columns=BEST_CLASSIFICATION_COLUMNS)

    ranked = prepare_results_frame(combined_results, experiment_order)
    label_order = _label_order_frame(combined_results)
    ranked = ranked.merge(label_order, on="label", how="left")
    ranked = ranked.sort_values(
        by=["label", *CLASSIFICATION_SORT_COLUMNS],
        ascending=[True, *CLASSIFICATION_ASCENDING],
        kind="mergesort",
    )
    best = ranked.groupby("label", sort=False).head(1).copy()
    best = best.sort_values("_label_position")
    best = best[BEST_CLASSIFICATION_COLUMNS]
    return best.reset_index(drop=True)


def build_best_regression_runs(
    combined_results: pd.DataFrame,
    experiment_order: list[str],
) -> pd.DataFrame:
    """Select the best regression row per label."""

    if combined_results.empty:
        return pd.DataFrame(columns=BEST_REGRESSION_COLUMNS)

    ranked = prepare_results_frame(combined_results, experiment_order)
    label_order = _label_order_frame(combined_results)
    ranked = ranked.merge(label_order, on="label", how="left")
    ranked = ranked.sort_values(
        by=["label", *REGRESSION_SORT_COLUMNS],
        ascending=[True, *REGRESSION_ASCENDING],
        kind="mergesort",
    )
    best = ranked.groupby("label", sort=False).head(1).copy()
    best = best.sort_values("_label_position")
    best = best[BEST_REGRESSION_COLUMNS]
    return best.reset_index(drop=True)


def _label_order_frame(combined_results: pd.DataFrame) -> pd.DataFrame:
    """Build a deterministic label ordering frame from source row order."""

    return (
        combined_results[["label"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(_label_position=lambda df: df.index)
    )
