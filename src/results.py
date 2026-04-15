from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from logger import Log
from plots import print_top_features
from split import Splits


log = Log()


def append_df_to_csv(df: pd.DataFrame, path: Path):
    """Append a DataFrame to a CSV, creating it if needed."""

    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)


def print_metrics(metrics):
    """Print classification and regression metrics by label."""

    if metrics is None:
        raise ValueError("No metrics available. Run evaluate() first.")

    for label_name, label_metrics in metrics.items():
        classification = label_metrics.classification
        regression = label_metrics.regression

        log.h1(f"Metrics for {label_name}")
        log.h2(f"{label_name} Classification Metrics")
        log.body(f"Positive Rate: {classification.positive_rate}")
        log.body(f"N Samples    : {classification.n_samples}")
        log.body(f"N Positive   : {classification.n_positive}")
        log.body(f"Accuracy     : {classification.accuracy}")
        log.body(f"Precision    : {classification.precision}")
        log.body(f"Recall       : {classification.recall}")
        log.body(f"F1 Score     : {classification.f1}")
        log.body(f"ROC AUC      : {classification.roc_auc}")

        log.h2(f"{label_name} {label_metrics.regression_display_name} Regression Metrics")
        log.body(f"N Samples: {regression.n_samples}")
        log.body(f"RMSE     : {regression.rmse}")
        log.body(f"MAE      : {regression.mae}")
        log.body(f"R²       : {regression.r2}")


def metrics_to_dataframe(metrics, experiment_config: ExperimentConfig, run_id=None, n_features=None):
    """Convert label metrics to a run-summary DataFrame."""

    if metrics is None:
        raise ValueError("No metrics available. Run evaluate() first.")

    rows = []

    for label_name, label_metrics in metrics.items():
        c = label_metrics.classification
        r = label_metrics.regression

        label_n_features = None if n_features is None else n_features.get(label_name)

        rows.append({
            "run_id": run_id,
            "label": label_name,
            "class_target": label_metrics.class_col,
            "reg_target": label_metrics.reg_col,
            "n_features": label_n_features,

            "smote": experiment_config.training.smote,
            "scale_pos_weight": experiment_config.training.scale_pos_weight,
            "log_features": experiment_config.preprocessing.log_features,
            "log_target_reg": experiment_config.training.log_target_reg,
            "outlier_threshold": experiment_config.training.outlier_threshold,

            "positive_rate": c.positive_rate,
            "class_n_samples": c.n_samples,
            "n_positive": c.n_positive,
            "accuracy": c.accuracy,
            "precision": c.precision,
            "recall": c.recall,
            "f1": c.f1,
            "roc_auc": c.roc_auc,

            "reg_n_samples": r.n_samples,
            "rmse": r.rmse,
            "mae": r.mae,
            "r2": r.r2,
        })

    return pd.DataFrame(rows)


def save_metrics_outputs(metrics, splits, experiment_config: ExperimentConfig, output_path, run_id):
    """Save aggregate and per-label metric CSV outputs."""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    n_features = {
        label_name: splits[label_name].X_test_class.shape[1]
        for label_name in metrics
    }

    metrics_df = metrics_to_dataframe(
        metrics=metrics,
        experiment_config=experiment_config,
        run_id=run_id,
        n_features=n_features,
    )

    append_df_to_csv(metrics_df, output_path / "all_runs.csv")

    for label_name in metrics_df["label"].unique():
        label_df = metrics_df[metrics_df["label"] == label_name]
        append_df_to_csv(label_df, output_path / f"{label_name}_metrics.csv")


def print_feature_importance(model, splits: dict[str, Splits], top_n=10):
    """Print top classifier and regressor features by label."""

    if not getattr(model, "is_fitted", False):
        raise ValueError("Model must be fitted before printing feature importance.")

    for label_name, info in model.models.items():
        pipeline = info["pipeline"]
        class_feature_names = pipeline.class_feature_names_
        reg_feature_names = pipeline.reg_feature_names_ or pipeline.class_feature_names_

        log.h1(f"FEATURE IMPORTANCE FOR {label_name}")

        print_top_features(
            model=pipeline.head1,
            feature_names=class_feature_names,
            label=f"{label_name} Classifier",
            top_n=top_n,
        )

        if pipeline.head2 is not None:
            print_top_features(
                model=pipeline.head2,
                feature_names=reg_feature_names,
                label=f"{label_name} Regressor",
                top_n=top_n,
            )
        else:
            log.warn(f"No regressor trained for {label_name}.")


def feature_importance_to_dataframe(model, splits: dict[str, Splits]):
    """Export raw feature importance values to a DataFrame."""

    if not getattr(model, "is_fitted", False):
        raise ValueError("Model must be fitted before exporting feature importance.")

    rows = []

    for label_name, info in model.models.items():
        pipeline = info["pipeline"]
        class_feature_names = pipeline.class_feature_names_
        reg_feature_names = pipeline.reg_feature_names_ or pipeline.class_feature_names_

        clf_importances = pipeline.head1.feature_importances_
        for feature, importance in zip(class_feature_names, clf_importances):
            rows.append({
                "label": label_name,
                "head_type": "classifier",
                "feature": feature,
                "importance": float(importance),
            })

        if pipeline.head2 is not None:
            reg_importances = pipeline.head2.feature_importances_
            for feature, importance in zip(reg_feature_names, reg_importances):
                rows.append({
                    "label": label_name,
                    "head_type": "regressor",
                    "feature": feature,
                    "importance": float(importance),
                })

    return pd.DataFrame(rows)


def feature_importance_rankings_to_dataframe(model, splits: dict[str, Splits]):
    """Add per-label/head ranks to feature importance values."""

    fi_df = feature_importance_to_dataframe(model=model, splits=splits).copy()

    fi_df["rank_within_head"] = (
        fi_df.groupby(["label", "head_type"])["importance"]
        .rank(method="min", ascending=False)
    )

    return fi_df


def summarize_feature_importance(fi_df, top_k=10):
    """Summarize feature importance across labels and heads."""

    fi_df = fi_df.copy()

    if "rank_within_head" not in fi_df.columns:
        fi_df["rank_within_head"] = (
            fi_df.groupby(["label", "head_type"])["importance"]
            .rank(method="min", ascending=False)
        )

    summary = (
        fi_df.groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            median_importance=("importance", "median"),
            max_importance=("importance", "max"),
            nonzero_count=("importance", lambda s: int((s > 0).sum())),
            topk_count=("rank_within_head", lambda s: int((s <= top_k).sum())),
        )
        .reset_index()
        .sort_values(["mean_importance", "max_importance"], ascending=False)
    )

    return fi_df, summary


def save_feature_importance_outputs(model, splits, output_path, top_k=10):
    """Save raw and summarized feature-importance CSV outputs."""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    fi_ranked_df = feature_importance_rankings_to_dataframe(model=model, splits=splits)
    fi_df, fi_summary_df = summarize_feature_importance(fi_ranked_df, top_k=top_k)

    fi_df.to_csv(output_path / "feature_importance_raw.csv", index=False)
    fi_summary_df.to_csv(output_path / "feature_importance_summary.csv", index=False)
