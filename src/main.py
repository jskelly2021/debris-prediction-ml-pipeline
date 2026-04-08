
import argparse
import pandas as pd

from logger import setup_logger
from config import load_config
from evaluation import evaluate_multilabel_model
from preprocess import load_and_preprocess_data
from split import make_label_specific_splits
from tune_mode import TuneMode
from multi_label_model import MultiLabelModel
from plots import save_multilabel_dashboards
from results import (
    feature_importance_rankings_to_dataframe,
    metrics_to_dataframe,
    print_feature_importance,
    print_metrics,
    summarize_feature_importance,
)
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label two-head XGBoost model")
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("--run-id", type=str, default="default_run")
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--plots", default=False, action="store_true")
    parser.add_argument("--feature_importance", default=False, action="store_true")

    return parser.parse_args()


def append_df_to_csv(df: pd.DataFrame, path: Path):
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)


def main():
    setup_logger()
    args = parse_args()
    config = load_config(args.config_path)

    X, y_class, y_reg = load_and_preprocess_data(config)

    splits = make_label_specific_splits(
        X=X,
        y_class=y_class,
        y_reg=y_reg,
        class_target_cols=config.class_target_cols,
        reg_target_cols=config.reg_target_cols,
        labels=config.label_names,
        apply_smote=config.smote,
        outlier_threshold=config.outlier_threshold
    )

    model = MultiLabelModel(
        train_config=config,
    )

    model.fit(
        splits=splits,
        class_tune_mode=TuneMode.NONE,
        reg_tune_mode=TuneMode.NONE
    )

    metrics = evaluate_multilabel_model(model, splits)

    print_feature_importance(model=model, splits=splits, top_n=5)
    print_metrics(metrics)

    if args.save:
        config.output_path.mkdir(parents=True, exist_ok=True)

        n_features = {
            label_name: splits[label_name].X_test_class.shape[1]
            for label_name in metrics
        }

        metrics_df = metrics_to_dataframe(
            metrics=metrics,
            train_config=config,
            run_id=args.run_id,
            n_features=n_features,
        )

        append_df_to_csv(metrics_df, config.output_path / "all_runs.csv")

        for label_name in metrics_df["label"].unique():
            label_df = metrics_df[metrics_df["label"] == label_name]
            append_df_to_csv(label_df, config.output_path / f"{label_name}_metrics.csv")

    if args.feature_importance:
        config.output_path.mkdir(parents=True, exist_ok=True)

        fi_ranked_df = feature_importance_rankings_to_dataframe(model=model, splits=splits)
        fi_df, fi_summary_df = summarize_feature_importance(fi_ranked_df, top_k=10)

        fi_df.to_csv(config.output_path / "feature_importance_raw.csv", index=False)
        fi_summary_df.to_csv(config.output_path / "feature_importance_summary.csv", index=False)

    if args.plots:
        config.output_path.mkdir(parents=True, exist_ok=True)
        save_multilabel_dashboards(
            model=model,
            splits=splits,
            output_dir=config.output_path / "plots",
            top_n_features=15
        )


if __name__ == "__main__":
    main()
