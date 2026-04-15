
import argparse

from logger import setup_logger
from config import build_pipeline_config, load_config
from evaluation import evaluate_multilabel_model
from preprocess import load_and_preprocess_data
from split import make_label_specific_splits
from tune_mode import parse_tune_mode
from multi_label_model import MultiLabelModel
from plots import save_multilabel_dashboards
from results import (
    print_feature_importance,
    print_metrics,
    save_feature_importance_outputs,
    save_metrics_outputs,
)


def parse_args():
    """Parse CLI arguments for a training run."""

    parser = argparse.ArgumentParser(description="Train multi-label two-head XGBoost model")
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("--run-id", type=str, default="default_run")
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--plots", default=False, action="store_true")
    parser.add_argument("--feature_importance", default=False, action="store_true")

    return parser.parse_args()


def main():
    """Run the configured training, evaluation, and optional exports."""

    setup_logger()
    args = parse_args()
    config = load_config(args.config_path)
    pipeline_config = build_pipeline_config(config)

    X, y_class, y_reg = load_and_preprocess_data(config)

    splits = make_label_specific_splits(
        X=X,
        y_class=y_class,
        y_reg=y_reg,
        label_specs=config.labels.label_specs,
        outlier_threshold=config.training.outlier_threshold,
        positive_only_regression=config.training.positive_only_regression,
    )

    model = MultiLabelModel(
        label_specs=config.labels.label_specs,
        pipeline_config=pipeline_config,
    )

    model.fit(
        splits=splits,
        class_tune_mode=parse_tune_mode(config.training.class_tune_mode),
        reg_tune_mode=parse_tune_mode(config.training.reg_tune_mode)
    )

    metrics = evaluate_multilabel_model(model, splits)

    print_feature_importance(model=model, splits=splits, top_n=5)
    print_metrics(metrics)

    if args.save:
        save_metrics_outputs(
            metrics=metrics,
            splits=splits,
            experiment_config=config,
            output_path=config.data.output_path,
            run_id=args.run_id,
        )

    if args.feature_importance:
        save_feature_importance_outputs(
            model=model,
            splits=splits,
            output_path=config.data.output_path,
            top_k=10,
        )

    if args.plots:
        config.data.output_path.mkdir(parents=True, exist_ok=True)
        save_multilabel_dashboards(
            model=model,
            splits=splits,
            output_dir=config.data.output_path / "plots",
            top_n_features=15
        )


if __name__ == "__main__":
    main()
