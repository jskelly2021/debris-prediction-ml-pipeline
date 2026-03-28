
import argparse

from logger import setup_logger
from config import load_config
from preprocess import load_preprocess_split_data
from tune_mode import TuneMode
from multi_label_model import MultiLabelModel
from plots import save_multilabel_dashboards


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label two-head XGBoost model")
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--smote", default=False, action="store_true")
    parser.add_argument("--scale_pos_weight", default=False, action="store_true")
    parser.add_argument("--log_regression_target", default=False, action="store_true")
    parser.add_argument("--positive_only_regression", default=False, action="store_true")
    return parser.parse_args()


def main():
    setup_logger()
    args = parse_args()
    config = load_config(args.config_path)

    splits = load_preprocess_split_data(
        config.data_path,
        class_target_cols=config.class_target_cols,
        reg_target_cols=config.reg_target_cols,
        drop_cols=config.drop_cols,
        add_labels=True,
    )

    model = MultiLabelModel(
        trainConfig=config,
    )

    model.fit(
        splits=splits,
        class_tune_mode=TuneMode.NONE,
        reg_tune_mode=TuneMode.NONE
    )

    model.evaluate(
        X=splits.X_test,
        y_class=splits.y_class_test,
        y_reg=splits.y_reg_test
    )

    model.print_metrics()

    if args.save:
        config.output_path.mkdir(parents=True, exist_ok=True)

        metrics_df = model.metrics_to_dataframe()
        metrics_df.to_csv(f"{config.output_path}/metrics_summary.csv", index=False)
    
        save_multilabel_dashboards(
            multilabel_model=model,
            splits=splits,
            output_dir=config.output_path
        )


if __name__ == "__main__":
    main()
