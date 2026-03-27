
import argparse

from src.config import load_config
from src.preprocessing import load_preprocess_split_data
from src.plots import save_multilabel_dashboards
from src.two_head_pipeline import ParamTuningMode, TwoHeadPipeline
from src.multi_label_model import MultiLabelModel
from xgboost import XGBClassifier, XGBRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label two-head XGBoost model")
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--use_smote", default=False, action="store_true")
    parser.add_argument("--use_scale_pos_weight", default=False, action="store_true")
    return parser.parse_args()


def build_pipeline_factory(config, args):
    def factory():
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
        )

        regressor = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
        )

        return TwoHeadPipeline(
            classifier=classifier,
            regressor=regressor,
            class_param_dist=config.class_params_dist,
            reg_param_dist=config.reg_params_dist,
            class_default_params=config.class_default_params,
            reg_default_params=config.reg_default_params,
            apply_smote=args.use_smote,
            apply_scale_pos_weight=args.use_scale_pos_weight,
            log_regression_target=True,
            positive_only_regression=True,
        )

    return factory


def main():
    args = parse_args()
    config = load_config(args.config_path)

    splits = load_preprocess_split_data(
        config.data_path,
        class_target_cols=config.class_target_cols,
        reg_target_cols=config.reg_target_cols,
        drop_cols=config.drop_cols,
        add_labels=True,
        verbose=False
    )

    model = MultiLabelModel(
        pipeline_factory=build_pipeline_factory(config, args),
        class_target_cols=config.class_target_cols,
        reg_target_cols=config.reg_target_cols,
        label_names=config.label_names,
        verbose=True
    )

    model.fit(
        splits=splits,
        class_tune_mode=ParamTuningMode.NONE,
        reg_tune_mode=ParamTuningMode.NONE
    )

    model.evaluate(
        X=splits.X_test,
        y_class=splits.y_class_test,
        y_reg=splits.y_reg_test
    )

    model.print_metrics()

    if args.save:
        config.output_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = model.metrics_to_dataframe()
        metrics_df.to_csv(f"{config.output_dir}/metrics_summary.csv", index=False)
    
        save_multilabel_dashboards(
            multilabel_model=model,
            splits=splits,
            output_dir=config.output_dir
        )


if __name__ == "__main__":
    main()
