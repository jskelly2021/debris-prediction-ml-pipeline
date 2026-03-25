
import numpy as np
import argparse

from xgboost import XGBClassifier, XGBRegressor
from plots import save_dashboard
from two_head_pipeline import ParamTuningMode, TwoHeadPipeline
from preprocessing import load_preprocess_split_data
from metrics import compute_classification_metrics, compute_regression_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-head XGBoost model")
    parser.add_argument("data_path", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save trained models and metrics")
    parser.add_argument("--no_plots", action="store_true", help="Whether to skip saving plots")
    return parser.parse_args()


def get_config():
    class_target_cols = ["has_VG_debris", "has_CD_debris"]
    reg_target_cols = ["VolVG_sum", "VolCD_sum"]

    drop_cols = [
        "OBJECTID","GRID_ID",
        "VolCD","VolCD_med","VolCD_sum",
        "VolVG","VolVG_med","VolVG_sum",
        "has_VG_debris","has_CD_debris"
    ]

    return class_target_cols, reg_target_cols, drop_cols


def print_metrics(splits, preds, class_target_col, reg_target_col, debris_type):
    y_class_test = np.asarray(splits.y_class_test[class_target_col])
    y_reg_test = np.asarray(splits.y_reg_test[reg_target_col])
    y_reg_pred = np.asarray(preds.reg_pred)

    positive_mask = (y_class_test == 1)

    class_metrics = compute_classification_metrics(
        splits.y_class_test[class_target_col],
        preds.class_pred,
        preds.class_prob
    )
    reg_metrics = compute_regression_metrics(
        splits.y_reg_test[reg_target_col],
        preds.reg_pred
    )
    if positive_mask.sum() > 0:
        pos_reg_metrics = compute_regression_metrics(
            y_reg_test[positive_mask],
            y_reg_pred[positive_mask],
        )
    class_metrics.print(f"{debris_type} Debris")
    reg_metrics.print(f"{debris_type} Debris")
    pos_reg_metrics.print(f"{debris_type.upper()} Positive-Only Volume")


def main():
    args = parse_args()

    class_target_cols, reg_target_cols, drop_cols = get_config()

    splits = load_preprocess_split_data(
        args.data_path,
        class_target_cols=class_target_cols,
        reg_target_cols=reg_target_cols,
        drop_cols=drop_cols,
        add_labels=True,
    )

    reg_params = {
        "n_estimators": [100, 300, 1000],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 6, 9],
        "min_child_weight": [1, 3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    }

    debris_types = ["VG", "CD"]

    for i, debris_type in enumerate(debris_types):
        print("\n" + "=" * 40)
        print(f"Processing {debris_type}")
        print("=" * 40)

        neg_count = (splits.y_class_train[class_target_cols[0]] == 0).sum()
        pos_count = (splits.y_class_train[class_target_cols[0]] == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        class_params = {
            "n_estimators": [100, 300, 1000],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 6, 9],
            "min_child_weight": [1, 3],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0],
            # "scale_pos_weight": [scale_pos_weight],
        }

        classifier = XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_jobs=-1)
        regressor = XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=-1)

        pipeline = TwoHeadPipeline(
            classifier=classifier,
            regressor=regressor,
            class_param_dist=class_params,
            reg_param_dist=reg_params,
            apply_smote=True,
            log_regression_target=True,
            positive_only_regression=True
        )

        pipeline.train(
            splits,
            class_target_cols[i],
            reg_target_cols[i],
            class_tune_mode=ParamTuningMode.NONE,
            reg_tune_mode=ParamTuningMode.NONE
        )

        preds = pipeline.predict(splits.X_test)

        print_metrics(splits, preds, class_target_cols[i], reg_target_cols[i], debris_type)

        if not args.no_plots:
            save_dashboard(
                preds=preds,
                splits=splits,
                model=pipeline,
                class_target_col=class_target_cols[i],
                reg_target_col=reg_target_cols[i],
                debris_type=debris_type,
                output_dir=args.output_dir,
            )

if __name__ == "__main__":
    main()
