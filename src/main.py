
import argparse

from pathlib import Path
from models import build_class_model, build_reg_model
from two_head_pipeline import TwoHeadDebrisModel
from preprocessing import load_preprocess_split_data
from metrics import show_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-head XGBoost model")
    parser.add_argument("data_path", type=str, help="Path to dataset")
    parser.add_argument("--block_level", action="store_true", help="Whether to preprocess dataset as block-level")
    parser.add_argument("--disable_head01", action="store_true", help="Whether to train head01 (classification)")
    parser.add_argument("--disable_head02", action="store_true", help="Whether to train head02 (regression)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save trained models and metrics")
    parser.add_argument("--no_plots", action="store_true", help="Whether to skip saving plots")
    return parser.parse_args()


def get_config(block_level: bool):
    if block_level:
        class_target_cols = ["has_veg_debris", "has_con_debris"]
        reg_target_cols = ["veg_volume_sum", "con_volume_sum"]

        drop_cols = [
            "fd_id", "cbfips", "cbgfips",
            "volume_count", "volume_sum", "volume_mean", "volume_median", "debris_majority",
            "veg_volume_count", "veg_volume_sum", "veg_volume_mean", "veg_volume_median",
            "con_volume_count", "con_volume_sum", "con_volume_mean", "con_volume_median",
            "has_veg_debris", "has_con_debris"
        ]
    else:
        class_target_cols = ["has_veg_debris", "has_con_debris"]
        reg_target_cols = ["veg_volume", "con_volume"]

        drop_cols = [
            "fd_id", "cbfips", "cbgfips",
            "veg_volume", "con_volume",
            "has_veg_debris", "has_con_debris"
        ]

    return class_target_cols, reg_target_cols, drop_cols


def main():
    args = parse_args()

    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    class_target_cols, reg_target_cols, drop_cols = get_config(args.block_level)

    splits = load_preprocess_split_data(
        args.data_path,
        class_target_cols=class_target_cols,
        reg_target_cols=reg_target_cols,
        drop_cols=drop_cols,
    )

    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]

    debris_types = ["veg", "con"]

    for i, debris_type in enumerate(debris_types):
        print("\n" + "=" * 40)
        print(f"Processing {debris_type}")
        print("=" * 40)

        y_class_train = splits["y_class_train"][class_target_cols[i]]
        y_reg_train = splits["y_reg_train"][reg_target_cols[i]]
        y_class_val = splits["y_class_val"][class_target_cols[i]]
        y_reg_val = splits["y_reg_val"][reg_target_cols[i]]
        y_class_test = splits["y_class_test"][class_target_cols[i]]
        y_reg_test = splits["y_reg_test"][reg_target_cols[i]]

        classifier = build_class_model(y_class_train)
        regressor = build_reg_model()

        model = TwoHeadDebrisModel(
            classifier=classifier,
            regressor=regressor,
            log_regression_target=True,
            positive_only_regression=True
        )

        model.train(
            X_train,
            y_class_train,
            y_reg_train,
            X_val,
            y_class_val,
            y_reg_val,
        )

        preds = model.predict(X_test)
        show_metrics(
            preds=preds,
            y_class_test=y_class_test,
            y_reg_test=y_reg_test,
            X_train=X_train,
            model=model,
            debris_type=debris_type,
            output_dir=args.output_dir,
            show_plots=not args.no_plots
        )

if __name__ == "__main__":
    main()
