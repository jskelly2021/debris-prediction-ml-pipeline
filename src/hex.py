
import argparse

from models import build_class_model, build_reg_model
from TwoHeadDebrisModel import TwoHeadDebrisModel
from preprocessing import load_preprocess_split_data
from metrics import show_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-head XGBoost model")
    parser.add_argument("data_path", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save trained models and metrics")
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

    X_train = splits.X_train
    X_val = splits.X_val
    X_test = splits.X_test

    debris_types = ["VG", "CD"]

    for i, debris_type in enumerate(debris_types):
        print("\n" + "=" * 40)
        print(f"Processing {debris_type}")
        print("=" * 40)

        y_class_train = splits.y_class_train[class_target_cols[i]]
        y_reg_train = splits.y_reg_train[reg_target_cols[i]]
        y_class_val = splits.y_class_val[class_target_cols[i]]
        y_reg_val = splits.y_reg_val[reg_target_cols[i]]
        y_class_test = splits.y_class_test[class_target_cols[i]]
        y_reg_test = splits.y_reg_test[reg_target_cols[i]]

        classifier = build_class_model(y_class_train)
        regressor = build_reg_model()

        model = TwoHeadDebrisModel(
            classifier=classifier,
            regressor=regressor,
            tune_hyperparams=True,
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
