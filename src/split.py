
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from config import LabelSpec
from logger import Log
from resampling import apply_smote_single_label


log = Log()


@dataclass
class Splits:
    X_train_class: pd.DataFrame
    X_val_class: pd.DataFrame
    X_test_class: pd.DataFrame
    y_train_class: pd.DataFrame
    y_val_class: pd.DataFrame
    y_test_class: pd.DataFrame

    X_train_reg: pd.DataFrame
    X_val_reg: pd.DataFrame
    X_test_reg: pd.DataFrame
    y_train_reg: pd.DataFrame
    y_val_reg: pd.DataFrame
    y_test_reg: pd.DataFrame


def _remove_outliers(X, y, outlier_threshold):
    if outlier_threshold is None:
        return X, y

    log.info("Removing outliers")

    mask = y < outlier_threshold

    return X.loc[mask], y.loc[mask]


def _filter_positive_regression_rows(X, y_reg, y_class):
    mask = y_class == 1
    return X.loc[mask], y_reg.loc[mask]


def _get_train_val_test_splits(X, y_class, y_reg, holdout_size=0.2, random_state=12) -> Splits:
    train_idx, temp_idx = train_test_split(
        X.index,
        test_size=holdout_size,
        random_state=random_state,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
        shuffle=True,
    )

    splits = Splits(
        X_train_class=X.loc[train_idx],
        X_val_class=X.loc[val_idx],
        X_test_class=X.loc[test_idx],
        X_train_reg=X.loc[train_idx],
        X_val_reg=X.loc[val_idx],
        X_test_reg=X.loc[test_idx],
        y_train_class=y_class.loc[train_idx],
        y_val_class=y_class.loc[val_idx],
        y_test_class=y_class.loc[test_idx],
        y_train_reg=y_reg.loc[train_idx],
        y_val_reg=y_reg.loc[val_idx],
        y_test_reg=y_reg.loc[test_idx],
    )

    return splits


def make_label_specific_splits(
    X,
    y_class,
    y_reg,
    label_specs: list[LabelSpec],
    apply_smote,
    outlier_threshold,
    positive_only_regression,
    holdout_size=0.2,
    random_state=12
) -> dict[str, Splits]:
    log.info("Creating train/val/test splits...")

    base = _get_train_val_test_splits(
        X,
        y_class,
        y_reg,
        holdout_size,
        random_state
    )

    splits = { } 

    for label_spec in label_specs:
        label_name = label_spec.label_name
        class_col = label_spec.class_target_col
        reg_col = label_spec.reg_target_col

        X_train_class = base.X_train_class.copy()
        y_train_class = base.y_train_class[class_col].copy()

        X_train_reg = base.X_train_reg.copy()
        y_train_reg = base.y_train_reg[reg_col].copy()

        X_val_reg = base.X_val_reg.copy()
        y_val_reg = base.y_val_reg[reg_col].copy()

        X_test_reg = base.X_test_reg.copy()
        y_test_reg = base.y_test_reg[reg_col].copy()

        if positive_only_regression:
            X_train_reg, y_train_reg = _filter_positive_regression_rows(
                X_train_reg,
                y_train_reg,
                base.y_train_class[class_col],
            )
            X_val_reg, y_val_reg = _filter_positive_regression_rows(
                X_val_reg,
                y_val_reg,
                base.y_val_class[class_col],
            )
            X_test_reg, y_test_reg = _filter_positive_regression_rows(
                X_test_reg,
                y_test_reg,
                base.y_test_class[class_col],
            )

        X_train_reg, y_train_reg = _remove_outliers(X_train_reg, y_train_reg, outlier_threshold)
        X_val_reg, y_val_reg = _remove_outliers(X_val_reg, y_val_reg, outlier_threshold)
        X_test_reg, y_test_reg = _remove_outliers(X_test_reg, y_test_reg, outlier_threshold)

        if apply_smote:
            X_train_class, y_train_class = apply_smote_single_label(
                X_train_class,
                y_train_class,
                label_name=class_col,
                random_state=random_state
            )

        splits[label_name] = Splits(
                X_train_class=X_train_class,
                X_val_class=base.X_val_class,
                X_test_class=base.X_test_class,
                y_train_class=y_train_class,
                y_val_class=base.y_val_class[class_col],
                y_test_class=base.y_test_class[class_col],

                X_train_reg=X_train_reg,
                X_val_reg=X_val_reg,
                X_test_reg=X_test_reg,
                y_train_reg=y_train_reg,
                y_val_reg=y_val_reg,
                y_test_reg=y_test_reg
            )

    log.info(f"Train/Val/Test splits complete: ")
    
    return splits
