
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from logger import Log


log = Log()


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_class_train: pd.DataFrame
    y_class_val: pd.DataFrame
    y_class_test: pd.DataFrame
    y_reg_train: pd.DataFrame
    y_reg_val: pd.DataFrame
    y_reg_test: pd.DataFrame


def _print_binary_label_breakdown(y_df, split_name):
    log.body(f"\n{split_name} label breakdown:")
    for col in y_df.columns:
        y = y_df[col]
        total = len(y)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        pos_rate = pos / total if total > 0 else 0.0

        log.body(
            f"  {col}: "
            f"n={total}, pos={pos}, neg={neg}, pos_rate={pos_rate:.4f}"
        )


def _print_multilabel_combo_breakdown(y_df, split_name):
    combo = y_df.astype(str).agg("_".join, axis=1)
    combo_counts = combo.value_counts().sort_index()

    log.body(f"\n{split_name} multilabel combo breakdown:")
    for combo_name, count in combo_counts.items():
        pct = count / len(combo) if len(combo) > 0 else 0.0
        log.body(f"  {combo_name}: {count} ({pct:.4%})")


def print_split_breakdown(splits):
    log.h1("TRAIN / VAL / TEST SPLIT BREAKDOWN")

    log.body(f"X_train shape: {splits.X_train.shape}")
    log.body(f"X_val   shape: {splits.X_val.shape}")
    log.body(f"X_test  shape: {splits.X_test.shape}")

    _print_binary_label_breakdown(splits.y_class_train, "TRAIN")
    _print_binary_label_breakdown(splits.y_class_val, "VAL")
    _print_binary_label_breakdown(splits.y_class_test, "TEST")

    _print_multilabel_combo_breakdown(splits.y_class_train, "TRAIN")
    _print_multilabel_combo_breakdown(splits.y_class_val, "VAL")
    _print_multilabel_combo_breakdown(splits.y_class_test, "TEST")


def make_train_val_test_splits(X, y_class, y_reg, labels, holdout_size=0.2, random_state=12) -> Splits:
    log.info("Creating train/val/test splits...")

    stratify_labels = y_class[labels[0]].astype(str)
    for label in labels[1:]:
        stratify_labels += "_" + y_class[label].astype(str)

    train_idx, temp_idx = train_test_split(
        X.index,
        test_size=holdout_size,
        random_state=random_state,
        stratify=stratify_labels
    )

    temp_stratify = y_class.loc[temp_idx, labels[0]].astype(str)
    for label in labels[1:]:
        temp_stratify += "_" + y_class.loc[temp_idx, label].astype(str)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_stratify
    )

    splits = Splits(
        X_train=X.loc[train_idx],
        X_val=X.loc[val_idx],
        X_test=X.loc[test_idx],
        y_class_train=y_class.loc[train_idx],
        y_class_val=y_class.loc[val_idx],
        y_class_test=y_class.loc[test_idx],
        y_reg_train=y_reg.loc[train_idx],
        y_reg_val=y_reg.loc[val_idx],
        y_reg_test=y_reg.loc[test_idx],
    )

    log.info(f"Train/Val/Test splits complete: ")
    print_split_breakdown(splits)

    return splits
