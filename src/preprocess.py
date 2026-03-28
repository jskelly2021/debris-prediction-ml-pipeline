
import pandas as pd
import numpy as np

from logger import Log
from split import make_train_val_test_splits


log = Log()


def preprocess_features(df, drop_cols):
    df = df.copy()

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=False,
        dummy_na=True
    )

    nunique = X_encoded.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X_encoded = X_encoded.drop(columns=constant_cols)

    return X_encoded


def preprocess_data(df, class_target_cols, reg_target_cols, drop_cols):
    log.info("Preprocessing data...")
    df = df.copy()

    missing_class = [c for c in class_target_cols if c not in df.columns]
    missing_reg = [c for c in reg_target_cols if c not in df.columns]
    if missing_reg:
        raise ValueError(f"Missing targets: class={missing_class}, reg={missing_reg}")

    X_encoded = preprocess_features(df, drop_cols)

    y_class = df[class_target_cols].copy()
    y_reg = df[reg_target_cols].copy()

    log.info(f"Preprocessing complete\n")
    return X_encoded, y_class, y_reg


def load_data(data_path):
    log.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    log.info(f"Data loaded successfully. Shape: {df.shape}\n")
    return df


def create_labels(df, class_target_cols, reg_target_cols):
    df = df.copy()

    for class_col, reg_col in zip(class_target_cols, reg_target_cols):
        df[class_col] = (df[reg_col] > 0).astype(int)

    return df


def load_preprocess_split_data(
        data_path,
        class_target_cols,
        reg_target_cols,
        drop_cols,
        add_labels=False,
        holdout_size=0.2,
        random_state=12,
    ):

    df = load_data(data_path)

    if add_labels:
        df = create_labels(df, class_target_cols, reg_target_cols)

    X, y_class, y_reg = preprocess_data(
        df,
        class_target_cols,
        reg_target_cols,
        drop_cols,
    )

    return make_train_val_test_splits(
        X,
        y_class,
        y_reg,
        labels=class_target_cols,
        holdout_size=holdout_size,
        random_state=random_state
    )
