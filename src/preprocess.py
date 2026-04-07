
import pandas as pd
import numpy as np

from logger import Log
from config import TrainConfig


log = Log()


def _apply_log_to_features(X, log_feature_cols):
    X = X.copy()

    for col in log_feature_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    
    return X


def _drop_columns(df, drop_cols):
    df = df.copy()

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def _one_hot_encode_features(X, categorical_cols):
    X = X.copy()

    categorical_cols = [col for col in categorical_cols if col in X.columns]

    X = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=False,
        dummy_na=True
    )

    return X    


def _remove_constant_columns(X):
    X = X.copy()

    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
    
    return X


def _preprocess_data(
    df,
    class_target_cols,
    reg_target_cols,
    drop_cols,
    categorical_cols,
    log_features,
    feature_cols_to_log
):
    df = df.copy()

    missing_class = [c for c in class_target_cols if c not in df.columns]
    missing_reg = [c for c in reg_target_cols if c not in df.columns]
    if missing_reg or missing_class:
        raise ValueError(f"Missing targets: class={missing_class}, reg={missing_reg}")

    X = _drop_columns(df, drop_cols)
    X = _apply_log_to_features(X, feature_cols_to_log) if log_features else X
    X_encoded = _one_hot_encode_features(X, categorical_cols)
    X_encoded = _remove_constant_columns(X_encoded)

    y_class = df[class_target_cols].copy()
    y_reg = df[reg_target_cols].copy()

    return X_encoded, y_class, y_reg


def _load_data(data_path):
    log.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    log.info(f"Data loaded successfully. Shape: {df.shape}\n")
    return df


def load_and_preprocess_data(config: TrainConfig):
    df = _load_data(config.data_path)

    log.info("Preprocessing data...")

    X, y_class, y_reg = _preprocess_data(
        df,
        class_target_cols=config.class_target_cols,
        reg_target_cols=config.reg_target_cols,
        drop_cols=config.drop_cols,
        categorical_cols=config.categorical_cols,
        log_features=config.log_features,
        feature_cols_to_log=config.feature_cols_to_log
    )

    log.info(f"Preprocessing complete\n")

    return X, y_class, y_reg
