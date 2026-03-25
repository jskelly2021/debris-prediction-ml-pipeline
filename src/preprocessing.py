
import pandas as pd
import numpy as np

from dataclasses import dataclass
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


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


def make_train_val_test_splits(X, y_class, y_reg, labels, holdout_size=0.2, random_state=12):
    print("Creating train/val/test splits...")

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

    print(f"Train/Val/Test splits complete: ")

    return Splits(
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


def preprocess_features(df, drop_cols, categorical_candidates=None):
    df = df.copy()

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)

    if categorical_candidates is None:
        categorical_candidates = []

    for col in categorical_candidates:
        if col in X.columns:
            X[col] = X[col].astype(str)

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


def preprocess_data(df, class_target_cols, reg_target_cols, drop_cols, categorical_candidates):
    print("Preprocessing data...")
    df = df.copy()

    missing_class = [c for c in class_target_cols if c not in df.columns]
    missing_reg = [c for c in reg_target_cols if c not in df.columns]
    if missing_reg:
        raise ValueError(f"Missing targets: class={missing_class}, reg={missing_reg}")

    X_encoded = preprocess_features(df, drop_cols, categorical_candidates)

    y_class = df[class_target_cols].copy()
    y_reg = df[reg_target_cols].copy()

    print(f"Preprocessing complete")
    print(f"Final feature set shape: {X_encoded.shape}")
    print(f"Classification target shape: {y_class.shape}")
    print(f"Regression target shape: {y_reg.shape}\n")

    return X_encoded, y_class, y_reg


def load_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}\n")
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
        categorical_candidates=None,
        holdout_size=0.2,
        random_state=12
    ):

    df = load_data(data_path)

    if add_labels:
        df = create_labels(df, class_target_cols, reg_target_cols)

    X, y_class, y_reg = preprocess_data(
        df,
        class_target_cols,
        reg_target_cols,
        drop_cols,
        categorical_candidates
    )

    return make_train_val_test_splits(
        X,
        y_class,
        y_reg,
        labels=class_target_cols,
        holdout_size=holdout_size,
        random_state=random_state
    )
