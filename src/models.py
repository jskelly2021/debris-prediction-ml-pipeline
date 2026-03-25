
from xgboost import XGBClassifier, XGBRegressor


def build_class_model(y_train) -> XGBClassifier:

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 12,
        "tree_method": ["hist"]
    }

    model = XGBClassifier(params=params, n_jobs=-1)

    return model


def build_reg_model() -> XGBRegressor:

    params = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 12,
        "tree_method": ["hist"]
    }

    model = XGBRegressor(params=params, n_jobs=-1)

    return model
