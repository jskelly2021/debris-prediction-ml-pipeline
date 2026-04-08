
import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from tune_mode import TuneMode
from regressor import train_regressor
from classifier import train_classifier
from dataclasses import dataclass
from logger import Log


log = Log()


@dataclass
class Preds:
    class_prob: np.ndarray
    class_pred: np.ndarray
    reg_pred: np.ndarray
    expected_volume_pred: np.ndarray


class TwoHeadPipeline:
    def __init__(
        self,
        pipelineConfig,
    ):
        self.class_param_dist = pipelineConfig.class_params_dist
        self.reg_param_dist = pipelineConfig.reg_params_dist
        self.class_default_params = pipelineConfig.class_default_params or {}
        self.reg_default_params = pipelineConfig.reg_default_params or {}
        self.apply_smote = pipelineConfig.smote
        self.apply_scale_pos_weight = pipelineConfig.scale_pos_weight
        self.log_regression_target = pipelineConfig.log_target_reg
        self.threshold = 0.5
        self.best_f1 = None
        self.head1 = self.__build_classifier()
        self.head2 = self.__build_regressor()
        self.is_fitted = False


    def __build_classifier(self):
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
        )


    def __build_regressor(self):
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
        )


    def train(
        self,
        splits,
        class_target_col,
        reg_target_col,
        class_tune_mode=TuneMode.NONE,
        reg_tune_mode=TuneMode.NONE
    ):
        classifierResults = train_classifier(
            estimator=self.head1,
            splits=splits,
            param_dist=self.class_param_dist,
            default_params=self.class_default_params,
            class_target_col=class_target_col,
            apply_scale_pos_weight=self.apply_scale_pos_weight,
            tune_mode=class_tune_mode
        )

        regressorResults = train_regressor(
            estimator=self.head2,
            splits=splits,
            param_dist=self.reg_param_dist,
            default_params=self.reg_default_params,
            reg_target_col=reg_target_col,
            log_target=self.log_regression_target,
            tune_mode=reg_tune_mode
        )
        
        self.head1 = classifierResults.estimator
        self.head2 = regressorResults.estimator
        self.is_fitted = True


    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")

        y_prob = self.head1.predict_proba(X)[:, 1]
        y_class = (y_prob >= self.threshold).astype(int)

        if self.head2 is None:
            y_reg = np.zeros(len(X))
        else:
            y_reg = self.head2.predict(X)
            if self.log_regression_target:
                y_reg = np.expm1(y_reg)

        y_expected = y_prob * y_reg

        return Preds(
            class_prob=y_prob,
            class_pred=y_class,
            reg_pred=y_reg,
            expected_volume_pred=y_expected
        )


    def predict_df(self, X, prefix):
        preds = self.predict(X)
        return pd.DataFrame({
            f"{prefix}_class_prob": preds.class_prob,
            f"{prefix}_class_pred": preds.class_pred,
            f"{prefix}_reg_pred": preds.reg_pred,
            f"{prefix}_expected_volume_pred": preds.expected_volume_pred,
        }, index=X.index)
