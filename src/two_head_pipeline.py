
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from tune_mode import TuneMode
from split import Splits
from regressor import train_regressor
from classifier import train_classifier
from resampling import apply_smote_single_label
from categorical_preprocessing import CategoricalPreprocessor
from feature_filter import FeatureFilter
from dataclasses import dataclass
from logger import Log


log = Log()


@dataclass
class Preds:
    """Store predictions from a classifier/regressor label pipeline."""

    class_prob: np.ndarray
    class_pred: np.ndarray
    reg_pred: np.ndarray
    expected_volume_pred: np.ndarray


class TwoHeadPipeline:
    """Train and run one label's classifier and regressor heads.

    Attributes:
        threshold: Classification threshold selected on validation data.
        class_feature_names_: Classifier output feature names.
    """

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
        self.positive_only_regression = pipelineConfig.positive_only_regression
        self.categorical_cols = pipelineConfig.categorical_cols
        self.categorical_encoding = pipelineConfig.categorical_encoding
        self.target_encoding_smoothing = pipelineConfig.target_encoding_smoothing
        self.feature_filtering = pipelineConfig.feature_filtering or {}
        self.classifier_model = pipelineConfig.classifier_model
        self.regressor_model = pipelineConfig.regressor_model
        self.threshold = 0.5
        self.best_f1 = None
        self.head1 = self.__build_classifier()
        self.head2 = self.__build_regressor()
        self.class_preprocessor = None
        self.reg_preprocessor = None
        self.class_feature_filter = None
        self.reg_feature_filter = None
        self.class_feature_names_ = []
        self.reg_feature_names_ = []
        self.is_fitted = False


    def __build_classifier(self):
        if self.classifier_model == "xgboost":
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=-1,
            )

        if self.classifier_model == "random_forest":
            return RandomForestClassifier(
                n_jobs=-1,
                random_state=12,
            )

        raise ValueError(f"Unsupported classifier model: {self.classifier_model}")


    def __build_regressor(self):
        if self.regressor_model == "xgboost":
            return XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
            )

        if self.regressor_model == "random_forest":
            return RandomForestRegressor(
                n_jobs=-1,
                random_state=12,
            )

        raise ValueError(f"Unsupported regressor model: {self.regressor_model}")


    def train(
        self,
        splits,
        class_target_col,
        reg_target_col,
        class_tune_mode=TuneMode.NONE,
        reg_tune_mode=TuneMode.NONE
    ):
        """Train classification and regression heads for one label."""

        class_splits = self._prepare_class_splits(splits, class_target_col)
        classifierResults = train_classifier(
            estimator=self.head1,
            splits=class_splits,
            param_dist=self.class_param_dist,
            default_params=self.class_default_params,
            class_target_col=class_target_col,
            apply_scale_pos_weight=self.apply_scale_pos_weight,
            tune_mode=class_tune_mode
        )

        reg_splits = self._prepare_reg_splits(splits, reg_target_col)
        regressorResults = train_regressor(
            estimator=self.head2,
            splits=reg_splits,
            param_dist=self.reg_param_dist,
            default_params=self.reg_default_params,
            reg_target_col=reg_target_col,
            log_target=self.log_regression_target,
            tune_mode=reg_tune_mode
        )
        
        self.head1 = classifierResults.estimator
        self.head2 = regressorResults.estimator
        self.threshold = classifierResults.best_threshold
        self.best_f1 = classifierResults.best_f1
        self.is_fitted = True


    def _prepare_class_splits(self, splits: Splits, class_target_col: str) -> Splits:
        """Preprocess and filter classifier split features."""

        self.class_preprocessor = CategoricalPreprocessor(
            categorical_cols=self.categorical_cols,
            categorical_encoding=self.categorical_encoding,
            target_smoothing=self.target_encoding_smoothing,
            head_name=f"{class_target_col} classifier",
        )

        X_train_class = self.class_preprocessor.fit_transform(splits.X_train_class, splits.y_train_class)
        X_val_class = self.class_preprocessor.transform(splits.X_val_class)
        X_test_class = self.class_preprocessor.transform(splits.X_test_class)
        self.class_feature_filter = self._build_feature_filter(
            head_name=f"{class_target_col} classifier",
        )
        X_train_class = self.class_feature_filter.fit_transform(X_train_class)
        X_val_class = self.class_feature_filter.transform(X_val_class)
        X_test_class = self.class_feature_filter.transform(X_test_class)
        self.class_feature_names_ = X_train_class.columns.tolist()

        y_train_class = splits.y_train_class
        if self.apply_smote:
            X_train_class, y_train_class = apply_smote_single_label(
                X_train_class,
                y_train_class,
                label_name=class_target_col,
            )

        return Splits(
            X_train_class=X_train_class,
            X_val_class=X_val_class,
            X_test_class=X_test_class,
            y_train_class=y_train_class,
            y_val_class=splits.y_val_class,
            y_test_class=splits.y_test_class,
            X_train_reg=splits.X_train_reg,
            X_val_reg=splits.X_val_reg,
            X_test_reg=splits.X_test_reg,
            y_train_reg=splits.y_train_reg,
            y_val_reg=splits.y_val_reg,
            y_test_reg=splits.y_test_reg,
        )


    def _prepare_reg_splits(self, splits: Splits, reg_target_col: str) -> Splits:
        """Preprocess and filter regressor split features."""

        if len(splits.X_train_reg) == 0:
            self.reg_preprocessor = None
            self.reg_feature_filter = None
            self.reg_feature_names_ = []
            return splits

        self.reg_preprocessor = CategoricalPreprocessor(
            categorical_cols=self.categorical_cols,
            categorical_encoding=self.categorical_encoding,
            target_smoothing=self.target_encoding_smoothing,
            head_name=f"{reg_target_col} regressor",
        )

        X_train_reg = self.reg_preprocessor.fit_transform(splits.X_train_reg, splits.y_train_reg)
        X_val_reg = self.reg_preprocessor.transform(splits.X_val_reg)
        X_test_reg = self.reg_preprocessor.transform(splits.X_test_reg)
        self.reg_feature_filter = self._build_feature_filter(
            head_name=f"{reg_target_col} regressor",
        )
        X_train_reg = self.reg_feature_filter.fit_transform(X_train_reg)
        X_val_reg = self.reg_feature_filter.transform(X_val_reg)
        X_test_reg = self.reg_feature_filter.transform(X_test_reg)
        self.reg_feature_names_ = X_train_reg.columns.tolist()

        return Splits(
            X_train_class=splits.X_train_class,
            X_val_class=splits.X_val_class,
            X_test_class=splits.X_test_class,
            y_train_class=splits.y_train_class,
            y_val_class=splits.y_val_class,
            y_test_class=splits.y_test_class,
            X_train_reg=X_train_reg,
            X_val_reg=X_val_reg,
            X_test_reg=X_test_reg,
            y_train_reg=splits.y_train_reg,
            y_val_reg=splits.y_val_reg,
            y_test_reg=splits.y_test_reg,
        )


    def predict(self, X):
        """Predict class, regression, and expected-volume outputs."""

        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")

        X_class = self.class_preprocessor.transform(X) if self.class_preprocessor is not None else X
        X_class = self.class_feature_filter.transform(X_class) if self.class_feature_filter is not None else X_class
        y_prob = self.head1.predict_proba(X_class)[:, 1]
        y_class = (y_prob >= self.threshold).astype(int)

        if self.head2 is None:
            y_reg = np.zeros(len(X))
        else:
            X_reg = self.reg_preprocessor.transform(X) if self.reg_preprocessor is not None else X
            X_reg = self.reg_feature_filter.transform(X_reg) if self.reg_feature_filter is not None else X_reg
            y_reg = self.head2.predict(X_reg)
            if self.log_regression_target:
                y_reg = np.expm1(y_reg)

        y_expected = y_prob * y_reg

        return Preds(
            class_prob=y_prob,
            class_pred=y_class,
            reg_pred=y_reg,
            expected_volume_pred=y_expected
        )


    def _build_feature_filter(self, head_name: str) -> FeatureFilter:
        """Create a feature filter from config settings."""

        return FeatureFilter(
            enabled=self.feature_filtering.get("enabled", False),
            drop_constant=self.feature_filtering.get("drop_constant", True),
            min_binary_positive_count=self.feature_filtering.get("min_binary_positive_count", 0),
            max_dominant_value_fraction=self.feature_filtering.get("max_dominant_value_fraction"),
            head_name=head_name,
        )


    def predict_df(self, X, prefix):
        """Return predictions as a DataFrame with label-prefixed columns."""

        preds = self.predict(X)
        return pd.DataFrame({
            f"{prefix}_class_prob": preds.class_prob,
            f"{prefix}_class_pred": preds.class_pred,
            f"{prefix}_reg_pred": preds.reg_pred,
            f"{prefix}_expected_volume_pred": preds.expected_volume_pred,
        }, index=X.index)
