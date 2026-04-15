
import pandas as pd

from two_head_pipeline import TuneMode, TwoHeadPipeline
from config import LabelSpec, PipelineConfig
from split import Splits
from logger import Log


log = Log()


class MultiLabelModel:
    """Coordinate one two-head pipeline per configured label.

    Attributes:
        models: Mapping of label names to fitted pipelines.
        is_fitted: Whether all label pipelines have been trained.
    """

    def __init__(
        self,
        label_specs: list[LabelSpec],
        pipeline_config: PipelineConfig,
    ):
        self.label_specs = label_specs
        self.pipeline_config = pipeline_config
        self.models = {}
        self.is_fitted = False


    def fit(
        self,
        splits: dict[str, Splits],
        class_tune_mode: TuneMode=TuneMode.NONE,
        reg_tune_mode: TuneMode=TuneMode.NONE
    ):
        """Fit all configured label pipelines.

        Args:
            splits: Label-specific train/validation/test split mapping.
        """

        for label_spec in self.label_specs:
            label_name = label_spec.label_name
            class_col = label_spec.class_target_col
            reg_col = label_spec.reg_target_col

            log.h1(f"Training model for label: {label_name}")

            pipeline = TwoHeadPipeline(
                pipelineConfig=self.pipeline_config
            )

            pipeline.train(
                splits=splits[label_name],
                class_target_col=class_col,
                reg_target_col=reg_col,
                class_tune_mode=class_tune_mode,
                reg_tune_mode=reg_tune_mode
            )

            self.models[label_name] = {
                "pipeline": pipeline,
                "class_col": class_col,
                "reg_col": reg_col,
            }

        self.is_fitted = True
        return self


    def predict(self, X):
        """Predict all labels for a feature DataFrame."""

        if not self.is_fitted:
            raise ValueError("MultiLabelModel must be fitted before prediction.")

        dfs = []
        for label_name, info in self.models.items():
            pipeline = info["pipeline"]
            pred_df = pipeline.predict_df(X, prefix=label_name)
            dfs.append(pred_df)

        return pd.concat(dfs, axis=1)
