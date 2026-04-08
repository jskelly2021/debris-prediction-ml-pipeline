
import pandas as pd

from two_head_pipeline import TuneMode, TwoHeadPipeline
from config import TrainConfig
from split import Splits
from logger import Log


log = Log()


class MultiLabelModel:
    def __init__(
        self,
        train_config: TrainConfig,
    ):
        if len(train_config.class_target_cols) != len(train_config.reg_target_cols):
            raise ValueError("class_target_cols and reg_target_cols must have the same length.")

        self.train_config = train_config
        self.models = {}
        self.is_fitted = False


    def fit(
        self,
        splits: dict[str, Splits],
        class_tune_mode: TuneMode=TuneMode.NONE,
        reg_tune_mode: TuneMode=TuneMode.NONE
    ):
        for i, label_name in enumerate(self.train_config.label_names):
            class_col = self.train_config.class_target_cols[i]
            reg_col = self.train_config.reg_target_cols[i]

            log.h1(f"Training model for label: {label_name}")

            pipeline = TwoHeadPipeline(
                pipelineConfig=self.train_config
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
        if not self.is_fitted:
            raise ValueError("MultiLabelModel must be fitted before prediction.")

        dfs = []
        for label_name, info in self.models.items():
            pipeline = info["pipeline"]
            pred_df = pipeline.predict_df(X, prefix=label_name)
            dfs.append(pred_df)

        return pd.concat(dfs, axis=1)
