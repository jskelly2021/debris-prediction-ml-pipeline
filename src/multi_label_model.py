
import pandas as pd

from src.two_head_pipeline import TwoHeadPipeline, ParamTuningMode
from src.metrics import compute_classification_metrics, compute_regression_metrics


class MultiLabelModel:
    def __init__(
        self,
        pipeline_factory,
        class_target_cols,
        reg_target_cols,
        label_names=None,
        verbose=False
    ):
        if len(class_target_cols) != len(reg_target_cols):
            raise ValueError("class_target_cols and reg_target_cols must have the same length.")

        self.pipeline_factory = pipeline_factory
        self.class_target_cols = class_target_cols
        self.reg_target_cols = reg_target_cols
        self.label_names = label_names or reg_target_cols
        self.verbose = verbose

        self.models = {}
        self.is_fitted = False


    def fit(
        self,
        splits,
        class_tune_mode=ParamTuningMode.NONE,
        reg_tune_mode=ParamTuningMode.NONE
    ):
        for i, label_name in enumerate(self.label_names):
            class_col = self.class_target_cols[i]
            reg_col = self.reg_target_cols[i]

            if self.verbose:
                print("\n" + "=" * 60)
                print(f"TRAINING LABEL: {label_name}")
                print(f"class target: {class_col}")
                print(f"reg target  : {reg_col}")
                print("=" * 60)

            pipeline = self.pipeline_factory()

            pipeline.train(
                splits=splits,
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


    def evaluate(self, X, y_class, y_reg):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        pred_df = self.predict(X)
        results = {}

        for label_name, info in self.models.items():
            class_col = info["class_col"]
            reg_col = info["reg_col"]

            class_metrics = compute_classification_metrics(
                y_true=y_class[class_col],
                y_pred=pred_df[f"{label_name}_class_pred"],
                y_prob=pred_df[f"{label_name}_class_prob"],
            )

            reg_metrics_all = compute_regression_metrics(
                y_true=y_reg[reg_col],
                y_pred=pred_df[f"{label_name}_expected_volume_pred"],
            )

            positive_mask = y_class[class_col] == 1

            reg_metrics_pos = None
            if positive_mask.sum() > 0:
                reg_metrics_pos = compute_regression_metrics(
                    y_true=y_reg.loc[positive_mask, reg_col],
                    y_pred=pred_df.loc[positive_mask, f"{label_name}_reg_pred"],
                )

            results[label_name] = {
                "classification": class_metrics,
                "regression_all": reg_metrics_all,
                "regression_positive_only": reg_metrics_pos,
            }

        self.metrics_ = results
        return results


    def print_metrics(self, metrics=None):
        metrics = metrics or getattr(self, "metrics_", None)

        if metrics is None:
            raise ValueError("No metrics available. Run evaluate() first.")

        for label_name, label_metrics in metrics.items():
            print("\n" + "=" * 60)
            print(f"METRICS FOR {label_name}")
            print("=" * 60)

            label_metrics["classification"].print(label_name)

            label_metrics["regression_all"].print(f"{label_name} Overall Expected Volume")

            pos_metrics = label_metrics["regression_positive_only"]
            if pos_metrics is not None:
                pos_metrics.print(f"{label_name} Positive-Only Volume")
            else:
                print(f"No positive-only regression metrics available for {label_name}.")


    def metrics_to_dataframe(self, metrics=None):
        metrics = metrics or getattr(self, "metrics_", None)

        if metrics is None:
            raise ValueError("No metrics available. Run evaluate() first.")

        rows = []

        for label_name, label_metrics in metrics.items():
            c = label_metrics["classification"]
            rows.append({
                "label": label_name,
                "metric_type": "classification",
                **c.to_dict()
            })

            r_all = label_metrics["regression_all"]
            rows.append({
                "label": label_name,
                "metric_type": "regression_all",
                **r_all.to_dict()
            })

            r_pos = label_metrics["regression_positive_only"]
            if r_pos is not None:
                rows.append({
                    "label": label_name,
                    "metric_type": "regression_positive_only",
                    **r_pos.to_dict()
                })

        return pd.DataFrame(rows)
