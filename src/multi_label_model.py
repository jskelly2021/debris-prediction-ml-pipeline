
import pandas as pd

from two_head_pipeline import TuneMode, TwoHeadPipeline
from metrics import compute_classification_metrics, compute_regression_metrics
from config import TrainConfig
from split import Splits
from logger import Log
from plots import print_top_features


log = Log()


class MultiLabelModel:
    def __init__(
        self,
        trainConfig: TrainConfig,
        splits: dict[str, Splits]
    ):
        if len(trainConfig.class_target_cols) != len(trainConfig.reg_target_cols):
            raise ValueError("class_target_cols and reg_target_cols must have the same length.")

        self.trainConfig = trainConfig
        self.splits = splits
        self.models = {}
        self.is_fitted = False


    def fit(
        self,
        class_tune_mode: TuneMode=TuneMode.NONE,
        reg_tune_mode: TuneMode=TuneMode.NONE
    ):
        for i, label_name in enumerate(self.trainConfig.label_names):
            class_col = self.trainConfig.class_target_cols[i]
            reg_col = self.trainConfig.reg_target_cols[i]

            log.h1(f"Training model for label: {label_name}")

            pipeline = TwoHeadPipeline(
                pipelineConfig=self.trainConfig
            )

            pipeline.train(
                splits=self.splits[label_name],
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


    def evaluate(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        results = {}

        for label_name, info in self.models.items():
            pipeline = info["pipeline"]
            class_col = info["class_col"]
            reg_col = info["reg_col"]

            class_preds = pipeline.predict_df(
                self.splits[label_name].X_test_class,
                prefix=label_name
            )

            reg_preds = pipeline.predict_df(
                self.splits[label_name].X_test_reg,
                prefix=label_name
            )

            class_metrics = compute_classification_metrics(
                y_true=self.splits[label_name].y_test_class,
                y_pred=class_preds[f"{label_name}_class_pred"],
                y_prob=class_preds[f"{label_name}_class_prob"],
            )

            reg_metrics = compute_regression_metrics(
                y_true=self.splits[label_name].y_test_reg,
                y_pred=reg_preds[f"{label_name}_expected_volume_pred"],
            )

            results[label_name] = {
                "classification": class_metrics,
                "regression": reg_metrics,
                "class_col": class_col,
                "reg_col": reg_col,
            }

        self.metrics_ = results
        return results


    def print_metrics(self, metrics=None):
        metrics = metrics or getattr(self, "metrics_", None)

        if metrics is None:
            raise ValueError("No metrics available. Run evaluate() first.")

        for label_name, label_metrics in metrics.items():
            log.h1(f"Metrics for {label_name}")
            label_metrics["classification"].print(label_name)
            label_metrics["regression"].print(f"{label_name} Expected Volume")


    def metrics_to_dataframe(self, run_id=None, metrics=None,):
        metrics = metrics or getattr(self, "metrics_", None)

        if metrics is None:
            raise ValueError("No metrics available. Run evaluate() first.")

        rows = []

        for label_name, label_metrics in metrics.items():
            c = label_metrics["classification"]
            r = label_metrics["regression"]

            n_features = self.splits[label_name].X_test_class.shape[1]

            rows.append({
                "run_id": run_id,
                "label": label_name,
                "class_target": label_metrics["class_col"],
                "reg_target": label_metrics["reg_col"],
                "n_features": n_features,

                "smote": self.trainConfig.smote,
                "scale_pos_weight": self.trainConfig.scale_pos_weight,
                "log_features": self.trainConfig.log_features,
                "log_target_reg": self.trainConfig.log_target_reg,
                "outlier_threshold": self.trainConfig.outlier_threshold,

                "positive_rate": c.positive_rate,
                "class_n_samples": c.n_samples,
                "n_positive": c.n_positive,
                "accuracy": c.accuracy,
                "precision": c.precision,
                "recall": c.recall,
                "f1": c.f1,
                "roc_auc": c.roc_auc,

                "reg_n_samples": r.n_samples,
                "rmse": r.rmse,
                "mae": r.mae,
                "r2": r.r2,
            })

        return pd.DataFrame(rows)


    def print_feature_importance(self, top_n=10):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before printing feature importance.")

        for label_name, info in self.models.items():
            feature_names = self.splits[label_name].X_train_class.columns
            pipeline = info["pipeline"]

            log.h1(f"FEATURE IMPORTANCE FOR {label_name}")

            print_top_features(
                model=pipeline.head1,
                feature_names=feature_names,
                label=f"{label_name} Classifier",
                top_n=top_n,
            )

            if pipeline.head2 is not None:
                print_top_features(
                    model=pipeline.head2,
                    feature_names=feature_names,
                    label=f"{label_name} Regressor",
                    top_n=top_n,
                )
            else:
                log.warn(f"No regressor trained for {label_name}.")


    def feature_importance_to_dataframe(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before exporting feature importance.")

        rows = []

        for label_name, info in self.models.items():
            pipeline = info["pipeline"]
            feature_names = self.splits[label_name].X_train_class.columns

            clf_importances = pipeline.head1.feature_importances_
            for feature, importance in zip(feature_names, clf_importances):
                rows.append({
                    "label": label_name,
                    "head_type": "classifier",
                    "feature": feature,
                    "importance": float(importance),
                })

            if pipeline.head2 is not None:
                reg_importances = pipeline.head2.feature_importances_
                for feature, importance in zip(feature_names, reg_importances):
                    rows.append({
                        "label": label_name,
                        "head_type": "regressor",
                        "feature": feature,
                        "importance": float(importance),
                    })

        return pd.DataFrame(rows)


    def summarize_feature_importance(self, top_k=10):
        fi_df = self.feature_importance_to_dataframe().copy()

        fi_df["rank_within_head"] = (
            fi_df.groupby(["label", "head_type"])["importance"]
            .rank(method="min", ascending=False)
        )

        summary = (
            fi_df.groupby("feature")
            .agg(
                mean_importance=("importance", "mean"),
                median_importance=("importance", "median"),
                max_importance=("importance", "max"),
                nonzero_count=("importance", lambda s: int((s > 0).sum())),
                topk_count=("rank_within_head", lambda s: int((s <= top_k).sum())),
            )
            .reset_index()
            .sort_values(["mean_importance", "max_importance"], ascending=False)
        )

        return fi_df, summary
