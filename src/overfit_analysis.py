from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from config import build_pipeline_config
from evaluation import evaluate_multilabel_model_on_split
from logger import Log
from multi_label_model import MultiLabelModel
from overfit_config import OverfitAnalysisConfig
from plots import save_overfit_metric_plot
from preprocess import load_and_preprocess_data
from split import make_label_specific_splits
from tune_mode import parse_tune_mode


log = Log()

BASE_PLOT_METRICS = ["f1", "roc_auc", "rmse", "r2", "accuracy", "mae"]
FINAL_PLOT_METRICS = ["final_rmse", "final_r2", "final_mae"]
COUNT_SUFFIXES = ("n_samples", "n_positive")
REPORT_IMAGE_WIDTH = 600


class OverfitAnalysisRunner:
    """Run a parameter sweep and summarize overfitting behavior for one label."""

    def __init__(self, analysis_config: OverfitAnalysisConfig, experiment_config, base_config_dict: dict):
        self.analysis_config = analysis_config
        self.experiment_config = experiment_config
        self.base_config_dict = deepcopy(base_config_dict)
        self.analysis_dir = analysis_config.output_root / analysis_config.analysis_name

    def run(self) -> dict:
        self._validate_inputs()
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self._write_resolved_config()
        X, y_class, y_reg = load_and_preprocess_data(self.experiment_config)
        splits = make_label_specific_splits(
            X=X,
            y_class=y_class,
            y_reg=y_reg,
            label_specs=self.experiment_config.labels.label_specs,
            outlier_threshold=self.experiment_config.training.outlier_threshold,
            positive_only_regression=self.experiment_config.training.positive_only_regression,
        )

        rows = []
        for parameter_value in self.analysis_config.values:
            log.h1(f"Overfit analysis run: {self.analysis_config.parameter}={parameter_value}")
            rows.append(self._run_single_value(parameter_value, splits))

        results_df = pd.DataFrame(rows).sort_values("parameter_value").reset_index(drop=True)
        results_df = self._add_gap_columns(results_df)

        results_path = self.analysis_dir / "results.csv"
        results_df.to_csv(results_path, index=False)

        plots_dir = self.analysis_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = self._write_plots(results_df, plots_dir)

        report_path = self.analysis_dir / "report.md"
        report_path.write_text(self._build_report(results_df, plot_paths))

        return {
            "analysis_dir": self.analysis_dir,
            "results_path": results_path,
            "report_path": report_path,
            "plots_dir": plots_dir,
        }

    def _validate_inputs(self) -> None:
        label_names = {label_spec.label_name for label_spec in self.experiment_config.labels.label_specs}
        if self.analysis_config.label_name not in label_names:
            raise ValueError(
                f"Unknown label_name: {self.analysis_config.label_name}. "
                f"Configured labels: {', '.join(sorted(label_names))}"
            )

        if self.experiment_config.models.classifier_model != "xgboost":
            raise ValueError("Overfit analysis v2 supports only xgboost classifier_model.")
        if self.experiment_config.models.regressor_model != "xgboost":
            raise ValueError("Overfit analysis v2 supports only xgboost regressor_model.")

    def _write_resolved_config(self) -> None:
        resolved_config = {
            "analysis_name": self.analysis_config.analysis_name,
            "base_config": str(self.analysis_config.base_config),
            "output_root": str(self.analysis_config.output_root),
            "label_name": self.analysis_config.label_name,
            "parameter": self.analysis_config.parameter,
            "values": self.analysis_config.values,
            "classifier_override": self.analysis_config.classifier_override,
            "regressor_override": self.analysis_config.regressor_override,
            "resolved_base_config": self.base_config_dict,
        }

        with (self.analysis_dir / "analysis_resolved.yaml").open("w") as file:
            yaml.safe_dump(resolved_config, file, sort_keys=False)

    def _run_single_value(self, parameter_value: int | float, splits) -> dict:
        pipeline_config = build_pipeline_config(self.experiment_config)
        pipeline_config.class_param_set.default_params.update(self.analysis_config.classifier_override)
        pipeline_config.reg_param_set.default_params.update(self.analysis_config.regressor_override)
        pipeline_config.class_param_set.default_params[self.analysis_config.parameter] = parameter_value
        pipeline_config.reg_param_set.default_params[self.analysis_config.parameter] = parameter_value

        model = MultiLabelModel(
            label_specs=self.experiment_config.labels.label_specs,
            pipeline_config=pipeline_config,
        )
        model.fit(
            splits=splits,
            class_tune_mode=parse_tune_mode(self.experiment_config.training.class_tune_mode),
            reg_tune_mode=parse_tune_mode(self.experiment_config.training.reg_tune_mode),
        )

        row = {
            "label": self.analysis_config.label_name,
            "parameter": self.analysis_config.parameter,
            "parameter_value": parameter_value,
            "classifier_model": self.experiment_config.models.classifier_model,
            "regressor_model": self.experiment_config.models.regressor_model,
        }

        for split_name in ("train", "val", "test"):
            split_results = evaluate_multilabel_model_on_split(
                model=model,
                splits=splits,
                split_name=split_name,
                label_names=[self.analysis_config.label_name],
            )
            label_result = split_results[self.analysis_config.label_name]
            row.update(self._flatten_split_metrics(split_name, label_result))

        return row

    def _flatten_split_metrics(self, split_name: str, label_result) -> dict:
        row = {}
        for metric_name, metric_value in label_result.classification.to_dict().items():
            row[f"{split_name}_{metric_name}"] = metric_value

        for metric_name, metric_value in label_result.regression.to_dict().items():
            row[f"{split_name}_{metric_name}"] = metric_value

        if label_result.final_regression is not None:
            for metric_name, metric_value in label_result.final_regression.to_dict().items():
                if metric_name == "n_samples":
                    continue
                row[f"{split_name}_final_{metric_name}"] = metric_value

        return row

    def _add_gap_columns(self, results_df: pd.DataFrame) -> pd.DataFrame:
        metric_suffixes = set()
        for column in results_df.columns:
            if column.startswith("train_"):
                metric_suffixes.add(column.removeprefix("train_"))

        for metric_suffix in sorted(metric_suffixes):
            if metric_suffix.endswith(COUNT_SUFFIXES):
                continue

            train_column = f"train_{metric_suffix}"
            val_column = f"val_{metric_suffix}"
            test_column = f"test_{metric_suffix}"
            if val_column in results_df.columns:
                results_df[f"gap_{metric_suffix}_train_val"] = results_df[train_column] - results_df[val_column]
            if test_column in results_df.columns:
                results_df[f"gap_{metric_suffix}_train_test"] = results_df[train_column] - results_df[test_column]

        return results_df

    def _write_plots(self, results_df: pd.DataFrame, plots_dir: Path) -> list[str]:
        plot_paths = []
        for metric_name in BASE_PLOT_METRICS + FINAL_PLOT_METRICS:
            if f"train_{metric_name}" not in results_df.columns:
                continue
            filename = f"{metric_name}_vs_{self.analysis_config.parameter}.png"
            save_overfit_metric_plot(
                results_df=results_df,
                metric_name=metric_name,
                parameter_name=self.analysis_config.parameter,
                out_path=plots_dir / filename,
            )
            plot_paths.append(f"plots/{filename}")

        return plot_paths

    def _build_report(self, results_df: pd.DataFrame, plot_paths: list[str]) -> str:
        best_row = self._select_best_validation_row(results_df)
        onset_value = self._detect_overfit_onset(results_df)
        train_trend = self._describe_series_trend(results_df["train_f1"])
        val_trend = self._describe_series_trend(results_df["val_f1"])
        reg_trend = self._describe_series_trend(results_df["val_rmse"], prefer_lower=True)
        gap_f1_trend = self._describe_series_trend(results_df["gap_f1_train_val"])
        gap_auc_trend = self._describe_series_trend(results_df["gap_roc_auc_train_val"])
        gap_rmse_trend = self._describe_series_trend(results_df["gap_rmse_train_val"])
        gap_r2_trend = self._describe_series_trend(results_df["gap_r2_train_val"])

        onset_text = (
            f"Suspected overfitting begins at `{self.analysis_config.parameter}={onset_value}`."
            if onset_value is not None
            else "No clear overfitting onset was detected in the tested range."
        )

        lines = [
            f"# Overfit Analysis: {self.analysis_config.analysis_name}",
            "",
            "## Summary",
            f"- Label: `{self.analysis_config.label_name}`",
            f"- Parameter swept: `{self.analysis_config.parameter}`",
            f"- Values tested: `{', '.join(str(value) for value in self.analysis_config.values)}`",
            (
                f"- Best validation value: `{best_row['parameter_value']}` "
                f"(selected by `val_f1`, tie-breaker `val_roc_auc`)"
            ),
            "- Early stopping: not used in the current codebase. XGBoost receives a validation `eval_set`, but no early-stopping setting or callback is configured.",
            "",
            "## Train And Validation Behavior",
            f"- Train F1 trend: {train_trend}",
            f"- Validation F1 trend: {val_trend}",
            f"- Validation RMSE trend: {reg_trend}",
            f"- Train-val F1 gap trend: {gap_f1_trend}",
            f"- Train-val ROC AUC gap trend: {gap_auc_trend}",
            f"- Train-val RMSE gap trend: {gap_rmse_trend}",
            f"- Train-val R2 gap trend: {gap_r2_trend}",
            (
                f"- Gap at best validation value: "
                f"`train-val F1 = {best_row['gap_f1_train_val']:.4f}`, "
                f"`train-val ROC AUC = {best_row['gap_roc_auc_train_val']:.4f}`, "
                f"`train-val RMSE = {best_row['gap_rmse_train_val']:.4f}`, "
                f"`train-val R2 = {best_row['gap_r2_train_val']:.4f}`"
            ),
            "",
            "## Overfitting Readout",
            f"- {onset_text}",
            (
                f"- Best validation metrics: "
                f"`F1 = {best_row['val_f1']:.4f}`, "
                f"`ROC AUC = {best_row['val_roc_auc']:.4f}`, "
                f"`RMSE = {best_row['val_rmse']:.4f}`, "
                f"`R2 = {best_row['val_r2']:.4f}`"
            ),
            "",
            "## Plots",
        ]

        for plot_path in plot_paths:
            lines.append(f'<img src="{plot_path}" width="{REPORT_IMAGE_WIDTH}">')
            lines.append("")

        lines.extend([
            "## Notes",
            "- TODO: support separate classifier/regressor sweep targets.",
            "- TODO: support multi-label aggregation or per-label output modes.",
        ])

        if self.experiment_config.training.positive_only_regression:
            lines.append("- TODO: add end-to-end expected-volume metrics for `positive_only_regression=true`.")
            lines.append("- Final-output regression metrics are omitted because this run is using conditional-only regression mode.")
        else:
            lines.append("- Final-output regression metrics are included as `final_*` columns and plots.")

        return "\n".join(lines) + "\n"

    def _select_best_validation_row(self, results_df: pd.DataFrame) -> pd.Series:
        sorted_df = results_df.sort_values(
            by=["val_f1", "val_roc_auc", "parameter_value"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        return sorted_df.iloc[0]

    def _detect_overfit_onset(self, results_df: pd.DataFrame) -> int | float | None:
        best_index = self._select_best_validation_row(results_df).name
        if best_index is None:
            return None

        ordered_df = results_df.reset_index(drop=True)
        best_position = int(ordered_df.index[ordered_df["parameter_value"] == results_df.loc[best_index, "parameter_value"]][0])

        for idx in range(best_position + 1, len(ordered_df)):
            current = ordered_df.iloc[idx]
            previous = ordered_df.iloc[idx - 1]
            if current["val_f1"] < previous["val_f1"] and current["gap_f1_train_val"] > previous["gap_f1_train_val"]:
                return current["parameter_value"]

        return None

    def _describe_series_trend(self, series: pd.Series, prefer_lower: bool = False) -> str:
        clean = series.dropna().reset_index(drop=True)
        if len(clean) < 2:
            return "insufficient data"

        diffs = clean.diff().dropna()
        nondecreasing = (diffs >= -1e-9).all()
        nonincreasing = (diffs <= 1e-9).all()

        if prefer_lower:
            if nonincreasing:
                return "improves monotonically across the tested range"
            if nondecreasing:
                return "worsens monotonically across the tested range"
        else:
            if nondecreasing:
                return "improves monotonically across the tested range"
            if nonincreasing:
                return "declines monotonically across the tested range"

        peak_idx = clean.idxmin() if prefer_lower else clean.idxmax()
        if peak_idx == len(clean) - 1:
            return "generally improves, with the best value at the largest tested complexity"
        if peak_idx == 0:
            return "starts strongest and then degrades as complexity increases"
        return "improves early and then plateaus or declines as complexity increases"
