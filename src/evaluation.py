from dataclasses import dataclass

from metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    compute_classification_metrics,
    compute_regression_metrics,
)


@dataclass
class LabelEvaluationResult:
    classification: ClassificationMetrics
    regression: RegressionMetrics
    class_col: str
    reg_col: str
    regression_display_name: str
    regression_prediction_column: str


def evaluate_multilabel_model(model, splits):
    if not getattr(model, "is_fitted", False):
        raise ValueError("Model must be fitted before evaluation.")

    results = {}

    for label_name, info in model.models.items():
        pipeline = info["pipeline"]
        class_col = info["class_col"]
        reg_col = info["reg_col"]

        class_preds = pipeline.predict_df(
            splits[label_name].X_test_class,
            prefix=label_name
        )

        reg_preds = pipeline.predict_df(
            splits[label_name].X_test_reg,
            prefix=label_name
        )

        if model.train_config.positive_only_regression:
            regression_display_name = "Conditional Volume"
            regression_prediction_column = f"{label_name}_reg_pred"
        else:
            regression_display_name = "Expected Volume"
            regression_prediction_column = f"{label_name}_expected_volume_pred"

        class_metrics = compute_classification_metrics(
            y_true=splits[label_name].y_test_class,
            y_pred=class_preds[f"{label_name}_class_pred"],
            y_prob=class_preds[f"{label_name}_class_prob"],
        )

        reg_metrics = compute_regression_metrics(
            y_true=splits[label_name].y_test_reg,
            y_pred=reg_preds[regression_prediction_column],
        )

        results[label_name] = LabelEvaluationResult(
            classification=class_metrics,
            regression=reg_metrics,
            class_col=class_col,
            reg_col=reg_col,
            regression_display_name=regression_display_name,
            regression_prediction_column=regression_prediction_column,
        )

    return results
