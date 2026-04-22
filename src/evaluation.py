from dataclasses import dataclass

from metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    compute_classification_metrics,
    compute_regression_metrics,
)


@dataclass
class LabelEvaluationResult:
    """Store classification and regression metrics for one label."""

    classification: ClassificationMetrics
    regression: RegressionMetrics
    final_regression: RegressionMetrics | None
    class_col: str
    reg_col: str
    regression_display_name: str
    regression_prediction_column: str


def _get_split_inputs(split, split_name: str):
    if split_name == "train":
        return (
            split.X_train_class,
            split.y_train_class,
            split.X_train_reg,
            split.y_train_reg,
        )
    if split_name == "val":
        return (
            split.X_val_class,
            split.y_val_class,
            split.X_val_reg,
            split.y_val_reg,
        )
    if split_name == "test":
        return (
            split.X_test_class,
            split.y_test_class,
            split.X_test_reg,
            split.y_test_reg,
        )

    raise ValueError(f"Unsupported split_name: {split_name}. Expected one of train, val, test.")


def evaluate_multilabel_model_on_split(model, splits, split_name: str, label_names: list[str] | None = None):
    """Evaluate a fitted multi-label model on one named split.

    Args:
        splits: Label-specific split mapping.
    """

    if not getattr(model, "is_fitted", False):
        raise ValueError("Model must be fitted before evaluation.")

    results = {}
    label_name_filter = None if label_names is None else set(label_names)

    for label_name, info in model.models.items():
        if label_name_filter is not None and label_name not in label_name_filter:
            continue

        pipeline = info["pipeline"]
        class_col = info["class_col"]
        reg_col = info["reg_col"]
        split = splits[label_name]
        X_class, y_class, X_reg, y_reg = _get_split_inputs(split, split_name)

        class_preds = pipeline.predict_df(
            X_class,
            prefix=label_name
        )

        reg_preds = pipeline.predict_df(X_reg, prefix=label_name)

        if model.pipeline_config.positive_only_regression:
            regression_display_name = "Conditional Volume"
            regression_prediction_column = f"{label_name}_reg_pred"
        else:
            regression_display_name = "Expected Volume"
            regression_prediction_column = f"{label_name}_expected_volume_pred"

        class_metrics = compute_classification_metrics(
            y_true=y_class,
            y_pred=class_preds[f"{label_name}_class_pred"],
            y_prob=class_preds[f"{label_name}_class_prob"],
        )

        reg_metrics = compute_regression_metrics(
            y_true=y_reg,
            y_pred=reg_preds[regression_prediction_column],
        )
        final_regression = None
        if not model.pipeline_config.positive_only_regression:
            final_regression = compute_regression_metrics(
                y_true=y_reg,
                y_pred=reg_preds[f"{label_name}_expected_volume_pred"],
            )

        results[label_name] = LabelEvaluationResult(
            classification=class_metrics,
            regression=reg_metrics,
            final_regression=final_regression,
            class_col=class_col,
            reg_col=reg_col,
            regression_display_name=regression_display_name,
            regression_prediction_column=regression_prediction_column,
        )

    return results


def evaluate_multilabel_model(model, splits):
    """Evaluate a fitted multi-label model on test splits."""

    return evaluate_multilabel_model_on_split(
        model=model,
        splits=splits,
        split_name="test",
    )
