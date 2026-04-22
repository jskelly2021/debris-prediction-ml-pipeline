from dataclasses import dataclass, field
from pathlib import Path

import yaml


INTEGER_SWEEP_PARAMETERS = {
    "max_depth",
    "n_estimators",
    "min_child_weight",
}
FLOAT_SWEEP_PARAMETERS = {
    "learning_rate",
    "gamma",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
}
SUPPORTED_OVERFIT_PARAMETERS = INTEGER_SWEEP_PARAMETERS | FLOAT_SWEEP_PARAMETERS


@dataclass(frozen=True)
class OverfitAnalysisConfig:
    analysis_name: str
    base_config: Path
    label_name: str
    parameter: str
    values: list[int | float]
    output_root: Path = Path("outputs/analyses")
    classifier_override: dict = field(default_factory=dict)
    regressor_override: dict = field(default_factory=dict)


def _normalize_sweep_values(parameter: str, values) -> list[int | float]:
    if not isinstance(values, list) or not values:
        raise ValueError("Overfit analysis config must define a non-empty values list.")

    normalized_values = []
    if parameter in INTEGER_SWEEP_PARAMETERS:
        for value in values:
            if not isinstance(value, int):
                raise ValueError(
                    f"Overfit analysis values for {parameter} must be integers."
                )
            normalized_values.append(value)
        return normalized_values

    if parameter in FLOAT_SWEEP_PARAMETERS:
        for value in values:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Overfit analysis values for {parameter} must be numeric."
                )
            normalized_values.append(float(value))
        return normalized_values

    supported = ", ".join(sorted(SUPPORTED_OVERFIT_PARAMETERS))
    raise ValueError(f"Unsupported overfit parameter: {parameter}. Supported parameters: {supported}")


def load_overfit_analysis_config(config_path: str | Path) -> OverfitAnalysisConfig:
    config_path = Path(config_path)
    with config_path.open("r") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"Overfit analysis config must contain a mapping: {config_path}")

    analysis_name = raw_config.get("analysis_name")
    if not isinstance(analysis_name, str) or not analysis_name.strip():
        raise ValueError("Overfit analysis config must define a non-empty analysis_name.")

    base_config = raw_config.get("base_config")
    if not base_config:
        raise ValueError("Overfit analysis config must define base_config.")

    label_name = raw_config.get("label_name")
    if not isinstance(label_name, str) or not label_name.strip():
        raise ValueError("Overfit analysis config must define a non-empty label_name.")

    parameter = raw_config.get("parameter")
    if parameter not in SUPPORTED_OVERFIT_PARAMETERS:
        supported = ", ".join(sorted(SUPPORTED_OVERFIT_PARAMETERS))
        raise ValueError(f"Unsupported overfit parameter: {parameter}. Supported parameters: {supported}")

    values = raw_config.get("values")
    normalized_values = _normalize_sweep_values(parameter, values)

    classifier_override = raw_config.get("classifier_override", {}) or {}
    regressor_override = raw_config.get("regressor_override", {}) or {}
    if not isinstance(classifier_override, dict):
        raise ValueError("classifier_override must be a mapping.")
    if not isinstance(regressor_override, dict):
        raise ValueError("regressor_override must be a mapping.")

    return OverfitAnalysisConfig(
        analysis_name=analysis_name,
        base_config=Path(base_config),
        output_root=Path(raw_config.get("output_root", "outputs/analyses")),
        label_name=label_name,
        parameter=parameter,
        values=normalized_values,
        classifier_override=classifier_override,
        regressor_override=regressor_override,
    )
