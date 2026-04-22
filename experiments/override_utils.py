from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path

import yaml

from config import (
    DataConfig,
    ExperimentConfig,
    LabelConfig,
    ModelSelectionConfig,
    PreprocessConfig,
    TrainingConfig,
    _nested_config_dict,
    build_pipeline_config,
    config_from_dict,
)


SUPPORTED_MODELS = {
    "xgboost",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "adaboost",
}

CONFIG_SCHEMA = {
    "data": DataConfig,
    "labels": LabelConfig,
    "preprocessing": PreprocessConfig,
    "training": TrainingConfig,
    "models": ModelSelectionConfig,
}
REPLACE_DICT_SENTINEL = "__replace__"


def load_config_dict(config_path: str | Path) -> dict:
    """Load and normalize an experiment YAML file as a nested dictionary."""

    config_path = Path(config_path)
    with config_path.open("r") as file:
        config_dict = yaml.safe_load(file) or {}

    if not isinstance(config_dict, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return _nested_config_dict(config_dict)


def deep_merge_config(base_config: dict, overrides: dict) -> dict:
    """Return a config dict with validated recursive overrides applied."""

    if not isinstance(overrides, dict):
        raise ValueError("Overrides must be a mapping.")

    merged_config = deepcopy(base_config)
    _deep_merge(merged_config, overrides, path=())
    return merged_config


def validate_resolved_config(config_dict: dict) -> ExperimentConfig:
    """Validate a merged config and return its dataclass representation."""

    experiment_config = config_from_dict(config_dict)
    _validate_model_name("classifier_model", experiment_config.models.classifier_model)
    _validate_model_name("regressor_model", experiment_config.models.regressor_model)
    _validate_params_path("classifier_params_path", experiment_config.models.classifier_params_path)
    _validate_params_path("regressor_params_path", experiment_config.models.regressor_params_path)
    build_pipeline_config(experiment_config)
    return experiment_config


def _deep_merge(target: dict, overrides: dict, path: tuple[str, ...]) -> None:
    for key, value in overrides.items():
        _validate_key(target, key, path)

        current_value = target.get(key)
        if isinstance(current_value, dict) and isinstance(value, dict):
            if value.get(REPLACE_DICT_SENTINEL) is True:
                replacement = deepcopy(value)
                replacement.pop(REPLACE_DICT_SENTINEL, None)
                _validate_nested_keys(current_value, replacement, path + (key,))
                target[key] = replacement
            else:
                _deep_merge(current_value, value, path + (key,))
        else:
            target[key] = deepcopy(value)


def _validate_key(target: dict, key: str, path: tuple[str, ...]) -> None:
    if not isinstance(key, str):
        raise ValueError(f"Override path contains non-string key at {_format_path(path)}.")

    if not path:
        if key not in CONFIG_SCHEMA:
            raise ValueError(f"Unknown config section: {key}")
        return

    section = path[0]
    if len(path) == 1 and section in CONFIG_SCHEMA:
        valid_fields = _dataclass_field_names(CONFIG_SCHEMA[section])
        if key not in valid_fields:
            raise ValueError(f"Unknown config key: {_format_path(path + (key,))}")
        return

    if key not in target:
        parent_path = _format_path(path)
        raise ValueError(f"Unknown nested config key under {parent_path}: {key}")


def _validate_nested_keys(target: dict, overrides: dict, path: tuple[str, ...]) -> None:
    for key, value in overrides.items():
        if key == REPLACE_DICT_SENTINEL:
            continue

        _validate_key(target, key, path)
        current_value = target.get(key)
        if isinstance(current_value, dict) and isinstance(value, dict):
            _validate_nested_keys(current_value, value, path + (key,))


def _dataclass_field_names(config_class) -> set[str]:
    if not is_dataclass(config_class):
        return set()
    return {
        field.name
        for field in fields(config_class)
        if field.init
    }


def _validate_model_name(field_name: str, model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported {field_name}: {model_name}. Supported models: {supported}")


def _validate_params_path(field_name: str, params_path: Path | None) -> None:
    if params_path is not None and not params_path.exists():
        raise ValueError(f"{field_name} does not exist: {params_path}")


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"
