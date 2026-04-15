from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LabelSpec:
    """Map one label name to its class and regression targets."""

    label_name: str
    class_target_col: str
    reg_target_col: str


@dataclass
class DataConfig:
    data_path: Path
    output_path: Path = Path("outputs")
    drop_cols: list[str] = field(default_factory=list)


@dataclass
class LabelConfig:
    label_names: list[str] = field(default_factory=list)
    class_target_cols: list[str] = field(default_factory=list)
    reg_target_cols: list[str] = field(default_factory=list)
    label_specs: list[LabelSpec] = field(init=False)

    def __post_init__(self):
        self.label_specs = _build_label_specs(
            label_names=self.label_names,
            class_target_cols=self.class_target_cols,
            reg_target_cols=self.reg_target_cols,
        )


@dataclass
class PreprocessConfig:
    categorical_cols: list[str] = field(default_factory=list)
    categorical_encoding: dict = field(default_factory=dict)
    feature_filtering: dict = field(default_factory=dict)
    feature_cols_to_log: list[str] = field(default_factory=list)
    log_features: bool = False
    target_encoding_smoothing: float = 10.0


@dataclass
class TrainingConfig:
    outlier_threshold: float = None
    smote: bool = False
    scale_pos_weight: bool = False
    log_target_reg: bool = False
    positive_only_regression: bool = False
    class_tune_mode: str = "none"
    reg_tune_mode: str = "none"


@dataclass
class ModelSelectionConfig:
    classifier_model: str = "xgboost"
    regressor_model: str = "xgboost"
    classifier_params_path: Path | None = None
    regressor_params_path: Path | None = None


@dataclass
class ParamSet:
    default_params: dict = field(default_factory=dict)
    params_dist: dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    classifier_model: str = "xgboost"
    regressor_model: str = "xgboost"
    categorical_cols: list[str] = field(default_factory=list)
    categorical_encoding: dict = field(default_factory=dict)
    feature_filtering: dict = field(default_factory=dict)
    target_encoding_smoothing: float = 10.0
    smote: bool = False
    scale_pos_weight: bool = False
    log_target_reg: bool = False
    positive_only_regression: bool = False
    class_param_set: ParamSet = field(default_factory=ParamSet)
    reg_param_set: ParamSet = field(default_factory=ParamSet)


@dataclass
class ExperimentConfig:
    data: DataConfig
    labels: LabelConfig
    preprocessing: PreprocessConfig
    training: TrainingConfig
    models: ModelSelectionConfig


def _build_label_specs(label_names: list[str], class_target_cols: list[str], reg_target_cols: list[str]) -> list[LabelSpec]:
    """Build aligned label specs from config target lists."""

    if not (len(label_names) == len(class_target_cols) == len(reg_target_cols)):
        raise ValueError("label_names, class_target_cols, and reg_target_cols must have the same length.")

    return [
        LabelSpec(
            label_name=label_name,
            class_target_col=class_target_col,
            reg_target_col=reg_target_col,
        )
        for label_name, class_target_col, reg_target_col in zip(
            label_names,
            class_target_cols,
            reg_target_cols,
        )
    ]


def _path_or_none(value) -> Path | None:
    """Convert a path-like config value to Path, preserving null values."""

    if value is None:
        return None
    return Path(value)


def _select_param_section(param_dict: dict, section: str | None) -> dict:
    """Select a model-specific param section when a shared param file is used."""

    if section is not None and section in param_dict:
        section_dict = param_dict[section] or {}
        if not isinstance(section_dict, dict):
            raise ValueError(f"Parameter section must contain a mapping: {section}")
        return section_dict

    return param_dict


def load_param_set(path: Path | None, section: str | None = None) -> ParamSet:
    """Load model defaults and search space from a parameter YAML file."""

    if path is None:
        return ParamSet()

    with Path(path).open("r") as file:
        param_dict = yaml.safe_load(file) or {}

    if not isinstance(param_dict, dict):
        raise ValueError(f"Parameter file must contain a mapping: {path}")

    param_dict = _select_param_section(param_dict, section)

    return ParamSet(
        default_params=param_dict.get("default_params", {}) or {},
        params_dist=param_dict.get("params_dist", param_dict.get("param_dist", {})) or {},
    )


def _nested_config_dict(config_dict: dict) -> dict:
    """Return a nested config dict, with temporary support for legacy flat YAML."""

    if any(key in config_dict for key in ("data", "labels", "preprocessing", "training", "models")):
        return config_dict

    return {
        "data": {
            "data_path": config_dict["data_path"],
            "output_path": config_dict.get("output_path", Path("outputs")),
            "drop_cols": config_dict.get("drop_cols", []),
        },
        "labels": {
            "label_names": config_dict.get("label_names", []),
            "class_target_cols": config_dict.get("class_target_cols", []),
            "reg_target_cols": config_dict.get("reg_target_cols", []),
        },
        "preprocessing": {
            "categorical_cols": config_dict.get("categorical_cols", []),
            "categorical_encoding": config_dict.get("categorical_encoding", {}),
            "feature_filtering": config_dict.get("feature_filtering", {}),
            "feature_cols_to_log": config_dict.get("feature_cols_to_log", []),
            "log_features": config_dict.get("log_features", False),
            "target_encoding_smoothing": config_dict.get("target_encoding_smoothing", 10.0),
        },
        "training": {
            "outlier_threshold": config_dict.get("outlier_threshold"),
            "smote": config_dict.get("smote", False),
            "scale_pos_weight": config_dict.get("scale_pos_weight", False),
            "log_target_reg": config_dict.get("log_target_reg", False),
            "positive_only_regression": config_dict.get("positive_only_regression", False),
            "class_tune_mode": config_dict.get("class_tune_mode", "none"),
            "reg_tune_mode": config_dict.get("reg_tune_mode", "none"),
        },
        "models": {
            "classifier_model": config_dict.get("classifier_model", "xgboost"),
            "regressor_model": config_dict.get("regressor_model", "xgboost"),
            "classifier_params_path": config_dict.get("classifier_params_path"),
            "regressor_params_path": config_dict.get("regressor_params_path"),
        },
    }


def build_pipeline_config(experiment_config: ExperimentConfig) -> PipelineConfig:
    """Resolve experiment settings into the config consumed by model training."""

    return PipelineConfig(
        classifier_model=experiment_config.models.classifier_model,
        regressor_model=experiment_config.models.regressor_model,
        categorical_cols=experiment_config.preprocessing.categorical_cols,
        categorical_encoding=experiment_config.preprocessing.categorical_encoding,
        feature_filtering=experiment_config.preprocessing.feature_filtering,
        target_encoding_smoothing=experiment_config.preprocessing.target_encoding_smoothing,
        smote=experiment_config.training.smote,
        scale_pos_weight=experiment_config.training.scale_pos_weight,
        log_target_reg=experiment_config.training.log_target_reg,
        positive_only_regression=experiment_config.training.positive_only_regression,
        class_param_set=load_param_set(experiment_config.models.classifier_params_path, section="classifier"),
        reg_param_set=load_param_set(experiment_config.models.regressor_params_path, section="regressor"),
    )


def config_from_dict(config_dict: dict) -> ExperimentConfig:
    """Build an experiment config from a plain dictionary."""

    config_dict = _nested_config_dict(deepcopy(config_dict))

    data_dict = config_dict.get("data", {})
    data_dict["data_path"] = Path(data_dict["data_path"])
    data_dict["output_path"] = Path(data_dict.get("output_path", Path("outputs")))

    models_dict = config_dict.get("models", {})
    models_dict["classifier_params_path"] = _path_or_none(models_dict.get("classifier_params_path"))
    models_dict["regressor_params_path"] = _path_or_none(models_dict.get("regressor_params_path"))

    return ExperimentConfig(
        data=DataConfig(**data_dict),
        labels=LabelConfig(**config_dict.get("labels", {})),
        preprocessing=PreprocessConfig(**config_dict.get("preprocessing", {})),
        training=TrainingConfig(**config_dict.get("training", {})),
        models=ModelSelectionConfig(**models_dict),
    )


def load_config(config_path: str) -> ExperimentConfig:
    """Load a YAML experiment config.

    Args:
        config_path: Path to a YAML config file.
    """

    config_path = Path(config_path)

    with config_path.open("r") as file:
        config_dict = yaml.safe_load(file) or {}

    return config_from_dict(config_dict)
