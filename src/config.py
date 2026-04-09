
import yaml

from pathlib import Path
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabelSpec:
    """Map one label name to its class and regression targets."""

    label_name: str
    class_target_col: str
    reg_target_col: str


@dataclass
class TrainConfig:
    """Hold YAML-driven training settings.

    Attributes:
        data_path: CSV input path.
        label_specs: Per-label class/regression target mappings.
    """

    data_path: Path
    output_path: Path = Path("outputs")
    class_target_cols: list[str] = field(default_factory=list)
    reg_target_cols: list[str] = field(default_factory=list)
    drop_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    categorical_encoding: dict = field(default_factory=dict)
    feature_filtering: dict = field(default_factory=dict)
    feature_cols_to_log: list[str] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)

    outlier_threshold: float = None
    smote: bool = False
    scale_pos_weight: bool = False
    log_features: bool = False
    log_target_reg: bool = False
    positive_only_regression: bool = False
    target_encoding_smoothing: float = 10.0

    class_params_dist: dict = field(default_factory=dict)
    reg_params_dist: dict = field(default_factory=dict)
    class_default_params: dict = field(default_factory=dict)
    reg_default_params: dict = field(default_factory=dict)
    label_specs: list[LabelSpec] = field(init=False)

    def __post_init__(self):
        self.label_specs = _build_label_specs(
            label_names=self.label_names,
            class_target_cols=self.class_target_cols,
            reg_target_cols=self.reg_target_cols,
        )


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


def load_config(config_path: str) -> TrainConfig:
    """Load a YAML training config.

    Args:
        config_path: Path to a YAML config file.
    """

    config_path = Path(config_path)

    with config_path.open("r") as file:
        config_dict = yaml.safe_load(file)

    config_dict["data_path"] = Path(config_dict["data_path"])
    if "output_path" in config_dict:
        config_dict["output_path"] = Path(config_dict["output_path"])

    return TrainConfig(**config_dict)
