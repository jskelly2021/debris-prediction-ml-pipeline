
import yaml

from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    data_path: Path
    output_path: Path = Path("outputs")
    class_target_cols: list[str] = field(default_factory=list)
    reg_target_cols: list[str] = field(default_factory=list)
    drop_cols: list[str] = field(default_factory=list)
    log_feature_cols: list[str] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)

    smote: bool = False
    scale_pos_weight: bool = False
    log_regression_features: bool = False
    log_regression_target: bool = False
    positive_only_regression: bool = False

    class_params_dist: dict = field(default_factory=dict)
    reg_params_dist: dict = field(default_factory=dict)
    class_default_params: dict = field(default_factory=dict)
    reg_default_params: dict = field(default_factory=dict)


def load_config(config_path: str) -> TrainConfig:
    config_path = Path(config_path)

    with config_path.open("r") as file:
        config_dict = yaml.safe_load(file)
    
    config_dict["data_path"] = Path(config_dict["data_path"])
    config_dict["output_path"] = Path(config_dict["output_path"])

    return TrainConfig(**config_dict)
