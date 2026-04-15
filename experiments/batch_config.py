from dataclasses import dataclass
from pathlib import Path

import yaml

from experiments.experiment_definition import ExperimentDefinition


@dataclass(frozen=True)
class BatchConfig:
    """Parsed batch experiment config."""

    batch_name: str
    base_config: Path
    output_root: Path
    experiments: list[ExperimentDefinition]


def load_batch_config(batch_config_path: str | Path) -> BatchConfig:
    """Load a batch YAML file."""

    batch_config_path = Path(batch_config_path)
    with batch_config_path.open("r") as file:
        raw_config = yaml.safe_load(file) or {}

    batch_name = raw_config.get("batch_name")
    if not isinstance(batch_name, str) or not batch_name.strip():
        raise ValueError("Batch config must define a non-empty batch_name.")

    base_config = raw_config.get("base_config")
    if not base_config:
        raise ValueError("Batch config must define base_config.")

    experiments = raw_config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Batch config must define a non-empty experiments list.")

    return BatchConfig(
        batch_name=batch_name,
        base_config=Path(base_config),
        output_root=Path(raw_config.get("output_root", "outputs/batches")),
        experiments=[
            ExperimentDefinition(
                name=experiment.get("name"),
                run_id=experiment.get("run_id", experiment.get("name")),
                overrides=experiment.get("overrides", {}) or {},
            )
            for experiment in experiments
        ],
    )
