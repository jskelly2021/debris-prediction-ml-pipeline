from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentDefinition:
    """One batch experiment and its config overrides."""

    name: str
    run_id: str
    overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Experiment name must be a non-empty string.")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(f"Experiment {self.name!r} must define a non-empty run_id.")
        if not isinstance(self.overrides, dict):
            raise ValueError(f"Experiment {self.name!r} overrides must be a mapping.")
