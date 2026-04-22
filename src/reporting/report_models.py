from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExperimentRunStatus:
    """One experiment status entry from the batch manifest."""

    name: str
    run_id: str
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class BatchManifest:
    """Loaded batch manifest metadata."""

    batch_name: str
    base_config: str
    output_path: str
    experiment_count: int
    succeeded_count: int
    failed_count: int
    timestamp: str
    runs: list[ExperimentRunStatus]


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Discovered artifacts for one experiment directory."""

    name: str
    run_id: str
    status: str
    experiment_dir: Path
    config_resolved_path: Path | None
    plots_dir: Path | None
    plots_dir_exists: bool
    plot_files: list[Path] = field(default_factory=list)
    error_message: str | None = None


@dataclass(frozen=True)
class BatchArtifacts:
    """All loaded inputs required to build a research packet."""

    batch_dir: Path
    manifest_path: Path
    combined_results_path: Path
    failed_runs_path: Path
    manifest: BatchManifest
    combined_results: pd.DataFrame
    failed_runs: pd.DataFrame
    experiments: list[ExperimentArtifacts]


@dataclass(frozen=True)
class NotableFinding:
    """A deterministic notable finding rendered into the report."""

    category: str
    message: str


@dataclass(frozen=True)
class PacketArtifacts:
    """Generated packet output metadata."""

    batch_dir: Path
    report_path: Path
    packet_dir: Path
    packet_manifest_path: Path
    figures_dir: Path
    tables_dir: Path
    figure_paths: dict[str, Path]
    table_paths: dict[str, Path]
