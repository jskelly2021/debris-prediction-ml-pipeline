import json
from pathlib import Path

import pandas as pd

from reporting.report_models import (
    BatchArtifacts,
    BatchManifest,
    ExperimentArtifacts,
    ExperimentRunStatus,
)


REQUIRED_BATCH_FILES = [
    "batch_manifest.json",
    "combined_results.csv",
    "failed_runs.csv",
]


def validate_batch_dir(batch_dir: Path) -> None:
    """Validate that the batch directory and root artifacts exist."""

    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
    if not batch_dir.is_dir():
        raise ValueError(f"Batch path is not a directory: {batch_dir}")

    for filename in REQUIRED_BATCH_FILES:
        path = batch_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required batch artifact is missing: {path}")


def load_manifest(batch_dir: Path) -> tuple[BatchManifest, Path]:
    """Load the batch manifest JSON."""

    manifest_path = batch_dir / "batch_manifest.json"
    with manifest_path.open("r") as file:
        raw_manifest = json.load(file)

    runs = [
        ExperimentRunStatus(
            name=run["name"],
            run_id=run["run_id"],
            status=run["status"],
            error_message=run.get("error_message"),
        )
        for run in raw_manifest.get("runs", [])
    ]

    manifest = BatchManifest(
        batch_name=raw_manifest["batch_name"],
        base_config=raw_manifest["base_config"],
        output_path=raw_manifest["output_path"],
        experiment_count=raw_manifest["experiment_count"],
        succeeded_count=raw_manifest["succeeded_count"],
        failed_count=raw_manifest["failed_count"],
        timestamp=raw_manifest["timestamp"],
        runs=runs,
    )
    return manifest, manifest_path


def load_combined_results(batch_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Load combined batch results."""

    combined_results_path = batch_dir / "combined_results.csv"
    combined_results = pd.read_csv(combined_results_path)
    return combined_results, combined_results_path


def load_failed_runs(batch_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Load failed run rows."""

    failed_runs_path = batch_dir / "failed_runs.csv"
    failed_runs = pd.read_csv(failed_runs_path)
    return failed_runs, failed_runs_path


def discover_experiment_artifacts(
    batch_dir: Path,
    manifest: BatchManifest,
) -> list[ExperimentArtifacts]:
    """Discover per-experiment packet inputs in manifest order."""

    experiments: list[ExperimentArtifacts] = []

    for run in manifest.runs:
        experiment_dir = batch_dir / run.name
        config_resolved_path = experiment_dir / "config_resolved.yaml"
        plots_dir = experiment_dir / "plots"
        plot_files = sorted(path for path in plots_dir.rglob("*.png")) if plots_dir.exists() else []

        experiments.append(
            ExperimentArtifacts(
                name=run.name,
                run_id=run.run_id,
                status=run.status,
                experiment_dir=experiment_dir,
                config_resolved_path=config_resolved_path if config_resolved_path.exists() else None,
                plots_dir=plots_dir if plots_dir.exists() else None,
                plots_dir_exists=plots_dir.exists(),
                plot_files=plot_files,
                error_message=run.error_message,
            )
        )

    return experiments


def load_batch_artifacts(batch_dir: Path) -> BatchArtifacts:
    """Load and discover all artifacts needed for packet generation."""

    batch_dir = Path(batch_dir)
    validate_batch_dir(batch_dir)
    manifest, manifest_path = load_manifest(batch_dir)
    combined_results, combined_results_path = load_combined_results(batch_dir)
    failed_runs, failed_runs_path = load_failed_runs(batch_dir)
    experiments = discover_experiment_artifacts(batch_dir=batch_dir, manifest=manifest)

    return BatchArtifacts(
        batch_dir=batch_dir,
        manifest_path=manifest_path,
        combined_results_path=combined_results_path,
        failed_runs_path=failed_runs_path,
        manifest=manifest,
        combined_results=combined_results,
        failed_runs=failed_runs,
        experiments=experiments,
    )
