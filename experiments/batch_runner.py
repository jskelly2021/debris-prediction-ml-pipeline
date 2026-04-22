import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from experiments.batch_config import BatchConfig
from experiments.override_utils import (
    deep_merge_config,
    load_config_dict,
    validate_resolved_config,
)
from run_single import run_experiment


FAILED_RUN_COLUMNS = ["name", "run_id", "status", "error_message"]


class BatchRunner:
    """Run a batch of single-experiment pipeline executions."""

    def __init__(self, batch_config: BatchConfig):
        self.batch_config = batch_config
        self.batch_dir = batch_config.output_root / batch_config.batch_name

    def run(self) -> dict:
        """Execute the batch and write aggregate outputs."""

        self.batch_dir.mkdir(parents=True, exist_ok=True)
        base_config = load_config_dict(self.batch_config.base_config)

        successful_results = []
        failed_runs = []
        run_statuses = []

        for experiment in self.batch_config.experiments:
            experiment_dir = self.batch_dir / experiment.name

            try:
                resolved_config_dict = deep_merge_config(base_config, experiment.overrides)
                resolved_config_dict.setdefault("data", {})
                resolved_config_dict["data"]["output_path"] = str(experiment_dir)

                experiment_dir.mkdir(parents=True, exist_ok=True)
                self._save_resolved_config(resolved_config_dict, experiment_dir)

                experiment_config = validate_resolved_config(resolved_config_dict)
                result_df = run_experiment(
                    config=experiment_config,
                    run_id=experiment.run_id,
                    save=True,
                    plots=True,
                    feature_importance=False,
                )
                result_df.insert(0, "experiment_name", experiment.name)
                successful_results.append(result_df)
                run_statuses.append(self._status_row(experiment, "success"))

            except Exception as exc:
                error_message = str(exc)
                failed_run = self._status_row(experiment, "failed", error_message)
                failed_runs.append(failed_run)
                run_statuses.append(failed_run)

        combined_results = self._write_combined_results(successful_results)
        failed_results = self._write_failed_runs(failed_runs)

        manifest = self._write_manifest(
            run_statuses=run_statuses,
            succeeded_count=len(successful_results),
            failed_count=len(failed_runs),
        )

        return {
            "batch_dir": self.batch_dir,
            "combined_results_path": combined_results,
            "failed_runs_path": failed_results,
            "manifest_path": manifest,
            "succeeded_count": len(successful_results),
            "failed_count": len(failed_runs),
        }

    def _save_resolved_config(self, resolved_config_dict: dict, experiment_dir: Path) -> None:
        with (experiment_dir / "config_resolved.yaml").open("w") as file:
            yaml.safe_dump(resolved_config_dict, file, sort_keys=False)

    def _write_combined_results(self, successful_results: list[pd.DataFrame]) -> Path:
        combined_results_path = self.batch_dir / "combined_results.csv"
        if successful_results:
            combined_df = pd.concat(successful_results, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        combined_df.to_csv(combined_results_path, index=False)
        return combined_results_path

    def _write_failed_runs(self, failed_runs: list[dict]) -> Path:
        failed_runs_path = self.batch_dir / "failed_runs.csv"
        failed_df = pd.DataFrame(failed_runs, columns=FAILED_RUN_COLUMNS)
        failed_df.to_csv(failed_runs_path, index=False)
        return failed_runs_path

    def _write_manifest(
        self,
        run_statuses: list[dict],
        succeeded_count: int,
        failed_count: int,
    ) -> Path:
        manifest_path = self.batch_dir / "batch_manifest.json"
        manifest = {
            "batch_name": self.batch_config.batch_name,
            "base_config": str(self.batch_config.base_config),
            "output_path": str(self.batch_dir),
            "experiment_count": len(self.batch_config.experiments),
            "succeeded_count": succeeded_count,
            "failed_count": failed_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runs": run_statuses,
        }

        with manifest_path.open("w") as file:
            json.dump(manifest, file, indent=2)
        return manifest_path

    def _status_row(self, experiment, status: str, error_message: str | None = None) -> dict:
        return {
            "name": experiment.name,
            "run_id": experiment.run_id,
            "status": status,
            "error_message": error_message,
        }
