import json
from datetime import datetime, timezone
from pathlib import Path

from logger import Log
from reporting.artifact_loader import load_batch_artifacts
from reporting.comparison_plots import FIGURE_FILENAMES, save_comparison_figures
from reporting.markdown_report import write_report
from reporting.report_models import NotableFinding, PacketArtifacts
from reporting.summary_tables import (
    build_best_classification_runs,
    build_best_regression_runs,
    build_summary_by_experiment,
    build_summary_by_label,
)


TABLE_FILENAMES = {
    "summary_by_label": "summary_by_label.csv",
    "summary_by_experiment": "summary_by_experiment.csv",
    "best_classification_runs": "best_classification_runs.csv",
    "best_regression_runs": "best_regression_runs.csv",
}


log = Log()


class ResearchPacketBuilder:
    """Build a research packet from an existing batch output directory."""

    def __init__(self, batch_dir: Path):
        self.batch_dir = Path(batch_dir)

    def build(self) -> dict:
        """Generate packet tables, figures, manifest, and markdown report."""

        log.info(f"Loading batch artifacts from {self.batch_dir}")
        batch_artifacts = load_batch_artifacts(self.batch_dir)
        experiment_order = [run.name for run in batch_artifacts.manifest.runs]

        packet_dir = batch_artifacts.batch_dir / "packet"
        tables_dir = packet_dir / "tables"
        figures_dir = packet_dir / "figures"
        report_path = batch_artifacts.batch_dir / "report.md"
        packet_manifest_path = packet_dir / "packet_manifest.json"

        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        log.info("Building summary tables")
        tables = {
            "summary_by_label": build_summary_by_label(
                batch_artifacts.combined_results,
                experiment_order=experiment_order,
            ),
            "summary_by_experiment": build_summary_by_experiment(
                batch_artifacts.combined_results,
                experiment_order=experiment_order,
            ),
            "best_classification_runs": build_best_classification_runs(
                batch_artifacts.combined_results,
                experiment_order=experiment_order,
            ),
            "best_regression_runs": build_best_regression_runs(
                batch_artifacts.combined_results,
                experiment_order=experiment_order,
            ),
        }
        table_paths = self._write_tables(tables=tables, tables_dir=tables_dir)

        log.info("Generating comparison figures")
        figure_paths = save_comparison_figures(
            combined_results=batch_artifacts.combined_results,
            figures_dir=figures_dir,
            experiment_order=experiment_order,
        )

        packet_artifacts = PacketArtifacts(
            batch_dir=batch_artifacts.batch_dir,
            report_path=report_path,
            packet_dir=packet_dir,
            packet_manifest_path=packet_manifest_path,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            figure_paths=figure_paths,
            table_paths=table_paths,
        )

        log.info("Computing notable findings")
        notable_findings = self._build_notable_findings(tables=tables, batch_artifacts=batch_artifacts)

        log.info("Writing packet manifest")
        self._write_packet_manifest(
            batch_artifacts=batch_artifacts,
            packet_artifacts=packet_artifacts,
        )

        log.info("Writing markdown report")
        write_report(
            report_path=report_path,
            batch_artifacts=batch_artifacts,
            packet_artifacts=packet_artifacts,
            tables=tables,
            notable_findings=notable_findings,
        )

        return {
            "batch_dir": batch_artifacts.batch_dir,
            "report_path": report_path,
            "packet_dir": packet_dir,
            "packet_manifest_path": packet_manifest_path,
            "figures_dir": figures_dir,
            "tables_dir": tables_dir,
        }

    def _write_tables(self, tables, tables_dir: Path) -> dict[str, Path]:
        """Write derived tables to disk."""

        table_paths: dict[str, Path] = {}
        for table_name, filename in TABLE_FILENAMES.items():
            path = tables_dir / filename
            tables[table_name].to_csv(path, index=False)
            table_paths[table_name] = path
        return table_paths

    def _build_notable_findings(self, tables, batch_artifacts) -> list[NotableFinding]:
        """Build deterministic notable findings from packet contents."""

        findings: list[NotableFinding] = []

        for row in tables["best_classification_runs"].itertuples(index=False):
            findings.append(
                NotableFinding(
                    category="best_classification",
                    message=f"Best F1 for label {row.label}: {row.experiment_name} ({row.f1:.4f})",
                )
            )

        for row in tables["best_regression_runs"].itertuples(index=False):
            findings.append(
                NotableFinding(
                    category="best_regression",
                    message=f"Best R2 for label {row.label}: {row.experiment_name} ({row.r2:.4f})",
                )
            )

        negative_r2 = batch_artifacts.combined_results[batch_artifacts.combined_results["r2"] < 0]
        for row in negative_r2.itertuples(index=False):
            findings.append(
                NotableFinding(
                    category="negative_r2",
                    message=f"Negative R2 for {row.experiment_name}:{row.label} ({row.r2:.4f})",
                )
            )

        if batch_artifacts.failed_runs.empty:
            findings.append(NotableFinding(category="failed_runs", message="No failed runs recorded."))
        else:
            findings.append(
                NotableFinding(
                    category="failed_runs",
                    message=f"Failed runs recorded: {len(batch_artifacts.failed_runs)}",
                )
            )

        return findings

    def _write_packet_manifest(self, batch_artifacts, packet_artifacts: PacketArtifacts) -> Path:
        """Write packet output metadata to JSON."""

        manifest = {
            "batch_name": batch_artifacts.manifest.batch_name,
            "base_config": batch_artifacts.manifest.base_config,
            "batch_dir": str(batch_artifacts.batch_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_count": batch_artifacts.manifest.experiment_count,
            "succeeded_count": batch_artifacts.manifest.succeeded_count,
            "failed_count": batch_artifacts.manifest.failed_count,
            "report_path": str(packet_artifacts.report_path),
            "packet_dir": str(packet_artifacts.packet_dir),
            "tables_dir": str(packet_artifacts.tables_dir),
            "figures_dir": str(packet_artifacts.figures_dir),
            "tables": {name: path.name for name, path in packet_artifacts.table_paths.items()},
            "figures": {name: path.name for name, path in packet_artifacts.figure_paths.items()},
        }

        with packet_artifacts.packet_manifest_path.open("w") as file:
            json.dump(manifest, file, indent=2)

        return packet_artifacts.packet_manifest_path
