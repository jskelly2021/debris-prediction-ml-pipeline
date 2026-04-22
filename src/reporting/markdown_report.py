from pathlib import Path

import pandas as pd

from reporting.report_models import BatchArtifacts, NotableFinding, PacketArtifacts


def render_report(
    batch_artifacts: BatchArtifacts,
    packet_artifacts: PacketArtifacts,
    tables: dict[str, pd.DataFrame],
    notable_findings: list[NotableFinding],
) -> str:
    """Render the research packet markdown report."""

    lines: list[str] = []
    manifest = batch_artifacts.manifest

    lines.append(f"# Research Packet: {manifest.batch_name}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Batch name: `{manifest.batch_name}`")
    lines.append(f"- Timestamp: `{manifest.timestamp}`")
    lines.append(f"- Base config path: `{manifest.base_config}`")
    lines.append(f"- Experiment count: `{manifest.experiment_count}`")
    lines.append(f"- Succeeded count: `{manifest.succeeded_count}`")
    lines.append(f"- Failed count: `{manifest.failed_count}`")
    lines.append("")

    inventory_rows = pd.DataFrame(
        [
            {
                "name": run.name,
                "run_id": run.run_id,
                "status": run.status,
            }
            for run in manifest.runs
        ]
    )
    lines.append("## Experiment Inventory")
    lines.append("")
    lines.append(dataframe_to_markdown(inventory_rows))
    lines.append("")

    lines.append("## Summary Tables")
    lines.append("")
    lines.append("### Summary by Label")
    lines.append("")
    lines.append(dataframe_to_markdown(tables["summary_by_label"]))
    lines.append("")
    lines.append("### Summary by Experiment")
    lines.append("")
    lines.append(dataframe_to_markdown(tables["summary_by_experiment"]))
    lines.append("")
    lines.append("### Best Classification Runs")
    lines.append("")
    lines.append(dataframe_to_markdown(tables["best_classification_runs"]))
    lines.append("")
    lines.append("### Best Regression Runs")
    lines.append("")
    lines.append(dataframe_to_markdown(tables["best_regression_runs"]))
    lines.append("")

    lines.append("## Comparison Figures")
    lines.append("")
    figure_paths = {
        name: path.relative_to(batch_artifacts.batch_dir).as_posix()
        for name, path in packet_artifacts.figure_paths.items()
    }
    lines.append("### Classification Metrics")
    lines.append("")
    lines.extend(
        [
            "<p>",
            f'  <img src="{figure_paths["f1_by_experiment"]}" width="45%" />',
            f'  <img src="{figure_paths["roc_auc_by_experiment"]}" width="45%" />',
            "</p>",
            "",
        ]
    )
    lines.append("### Regression Metrics")
    lines.append("")
    lines.extend(
        [
            "<p>",
            f'  <img src="{figure_paths["r2_by_experiment"]}" width="45%" />',
            f'  <img src="{figure_paths["rmse_by_experiment"]}" width="45%" />',
            "</p>",
            "",
        ]
    )
    lines.append("### Tradeoff")
    lines.append("")
    lines.extend(
        [
            "<p>",
            f'  <img src="{figure_paths["f1_vs_r2_scatter"]}" width="60%" />',
            "</p>",
            "",
        ]
    )

    lines.append("## Notable Findings")
    lines.append("")
    if notable_findings:
        for finding in notable_findings:
            lines.append(f"- {finding.message}")
    else:
        lines.append("- No notable findings generated.")
    lines.append("")

    if not batch_artifacts.failed_runs.empty:
        lines.append("## Failures")
        lines.append("")
        lines.append(dataframe_to_markdown(batch_artifacts.failed_runs))
        lines.append("")

    lines.append("## Experiment Drill-Down")
    lines.append("")
    for experiment in batch_artifacts.experiments:
        lines.append(f"### {experiment.name}")
        lines.append("")
        lines.append(f"- Run ID: `{experiment.run_id}`")
        if experiment.config_resolved_path is not None:
            config_path = experiment.config_resolved_path.relative_to(batch_artifacts.batch_dir).as_posix()
            lines.append(f"- Resolved config: `{config_path}`")
        else:
            lines.append("- Resolved config: missing")
        lines.append(f"- Plots directory exists: `{experiment.plots_dir_exists}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(
    report_path: Path,
    batch_artifacts: BatchArtifacts,
    packet_artifacts: PacketArtifacts,
    tables: dict[str, pd.DataFrame],
    notable_findings: list[NotableFinding],
) -> Path:
    """Write the report markdown file."""

    report_text = render_report(
        batch_artifacts=batch_artifacts,
        packet_artifacts=packet_artifacts,
        tables=tables,
        notable_findings=notable_findings,
    )
    report_path.write_text(report_text)
    return report_path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a simple markdown table."""

    if df.empty:
        return "_No rows._"

    safe_df = df.copy()
    headers = [str(column) for column in safe_df.columns]
    rows = [[_format_markdown_cell(value) for value in row] for row in safe_df.itertuples(index=False, name=None)]

    separator = ["---"] * len(headers)
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    markdown_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(markdown_lines)


def _format_markdown_cell(value) -> str:
    """Format one markdown table cell."""

    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ")
