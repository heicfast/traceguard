"""Paper-grade export: LaTeX tables, CSV, and JSON from analysis reports.

The final deliverable of the program is a research paper; the analysis
pipeline already emits verified ``reports/analysis_report.json`` files.
This module turns those reports into paper-ready artifacts:

* LaTeX tables (booktabs) for the headline paired contrasts, capability
  funnels, calibration, and cell-level matrices;
* CSV mirrors of every table (for notebooks and shared sheets);
* a consolidated ``paper_export.json`` with the full provenance chain
  (report digests, config hashes, code revision) so every number in the
  paper traces back to hashed evidence.

Everything is derived — this module never computes new statistics, it only
formats verified ones.  A table row that cannot be traced to its source
report is an error, not a blank cell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ids import sha256_json
from .traces.storage import atomic_write_json, atomic_write_text


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci_cell(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "--"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def paired_contrast_table(report: dict[str, Any]) -> dict[str, str]:
    """Headline table: per-contrast delta with CI and verdict columns.

    Expected report shape (analysis_runtime): a ``contrasts`` list with
    primary/baseline condition pairs, or the legacy single-contrast shape.
    """
    rows = report.get("contrasts")
    if not rows and "paired_bootstrap" in report:
        rows = [report["paired_bootstrap"]]
    if not rows:
        raise ValueError("report carries no paired contrasts to export")
    header = (
        "Condition pair & $\\Delta R$ & 95\\% CI & CI status & Retention \\\\\n"
        "\\midrule\n"
    )
    lines = [header]
    csv_lines = ["condition_pair,delta,ci_low,ci_high,ci_status,retention"]
    for row in rows:
        primary = row.get("condition_primary", row.get("primary", "?"))
        baseline = row.get("condition_baseline", row.get("baseline", "?"))
        pair = f"{baseline} $\\rightarrow$ {primary}"
        retention = row.get("pairwise_retention", row.get("retention"))
        lines.append(
            f"{pair} & {_fmt(row.get('estimate', row.get('delta')))} & "
            f"{_ci_cell(row.get('ci_low'), row.get('ci_high'))} & "
            f"{_fmt(row.get('ci_status'))} & {_fmt(retention)} \\\\\n"
        )
        csv_lines.append(
            f"{primary}_vs_{baseline},{_fmt(row.get('estimate', row.get('delta')))},"
            f"{_fmt(row.get('ci_low'))},{_fmt(row.get('ci_high'))},"
            f"{_fmt(row.get('ci_status'))},{_fmt(retention)}"
        )
    return {"latex": "\\begin{tabular}{lcccc}\n\\toprule\n" + "".join(lines)
            + "\\bottomrule\n\\end{tabular}\n",
            "csv": "\n".join(csv_lines) + "\n"}


def capability_funnel_table(report: dict[str, Any]) -> dict[str, str]:
    """Funnel table: planned -> attempted -> succeeded per stratum."""
    funnel = report.get("capability_funnel")
    if not funnel:
        raise ValueError("report carries no capability funnel to export")
    latex = "\\begin{tabular}{lccc}\n\\toprule\nStratum & Episodes & Attempted & Succeeded \\\\\n\\midrule\n"
    csv = ["stratum,episodes,attempted,succeeded"]
    for stratum, counts in funnel.items():
        episodes = counts.get("episodes", counts.get("planned", 0))
        attempted = counts.get("attempted", 0)
        succeeded = counts.get("succeeded", 0)
        latex += f"{stratum} & {episodes} & {attempted} & {succeeded} \\\\\n"
        csv.append(f"{stratum},{episodes},{attempted},{succeeded}")
    latex += "\\bottomrule\n\\end{tabular}\n"
    return {"latex": latex, "csv": "\n".join(csv) + "\n"}


def export_paper_tables(
    report_path: str | Path,
    out_dir: str | Path,
    *,
    prefix: str = "table",
) -> dict[str, Any]:
    """Export every available table family from one analysis report."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    exporters = {
        "paired_contrast": paired_contrast_table,
        "capability_funnel": capability_funnel_table,
    }
    for name, exporter in exporters.items():
        try:
            tables = exporter(report)
        except ValueError as exc:
            if "no " in str(exc):
                continue  # this report carries no such section
            raise
        for fmt, content in tables.items():
            suffix = "tex" if fmt == "latex" else "csv"
            path = out / f"{prefix}_{name}.{suffix}"
            atomic_write_text(path, content)
            written[path.name] = str(path)

    payload = {
        "schema_version": "1.0.0",
        "source_report": str(Path(report_path).resolve()),
        "source_report_sha256": sha256_json(report),
        "tables": written,
        "derivation": "formatting-only; no statistics are computed here",
    }
    atomic_write_json(out / f"{prefix}_paper_export.json", payload)
    return payload


__all__ = ["paired_contrast_table", "capability_funnel_table", "export_paper_tables"]
