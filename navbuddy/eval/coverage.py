"""Coverage and readiness reports for modality-matrix outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from navbuddy.eval.augment_assignment import load_assignment_file
from navbuddy.eval.schemas import VALID_AUGMENTS


def _load_samples(dataset_path: Path) -> tuple[Dict[str, dict], Dict[str, Set[str]]]:
    samples_by_id: Dict[str, dict] = {}
    route_to_ids: Dict[str, Set[str]] = defaultdict(set)
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            route_id = row.get("route_id")
            if not sample_id or not route_id:
                continue
            samples_by_id[sample_id] = row
            route_to_ids[route_id].add(sample_id)
    return samples_by_id, route_to_ids


def _iter_results(results_dir: Path) -> Iterable[dict]:
    for path in sorted(results_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield row


def _coverage_pct(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((done / total) * 100.0, 2)


def compute_modality_coverage(
    *,
    dataset_path: Path,
    results_dir: Path,
    models: Optional[List[str]] = None,
    augment_assignment_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute baseline/prior/augmented coverage per model."""
    samples_by_id, route_to_ids = _load_samples(dataset_path)
    total_samples = len(samples_by_id)
    model_filter = set(models or [])
    assignments: Dict[str, str] = {}

    if augment_assignment_file and augment_assignment_file.exists():
        assignments = load_assignment_file(augment_assignment_file)

    per_model: Dict[str, Dict[str, Any]] = {}
    seen_keys = set()

    for row in _iter_results(results_dir):
        sample_id = row.get("sample_id") or row.get("id")
        model_id = row.get("model_id")
        modality = row.get("modality")
        variant = row.get("variant") or None
        augment = row.get("augment") or None
        if not sample_id or not model_id or not modality:
            continue
        if sample_id not in samples_by_id:
            continue
        if model_filter and model_id not in model_filter:
            continue

        dedupe_key = (sample_id, model_id, modality, variant, augment)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        if model_id not in per_model:
            per_model[model_id] = {
                "baseline_ids": set(),
                "prior_ids": set(),
                "aug_any_ids": set(),
                "aug_specific_ids": defaultdict(set),
            }

        bucket = per_model[model_id]
        if modality == "video + prior" and augment is None:
            bucket["baseline_ids"].add(sample_id)
        elif modality == "prior" and augment is None:
            bucket["prior_ids"].add(sample_id)
        elif modality == "video + prior" and augment in VALID_AUGMENTS:
            bucket["aug_any_ids"].add(sample_id)
            route_id = samples_by_id[sample_id]["route_id"]
            bucket["aug_specific_ids"][(route_id, augment)].add(sample_id)

    report_models: Dict[str, Any] = {}
    for model_id, bucket in per_model.items():
        baseline_ids = bucket["baseline_ids"]
        prior_ids = bucket["prior_ids"]
        aug_any_ids = bucket["aug_any_ids"]
        aug_specific_ids = bucket["aug_specific_ids"]

        by_route = {}
        for route_id, route_sample_ids in sorted(route_to_ids.items()):
            route_total = len(route_sample_ids)
            baseline_missing = len(route_sample_ids - baseline_ids)
            prior_missing = len(route_sample_ids - prior_ids)

            if route_id in assignments:
                assigned_augment = assignments[route_id]
                covered_aug_ids = aug_specific_ids.get((route_id, assigned_augment), set())
            else:
                assigned_augment = None
                covered_aug_ids = route_sample_ids & aug_any_ids

            augmented_missing = len(route_sample_ids - covered_aug_ids)
            by_route[route_id] = {
                "route_samples": route_total,
                "assigned_augment": assigned_augment,
                "baseline_missing": baseline_missing,
                "prior_missing": prior_missing,
                "augmented_missing": augmented_missing,
            }

        report_models[model_id] = {
            "counts": {
                "total_samples": total_samples,
                "baseline_done": len(baseline_ids),
                "prior_done": len(prior_ids),
                "augmented_done": len(aug_any_ids),
            },
            "coverage_pct": {
                "baseline": _coverage_pct(len(baseline_ids), total_samples),
                "prior_only": _coverage_pct(len(prior_ids), total_samples),
                "augmented": _coverage_pct(len(aug_any_ids), total_samples),
            },
            "missing": {
                "baseline": total_samples - len(baseline_ids),
                "prior_only": total_samples - len(prior_ids),
                "augmented": total_samples - len(aug_any_ids),
            },
            "by_route": by_route,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "results_dir": str(results_dir),
        "total_samples": total_samples,
        "total_routes": len(route_to_ids),
        "models": report_models,
    }


def coverage_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Modality Coverage Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Dataset: `{report.get('dataset_path', '')}`",
        f"- Results dir: `{report.get('results_dir', '')}`",
        f"- Samples: {report.get('total_samples', 0)}",
        f"- Routes: {report.get('total_routes', 0)}",
        "",
    ]
    models = report.get("models", {})
    if not models:
        lines.append("No model results found.")
        return "\n".join(lines) + "\n"

    for model_id, payload in sorted(models.items()):
        counts = payload.get("counts", {})
        pct = payload.get("coverage_pct", {})
        missing = payload.get("missing", {})
        lines.extend(
            [
                f"## {model_id}",
                "",
                f"- Baseline (`video + prior`, normal): {counts.get('baseline_done', 0)}/{counts.get('total_samples', 0)} ({pct.get('baseline', 0)}%)",
                f"- Prior-only: {counts.get('prior_done', 0)}/{counts.get('total_samples', 0)} ({pct.get('prior_only', 0)}%)",
                f"- Augmented (`video + prior`, assigned augment): {counts.get('augmented_done', 0)}/{counts.get('total_samples', 0)} ({pct.get('augmented', 0)}%)",
                f"- Missing baseline/prior/augmented: {missing.get('baseline', 0)} / {missing.get('prior_only', 0)} / {missing.get('augmented', 0)}",
                "",
                "| Route | Assigned augment | Baseline missing | Prior missing | Augmented missing |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for route_id, route_stats in sorted(payload.get("by_route", {}).items()):
            lines.append(
                "| "
                + f"{route_id} | "
                + f"{route_stats.get('assigned_augment') or '-'} | "
                + f"{route_stats.get('baseline_missing', 0)} | "
                + f"{route_stats.get('prior_missing', 0)} | "
                + f"{route_stats.get('augmented_missing', 0)} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
