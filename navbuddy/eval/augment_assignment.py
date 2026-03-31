"""Deterministic route -> augment assignment for modality-matrix evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from navbuddy.eval.schemas import VALID_AUGMENTS


DEFAULT_AUGMENT_CYCLE = ("rain", "night", "fog", "motion_blur")


def load_route_ids(dataset_path: Path) -> List[str]:
    """Load unique route IDs from a samples JSONL file."""
    route_ids = set()
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            route_id = row.get("route_id")
            if route_id:
                route_ids.add(route_id)
    return sorted(route_ids)


def assign_route_augments(
    route_ids: Iterable[str],
    cycle: Iterable[str] = DEFAULT_AUGMENT_CYCLE,
) -> Dict[str, str]:
    """Assign one augment per route using lexicographic round-robin."""
    ordered_routes = sorted(route_ids)
    ordered_cycle = list(cycle)
    if not ordered_cycle:
        raise ValueError("augment cycle cannot be empty")

    invalid = [a for a in ordered_cycle if a not in VALID_AUGMENTS]
    if invalid:
        raise ValueError(f"invalid augments in cycle: {invalid}")

    assignments: Dict[str, str] = {}
    for idx, route_id in enumerate(ordered_routes):
        assignments[route_id] = ordered_cycle[idx % len(ordered_cycle)]
    return assignments


def build_assignment_payload(
    dataset_path: Path,
    cycle: Iterable[str] = DEFAULT_AUGMENT_CYCLE,
) -> Dict[str, object]:
    """Build serializable assignment payload for config storage."""
    route_ids = load_route_ids(dataset_path)
    assignments = assign_route_augments(route_ids, cycle=cycle)
    return {
        "version": "v1",
        "strategy": "round_robin_sorted_routes",
        "dataset": str(dataset_path),
        "augment_cycle": list(cycle),
        "routes_total": len(route_ids),
        "assignments": assignments,
    }


def save_assignment_file(payload: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_assignment_file(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text())
    assignments = data.get("assignments", {})
    if not isinstance(assignments, dict):
        raise ValueError(f"Invalid assignment file: {path}")
    normalized: Dict[str, str] = {}
    for route_id, augment in assignments.items():
        if augment not in VALID_AUGMENTS:
            raise ValueError(f"Invalid augment '{augment}' for route {route_id}")
        normalized[str(route_id)] = str(augment)
    return normalized
