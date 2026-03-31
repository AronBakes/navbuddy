"""Dataset frame regeneration utilities."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from navbuddy.streetview_client import (
    check_streetview_coverage,
    download_streetview_image,
    sample_frames_for_step,
)
from navbuddy.utils import generate_frame_filename


@dataclass
class RegenerateStats:
    samples_total: int = 0
    samples_updated: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    removed_old: int = 0


def regenerate_frames_dataset(
    *,
    data_root: Path,
    api_key: str,
    frame_profile: str = "sparse4",
    spacing: float = 20.0,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    replace: bool = False,
    concurrency: int = 4,
) -> RegenerateStats:
    """Rebuild frame files and rewrite sample frame lists in samples.jsonl."""
    samples_path = data_root / "samples.jsonl"
    frames_dir = data_root / "frames"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.jsonl not found: {samples_path}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    stats = RegenerateStats()

    samples: List[Dict] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    stats.samples_total = len(samples)

    # Build all download tasks first so we can parallelize network IO.
    # task: (sample_index, rel_path, params_dict, abs_path)
    tasks: List[Tuple[int, str, Dict, Path]] = []
    expected_rel_paths: Dict[int, List[str]] = {}
    old_rel_paths: Dict[int, List[str]] = {}

    for idx, sample in enumerate(samples):
        route_id = str(sample.get("route_id", "route"))
        step_idx = int(sample.get("step_index", 0) or 0)
        distances = sample.get("distances", {}) if isinstance(sample.get("distances"), dict) else {}
        step_distance_m = int(distances.get("step_distance_m", 0) or 0)

        step_polyline = (
            ((sample.get("geometry") or {}).get("step_polyline") if isinstance(sample.get("geometry"), dict) else "")
            or ""
        )
        step = {
            "distanceMeters": step_distance_m,
            "polyline": {"encodedPolyline": step_polyline},
        }

        params_list = sample_frames_for_step(
            step,
            frame_profile=frame_profile,
            mode="custom" if frame_profile == "custom" else "sparse",
            spacing=spacing,
            sample_start=sample_start,
            sample_end=sample_end,
        )

        new_rel_paths: List[str] = []
        for params in params_list:
            filename = generate_frame_filename(
                route_id=route_id,
                step_index=step_idx,
                distance_from_end_m=int(params.distance_m),
                step_distance_m=step_distance_m,
            )
            rel = f"frames/{filename}"
            abs_path = frames_dir / filename
            new_rel_paths.append(rel)

            if abs_path.exists() and not replace:
                stats.skipped_existing += 1
                continue

            tasks.append((idx, rel, params.to_dict(), abs_path))

        expected_rel_paths[idx] = new_rel_paths
        old_rel_paths[idx] = list(((sample.get("images") or {}).get("frames") or []))

    # Coverage check: query metadata in parallel to get pano_ids, then deduplicate
    # within each sample (prevents downloading the same panorama twice for one step).
    # This also enforces source=outdoor via check_streetview_coverage's default.
    if tasks:
        workers = max(1, int(concurrency))

        # Map each task to its coverage metadata
        coverage_results: Dict[Tuple[int, str], Dict] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cov_futures = {
                pool.submit(
                    check_streetview_coverage,
                    params_dict["lat"],
                    params_dict["lng"],
                    api_key=api_key,
                ): (idx, rel)
                for idx, rel, params_dict, _ in tasks
            }
            for future in as_completed(cov_futures):
                key = cov_futures[future]
                try:
                    coverage_results[key] = future.result()
                except Exception:
                    coverage_results[key] = {"status": "ERROR"}

        # Deduplicate: within each sample, skip tasks with a repeated pano_id
        seen_pano_ids_by_sample: Dict[int, set] = {}
        deduped_tasks = []
        for task in tasks:
            idx, rel, params_dict, abs_path = task
            cov = coverage_results.get((idx, rel), {})
            if cov.get("status") != "OK":
                continue  # No outdoor coverage — skip
            pano_id = cov.get("pano_id")
            if pano_id:
                seen = seen_pano_ids_by_sample.setdefault(idx, set())
                if pano_id in seen:
                    continue  # Duplicate panorama for this step — skip
                seen.add(pano_id)
            deduped_tasks.append(task)

        tasks = deduped_tasks

    # Execute downloads with bounded concurrency.
    task_results: Dict[Tuple[int, str], bool] = {}
    if tasks:
        workers = max(1, int(concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(
                    download_streetview_image,
                    params_dict,
                    abs_path,
                    api_key=api_key,
                ): (idx, rel)
                for idx, rel, params_dict, abs_path in tasks
            }
            for future in as_completed(future_map):
                key = future_map[future]
                try:
                    ok = bool(future.result())
                except Exception:
                    ok = False
                task_results[key] = ok
                if ok:
                    stats.downloaded += 1
                else:
                    stats.failed += 1

    # Update sample rows and optionally remove stale old files.
    for idx, sample in enumerate(samples):
        desired = expected_rel_paths.get(idx, [])
        existing_desired = [rel for rel in desired if (data_root / rel).exists()]

        if not existing_desired and not replace:
            # Non-destructive mode: keep old frame list if regeneration yielded nothing.
            continue

        sample.setdefault("images", {})
        sample["images"]["frames"] = existing_desired
        stats.samples_updated += 1

        if replace:
            keep = set(existing_desired)
            for rel in old_rel_paths.get(idx, []):
                if rel in keep:
                    continue
                path = data_root / rel
                if path.exists():
                    try:
                        path.unlink()
                        stats.removed_old += 1
                    except Exception:
                        pass

    # Atomic rewrite
    tmp_path = samples_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    tmp_path.replace(samples_path)

    return stats


__all__ = ["RegenerateStats", "regenerate_frames_dataset"]

