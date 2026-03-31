"""Run deterministic modality-matrix evaluations with missing-only support."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator

from navbuddy.eval.augment_assignment import (
    DEFAULT_AUGMENT_CYCLE,
    assign_route_augments,
    load_assignment_file,
)
from navbuddy.eval.inference import run_inference


CANONICAL_MODALITIES = {
    "video_prior_normal",
    "prior_only",
    "video_prior_augmented",
    "image_prior_normal",      # single last frame (40m), no augmentation
    "image_prior_augmented",   # single last frame (40m) + augmentation
}


class MatrixModelSpec(BaseModel):
    model_id: str
    provider: Optional[str] = None
    variant: Optional[str] = None
    limit: Optional[int] = None
    provider_order: Optional[List[str]] = None
    use_segformer_context: Optional[bool] = None
    segformer_model_id: Optional[str] = None
    segformer_device: Optional[str] = None
    segformer_cache_dir: Optional[str] = None
    local_device: Optional[str] = None
    local_dtype: Optional[str] = None
    local_load_in_4bit: Optional[bool] = None
    local_max_new_tokens: Optional[int] = None
    local_temperature: Optional[float] = None


class MatrixConfig(BaseModel):
    dataset_path: Path
    data_root: Optional[Path] = None
    results_dir: Path = Path("results")
    augment_assignment_file: Optional[Path] = None
    modalities: List[str] = Field(
        default_factory=lambda: [
            "video_prior_normal",
            "prior_only",
            "video_prior_augmented",
        ]
    )
    missing_only_default: bool = False
    provider: str = "openrouter"
    limit: Optional[int] = None
    use_segformer_context: bool = False
    segformer_model_id: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    segformer_device: str = "auto"
    segformer_cache_dir: Optional[str] = None
    local_device: str = "auto"
    local_dtype: str = "auto"
    local_load_in_4bit: bool = True
    local_max_new_tokens: int = 96
    local_temperature: float = 0.0
    models: List[MatrixModelSpec]

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_models(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("models must be a list")
        normalized = []
        for item in value:
            if isinstance(item, str):
                normalized.append({"model_id": item})
            else:
                normalized.append(item)
        return normalized

    @field_validator("modalities")
    @classmethod
    def _validate_modalities(cls, value: List[str]) -> List[str]:
        invalid = [m for m in value if m not in CANONICAL_MODALITIES]
        if invalid:
            raise ValueError(
                f"invalid modalities {invalid}; allowed={sorted(CANONICAL_MODALITIES)}"
            )
        return value

    @classmethod
    def from_file(cls, path: Path) -> "MatrixConfig":
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(path.read_text())
        else:
            data = json.loads(path.read_text())
        return cls.model_validate(data)


@dataclass
class TaskRunSummary:
    output_path: Path
    attempted: int
    written: int
    skipped_existing: int
    errors: int


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def _load_dataset_rows(dataset_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_rows_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count


def _load_existing_index(results_dir: Path) -> set[tuple[str, str, str, Optional[str], Optional[str]]]:
    index: set[tuple[str, str, str, Optional[str], Optional[str]]] = set()
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
                sample_id = row.get("sample_id") or row.get("id")
                model_id = row.get("model_id")
                modality = row.get("modality")
                if not sample_id or not model_id or not modality:
                    continue
                variant = row.get("variant") or None
                augment = row.get("augment") or None
                index.add((sample_id, model_id, modality, variant, augment))
    return index


def _append_jsonl(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(src, "r", encoding="utf-8") as in_handle, open(dst, "a", encoding="utf-8") as out_handle:
        for line in in_handle:
            line = line.strip()
            if not line:
                continue
            out_handle.write(line + "\n")
            written += 1
    return written


def _resolve_assignments(
    route_ids: List[str],
    assignment_file: Optional[Path],
) -> Dict[str, str]:
    if assignment_file and assignment_file.exists():
        assignments = load_assignment_file(assignment_file)
    else:
        assignments = assign_route_augments(route_ids, cycle=DEFAULT_AUGMENT_CYCLE)

    missing = sorted(set(route_ids) - set(assignments))
    if missing:
        raise ValueError(
            "augment assignment file is missing routes: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )
    return {rid: assignments[rid] for rid in route_ids}


def _effective_option(model_value: Any, default_value: Any) -> Any:
    return default_value if model_value is None else model_value


def _run_single_task(
    *,
    rows: List[Dict[str, Any]],
    dataset_path: Path,
    output_path: Path,
    model: MatrixModelSpec,
    modality: str,
    augment: Optional[str],
    variant: Optional[str],
    config: MatrixConfig,
    missing_only: bool,
    existing_index: set[tuple[str, str, str, Optional[str], Optional[str]]],
    verbose: bool,
) -> TaskRunSummary:
    attempted = len(rows)
    if attempted == 0:
        return TaskRunSummary(
            output_path=output_path,
            attempted=0,
            written=0,
            skipped_existing=0,
            errors=0,
        )

    selected_rows = rows
    skipped_existing = 0

    if missing_only:
        filtered = []
        for row in rows:
            sample_id = row.get("id") or row.get("sample_id")
            key = (sample_id, model.model_id, modality, variant, augment)
            if key in existing_index:
                skipped_existing += 1
                continue
            filtered.append(row)
        selected_rows = filtered

    if not selected_rows:
        return TaskRunSummary(
            output_path=output_path,
            attempted=attempted,
            written=0,
            skipped_existing=skipped_existing,
            errors=0,
        )

    tmp_dir = output_path.parent / ".tmp_matrix"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dataset_tmp = tmp_dir / f"dataset_{_slug(output_path.stem)}.jsonl"
    result_tmp = tmp_dir / f"result_{_slug(output_path.stem)}.jsonl"
    _write_rows_jsonl(selected_rows, dataset_tmp)

    try:
        results = run_inference(
            dataset_path=dataset_tmp,
            model_id=model.model_id,
            output_path=result_tmp if missing_only else output_path,
            modality=modality,
            provider=_effective_option(model.provider, config.provider),
            data_root=config.data_root or dataset_path.parent,
            limit=_effective_option(model.limit, config.limit),
            verbose=verbose,
            augment=augment,
            variant=variant,
            use_segformer_context=_effective_option(
                model.use_segformer_context, config.use_segformer_context
            ),
            segformer_model_id=_effective_option(
                model.segformer_model_id, config.segformer_model_id
            ),
            segformer_device=_effective_option(model.segformer_device, config.segformer_device),
            segformer_cache_dir=_effective_option(
                model.segformer_cache_dir, config.segformer_cache_dir
            ),
            local_device=_effective_option(model.local_device, config.local_device),
            local_dtype=_effective_option(model.local_dtype, config.local_dtype),
            local_load_in_4bit=_effective_option(
                model.local_load_in_4bit, config.local_load_in_4bit
            ),
            local_max_new_tokens=_effective_option(
                model.local_max_new_tokens, config.local_max_new_tokens
            ),
            local_temperature=_effective_option(
                model.local_temperature, config.local_temperature
            ),
            provider_order=model.provider_order,
        )

        if missing_only:
            written = _append_jsonl(result_tmp, output_path)
        else:
            written = len(results)

        errors = sum(1 for item in results if item.error)
        for item in results:
            existing_index.add(
                (
                    item.id,
                    item.model_id,
                    item.modality,
                    item.variant,
                    item.augment,
                )
            )
    finally:
        if dataset_tmp.exists():
            dataset_tmp.unlink()
        if result_tmp.exists():
            result_tmp.unlink()

    return TaskRunSummary(
        output_path=output_path,
        attempted=attempted,
        written=written,
        skipped_existing=skipped_existing,
        errors=errors,
    )


def run_evaluation_matrix(
    config_path: Path,
    *,
    missing_only: bool = False,
    verbose: bool = True,
    sample_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run modality matrix for all configured models."""
    config = MatrixConfig.from_file(config_path)
    dataset_path = config.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = _load_dataset_rows(dataset_path)
    if sample_ids:
        id_set = set(sample_ids)
        rows = [r for r in rows if (r.get("id") or r.get("sample_id")) in id_set]
        if not rows:
            raise ValueError(f"None of the requested sample_ids found in dataset: {sample_ids}")
    route_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        route_to_rows[str(row.get("route_id"))].append(row)
    route_ids = sorted(route_to_rows)
    assignments = _resolve_assignments(route_ids, config.augment_assignment_file)

    effective_missing_only = bool(missing_only or config.missing_only_default)
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_index = _load_existing_index(results_dir) if effective_missing_only else set()

    summaries: List[TaskRunSummary] = []

    for model in config.models:
        model_slug = _slug(model.model_id)
        variant = model.variant
        variant_suffix = f"_{_slug(variant)}" if variant else ""

        if "video_prior_normal" in config.modalities:
            summaries.append(
                _run_single_task(
                    rows=rows,
                    dataset_path=dataset_path,
                    output_path=results_dir
                    / f"results_matrix_{model_slug}{variant_suffix}_video_prior_normal.jsonl",
                    model=model,
                    modality="video + prior",
                    augment=None,
                    variant=variant,
                    config=config,
                    missing_only=effective_missing_only,
                    existing_index=existing_index,
                    verbose=verbose,
                )
            )

        if "prior_only" in config.modalities:
            summaries.append(
                _run_single_task(
                    rows=rows,
                    dataset_path=dataset_path,
                    output_path=results_dir
                    / f"results_matrix_{model_slug}{variant_suffix}_prior_only.jsonl",
                    model=model,
                    modality="prior",
                    augment=None,
                    variant=variant,
                    config=config,
                    missing_only=effective_missing_only,
                    existing_index=existing_index,
                    verbose=verbose,
                )
            )

        # Build augment-to-rows mapping once for both video and image augmented modalities.
        augment_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if "video_prior_augmented" in config.modalities or "image_prior_augmented" in config.modalities:
            for route_id, assigned_augment in assignments.items():
                augment_to_rows[assigned_augment].extend(route_to_rows.get(route_id, []))

        if "video_prior_augmented" in config.modalities:
            for augment_name in sorted(augment_to_rows):
                summaries.append(
                    _run_single_task(
                        rows=augment_to_rows[augment_name],
                        dataset_path=dataset_path,
                        output_path=results_dir
                        / (
                            f"results_matrix_{model_slug}{variant_suffix}"
                            f"_video_prior_augmented_{augment_name}.jsonl"
                        ),
                        model=model,
                        modality="video + prior",
                        augment=augment_name,
                        variant=variant,
                        config=config,
                        missing_only=effective_missing_only,
                        existing_index=existing_index,
                        verbose=verbose,
                    )
                )

        if "image_prior_normal" in config.modalities:
            summaries.append(
                _run_single_task(
                    rows=rows,
                    dataset_path=dataset_path,
                    output_path=results_dir
                    / f"results_matrix_{model_slug}{variant_suffix}_image_prior_normal.jsonl",
                    model=model,
                    modality="image + prior",
                    augment=None,
                    variant=variant,
                    config=config,
                    missing_only=effective_missing_only,
                    existing_index=existing_index,
                    verbose=verbose,
                )
            )

        if "image_prior_augmented" in config.modalities:
            # Single last frame (closest to maneuver) with augmentation.
            # Uses modality="image + prior" which selects frame_paths[-1] only.
            for augment_name in sorted(augment_to_rows):
                summaries.append(
                    _run_single_task(
                        rows=augment_to_rows[augment_name],
                        dataset_path=dataset_path,
                        output_path=results_dir
                        / (
                            f"results_matrix_{model_slug}{variant_suffix}"
                            f"_image_prior_augmented_{augment_name}.jsonl"
                        ),
                        model=model,
                        modality="image + prior",
                        augment=augment_name,
                        variant=variant,
                        config=config,
                        missing_only=effective_missing_only,
                        existing_index=existing_index,
                        verbose=verbose,
                    )
                )

    return {
        "config": str(config_path),
        "dataset": str(dataset_path),
        "routes_total": len(route_ids),
        "samples_total": len(rows),
        "missing_only": effective_missing_only,
        "summaries": [
            {
                "output_path": str(item.output_path),
                "attempted": item.attempted,
                "written": item.written,
                "skipped_existing": item.skipped_existing,
                "errors": item.errors,
            }
            for item in summaries
        ],
    }
