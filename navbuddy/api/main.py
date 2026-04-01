"""Minimal FastAPI backend for the NavBuddy-100 sample viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


def create_app(data_root: Path) -> FastAPI:
    """Create the FastAPI app with the given data root."""

    app = FastAPI(title="NavBuddy-100 Viewer", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Load data on startup ──

    samples: List[Dict[str, Any]] = []
    samples_by_id: Dict[str, Dict[str, Any]] = {}
    canonical_gt: Dict[str, Dict[str, Any]] = {}
    split_config: Dict[str, Any] = {}
    models_config: List[Dict[str, Any]] = []
    # model results: sample_id -> list of results
    results_by_sample: Dict[str, List[Dict[str, Any]]] = {}

    def _load():
        nonlocal samples, samples_by_id, canonical_gt, split_config, models_config, results_by_sample

        # Samples
        samples_path = data_root / "gt_split_samples.jsonl"
        if samples_path.exists():
            with open(samples_path) as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            samples_by_id = {s["id"]: s for s in samples}

        # Canonical GT
        gt_path = data_root / "canonical_gt.jsonl"
        if gt_path.exists():
            with open(gt_path) as f:
                for line in f:
                    if line.strip():
                        gt = json.loads(line)
                        canonical_gt[gt["sample_id"]] = gt

        # Split config
        split_path = data_root / "gt_split_config.json"
        if split_path.exists():
            with open(split_path) as f:
                split_config.update(json.load(f))

        # Models
        models_path = data_root / "models.json"
        if models_path.exists():
            with open(models_path) as f:
                data = json.load(f)
                models_config.extend(data.get("models", []))

        # Results
        results_dir = data_root / "results"
        if results_dir.is_dir():
            for result_file in sorted(results_dir.glob("*.jsonl")):
                with open(result_file) as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            sid = r.get("id", "")
                            if sid not in results_by_sample:
                                results_by_sample[sid] = []
                            results_by_sample[sid].append(r)

    _load()

    # Build split lookup
    split_lookup: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for sid in split_config.get(split_name, []):
            split_lookup[sid] = split_name

    # ── Endpoints ──

    @app.get("/api/stats")
    def get_stats():
        cities: Dict[str, int] = {}
        for s in samples:
            city = s.get("_city", s["id"].split("_route")[0])
            cities[city] = cities.get(city, 0) + 1

        models_with_results = set()
        for results in results_by_sample.values():
            for r in results:
                models_with_results.add(r["model_id"])

        return {
            "total_samples": len(samples),
            "total_models": len(models_with_results),
            "total_results": sum(len(r) for r in results_by_sample.values()),
            "cities": [{"name": k, "count": v} for k, v in sorted(cities.items())],
        }

    @app.get("/api/samples")
    def get_samples():
        out = []
        for s in samples:
            sid = s["id"]
            city = s.get("_city", sid.split("_route")[0])
            maneuver = s.get("maneuver", "")
            instruction = s.get("prior", {}).get("instruction", "")
            frames = s.get("images", {}).get("frames", [])
            overhead = s.get("images", {}).get("overhead")

            out.append({
                "id": sid,
                "route_id": s.get("route_id", ""),
                "step_index": s.get("step_index", 0),
                "city": city,
                "maneuver": maneuver,
                "instruction": instruction,
                "split": split_lookup.get(sid),
                "frame": frames[-1] if frames else None,
                "map": overhead,
                "distances": s.get("distances", {}),
                "result_count": len(results_by_sample.get(sid, [])),
            })
        return out

    @app.get("/api/samples/{sample_id}")
    def get_sample(sample_id: str):
        s = samples_by_id.get(sample_id)
        if not s:
            raise HTTPException(404, f"Sample {sample_id} not found")

        city = s.get("_city", sample_id.split("_route")[0])
        frames = s.get("images", {}).get("frames", [])
        overhead = s.get("images", {}).get("overhead")

        # Find prev/next
        ids = [ss["id"] for ss in samples]
        idx = ids.index(sample_id) if sample_id in ids else -1
        prev_id = ids[idx - 1] if idx > 0 else None
        next_id = ids[idx + 1] if idx < len(ids) - 1 else None

        return {
            "id": sample_id,
            "route_id": s.get("route_id", ""),
            "step_index": s.get("step_index", 0),
            "city": city,
            "maneuver": s.get("maneuver", ""),
            "instruction": s.get("prior", {}).get("instruction", ""),
            "split": split_lookup.get(sample_id),
            "frames": frames,
            "map": overhead,
            "distances": s.get("distances", {}),
            "geometry": s.get("geometry", {}),
            "osm_road": s.get("osm_road", {}),
            "result_count": len(results_by_sample.get(sample_id, [])),
            "prev_id": prev_id,
            "next_id": next_id,
        }

    @app.get("/api/samples/{sample_id}/results")
    def get_sample_results(sample_id: str):
        results = results_by_sample.get(sample_id, [])
        return {"sample_id": sample_id, "results": results}

    @app.get("/api/canonical-gt/{sample_id}")
    def get_canonical_gt(sample_id: str):
        gt = canonical_gt.get(sample_id)
        if not gt:
            raise HTTPException(404, f"No canonical GT for {sample_id}")
        return gt

    @app.get("/api/models")
    def get_models():
        return models_config

    # ── Static file serving (frames + maps) ──

    @app.get("/api/frames/{filename:path}")
    def get_frame(filename: str):
        path = data_root / "frames" / filename
        if not path.exists():
            raise HTTPException(404, f"Frame not found: {filename}")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/maps/{filename:path}")
    def get_map(filename: str):
        path = data_root / "maps" / filename
        if not path.exists():
            raise HTTPException(404, f"Map not found: {filename}")
        return FileResponse(path, media_type="image/png")

    return app
