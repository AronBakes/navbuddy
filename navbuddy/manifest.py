"""Manifest generation and download for sharing datasets without images.

The manifest contains all route/sample metadata but no actual images.
Users can download images using their own Google Maps API key.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from navbuddy.polylines import pose_from_polyline
from navbuddy.sampling import FRAME_PROFILES, SPARSE4_TARGETS_REMAINING_M, clamp_targets, profile_distances
from navbuddy.utils import generate_frame_filename

DOWNLOAD_FRAME_PROFILES = {"manifest", *FRAME_PROFILES}


class ManifestFrame(BaseModel):
    """Frame metadata in manifest (no actual image data)."""
    filename: str
    distance_into_step_m: int
    remaining_m: int
    pano_id: Optional[str] = None  # Google Street View pano ID for re-download
    lat: Optional[float] = None
    lng: Optional[float] = None
    heading: Optional[float] = None
    pitch: Optional[int] = None
    fov: Optional[int] = None
    size: Optional[str] = None


class ManifestStep(BaseModel):
    """Step metadata in manifest."""
    step_index: int
    maneuver: str = ""
    instruction: str = ""
    polyline: str = ""
    distance_m: Optional[int] = 0
    start_lat: float = 0.0
    start_lng: float = 0.0
    end_lat: float = 0.0
    end_lng: float = 0.0
    heading: Optional[float] = None
    frames: List[ManifestFrame] = Field(default_factory=list)
    osm_road: Optional[Dict[str, Any]] = None


class ManifestRoute(BaseModel):
    """Route metadata in manifest."""
    route_id: str
    city: str = ""
    origin: Optional[Dict[str, float]] = None
    destination: Optional[Dict[str, float]] = None
    total_distance_m: Optional[int] = 0
    total_duration_s: Optional[int] = 0
    steps_count: Optional[int] = 0
    frames_count: Optional[int] = 0
    routing_engine: str = "google"
    steps: List[ManifestStep] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class DatasetManifest(BaseModel):
    """Complete dataset manifest."""
    version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    name: str = "navbuddy-dataset"
    description: str = "Navigation VLM training dataset"
    license: str = "CC-BY-NC-4.0"

    # Stats
    routes_count: int = 0
    samples_count: int = 0
    total_frames: int = 0

    # Download info
    download_instructions: str = """
To download images, you need a Google Maps API key with Street View Static API enabled.

1. Get an API key: https://developers.google.com/maps/documentation/streetview/get-api-key
2. Run: navbuddy download-manifest --manifest manifest.json --api-key YOUR_KEY
"""

    # Routes
    routes: List[ManifestRoute] = Field(default_factory=list)

    # Optional: embedded GT labels keyed by sample_id
    gt_labels: Optional[Dict[str, Any]] = None


def parse_frame_filename(filename: str) -> Dict[str, Any]:
    """Parse frame filename to extract distance info.

    Format: {route_id}_step{idx:03d}_{into:04d}m_{remaining:03d}m.jpg
    """
    # Remove extension
    name = Path(filename).stem

    # Find step and distance parts
    parts = name.split("_")

    result = {
        "filename": filename,
        "distance_into_step_m": 0,
        "remaining_m": 0,
    }

    for i, part in enumerate(parts):
        if part.startswith("step") and len(part) == 7:  # step000
            # Next parts should be distances
            if i + 2 < len(parts):
                into_part = parts[i + 1]  # 0000m
                remaining_part = parts[i + 2]  # 050m
                if into_part.endswith("m") and remaining_part.endswith("m"):
                    try:
                        result["distance_into_step_m"] = int(into_part[:-1])
                        result["remaining_m"] = int(remaining_part[:-1])
                    except ValueError:
                        pass
            break

    return result


def _frame_params_from_sample(
    *,
    sample: Dict[str, Any],
    remaining_m: int,
) -> Dict[str, Any]:
    """Reconstruct per-frame request params from sample geometry."""
    geometry = sample.get("geometry", {}) if isinstance(sample.get("geometry"), dict) else {}
    polyline = str(geometry.get("step_polyline") or "")
    point = pose_from_polyline(polyline, float(remaining_m))
    if point is None:
        return {
            "lat": geometry.get("start_lat"),
            "lng": geometry.get("start_lng"),
            "heading": geometry.get("heading"),
            "pitch": 0,
            "fov": 90,
            "size": "640x400",
        }
    lat, lng, heading = point
    return {
        "lat": lat,
        "lng": lng,
        "heading": heading,
        "pitch": 0,
        "fov": 90,
        "size": "640x400",
    }


def _filter_frames_by_profile(
    frames: List[ManifestFrame],
    *,
    step_distance_m: int,
    manifest_frame_profile: str,
) -> List[ManifestFrame]:
    profile = manifest_frame_profile.strip().lower()
    if profile == "all":
        return frames
    if profile != "sparse4":
        raise ValueError("manifest_frame_profile must be 'sparse4' or 'all'")

    targets = set(clamp_targets(step_distance_m, SPARSE4_TARGETS_REMAINING_M))
    filtered: List[ManifestFrame] = []
    seen_remaining = set()
    for frame in frames:
        rem = int(frame.remaining_m)
        if rem not in targets or rem in seen_remaining:
            continue
        seen_remaining.add(rem)
        filtered.append(frame)
    filtered.sort(key=lambda f: int(f.remaining_m), reverse=True)
    return filtered


def _resolve_step_frames(
    *,
    route_id: str,
    step: ManifestStep,
    frame_profile: str,
    spacing: float,
    sample_start: Optional[float],
    sample_end: Optional[float],
) -> List[ManifestFrame]:
    """Resolve frames for a step using manifest frames or profile resampling."""
    if frame_profile == "manifest":
        # Default: single closest-to-maneuver frame (smallest remaining_m)
        if step.frames:
            closest = min(step.frames, key=lambda f: int(f.remaining_m))
            return [closest]
        return []
    if frame_profile == "sparse4" and step.frames:
        # Return all manifest frames (already sparse4 in NavBuddy-100)
        return list(step.frames)

    distances = profile_distances(
        step_distance_m=float(step.distance_m or 0),
        frame_profile=frame_profile,
        spacing_m=spacing,
        sample_start=sample_start,
        sample_end=sample_end,
    )

    frames: List[ManifestFrame] = []
    for remaining in distances:
        remaining_i = int(remaining)
        point = pose_from_polyline(step.polyline, float(remaining_i))
        lat: Optional[float] = None
        lng: Optional[float] = None
        heading: Optional[float] = None
        if point is not None:
            lat, lng, heading = point
        else:
            lat = step.start_lat
            lng = step.start_lng
            heading = step.heading

        filename = generate_frame_filename(
            route_id=route_id,
            step_index=int(step.step_index),
            distance_from_end_m=remaining_i,
            step_distance_m=int(step.distance_m or 0),
        )
        frames.append(
            ManifestFrame(
                filename=filename,
                distance_into_step_m=max(0, int(round(float(step.distance_m or 0) - remaining_i))),
                remaining_m=remaining_i,
                lat=lat,
                lng=lng,
                heading=heading,
                pitch=0,
                fov=90,
                size="640x400",
            )
        )
    return frames


def build_manifest(
    data_root: Path,
    name: str = "navbuddy-dataset",
    description: str = "Navigation VLM training dataset",
    manifest_frame_profile: str = "sparse4",
) -> DatasetManifest:
    """Build a manifest from a data directory.

    Args:
        data_root: Root data directory (e.g., ./data/brisbane)
        name: Dataset name
        description: Dataset description
        manifest_frame_profile: sparse4 or all

    Returns:
        DatasetManifest object
    """
    samples_file = data_root / "samples.jsonl"
    routes_dir = data_root / "routes"

    # Load all samples grouped by route
    samples_by_route: Dict[str, List[Dict]] = {}

    if samples_file.exists():
        with open(samples_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                route_id = sample.get("route_id", "")
                if route_id not in samples_by_route:
                    samples_by_route[route_id] = []
                samples_by_route[route_id].append(sample)

    # Build routes
    manifest_routes = []
    total_frames = 0

    for route_id, samples in sorted(samples_by_route.items()):
        # Load route metadata
        metadata_file = routes_dir / route_id / "metadata.json"
        route_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                route_metadata = json.load(f)

        # Sort samples by step index
        samples.sort(key=lambda x: x.get("step_index", 0))

        # Build steps
        manifest_steps = []
        route_frames = 0

        for sample in samples:
            geometry = sample.get("geometry", {})
            distances = sample.get("distances", {})
            step_distance_m = int(distances.get("step_distance_m", 0) or 0)

            # Build frames list
            frames = []
            for frame_path in sample.get("images", {}).get("frames", []):
                if frame_path:
                    frame_info = parse_frame_filename(Path(frame_path).name)
                    frame_info.update(
                        _frame_params_from_sample(
                            sample=sample,
                            remaining_m=int(frame_info.get("remaining_m", 0) or 0),
                        )
                    )
                    frames.append(ManifestFrame(**frame_info))

            frames = _filter_frames_by_profile(
                frames,
                step_distance_m=step_distance_m,
                manifest_frame_profile=manifest_frame_profile,
            )
            route_frames += len(frames)

            step = ManifestStep(
                step_index=sample.get("step_index", 0),
                maneuver=sample.get("maneuver", "UNKNOWN"),
                instruction=sample.get("prior", {}).get("instruction", ""),
                polyline=geometry.get("step_polyline", ""),
                distance_m=step_distance_m,
                start_lat=geometry.get("start_lat", 0),
                start_lng=geometry.get("start_lng", 0),
                end_lat=geometry.get("end_lat", 0),
                end_lng=geometry.get("end_lng", 0),
                heading=geometry.get("heading"),
                frames=frames,
                osm_road=sample.get("osm_road"),
            )
            manifest_steps.append(step)

        total_frames += route_frames

        # Build route
        route = ManifestRoute(
            route_id=route_id,
            origin=route_metadata.get("origin", {"lat": 0, "lng": 0}),
            destination=route_metadata.get("destination", {"lat": 0, "lng": 0}),
            total_distance_m=route_metadata.get("total_distance_m", 0),
            total_duration_s=route_metadata.get("total_duration_s", 0),
            steps_count=len(manifest_steps),
            frames_count=route_frames,
            routing_engine=route_metadata.get("routing_engine", "google"),
            steps=manifest_steps,
        )
        manifest_routes.append(route)

    # Build manifest
    manifest = DatasetManifest(
        name=name,
        description=description,
        routes_count=len(manifest_routes),
        samples_count=sum(len(r.steps) for r in manifest_routes),
        total_frames=total_frames,
        routes=manifest_routes,
    )

    return manifest


def export_manifest(
    data_root: Path,
    output_file: Path,
    name: str = "navbuddy-dataset",
    description: str = "Navigation VLM training dataset",
    manifest_frame_profile: str = "sparse4",
    pretty: bool = True,
) -> DatasetManifest:
    """Export manifest to JSON file.

    Args:
        data_root: Root data directory
        output_file: Where to save the manifest
        name: Dataset name
        description: Dataset description
        manifest_frame_profile: sparse4 or all
        pretty: Use pretty JSON formatting

    Returns:
        The generated manifest
    """
    manifest = build_manifest(
        data_root,
        name,
        description,
        manifest_frame_profile=manifest_frame_profile,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        if pretty:
            f.write(manifest.model_dump_json(indent=2))
        else:
            f.write(manifest.model_dump_json())

    return manifest


def estimate_download_from_manifest(
    manifest_file: Path,
    output_dir: Path,
    *,
    frame_profile: str = "manifest",
    spacing: float = 5.0,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """Estimate frame counts/requests for a manifest download run."""
    profile = (frame_profile or "manifest").strip().lower()
    if profile not in DOWNLOAD_FRAME_PROFILES:
        raise ValueError(
            f"frame_profile must be one of: {', '.join(sorted(DOWNLOAD_FRAME_PROFILES))}"
        )

    with open(manifest_file, encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest = DatasetManifest(**manifest_data)

    frames_dir = output_dir / "frames"
    total_targets = 0
    existing = 0

    for route in manifest.routes:
        for step in route.steps:
            step_frames = _resolve_step_frames(
                route_id=route.route_id,
                step=step,
                frame_profile=profile,
                spacing=spacing,
                sample_start=sample_start,
                sample_end=sample_end,
            )
            total_targets += len(step_frames)
            for frame in step_frames:
                if (frames_dir / frame.filename).exists():
                    existing += 1

    to_download = max(0, total_targets - existing)
    estimated_requests = min(to_download, int(limit)) if limit is not None else to_download

    return {
        "routes": len(manifest.routes),
        "steps": sum(len(route.steps) for route in manifest.routes),
        "total_targets": total_targets,
        "existing": existing,
        "to_download": to_download,
        "estimated_requests": estimated_requests,
    }


def download_from_manifest(
    manifest_file: Path,
    output_dir: Path,
    api_key: str,
    limit: Optional[int] = None,
    frame_profile: str = "manifest",
    spacing: float = 5.0,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    render_maps: bool = False,
    car_icon: str = "arrow",
    verbose: bool = True,
) -> Dict[str, int]:
    """Download images from a manifest using Google Street View API.

    Args:
        manifest_file: Path to manifest JSON
        output_dir: Where to save images
        api_key: Google Maps API key
        limit: Max frames to download
        verbose: Print progress

    Returns:
        Dict with download stats
    """
    import urllib.request

    profile = (frame_profile or "manifest").strip().lower()
    if profile not in DOWNLOAD_FRAME_PROFILES:
        raise ValueError(
            f"frame_profile must be one of: {', '.join(sorted(DOWNLOAD_FRAME_PROFILES))}"
        )

    # Load manifest
    with open(manifest_file, encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest = DatasetManifest(**manifest_data)

    route_step_frames: Dict[str, Dict[int, List[ManifestFrame]]] = {}
    total_target_frames = 0
    for route in manifest.routes:
        route_steps: Dict[int, List[ManifestFrame]] = {}
        for step in route.steps:
            resolved = _resolve_step_frames(
                route_id=route.route_id,
                step=step,
                frame_profile=profile,
                spacing=spacing,
                sample_start=sample_start,
                sample_end=sample_end,
            )
            route_steps[int(step.step_index)] = resolved
            total_target_frames += len(resolved)
        route_step_frames[route.route_id] = route_steps

    if verbose:
        print(f"Manifest: {manifest.name}")
        print(f"Routes: {manifest.routes_count}")
        print(f"Frames: {total_target_frames}")
        print(f"Frame profile: {profile}")
        if profile == "custom":
            print(f"Spacing: {spacing}m")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    maps_dir = output_dir / "maps"
    routes_dir = output_dir / "routes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    routes_dir.mkdir(parents=True, exist_ok=True)

    # Persist manifest + reconstructed dataset metadata so output_dir is usable directly.
    # This allows `navbuddy stats/play/view` on downloaded datasets.
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    samples_path = output_dir / "samples.jsonl"
    samples_written = 0
    routes_written = 0

    # Optionally import map renderer (requires playwright)
    _render_map_fn = None
    _add_overlay_fn = None
    if render_maps:
        try:
            from navbuddy.map_renderer_osm import render_map as _render_map_fn, HAS_PLAYWRIGHT  # type: ignore
            if not HAS_PLAYWRIGHT:
                raise ImportError("playwright not available")
            from navbuddy.polylines import decode_polyline as _decode_polyline
            from navbuddy.overlays import (  # type: ignore
                add_overlay_to_map as _add_overlay_fn,
                estimate_eta_from_sample,
                build_step_payload_from_sample,
                overlay_scale_for_map,
            )
        except ImportError:
            if verbose:
                print("  [warning] map rendering unavailable — run: pip install navbuddy[render] && playwright install chromium")
            _render_map_fn = None

    with open(samples_path, "w", encoding="utf-8") as samples_f:
        for route in manifest.routes:
            route_dir = routes_dir / route.route_id
            route_dir.mkdir(parents=True, exist_ok=True)
            route_frames_count = sum(
                len(route_step_frames[route.route_id].get(int(step.step_index), []))
                for step in route.steps
            )

            route_metadata = {
                "route_id": route.route_id,
                "origin": route.origin,
                "destination": route.destination,
                "total_distance_m": route.total_distance_m,
                "total_duration_s": route.total_duration_s,
                "steps_count": route.steps_count,
                "frames_count": route_frames_count,
                "created_at": manifest.created_at,
                "routing_engine": route.routing_engine,
                "source_manifest": str(manifest_file),
                "frame_profile": profile,
            }
            with open(route_dir / "metadata.json", "w", encoding="utf-8") as rf:
                json.dump(route_metadata, rf, indent=2)
            routes_written += 1

            # Precompute remaining distance at each step start.
            remaining_per_step: List[int] = []
            running = 0
            for step in reversed(route.steps):
                running += int(step.distance_m or 0)
                remaining_per_step.append(running)
            remaining_per_step.reverse()

            for idx, step in enumerate(route.steps):
                sample_id = f"{route.route_id}_step{step.step_index:03d}"
                frames = route_step_frames[route.route_id].get(int(step.step_index), [])
                frame_paths = [f"frames/{frame.filename}" for frame in frames]

                # Render OSM overhead map if requested
                map_rel: Optional[str] = None
                map_filename = f"{route.route_id}_step{step.step_index:03d}_map.png"
                map_path = maps_dir / map_filename
                if _render_map_fn is not None and not map_path.exists():
                    try:
                        coords = _decode_polyline(step.polyline)
                        ok = _render_map_fn(
                            step_polyline_coords=coords,
                            car_lat=step.start_lat,
                            car_lng=step.start_lng,
                            heading=step.heading or 0.0,
                            start_lat=step.start_lat,
                            start_lng=step.start_lng,
                            end_lat=step.end_lat,
                            end_lng=step.end_lng,
                            output_path=map_path,
                            car_icon=car_icon,
                        )
                        if ok:
                            if verbose:
                                print(f"  [map] {map_filename}")
                            # Apply navigation overlay (nav card + ETA) if available
                            if _add_overlay_fn is not None:
                                try:
                                    remaining_m = remaining_per_step[idx]
                                    # Build a minimal sample-like dict for the helpers
                                    _sample_proxy = {
                                        "prior": {"instruction": step.instruction},
                                        "maneuver": step.maneuver,
                                        "distances": {
                                            "step_distance_m": step.distance_m,
                                            "remaining_distance_m": remaining_m,
                                        },
                                    }
                                    _next_step = route.steps[idx + 1] if idx + 1 < len(route.steps) else None
                                    _next_proxy = {
                                        "prior": {"instruction": _next_step.instruction},
                                        "maneuver": _next_step.maneuver,
                                        "distances": {"step_distance_m": _next_step.distance_m, "remaining_distance_m": 0},
                                    } if _next_step else None
                                    _step_payload = build_step_payload_from_sample(_sample_proxy)
                                    _next_payload = build_step_payload_from_sample(_next_proxy) if _next_proxy else None
                                    _route_meta = {"total_distance_m": route.total_distance_m, "total_duration_s": route.total_duration_s}
                                    _arrival, _mins, _dist_km = estimate_eta_from_sample(_sample_proxy, _route_meta)
                                    _scale = overlay_scale_for_map(640, 400)
                                    _add_overlay_fn(
                                        map_path,
                                        _step_payload,
                                        _next_payload,
                                        arrival_time=_arrival,
                                        minutes_remaining=_mins,
                                        distance_km=_dist_km,
                                        overlay_scale=_scale,
                                    )
                                except Exception as oe:
                                    if verbose:
                                        print(f"  [map] overlay failed {map_filename}: {oe}")
                    except Exception as e:
                        if verbose:
                            print(f"  [map] failed {map_filename}: {e}")
                if map_path.exists():
                    map_rel = f"maps/{map_filename}"

                sample = {
                    "id": sample_id,
                    "route_id": route.route_id,
                    "step_index": step.step_index,
                    "dataset_version": "v1.0",
                    "split": "train",
                    "maneuver": step.maneuver,
                    "prior": {"instruction": step.instruction},
                    "images": {
                        "overhead": map_rel,
                        "frames": frame_paths,
                    },
                    "geometry": {
                        "step_polyline": step.polyline,
                        "start_lat": step.start_lat,
                        "start_lng": step.start_lng,
                        "end_lat": step.end_lat,
                        "end_lng": step.end_lng,
                        "heading": step.heading,
                    },
                    "distances": {
                        "step_distance_m": int(step.distance_m or 0),
                        "remaining_distance_m": remaining_per_step[idx],
                    },
                    "osm_road": step.osm_road,
                    "metadata": {
                        "source": "manifest_download",
                        "created_at": manifest.created_at,
                    },
                }

                samples_f.write(json.dumps(sample) + "\n")
                samples_written += 1

    # Write embedded GT labels if present
    if manifest.gt_labels:
        labels_path = output_dir / "labels.jsonl"
        with open(labels_path, "w", encoding="utf-8") as lf:
            for sid, lbl in manifest.gt_labels.items():
                lf.write(json.dumps({"sample_id": sid, **lbl}) + "\n")
        if verbose:
            print(f"\nLabels written: {labels_path} ({len(manifest.gt_labels)} entries)")

    # Download frames
    downloaded = 0
    skipped = 0
    failed = 0

    for route in manifest.routes:
        if verbose:
            print(f"\nRoute: {route.route_id}")

        for step in route.steps:
            frames = route_step_frames[route.route_id].get(int(step.step_index), [])
            for frame in frames:
                if limit and downloaded >= limit:
                    break

                output_path = frames_dir / frame.filename

                # Skip if already exists
                if output_path.exists():
                    skipped += 1
                    continue

                # Build Street View URL
                # Prefer exact per-frame geometry when provided.
                lat = frame.lat
                lng = frame.lng
                heading = frame.heading
                pitch = frame.pitch if frame.pitch is not None else 0
                fov = frame.fov if frame.fov is not None else 90
                size = frame.size or "640x400"

                # Backward-compatible fallback: reconstruct from step polyline + remaining distance.
                if lat is None or lng is None or heading is None:
                    point = pose_from_polyline(step.polyline, float(frame.remaining_m))
                    if point is not None:
                        lat, lng, heading = point

                # Last resort fallback for legacy manifests with no geometry info.
                if lat is None or lng is None:
                    lat = step.start_lat
                    lng = step.start_lng
                if heading is None:
                    heading = step.heading or 0

                url = (
                    f"https://maps.googleapis.com/maps/api/streetview"
                    f"?size={size}"
                    f"&location={lat},{lng}"
                    f"&heading={heading}"
                    f"&pitch={pitch}"
                    f"&fov={fov}"
                    f"&key={api_key}"
                )

                try:
                    urllib.request.urlretrieve(url, output_path)
                    downloaded += 1
                    if verbose:
                        print(f"  ✓ {frame.filename}")
                except Exception as e:
                    failed += 1
                    if verbose:
                        print(f"  ✗ {frame.filename}: {e}")

            if limit and downloaded >= limit:
                break
        if limit and downloaded >= limit:
            break

    maps_rendered = sum(
        1 for route in manifest.routes for step in route.steps
        if (maps_dir / f"{route.route_id}_step{step.step_index:03d}_map.png").exists()
    ) if render_maps else 0

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "routes_written": routes_written,
        "samples_written": samples_written,
        "maps_rendered": maps_rendered,
    }


__all__ = [
    "DatasetManifest",
    "ManifestRoute",
    "ManifestStep",
    "ManifestFrame",
    "build_manifest",
    "export_manifest",
    "estimate_download_from_manifest",
    "download_from_manifest",
]
