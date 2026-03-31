"""NavBuddy route generation pipeline.

Orchestrates: Routing (Google Directions API) -> frame sampling -> Street View download -> output.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from navbuddy.routing_client import get_route
from navbuddy.streetview_client import (
    sample_frames_for_step,
    check_streetview_coverage,
    download_streetview_image,
)
from navbuddy.osm_client import enrich_step_with_osm
from navbuddy.map_renderer_osm import generate_step_map_osm, CAR_ICONS, DEFAULT_CAR_ICON_SCALE
from navbuddy.overlays import add_overlay_to_map
from navbuddy.utils import (
    generate_route_id,
    generate_sample_id,
    resolve_effective_instructions,
    generate_frame_filename,
    generate_map_filename,
    get_api_key,
)
from navbuddy.polylines import pose_from_polyline

LatLon = Tuple[float, float]

# Default assets directory (relative to package)
DEFAULT_ASSETS_DIR = Path(__file__).parent.parent / "assets"

__all__ = ["generate_route", "preflight_route"]


STREETVIEW_COST_PER_1000 = 7.00  # USD per 1000 Street View requests


def preflight_route(
    origin: LatLon,
    destination: LatLon,
    *,
    frame_profile: str = "sparse4",
    spacing: float = 20.0,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    map_renderer: str = "osm",
    skip_images: bool = False,
) -> Dict[str, Any]:
    """Fetch route and estimate API calls/cost without downloading anything.

    Returns dict with route info, step breakdown, frame counts, and cost estimate.
    """
    from navbuddy.sampling import profile_distances

    api_key = get_api_key("GOOGLE_MAPS_API_KEY") or get_api_key("GOOGLE_STREETVIEW_API_KEY")

    route = get_route(
        origin,
        destination,
        api_key=api_key,
    )

    all_steps = []
    for leg in route.get("legs", []):
        all_steps.extend(leg.get("steps", []))

    if not all_steps:
        raise RuntimeError("No steps found in route")

    steps = resolve_effective_instructions(all_steps)

    total_distance_m = route.get("distanceMeters", 0)
    total_duration_s = _parse_duration(route.get("duration", "0s"))

    step_details = []
    total_frames = 0

    for step_idx, step in enumerate(steps):
        step_dist = step.get("distanceMeters", 0) or 0
        distances = profile_distances(
            step_distance_m=step_dist,
            frame_profile=frame_profile,
            spacing_m=spacing,
            sample_start=sample_start,
            sample_end=sample_end,
        )
        n_frames = len(distances)
        total_frames += n_frames

        maneuver = _get_maneuver_type(step)
        instruction = step.get("effective_instruction", "")

        step_details.append({
            "step_index": step_idx,
            "maneuver": maneuver,
            "instruction": instruction,
            "distance_m": step_dist,
            "frames": n_frames,
            "remaining_targets_m": distances,
        })

    # API call counts
    # Street View: 1 metadata check + 1 image download per frame
    sv_requests = total_frames * 2 if not skip_images else 0
    # Maps: 1 per step (free for OSM, 1 Static Maps API call for Google)
    map_requests = len(steps) if map_renderer == "google" else 0
    # Routing: already done (1 call)
    routing_requests = 1

    # Cost: only Street View image downloads are billed ($7/1000)
    # Metadata checks are free
    sv_image_requests = total_frames if not skip_images else 0
    sv_cost = (sv_image_requests / 1000.0) * STREETVIEW_COST_PER_1000
    maps_cost = (map_requests / 1000.0) * 7.00  # Static Maps ~$7/1000
    total_cost = sv_cost + maps_cost

    return {
        "steps_count": len(steps),
        "total_distance_m": total_distance_m,
        "total_duration_s": total_duration_s,
        "frame_profile": frame_profile,
        "total_frames": total_frames,
        "total_maps": len(steps),
        "step_details": step_details,
        "api_calls": {
            "routing": routing_requests,
            "streetview_metadata": total_frames if not skip_images else 0,
            "streetview_images": sv_image_requests,
            "osm_overpass": len(steps),
            "static_maps": map_requests,
            "playwright_maps": len(steps) if map_renderer == "osm" else 0,
        },
        "estimated_cost_usd": round(total_cost, 4),
        "cost_breakdown": {
            "streetview": round(sv_cost, 4),
            "static_maps": round(maps_cost, 4),
            "osm_overpass": 0.0,
            "playwright_maps": 0.0,
        },
    }


def generate_route(
    origin: LatLon,
    destination: LatLon,
    *,
    output_dir: Path,
    city: Optional[str] = None,
    route_id: Optional[str] = None,
    skip_images: bool = False,
    frame_profile: str = "sparse4",
    sample_mode: str = "sparse",
    spacing: float = 20.0,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    map_renderer: str = "osm",
    car_icon: str = "sedan",
    car_icon_scale: float = DEFAULT_CAR_ICON_SCALE,
    assets_dir: Optional[Path] = None,
    add_overlays: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Generate a complete route with frames and metadata.

    Args:
        origin: (lat, lon) tuple for start
        destination: (lat, lon) tuple for end
        output_dir: Base output directory
        city: Optional city name for ID prefix
        route_id: Optional custom route ID
        skip_images: Skip downloading Street View images
        frame_profile: Canonical frame profile (sparse4, video5m, custom)
        sample_mode: Legacy alias (sparse/dense/custom), retained for compatibility
        spacing: Custom-profile spacing between samples in meters
        sample_start: Custom-profile sampling window start (meters from step end)
        sample_end: Custom-profile sampling window end (meters from step end)
        map_renderer: "osm" (default, Playwright + Leaflet) or "google" (Static Maps API)
        car_icon: Car icon type - "sedan" (default), "cybertruck", "f1", "model3", "wrx", "arrow"
        car_icon_scale: Scale factor for car icon (default 0.025, maintains aspect ratio)
        assets_dir: Directory containing car icon images (default: package assets/)
        add_overlays: Add navigation overlays (header + ETA) to map images
        progress_callback: Optional callback for progress updates

    Returns:
        Dict with route_id, steps_count, frames_count, output_dir, engine
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Generate route ID
    if not route_id:
        route_id = generate_route_id(city=city)

    # Get API key for routing (Google) and Street View
    api_key = get_api_key("GOOGLE_MAPS_API_KEY") or get_api_key("GOOGLE_STREETVIEW_API_KEY")

    log("Requesting route from Google Directions...")

    try:
        route = get_route(
            origin,
            destination,
            api_key=api_key,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get route: {e}")

    # Extract steps from all legs
    all_steps = []
    for leg in route.get("legs", []):
        all_steps.extend(leg.get("steps", []))

    if not all_steps:
        raise RuntimeError("No steps found in route")

    log(f"Got {len(all_steps)} steps, applying N+1 instruction policy...")

    # Apply N+1 instruction resolution
    steps = resolve_effective_instructions(all_steps)

    # Extract full route polyline for dual-layer map rendering
    route_polyline = route.get("polyline", {}).get("encodedPolyline5", "")
    if not route_polyline:
        route_polyline = route.get("polyline", {}).get("encodedPolyline", "")

    if route_polyline:
        log(f"Route polyline: {len(route_polyline)} chars")
    else:
        log("Warning: No route polyline found - only step polylines will be shown")

    # Set up assets directory for custom car icons
    if assets_dir is None:
        assets_dir = DEFAULT_ASSETS_DIR
    assets_dir = Path(assets_dir)

    # Create output directories
    output_dir = Path(output_dir)
    routes_dir = output_dir / "routes" / route_id
    frames_dir = output_dir / "frames"
    maps_dir = output_dir / "maps"

    routes_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total route distance and duration for overlays
    total_distance_m = route.get("distanceMeters", 0)
    total_duration_s = _parse_duration(route.get("duration", "0s"))

    # Process each step
    samples = []
    total_frames = 0
    total_maps = 0

    for step_idx, step in enumerate(steps):
        log(f"Processing step {step_idx + 1}/{len(steps)}...")

        # Enrich step with OSM data
        log(f"  Querying OSM for road metadata...")
        enrich_step_with_osm(step, radius_m=25.0)

        sample_id = generate_sample_id(route_id, step_idx)

        # Sample frame positions
        frame_params = sample_frames_for_step(
            step,
            frame_profile=frame_profile,
            mode=sample_mode,
            spacing=spacing,
            sample_start=sample_start,
            sample_end=sample_end,
        )

        # Track frame paths
        frame_paths = []

        if not skip_images and api_key and frame_params:
            seen_pano_ids: set = set()  # deduplicate within this step
            for i, params in enumerate(frame_params):
                log(f"  Downloading frame {i + 1}/{len(frame_params)} at {params.distance_m}m...")

                # Check coverage first (returns pano_id and snapped location)
                coverage = check_streetview_coverage(
                    params.lat, params.lng, api_key=api_key
                )

                if coverage.get("status") != "OK":
                    continue  # No outdoor coverage at this location

                # Skip if this panorama was already downloaded for this step
                pano_id = coverage.get("pano_id")
                if pano_id and pano_id in seen_pano_ids:
                    log(f"  Skipping frame {i + 1} — duplicate panorama (pano_id {pano_id[:8]}…)")
                    continue
                if pano_id:
                    seen_pano_ids.add(pano_id)

                # Generate filename
                step_distance_m = step.get("distanceMeters", 0)
                filename = generate_frame_filename(
                    route_id, step_idx, params.distance_m, step_distance_m
                )
                frame_path = frames_dir / filename

                # Download image
                success = download_streetview_image(
                    params.to_dict(), frame_path, api_key=api_key
                )

                if success:
                    frame_paths.append(f"frames/{filename}")
                    total_frames += 1

        # Generate overhead map (always generate maps, skip_images only affects Street View)
        map_filename = generate_map_filename(route_id, step_idx)
        map_path = maps_dir / map_filename
        map_generated = False

        # Prefer a near-maneuver map pose: 20m from step end,
        # or midpoint for short steps (<40m).
        car_lat = None
        car_lng = None
        car_heading = None
        step_poly = step.get("polyline", {})
        step_polyline = step_poly.get("encodedPolyline5") or step_poly.get("encodedPolyline", "")
        step_distance_m = float(step.get("distanceMeters", 0) or 0)
        if step_polyline and step_distance_m > 0:
            target_from_end = (step_distance_m / 2.0) if step_distance_m < 40.0 else 20.0
            pose = pose_from_polyline(step_polyline, target_from_end)
            if pose:
                car_lat, car_lng, car_heading = pose

        # Fallback to first sampled frame if available.
        if car_lat is None or car_lng is None or car_heading is None:
            if frame_params:
                car_lat = frame_params[0].lat
                car_lng = frame_params[0].lng
                car_heading = frame_params[0].heading

        log(f"  Generating OSM overhead map...")
        success = generate_step_map_osm(
            step, map_path,
            car_lat=car_lat, car_lng=car_lng, car_heading=car_heading,
            route_polyline=route_polyline,
            car_icon=car_icon,
            car_icon_scale=car_icon_scale,
            assets_dir=assets_dir,
        )
        if success:
            map_generated = True
            total_maps += 1

        # Add navigation overlays if requested
        if map_generated and add_overlays:
            log(f"  Adding navigation overlay...")
            # Get next step for secondary instruction
            next_step = steps[step_idx + 1] if step_idx + 1 < len(steps) else None
            # Calculate remaining distance, accounting for car position within current step
            step_distance_m = step.get("distanceMeters", 0)
            distance_into_step = frame_params[0].distance_m if frame_params else 0
            current_step_remaining_m = max(0, step_distance_m - distance_into_step)
            future_steps_m = _calc_remaining_distance(steps, step_idx + 1) if step_idx + 1 < len(steps) else 0
            remaining_m = current_step_remaining_m + future_steps_m
            remaining_km = remaining_m / 1000.0
            # Calculate remaining time, accounting for car position within current step
            fraction_done = (distance_into_step / step_distance_m) if step_distance_m > 0 else 0.0
            current_step_dur_s = _parse_duration(step.get("staticDuration", ""))
            current_step_remaining_dur_s = current_step_dur_s * (1.0 - fraction_done)
            future_steps_dur_s = _calc_remaining_duration_s(steps, step_idx + 1) if step_idx + 1 < len(steps) else None
            if future_steps_dur_s is not None or current_step_dur_s > 0:
                remaining_dur_s = current_step_remaining_dur_s + (future_steps_dur_s or 0)
                remaining_min = remaining_dur_s / 60.0
            else:
                # Fallback: assume 40 km/h average
                remaining_min = (remaining_m / 1000.0) / 40.0 * 60.0
                remaining_dur_s = remaining_min * 60
            # Calculate arrival time
            arrival_time = _calc_arrival_time(remaining_dur_s)
            # Further reduce overlay card size for OSM renders.
            overlay_scale = 2.1 if map_renderer == "osm" else 1.0
            try:
                add_overlay_to_map(
                    map_path, step, next_step,
                    arrival_time=arrival_time,
                    minutes_remaining=remaining_min,
                    distance_km=remaining_km,
                    use_playwright=True,
                    overlay_scale=overlay_scale,
                )
            except Exception as e:
                log(f"  Warning: Failed to add overlay: {e}")

        # Build sample entry
        start_loc = step.get("startLocation", {}).get("latLng", {})
        end_loc = step.get("endLocation", {}).get("latLng", {})

        # Get polyline (try encodedPolyline5 first, fallback to encodedPolyline)
        step_poly = step.get("polyline", {})
        step_polyline = step_poly.get("encodedPolyline5") or step_poly.get("encodedPolyline", "")

        sample = {
            "id": sample_id,
            "route_id": route_id,
            "step_index": step_idx,
            "dataset_version": "v1.0",
            "split": "train",  # Default to train, can be reassigned later
            "maneuver": _get_maneuver_type(step),
            "prior": {
                "instruction": step.get("effective_instruction", ""),
            },
            "images": {
                "overhead": f"maps/{map_filename}" if map_generated else None,
                "frames": frame_paths,
            },
            "geometry": {
                "step_polyline": step_polyline,
                "start_lat": start_loc.get("latitude"),
                "start_lng": start_loc.get("longitude"),
                "end_lat": end_loc.get("latitude"),
                "end_lng": end_loc.get("longitude"),
                "heading": frame_params[0].heading if frame_params else None,
            },
            "distances": {
                "step_distance_m": step.get("distanceMeters", 0),
                "remaining_distance_m": _calc_remaining_distance(steps, step_idx),
            },
            "osm_road": _build_osm_road_data(step),
            "metadata": {
                "source": "google",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        samples.append(sample)

    # Save route metadata
    log("Saving route metadata...")

    metadata = {
        "route_id": route_id,
        "origin": {"lat": origin[0], "lng": origin[1]},
        "destination": {"lat": destination[0], "lng": destination[1]},
        "total_distance_m": route.get("distanceMeters", 0),
        "total_duration_s": _parse_duration(route.get("duration", "0s")),
        "steps_count": len(steps),
        "frames_count": total_frames,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame_profile": frame_profile,
        "sample_mode": sample_mode,
    }

    with open(routes_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save polyline
    polyline_data = {
        "encoded": route.get("polyline", {}).get("encodedPolyline", ""),
        "encoded5": route.get("polyline", {}).get("encodedPolyline5", ""),
    }
    with open(routes_dir / "polyline.json", "w", encoding="utf-8") as f:
        json.dump(polyline_data, f, indent=2)

    # Save raw guidance steps
    with open(routes_dir / "guidance.json", "w", encoding="utf-8") as f:
        json.dump({"steps": steps}, f, indent=2)

    # Save samples.jsonl
    samples_file = output_dir / "samples.jsonl"
    with open(samples_file, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return {
        "route_id": route_id,
        "engine": "google",
        "steps_count": len(steps),
        "frames_count": total_frames,
        "maps_count": total_maps,
        "output_dir": str(output_dir),
    }


def _get_maneuver_type(step: Dict[str, Any]) -> str:
    """Extract maneuver type from step.

    Handles both Routes API v2 (maneuver as SCREAMING_SNAKE string)
    and legacy Directions API (maneuver as kebab-case or integer type code).
    """
    nav = step.get("navigationInstruction", {})

    # Prefer the maneuver string field (present in both APIs)
    maneuver = nav.get("maneuver")
    if maneuver and isinstance(maneuver, str) and maneuver not in ("", "?"):
        # Normalize: kebab-case → SCREAMING_SNAKE (legacy "turn-left" → "TURN_LEFT")
        return maneuver.upper().replace("-", "_")

    # Fallback: legacy integer type code → string
    type_code = nav.get("type")
    if type_code is None:
        return "UNKNOWN"

    type_map = {
        0: "NONE",
        1: "DEPART",
        2: "DEPART_RIGHT",
        3: "DEPART_LEFT",
        4: "DESTINATION",
        5: "DESTINATION_RIGHT",
        6: "DESTINATION_LEFT",
        7: "NAME_CHANGE",
        8: "CONTINUE",
        9: "SLIGHT_RIGHT",
        10: "TURN_RIGHT",
        11: "SHARP_RIGHT",
        12: "UTURN_RIGHT",
        13: "UTURN_LEFT",
        14: "SHARP_LEFT",
        15: "TURN_LEFT",
        16: "SLIGHT_LEFT",
        17: "RAMP_STRAIGHT",
        18: "RAMP_RIGHT",
        19: "RAMP_LEFT",
        20: "EXIT_RIGHT",
        21: "EXIT_LEFT",
        22: "STRAIGHT",
        23: "KEEP_RIGHT",
        24: "KEEP_LEFT",
        25: "MERGE",
        26: "ROUNDABOUT_ENTER",
        27: "ROUNDABOUT_EXIT",
        28: "FERRY",
        29: "FERRY_TRAIN",
    }

    return type_map.get(type_code, f"TYPE_{type_code}")


def _extract_road_name(step: Dict[str, Any]) -> Optional[str]:
    """Extract road name from step instruction."""
    instruction = step.get("effective_instruction", "")

    # Try to extract road name after "onto" or "on"
    for keyword in [" onto ", " on "]:
        if keyword in instruction.lower():
            idx = instruction.lower().find(keyword)
            return instruction[idx + len(keyword):].strip().rstrip(".")

    return None


def _build_osm_road_data(step: Dict[str, Any]) -> Dict[str, Any]:
    """Build osm_road dict from step data.

    Combines OSM data from Overpass query with step metadata.
    """
    # Get OSM data if available (from enrich_step_with_osm)
    osm = step.get("osm_road", {})

    # Get street names from step
    street_names = step.get("street_names", [])

    # Build combined road data
    road_data = {
        "highway": osm.get("highway"),
        "name": osm.get("name") or (street_names[0] if street_names else _extract_road_name(step)),
        "ref": osm.get("ref"),
        "maxspeed": osm.get("maxspeed"),
        "lanes": osm.get("lanes"),
        "surface": osm.get("surface"),
        "oneway": osm.get("oneway"),
        "lit": osm.get("lit"),
        "bridge": osm.get("bridge"),
        "tunnel": osm.get("tunnel"),
        "toll": osm.get("toll"),
    }

    # Include street_names array if available
    if street_names:
        road_data["street_names"] = street_names

    return road_data


def _calc_remaining_distance(steps: List[Dict], current_idx: int) -> int:
    """Calculate remaining distance from current step to end."""
    remaining = 0
    for step in steps[current_idx:]:
        remaining += step.get("distanceMeters", 0)
    return remaining


def _parse_duration(duration_str: str) -> int:
    """Parse duration string like '120s' to seconds."""
    if duration_str.endswith("s"):
        try:
            return int(duration_str[:-1])
        except ValueError:
            pass
    return 0


def _calc_remaining_duration_s(steps: List[Dict], current_idx: int) -> Optional[int]:
    """Calculate remaining duration in seconds from step durations.

    Uses staticDuration fields from Google Directions API routing responses.
    Returns None if no duration data is available.
    """
    total = 0
    has_data = False
    for step in steps[current_idx:]:
        dur_str = step.get("staticDuration", "")
        dur_s = _parse_duration(dur_str)
        if dur_s > 0:
            has_data = True
        total += dur_s
    return total if has_data else None


def _calc_arrival_time(remaining_seconds: float) -> str:
    """Calculate arrival time string from remaining seconds."""
    now = datetime.now()
    arrival = now + timedelta(seconds=remaining_seconds)
    return arrival.strftime("%I:%M").lstrip("0")
