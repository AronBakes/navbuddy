"""Google Street View Static API client with frame sampling."""

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from navbuddy.polylines import (
    bearing_deg,
    decode_polyline,
    haversine_m,
    pose_at_remaining_m,
)
from navbuddy.sampling import (
    profile_distances,
    profile_from_sample_mode,
)

LatLon = Tuple[float, float]

STREETVIEW_BASE_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_BASE_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

__all__ = [
    "StreetViewParams",
    "build_streetview_url",
    "check_streetview_coverage",
    "download_streetview_image",
    "sample_frames_for_step",
]


class StreetViewParams:
    """Parameters for a Street View image request."""

    def __init__(
        self,
        lat: float,
        lng: float,
        heading: float,
        distance_m: int,
        pitch: int = 0,
        fov: int = 90,
        size: str = "640x400",
    ):
        self.lat = lat
        self.lng = lng
        self.heading = heading
        self.distance_m = distance_m
        self.pitch = pitch
        self.fov = fov
        self.size = size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lng": self.lng,
            "heading": self.heading,
            "distance_m": self.distance_m,
            "pitch": self.pitch,
            "fov": self.fov,
            "size": self.size,
        }


def build_streetview_url(
    lat: float,
    lng: float,
    heading: float,
    *,
    api_key: str,
    pitch: int = 0,
    fov: int = 90,
    size: str = "640x400",
    source: str = "outdoor",
) -> str:
    """Build Google Street View Static API URL.

    Args:
        lat: Latitude
        lng: Longitude
        heading: Camera heading (0-360 degrees, 0=North)
        api_key: Google API key
        pitch: Camera pitch (-90 to 90, 0=horizontal)
        fov: Field of view (10-120 degrees)
        size: Image size as "WIDTHxHEIGHT"
        source: Panorama source filter — "outdoor" (default) excludes indoor
                panoramas from businesses/lobbies. Use "default" to include all.

    Returns:
        Complete Street View image URL
    """
    params = {
        "size": size,
        "location": f"{lat},{lng}",
        "heading": str(int(heading)),
        "pitch": str(pitch),
        "fov": str(fov),
        "source": source,
        "key": api_key,
    }
    return f"{STREETVIEW_BASE_URL}?{urllib.parse.urlencode(params)}"


def check_streetview_coverage(
    lat: float,
    lng: float,
    *,
    api_key: str,
    radius: int = 50,
    source: str = "outdoor",
) -> Dict[str, Any]:
    """Check if Street View imagery is available at location.

    Args:
        lat: Latitude
        lng: Longitude
        api_key: Google API key
        radius: Search radius in meters
        source: Panorama source filter ("outdoor" excludes indoor panoramas).
                The response includes 'pano_id' which can be used to deduplicate
                frames that snap to the same panorama.

    Returns:
        Metadata dict with 'status', 'pano_id', and 'location' keys.
    """
    params = {
        "location": f"{lat},{lng}",
        "radius": str(radius),
        "source": source,
        "key": api_key,
    }
    url = f"{METADATA_BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def download_streetview_image(
    params: Dict[str, Any],
    output_path: Path,
    *,
    api_key: str,
    timeout: float = 20.0,
    source: str = "outdoor",
) -> bool:
    """Download Street View image to disk.

    Args:
        params: Dict with lat, lng, heading, pitch, fov, size
        output_path: Path to save image
        api_key: Google API key
        timeout: Request timeout in seconds
        source: Panorama source filter ("outdoor" excludes indoor panoramas).

    Returns:
        True if download succeeded
    """
    url = build_streetview_url(
        params["lat"],
        params["lng"],
        params["heading"],
        api_key=api_key,
        pitch=params.get("pitch", 0),
        fov=params.get("fov", 90),
        size=params.get("size", "640x400"),
        source=source,
    )

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception:
        return False


def sample_frames_for_step(
    step: Dict[str, Any],
    *,
    frame_profile: Optional[str] = None,
    mode: str = "sparse",
    spacing: float = 20.0,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    size: str = "640x400",
    fov: int = 90,
    pitch: int = 0,
) -> List[StreetViewParams]:
    """Sample frame positions along a step polyline.

    Uses canonical frame profiles:
    - sparse4: 100/80/60/40m before maneuver (clamped for short steps)
    - video5m: full step sampled every 5m with 5m floor near maneuver
    - custom: windowed/spacing sampling from provided options

    Distances are measured back from the end of the polyline.

    Args:
        step: Route step dict with polyline
        frame_profile: Canonical profile (sparse4, video5m, custom). If omitted,
            legacy `mode` is mapped for backward compatibility.
        mode: Legacy mode alias: sparse|dense|custom
        spacing: Spacing between samples in meters (used with custom mode)
        sample_start: Start of sampling window in meters from end (custom mode)
        sample_end: End of sampling window in meters from end (custom mode)
        size: Image size
        fov: Field of view
        pitch: Camera pitch

    Returns:
        List of StreetViewParams for each sample point

    Examples:
        # Sample from 150m to 30m before end, every 20m
        sample_frames_for_step(step, mode="custom", spacing=20, sample_start=150, sample_end=30)
        # -> distances: [150, 130, 110, 90, 70, 50, 30]

        # If step is only 80m but user wants 150-30m, clamps to [80, 60, 40, 30]
        sample_frames_for_step(step, mode="custom", spacing=20, sample_start=150, sample_end=30)

        # Sample from start of step to 30m before end
        sample_frames_for_step(step, mode="custom", sample_end=30)

        # Sample entire step at 20m intervals
        sample_frames_for_step(step, mode="custom", spacing=20)
    """
    # Get polyline
    polyline_str = step.get("polyline", {}).get("encodedPolyline", "")
    if not polyline_str:
        # Try encodedPolyline5 (precision 5)
        polyline_str = step.get("polyline", {}).get("encodedPolyline5", "")

    if not polyline_str:
        return []

    try:
        coords = decode_polyline(polyline_str)
    except Exception:
        return []

    if len(coords) < 2:
        return []

    # Get step distance
    step_distance_m = step.get("distanceMeters", 0)
    if not step_distance_m:
        # Calculate from polyline
        step_distance_m = sum(
            haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
            for i in range(len(coords) - 1)
        )

    # Resolve profile from explicit frame_profile or legacy mode alias.
    resolved_profile = (frame_profile or profile_from_sample_mode(mode)).lower()
    distances = profile_distances(
        step_distance_m=step_distance_m,
        frame_profile=resolved_profile,
        spacing_m=spacing,
        sample_start=sample_start,
        sample_end=sample_end,
    )

    results = []
    for distance_m in distances:
        # Skip if step is too short
        if distance_m > step_distance_m:
            continue

        lat, lng, heading = pose_at_remaining_m(coords, step_distance_m, distance_m)
        results.append(
                StreetViewParams(
                    lat=lat,
                    lng=lng,
                    heading=heading,
                    distance_m=int(distance_m),
                    pitch=pitch,
                    fov=fov,
                    size=size,
                )
            )

    return results


