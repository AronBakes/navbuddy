"""Google Directions API routing client."""

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

from navbuddy.polylines import decode_polyline, encode_polyline, bearing_deg


def _strip_html(text: str) -> str:
    """Remove HTML tags from text, inserting spaces at block-level tag boundaries."""
    # Replace block-level tags with a space so adjacent sentences don't merge
    text = re.sub(r"<(?:div|br|p|span)[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple spaces
    return re.sub(r" {2,}", " ", text).strip()

LatLon = Tuple[float, float]

GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

__all__ = [
    "get_route",
    "get_route_google",
    "normalize_route_response",
    "geocode",
    "reverse_geocode",
]


def geocode(address: str, *, api_key: Optional[str] = None) -> LatLon:
    """Convert an address string to (lat, lng) using Google Geocoding API."""
    import os

    api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("Google Geocoding API requires GOOGLE_MAPS_API_KEY")

    params = urllib.parse.urlencode({"address": address, "key": api_key})
    url = f"{GOOGLE_GEOCODE_URL}?{params}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    if data.get("status") != "OK" or not data.get("results"):
        raise ValueError(f"Geocoding failed for '{address}': {data.get('status')}")

    loc = data["results"][0]["geometry"]["location"]
    return (loc["lat"], loc["lng"])


def reverse_geocode(lat: float, lng: float, *, api_key: Optional[str] = None) -> str:
    """Convert (lat, lng) to a human-readable address using Google Geocoding API."""
    import os

    api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("Google Geocoding API requires GOOGLE_MAPS_API_KEY")

    params = urllib.parse.urlencode({"latlng": f"{lat},{lng}", "key": api_key})
    url = f"{GOOGLE_GEOCODE_URL}?{params}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    if data.get("status") != "OK" or not data.get("results"):
        raise ValueError(f"Reverse geocoding failed for ({lat}, {lng}): {data.get('status')}")

    return data["results"][0]["formatted_address"]


def get_route(
    origin: LatLon,
    destination: LatLon,
    *,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Get a route using Google Directions API.

    Args:
        origin: (lat, lon) start point
        destination: (lat, lon) end point
        api_key: Google Maps API key (required)
        **kwargs: Additional engine-specific options

    Returns:
        Normalized route response with steps, polylines, distances

    Raises:
        ValueError: If no API key provided
        RuntimeError: If routing request fails
    """
    if not api_key:
        raise ValueError(
            "Google Directions API requires GOOGLE_MAPS_API_KEY. Set it in .env"
        )
    return get_route_google(origin, destination, api_key=api_key, **kwargs)


def get_route_google(
    origin: LatLon,
    destination: LatLon,
    *,
    api_key: str,
    mode: str = "driving",
    language: str = "en",
    units: str = "metric",
) -> Dict[str, Any]:
    """Get route from Google Directions API.

    Args:
        origin: (lat, lon) start point
        destination: (lat, lon) end point
        api_key: Google Maps API key
        mode: Travel mode (driving, walking, bicycling, transit)
        language: Response language
        units: metric or imperial

    Returns:
        Normalized route response
    """
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": mode,
        "language": language,
        "units": units,
        "key": api_key,
    }

    url = f"{GOOGLE_DIRECTIONS_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except Exception as e:
        raise RuntimeError(f"Google Directions API request failed: {e}")

    if data.get("status") != "OK":
        error_msg = data.get("error_message", data.get("status", "Unknown error"))
        raise RuntimeError(f"Google Directions API error: {error_msg}")

    return _normalize_google_response(data)


def _normalize_google_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Google Directions API response to normalized format.

    Normalizes the response to the canonical NavBuddy route structure.
    """
    if not data.get("routes"):
        raise RuntimeError("No routes found")

    route = data["routes"][0]
    legs = route.get("legs", [])

    # Track bounds for raw data
    min_lat, max_lat = 90.0, -90.0
    min_lon, max_lon = 180.0, -180.0
    has_highway = False
    has_toll = False
    has_ferry = False

    # Build normalized steps
    all_steps = []
    shape_index = 0

    for leg in legs:
        for step_idx, step in enumerate(leg.get("steps", [])):
            # Decode Google's polyline (precision 5)
            polyline_str = step.get("polyline", {}).get("points", "")
            coords = decode_polyline(polyline_str) if polyline_str else []

            # Update bounds
            for lat, lon in coords:
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)

            # Calculate bearings from coordinates
            bearing_after = 0.0
            bearing_before = 0.0
            if len(coords) >= 2:
                bearing_after = bearing_deg(coords[0][0], coords[0][1], coords[1][0], coords[1][1])
                bearing_before = bearing_deg(coords[-2][0], coords[-2][1], coords[-1][0], coords[-1][1])

            start_loc = step.get("start_location", {})
            end_loc = step.get("end_location", {})

            # Strip HTML from instructions
            instruction = _strip_html(step.get("html_instructions", ""))
            maneuver = step.get("maneuver", "")
            maneuver_type = _google_maneuver_to_type(maneuver)

            # Extract street names from instruction
            street_names = _extract_street_names(instruction)

            # Check for highway/toll indicators
            if "highway" in instruction.lower() or "motorway" in instruction.lower():
                has_highway = True
            if "toll" in instruction.lower():
                has_toll = True
            if "ferry" in instruction.lower():
                has_ferry = True

            # Calculate time and length
            distance_m = step.get("distance", {}).get("value", 0)
            duration_s = step.get("duration", {}).get("value", 0)
            length_km = distance_m / 1000.0

            # Shape indices
            begin_shape_index = shape_index
            end_shape_index = shape_index + len(coords) - 1 if coords else shape_index
            shape_index = end_shape_index

            normalized_step = {
                "maneuverIndex": step_idx,
                "distanceMeters": distance_m,
                "staticDuration": f"{duration_s}s",
                "polyline": {
                    "encodedPolyline": polyline_str,
                    "encodedPolyline5": polyline_str,  # Google uses precision 5
                },
                "startLocation": {
                    "latLng": {
                        "latitude": start_loc.get("lat"),
                        "longitude": start_loc.get("lng"),
                    }
                },
                "endLocation": {
                    "latLng": {
                        "latitude": end_loc.get("lat"),
                        "longitude": end_loc.get("lng"),
                    }
                },
                "navigationInstruction": {
                    "type": maneuver_type,
                    "instruction": instruction,
                    "maneuver": maneuver,
                    "verbal_pre_transition_instruction": instruction,
                    "verbal_post_transition_instruction": _make_continue_instruction(distance_m),
                },
                "street_names": street_names,
                "bearing_before": int(bearing_before),
                "bearing_after": int(bearing_after),
                "time": float(duration_s),
                "length": length_km,
                "cost": float(duration_s),  # Use duration as cost proxy
                "begin_shape_index": begin_shape_index,
                "end_shape_index": end_shape_index,
                "travel_mode": "drive",
                "travel_type": "car",
            }

            # Add highway flag if this looks like a highway step
            if _is_highway_step(instruction, maneuver):
                normalized_step["highway"] = True
                has_highway = True

            all_steps.append(normalized_step)

    # Build route-level data
    total_distance = sum(leg.get("distance", {}).get("value", 0) for leg in legs)
    total_duration = sum(leg.get("duration", {}).get("value", 0) for leg in legs)

    # Get overview polyline
    overview_polyline = route.get("overview_polyline", {}).get("points", "")

    return {
        "engine": "google",
        "legs": [{
            "steps": all_steps,
            "distanceMeters": total_distance,
            "duration": f"{total_duration}s",
        }],
        "distanceMeters": total_distance,
        "duration": f"{total_duration}s",
        "durationSeconds": total_duration,
        "cost": float(total_duration),
        "polyline": {
            "encodedPolyline": overview_polyline,
            "encodedPolyline5": overview_polyline,
        },
        "bounds": route.get("bounds", {}),
        "copyrights": route.get("copyrights", ""),
        "raw": {
            "has_time_restrictions": False,
            "has_toll": has_toll,
            "has_highway": has_highway,
            "has_ferry": has_ferry,
            "min_lat": min_lat if min_lat != 90.0 else None,
            "min_lon": min_lon if min_lon != 180.0 else None,
            "max_lat": max_lat if max_lat != -90.0 else None,
            "max_lon": max_lon if max_lon != -180.0 else None,
            "time": float(total_duration),
            "length": total_distance / 1000.0,
            "cost": float(total_duration),
        },
    }


def _google_maneuver_to_type(maneuver: str) -> int:
    """Convert Google maneuver string to numeric type."""
    mapping = {
        "": 8,  # Continue
        "turn-left": 15,
        "turn-right": 10,
        "turn-slight-left": 16,
        "turn-slight-right": 9,
        "turn-sharp-left": 14,
        "turn-sharp-right": 11,
        "uturn-left": 13,
        "uturn-right": 12,
        "straight": 8,
        "ramp-left": 19,
        "ramp-right": 18,
        "merge": 25,
        "fork-left": 24,
        "fork-right": 23,
        "roundabout-left": 26,
        "roundabout-right": 26,
        "keep-left": 24,
        "keep-right": 23,
    }
    return mapping.get(maneuver, 8)


def _extract_street_names(instruction: str) -> List[str]:
    """Extract street names from instruction text."""
    names = []

    # Common patterns: "onto X", "on X"
    for pattern in [" onto ", " on "]:
        if pattern in instruction.lower():
            idx = instruction.lower().find(pattern)
            rest = instruction[idx + len(pattern):].strip()
            # Take until next preposition or end
            for end_pattern in [" toward ", " then ", " and ", ".", ",", "Partial", "partial"]:
                if end_pattern in rest:
                    end_idx = rest.find(end_pattern)
                    rest = rest[:end_idx].strip()
                    break
            if rest:
                # Split on "/" for multi-name roads
                for name in rest.split("/"):
                    name = name.strip()
                    if name and len(name) > 1:
                        names.append(name)
            break

    return names


def _make_continue_instruction(distance_m: float) -> str:
    """Generate a 'Continue for X' instruction."""
    if distance_m < 100:
        return f"Continue for {int(distance_m)} meters."
    elif distance_m < 1000:
        return f"Continue for {int(distance_m / 10) * 10} meters."
    else:
        km = distance_m / 1000.0
        if km < 10:
            return f"Continue for {km:.1f} kilometers."
        else:
            return f"Continue for {int(km)} kilometers."


def _is_highway_step(instruction: str, maneuver: str) -> bool:
    """Check if a step is on a highway/motorway."""
    instruction_lower = instruction.lower()

    # Keywords indicating highway
    highway_keywords = [
        "highway", "motorway", "freeway", "expressway",
        "interstate", "bypass", "m1", "m2", "m3", "m4", "m5",
    ]

    for keyword in highway_keywords:
        if keyword in instruction_lower:
            return True

    # Ramp maneuvers often indicate highway
    if maneuver in ["ramp-left", "ramp-right", "merge"]:
        return True

    return False



def normalize_route_response(route: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure route response has all expected fields.

    Args:
        route: Route response from any engine

    Returns:
        Route with all fields normalized
    """
    # Extract all steps
    all_steps = []
    for leg in route.get("legs", []):
        all_steps.extend(leg.get("steps", []))

    # Ensure each step has required fields
    for step in all_steps:
        # Ensure polyline exists
        if "polyline" not in step:
            step["polyline"] = {"encodedPolyline": "", "encodedPolyline5": ""}

        # Ensure navigation instruction exists
        if "navigationInstruction" not in step:
            step["navigationInstruction"] = {"instruction": "", "type": 8}

        # Ensure locations exist
        if "startLocation" not in step:
            step["startLocation"] = {"latLng": {"latitude": None, "longitude": None}}
        if "endLocation" not in step:
            step["endLocation"] = {"latLng": {"latitude": None, "longitude": None}}

    return route
