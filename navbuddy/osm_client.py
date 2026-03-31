"""OSM client for enriching route data with OpenStreetMap metadata.

Uses Overpass API to query road information (highway type, lanes, speed limits, etc.)
"""

import json
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

LatLon = Tuple[float, float]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Cache to avoid repeated queries for same location
_road_cache: Dict[str, Dict[str, Any]] = {}


def get_road_info(
    lat: float,
    lon: float,
    *,
    radius_m: float = 20.0,
    timeout: int = 10,
) -> Dict[str, Any]:
    """Get OSM road information for a point.

    Args:
        lat: Latitude
        lon: Longitude
        radius_m: Search radius in meters
        timeout: Request timeout in seconds

    Returns:
        Dict with OSM road tags: highway, name, maxspeed, lanes, surface, etc.
    """
    # Check cache first (rounded to 5 decimal places ~1m precision)
    cache_key = f"{lat:.5f},{lon:.5f}"
    if cache_key in _road_cache:
        return _road_cache[cache_key]

    # Overpass QL query to find roads near the point
    query = f"""
    [out:json][timeout:{timeout}];
    way(around:{radius_m},{lat},{lon})["highway"];
    out tags;
    """

    try:
        data = urllib.parse.urlencode({"data": query}).encode()
        req = urllib.request.Request(OVERPASS_URL, data=data)
        req.add_header("User-Agent", "NavBuddy/1.0")

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)

    except Exception as e:
        # Return empty dict on failure (don't block route generation)
        return {"_error": str(e)}

    # Parse response - find the most relevant road
    roads = result.get("elements", [])
    if not roads:
        _road_cache[cache_key] = {}
        return {}

    # Prefer higher-class roads (primary > secondary > tertiary > residential)
    road_priority = {
        "motorway": 10,
        "motorway_link": 9,
        "trunk": 8,
        "trunk_link": 7,
        "primary": 6,
        "primary_link": 5,
        "secondary": 4,
        "secondary_link": 3,
        "tertiary": 2,
        "tertiary_link": 1,
        "residential": 0,
        "unclassified": 0,
    }

    best_road = None
    best_priority = -1

    for road in roads:
        tags = road.get("tags", {})
        highway = tags.get("highway", "")
        priority = road_priority.get(highway, -1)
        if priority > best_priority:
            best_priority = priority
            best_road = tags

    if not best_road:
        best_road = roads[0].get("tags", {})

    # Extract relevant fields
    road_info = {
        "highway": best_road.get("highway"),
        "name": best_road.get("name"),
        "ref": best_road.get("ref"),  # Road reference number (e.g., "M3")
        "maxspeed": best_road.get("maxspeed"),
        "lanes": _parse_lanes(best_road.get("lanes")),
        "surface": best_road.get("surface"),
        "oneway": best_road.get("oneway") == "yes",
        "lit": best_road.get("lit") == "yes",
        "sidewalk": best_road.get("sidewalk"),
        "cycleway": best_road.get("cycleway"),
        "bridge": best_road.get("bridge") == "yes",
        "tunnel": best_road.get("tunnel") == "yes",
        "toll": best_road.get("toll") == "yes",
    }

    # Cache result
    _road_cache[cache_key] = road_info
    return road_info


def _parse_lanes(lanes_str: Optional[str]) -> Optional[int]:
    """Parse lanes string to integer."""
    if not lanes_str:
        return None
    try:
        return int(lanes_str)
    except ValueError:
        return None


def get_road_info_batch(
    points: List[LatLon],
    *,
    radius_m: float = 20.0,
    delay_s: float = 0.1,
) -> List[Dict[str, Any]]:
    """Get OSM road info for multiple points.

    Args:
        points: List of (lat, lon) tuples
        radius_m: Search radius in meters
        delay_s: Delay between requests (to be nice to Overpass)

    Returns:
        List of road info dicts, one per point
    """
    results = []
    for i, (lat, lon) in enumerate(points):
        info = get_road_info(lat, lon, radius_m=radius_m)
        results.append(info)

        # Rate limit (skip delay for cached results)
        cache_key = f"{lat:.5f},{lon:.5f}"
        if cache_key not in _road_cache and i < len(points) - 1:
            time.sleep(delay_s)

    return results


def enrich_step_with_osm(
    step: Dict[str, Any],
    *,
    radius_m: float = 20.0,
) -> Dict[str, Any]:
    """Enrich a route step with OSM road data.

    Args:
        step: Route step dict with startLocation
        radius_m: Search radius in meters

    Returns:
        Step dict with added osm_road field
    """
    # Get coordinates from step
    start_loc = step.get("startLocation", {}).get("latLng", {})
    lat = start_loc.get("latitude")
    lon = start_loc.get("longitude")

    if lat is None or lon is None:
        return step

    # Query OSM
    road_info = get_road_info(lat, lon, radius_m=radius_m)

    # Add to step (don't overwrite existing data)
    if "osm_road" not in step:
        step["osm_road"] = {}

    # Merge OSM data
    for key, value in road_info.items():
        if value is not None and not key.startswith("_"):
            step["osm_road"][key] = value

    return step


def enrich_route_with_osm(
    route: Dict[str, Any],
    *,
    radius_m: float = 20.0,
    delay_s: float = 0.1,
) -> Dict[str, Any]:
    """Enrich a route with OSM data for all steps.

    Args:
        route: Route dict with legs and steps
        radius_m: Search radius in meters
        delay_s: Delay between requests

    Returns:
        Route dict with osm_road added to each step
    """
    for leg in route.get("legs", []):
        for i, step in enumerate(leg.get("steps", [])):
            enrich_step_with_osm(step, radius_m=radius_m)

            # Rate limit between steps
            if i < len(leg.get("steps", [])) - 1:
                time.sleep(delay_s)

    return route


def clear_cache():
    """Clear the road info cache."""
    global _road_cache
    _road_cache = {}


__all__ = [
    "get_road_info",
    "get_road_info_batch",
    "enrich_step_with_osm",
    "enrich_route_with_osm",
    "clear_cache",
]
