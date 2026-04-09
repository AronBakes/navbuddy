"""Load rendering and generation config from config/*.yaml files.

Provides module-level defaults that match config/osm_map.yaml and
config/generation.yaml. Falls back to hardcoded values if the YAML
files are missing (e.g., pip-installed without the config dir).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Config directory: repo root / config /
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML file from the config directory, returning {} on failure."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    try:
        import yaml  # pyyaml — already a dependency via other packages
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ── Load configs ──────────────────────────────────────────────────

_osm = _load_yaml("osm_map.yaml")
_gen = _load_yaml("generation.yaml")

# ── OSM map defaults ─────────────────────────────────────────────

MAP_WIDTH: int = _osm.get("map", {}).get("width", 1280)
MAP_HEIGHT: int = _osm.get("map", {}).get("height", 800)
MAP_ZOOM: int = _osm.get("map", {}).get("zoom", 18)
MAP_TILE_URL: str = _osm.get("map", {}).get(
    "tile_url", "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
)

CAR_ICON: str = _osm.get("car_icon", {}).get("type", "sedan")
CAR_ICON_SCALE: float = _osm.get("car_icon", {}).get("scale", 0.025)

OVERLAY_ENABLED: bool = _osm.get("overlay", {}).get("enabled", True)
OVERLAY_SCALE: float = _osm.get("overlay", {}).get("scale", 1.4)
OVERLAY_DEVICE_SCALE: int = _osm.get("overlay", {}).get("device_scale_factor", 1)

POLYLINE_STEP_COLOR: str = _osm.get("polyline", {}).get("step_color", "#4285F4")
POLYLINE_STEP_WEIGHT: int = _osm.get("polyline", {}).get("step_weight", 6)
POLYLINE_STEP_OPACITY: float = _osm.get("polyline", {}).get("step_opacity", 0.9)
POLYLINE_ROUTE_COLOR: str = _osm.get("polyline", {}).get("route_color", "#999999")
POLYLINE_ROUTE_WEIGHT: int = _osm.get("polyline", {}).get("route_weight", 3)
POLYLINE_ROUTE_OPACITY: float = _osm.get("polyline", {}).get("route_opacity", 0.4)

# ── Generation defaults ──────────────────────────────────────────

FRAME_PROFILE: str = _gen.get("frames", {}).get("profile", "manifest")
FRAME_SPACING: float = _gen.get("frames", {}).get("spacing", 20)
STREETVIEW_SIZE: str = _gen.get("streetview", {}).get("size", "640x400")
STREETVIEW_FOV: int = _gen.get("streetview", {}).get("fov", 90)
ROUTING_ENGINE: str = _gen.get("routing", {}).get("engine", "google")
