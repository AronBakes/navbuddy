"""OSM-based overhead map rendering using Playwright + Leaflet.

Generates bird's-eye view maps with OpenStreetMap tiles:
- Route polyline overlay (full route, dimmed)
- Step polyline overlay (current step, highlighted)
- Car icon at current position (rotated to heading)
- Start marker (green) and end marker (red)
- Optional navigation overlays (header + ETA card)

No Google API required - uses free OSM tiles.
"""

import asyncio
import base64
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Playwright is optional
try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


# Available car icon presets
CAR_ICONS = {
    "arrow": "arrow",  # Default SVG arrow
    "sedan": "navbuddy-sedan.png",  # Generic yellow cab — recommended for paper figures
    "cybertruck": "navbuddy-cybertruck.png",
    "f1": "navbuddy-f1.png",
    "model3": "navbuddy-model3.png",
    "wrx": "navbuddy-wrx.png",
}

# Car icon assets point east (90°) by default, so we subtract 90° to align with north
CAR_ICON_HEADING_OFFSET = -90.0

# Default scale factor for car icons (1.0 = original size)
DEFAULT_CAR_ICON_SCALE = 0.025


from navbuddy.polylines import haversine_m, bearing_deg  # canonical implementations


def _polyline_length_m(coords: List[Tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    return sum(haversine_m(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1))


def _interpolate_car_pose_from_end(
    coords: List[Tuple[float, float]],
    distance_from_end_m: float,
) -> Optional[Tuple[float, float, float]]:
    """Return (lat, lng, heading) interpolated along polyline at given distance from end."""
    if len(coords) < 2:
        return None
    seg_lens = [haversine_m(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1)]
    total = sum(seg_lens)
    if total <= 0:
        return None
    target = max(0.0, total - float(distance_from_end_m))
    cum = 0.0
    for i, seg in enumerate(seg_lens):
        if cum + seg >= target:
            t = 0.0 if seg <= 0 else (target - cum) / seg
            lat1, lng1 = coords[i]
            lat2, lng2 = coords[i + 1]
            hdg = bearing_deg(lat1, lng1, lat2, lng2)
            return lat1 + t * (lat2 - lat1), lng1 + t * (lng2 - lng1), hdg if hdg is not None else 0.0
        cum += seg
    lat1, lng1 = coords[-2]
    lat2, lng2 = coords[-1]
    hdg = bearing_deg(lat1, lng1, lat2, lng2)
    return lat2, lng2, hdg if hdg is not None else 0.0


def choose_car_pose(
    step_coords: List[Tuple[float, float]],
    start_lat: Optional[float] = None,
    start_lng: Optional[float] = None,
    end_lat: Optional[float] = None,
    end_lng: Optional[float] = None,
    fallback_heading: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Place car near the maneuver point: 20 m from step end (midpoint if step < 40 m).

    Returns:
        (car_lat, car_lng, heading_degrees)
    """
    if step_coords:
        total_m = _polyline_length_m(step_coords)
        if total_m > 0:
            dist_from_end = (total_m / 2.0) if total_m < 40.0 else 20.0
            pose = _interpolate_car_pose_from_end(step_coords, dist_from_end)
            if pose is not None:
                return pose

    # Fallback: use start position + inferred heading
    fb_lat = start_lat if start_lat is not None else (step_coords[0][0] if step_coords else 0.0)
    fb_lng = start_lng if start_lng is not None else (step_coords[0][1] if step_coords else 0.0)
    hdg: Optional[float] = None
    if len(step_coords) >= 2:
        for i in range(1, len(step_coords)):
            if haversine_m(*step_coords[i - 1], *step_coords[i]) > 0.05:
                hdg = bearing_deg(*step_coords[i - 1], *step_coords[i])
                break
    if hdg is None and start_lat is not None and end_lat is not None:
        hdg = bearing_deg(start_lat, start_lng, end_lat, end_lng)  # type: ignore[arg-type]
    if hdg is None and isinstance(fallback_heading, (int, float)):
        hdg = float(fallback_heading)
    return fb_lat, fb_lng, hdg if hdg is not None else 0.0


def _get_arrow_svg(size: int = 32) -> str:
    """Get arrow SVG for car icon."""
    return f'''<svg viewBox="0 0 24 24" width="{size}" height="{size}">
  <path fill="#1a73e8" stroke="#fff" stroke-width="1.5" d="M12 2L4 14h5v8h6v-8h5z"/>
</svg>'''


def _encode_image_to_data_uri(image_path: Path) -> str:
    """Convert image file to data URI for embedding in HTML."""
    if not image_path.exists():
        return ""
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions without loading full image."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (100, 100)  # fallback


def _generate_leaflet_html(
    step_polyline_coords: List[Tuple[float, float]],
    car_lat: float,
    car_lng: float,
    heading: float,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    *,
    route_polyline_coords: Optional[List[Tuple[float, float]]] = None,
    car_icon: str = "arrow",
    car_icon_data_uri: Optional[str] = None,
    car_icon_width: int = 48,
    car_icon_height: int = 48,
    zoom: int = 18,
    width: int = 640,
    height: int = 400,
    nav_instruction: Optional[str] = None,
    next_instruction: Optional[str] = None,
    remaining_distance_m: Optional[float] = None,
    eta_minutes: Optional[int] = None,
    show_nav_overlay: bool = True,
) -> str:
    """Generate HTML with Leaflet map for screenshot.

    Args:
        step_polyline_coords: Current step polyline (highlighted on top)
        car_lat, car_lng: Car position
        heading: Car heading in degrees
        start_lat, start_lng: Step start position
        end_lat, end_lng: Step end position
        route_polyline_coords: Full route polyline (dimmed underneath)
        car_icon: Icon type - "arrow" or custom image name
        car_icon_data_uri: Base64 data URI for custom car icon
        car_icon_width: Width of car icon in pixels
        car_icon_height: Height of car icon in pixels
        zoom: Map zoom level
        width, height: Image dimensions
    """

    # Convert polylines to Leaflet format
    step_path_js = json.dumps([[lat, lng] for lat, lng in step_polyline_coords])
    route_path_js = json.dumps([[lat, lng] for lat, lng in (route_polyline_coords or [])])

    # Generate car icon HTML
    if car_icon_data_uri:
        # Custom image icon - apply heading offset since assets point east
        adjusted_heading = heading + CAR_ICON_HEADING_OFFSET
        car_icon_html = f'''<img class="car-icon" src="{car_icon_data_uri}" width="{car_icon_width}" height="{car_icon_height}" style="transform: rotate({adjusted_heading}deg);">'''
        icon_anchor_x = car_icon_width // 2
        icon_anchor_y = car_icon_height // 2
    else:
        # Default arrow SVG (points north, no offset needed)
        car_icon_html = f'''<svg class="car-icon" viewBox="0 0 24 24" width="{car_icon_width}" height="{car_icon_height}" style="transform: rotate({heading}deg);">
  <path fill="#1a73e8" stroke="#fff" stroke-width="1.5" d="M12 2L4 14h5v8h6v-8h5z"/>
</svg>'''
        icon_anchor_x = car_icon_width // 2
        icon_anchor_y = car_icon_height // 2

    # Build nav overlay HTML
    nav_overlay_html = ""
    if show_nav_overlay and nav_instruction:
        # Format remaining distance
        if remaining_distance_m is not None:
            dist_km = remaining_distance_m / 1000
            dist_str = f"{dist_km:.1f} km" if dist_km >= 1.0 else f"{int(remaining_distance_m)} m"
        else:
            dist_str = ""

        # ETA: use provided or estimate at 30 km/h urban average
        if eta_minutes is None and remaining_distance_m is not None:
            eta_minutes = max(1, int(remaining_distance_m / 1000 / 30 * 60))

        eta_str = f"{eta_minutes}m" if eta_minutes is not None else ""

        # Truncate instruction to fit header
        instr_display = nav_instruction[:52] + "…" if len(nav_instruction) > 52 else nav_instruction
        next_instr_display = ""
        if next_instruction:
            next_instr_display = next_instruction[:52] + "…" if len(next_instruction) > 52 else next_instruction

        next_row_html = f'''
        <div class="nav-row">
          <div class="nav-arrow">↑</div>
          <div class="nav-text">{next_instr_display}</div>
        </div>''' if next_instr_display else ""

        eta_html = f'''
    <div class="eta-card">
      <div class="eta-time">{eta_str}</div>
      <div class="eta-label">ETA</div>
      <div class="eta-dist">{dist_str}</div>
    </div>''' if (eta_str or dist_str) else ""

        nav_overlay_html = f'''
    <div class="nav-header">
        <div class="nav-row">
          <div class="nav-arrow">↑</div>
          <div class="nav-dist">{dist_str}</div>
        </div>
        <div class="nav-row nav-main">
          <div class="nav-text">{instr_display}</div>
        </div>{next_row_html}
    </div>{eta_html}'''

    nav_overlay_css = """
        /* Navigation header overlay */
        .nav-header {
            position: absolute;
            top: 0; left: 0; right: 0;
            background: #1b6e38;
            color: white;
            padding: 8px 12px 6px;
            z-index: 1000;
            font-family: -apple-system, Roboto, Arial, sans-serif;
            box-shadow: 0 2px 6px rgba(0,0,0,0.35);
        }
        .nav-row {
            display: flex;
            align-items: center;
            gap: 8px;
            line-height: 1.25;
        }
        .nav-arrow {
            font-size: 15px;
            font-weight: bold;
            min-width: 16px;
        }
        .nav-dist {
            font-size: 15px;
            font-weight: 700;
            white-space: nowrap;
        }
        .nav-main .nav-text {
            font-size: 12px;
            font-weight: 400;
            opacity: 0.92;
        }
        .nav-row:not(.nav-main) .nav-text {
            font-size: 11px;
            opacity: 0.78;
        }
        /* ETA card */
        .eta-card {
            position: absolute;
            bottom: 0; left: 0;
            background: white;
            border-radius: 0 6px 0 0;
            padding: 5px 10px;
            z-index: 1000;
            font-family: -apple-system, Roboto, Arial, sans-serif;
            box-shadow: 0 -1px 4px rgba(0,0,0,0.2);
            display: flex;
            align-items: baseline;
            gap: 6px;
        }
        .eta-time {
            font-size: 16px;
            font-weight: 700;
            color: #1a1a1a;
        }
        .eta-label {
            font-size: 10px;
            color: #666;
            text-transform: uppercase;
        }
        .eta-dist {
            font-size: 12px;
            color: #444;
            margin-left: 4px;
        }
    """ if show_nav_overlay and nav_instruction else ""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NavBuddy Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; }}
        html, body {{ width: 100%; height: 100%; position: relative; }}
        #map {{ width: {width}px; height: {height}px; }}

        /* Car icon marker */
        .car-icon {{
            transform-origin: center center;
        }}

        /* Hide Leaflet attribution for cleaner screenshots */
        .leaflet-control-attribution {{ display: none !important; }}
        {nav_overlay_css}
    </style>
</head>
<body>
    <div id="map"></div>
    {nav_overlay_html}
    <script>
        // Initialize map centered on car position
        const map = L.map('map', {{
            center: [{car_lat}, {car_lng}],
            zoom: {zoom},
            zoomControl: false,
            attributionControl: false
        }});

        // Add OSM tiles
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
        }}).addTo(map);

        // Full route polyline (underneath, dimmed)
        const routePath = {route_path_js};
        if (routePath.length > 0) {{
            L.polyline(routePath, {{
                color: '#8FAFD6',
                weight: 5,
                opacity: 0.58
            }}).addTo(map);
        }}

        // Current step polyline (on top, highlighted)
        const stepPath = {step_path_js};
        if (stepPath.length > 0) {{
            L.polyline(stepPath, {{
                color: '#4285F4',
                weight: 7,
                opacity: 0.95
            }}).addTo(map);
        }}

        // Start marker (green)
        L.circleMarker([{start_lat}, {start_lng}], {{
            radius: 10,
            fillColor: '#34A853',
            color: '#fff',
            weight: 3,
            opacity: 1,
            fillOpacity: 1
        }}).addTo(map);

        // End marker (red)
        L.circleMarker([{end_lat}, {end_lng}], {{
            radius: 10,
            fillColor: '#EA4335',
            color: '#fff',
            weight: 3,
            opacity: 1,
            fillOpacity: 1
        }}).addTo(map);

        // Car icon (arrow or custom image)
        const carIcon = L.divIcon({{
            className: 'car-marker',
            html: `{car_icon_html}`,
            iconSize: [{car_icon_width}, {car_icon_height}],
            iconAnchor: [{icon_anchor_x}, {icon_anchor_y}]
        }});
        L.marker([{car_lat}, {car_lng}], {{ icon: carIcon }}).addTo(map);

        // Signal ready after tiles load
        map.whenReady(function() {{
            setTimeout(function() {{
                window.__mapReady = true;
            }}, 1000);  // Wait for tiles to render
        }});
    </script>
</body>
</html>"""


async def render_map_async(
    step_polyline_coords: List[Tuple[float, float]],
    car_lat: float,
    car_lng: float,
    heading: float,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    output_path: Path,
    *,
    route_polyline_coords: Optional[List[Tuple[float, float]]] = None,
    car_icon: str = "arrow",
    assets_dir: Optional[Path] = None,
    car_icon_scale: float = DEFAULT_CAR_ICON_SCALE,
    zoom: int = 18,
    width: int = 640,
    height: int = 400,
    nav_instruction: Optional[str] = None,
    next_instruction: Optional[str] = None,
    remaining_distance_m: Optional[float] = None,
    eta_minutes: Optional[int] = None,
    show_nav_overlay: bool = True,
) -> bool:
    """Render OSM map to image using Playwright.

    Args:
        step_polyline_coords: List of (lat, lng) tuples for current step
        car_lat, car_lng: Car position
        heading: Car heading in degrees
        start_lat, start_lng: Step start position
        end_lat, end_lng: Step end position
        output_path: Where to save the PNG
        route_polyline_coords: Optional full route polyline (dimmed underneath)
        car_icon: "arrow" (default) or one of: "cybertruck", "f1", "model3", "wrx"
        assets_dir: Directory containing car icon images
        car_icon_scale: Scale factor for car icon (default 0.4, maintains aspect ratio)
        zoom: Map zoom level
        width, height: Image dimensions

    Returns:
        True if successful
    """
    if not HAS_PLAYWRIGHT:
        if not getattr(render_map_async, "_warned", False):
            print("Playwright not installed. Run: pip install navbuddy[render] && playwright install chromium")
            render_map_async._warned = True
        return False

    # Load car icon if custom
    car_icon_data_uri = None
    car_icon_width = 48  # default for arrow
    car_icon_height = 48

    if car_icon != "arrow" and assets_dir:
        icon_filename = CAR_ICONS.get(car_icon)
        if icon_filename and icon_filename != "arrow":
            icon_path = assets_dir / icon_filename
            if icon_path.exists():
                car_icon_data_uri = _encode_image_to_data_uri(icon_path)
                # Get original dimensions and apply scale
                orig_width, orig_height = _get_image_dimensions(icon_path)
                car_icon_width = int(orig_width * car_icon_scale)
                car_icon_height = int(orig_height * car_icon_scale)

    # Generate HTML
    html = _generate_leaflet_html(
        step_polyline_coords, car_lat, car_lng, heading,
        start_lat, start_lng, end_lat, end_lng,
        route_polyline_coords=route_polyline_coords,
        car_icon=car_icon,
        car_icon_data_uri=car_icon_data_uri,
        car_icon_width=car_icon_width,
        car_icon_height=car_icon_height,
        zoom=zoom, width=width, height=height,
        nav_instruction=nav_instruction,
        next_instruction=next_instruction,
        remaining_distance_m=remaining_distance_m,
        eta_minutes=eta_minutes,
        show_nav_overlay=show_nav_overlay,
    )

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html)
        html_path = f.name

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )

            context = await browser.new_context(
                viewport={'width': width, 'height': height},
                device_scale_factor=1,  # Exact pixel output (avoids HiDPI 4× scaling)
                extra_http_headers={
                    'Referer': 'https://www.openstreetmap.org/',
                    'User-Agent': 'Mozilla/5.0 NavBuddy/1.0 (research; openstreetmap.org/copyright)',
                }
            )
            page = await context.new_page()

            # Load the HTML
            await page.goto(f'file://{html_path}', wait_until='networkidle')

            # Wait for map to be ready
            try:
                await page.wait_for_function('window.__mapReady === true', timeout=10000)
            except Exception:
                # Fallback: just wait a bit
                await page.wait_for_timeout(3000)

            # Extra wait for tile loading
            await page.wait_for_timeout(1500)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Screenshot
            await page.screenshot(path=str(output_path))

            await context.close()
            await browser.close()

        return True

    except Exception as e:
        print(f"Map render failed: {e}")
        return False
    finally:
        # Clean up temp file
        Path(html_path).unlink(missing_ok=True)


def render_map(
    step_polyline_coords: List[Tuple[float, float]],
    car_lat: float,
    car_lng: float,
    heading: float,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    output_path: Path,
    *,
    route_polyline_coords: Optional[List[Tuple[float, float]]] = None,
    car_icon: str = "arrow",
    assets_dir: Optional[Path] = None,
    car_icon_scale: float = DEFAULT_CAR_ICON_SCALE,
    **kwargs,
) -> bool:
    """Synchronous wrapper for render_map_async."""
    return asyncio.run(render_map_async(
        step_polyline_coords, car_lat, car_lng, heading,
        start_lat, start_lng, end_lat, end_lng,
        output_path,
        route_polyline_coords=route_polyline_coords,
        car_icon=car_icon,
        assets_dir=assets_dir,
        car_icon_scale=car_icon_scale,
        **kwargs
    ))


def generate_step_map_osm(
    step: Dict[str, Any],
    output_path: Path,
    *,
    car_lat: Optional[float] = None,
    car_lng: Optional[float] = None,
    car_heading: Optional[float] = None,
    route_polyline: Optional[str] = None,
    car_icon: str = "arrow",
    assets_dir: Optional[Path] = None,
    car_icon_scale: float = DEFAULT_CAR_ICON_SCALE,
    zoom: int = 18,
    width: int = 640,
    height: int = 400,
) -> bool:
    """Generate OSM overhead map for a route step.

    Args:
        step: Route step dict with startLocation, endLocation, polyline
        output_path: Where to save the map image
        car_lat, car_lng: Override car position (default: step start)
        car_heading: Override car heading (default: step bearing_after)
        route_polyline: Encoded polyline for full route (dimmed underneath)
        car_icon: "arrow" (default) or one of: "cybertruck", "f1", "model3", "wrx"
        assets_dir: Directory containing car icon images
        car_icon_scale: Scale factor for car icon (default 0.4, maintains aspect ratio)
        zoom: Map zoom level
        width, height: Image dimensions

    Returns:
        True if successful
    """
    # Import here to avoid circular imports
    from navbuddy.polylines import decode_polyline

    # Extract locations
    start_loc = step.get("startLocation", {}).get("latLng", {})
    end_loc = step.get("endLocation", {}).get("latLng", {})

    start_lat = start_loc.get("latitude")
    start_lng = start_loc.get("longitude")
    end_lat = end_loc.get("latitude")
    end_lng = end_loc.get("longitude")

    if start_lat is None or start_lng is None:
        return False

    # Use provided car position or default to start
    lat = car_lat if car_lat is not None else start_lat
    lng = car_lng if car_lng is not None else start_lng
    heading = car_heading if car_heading is not None else step.get("bearing_after", 0)

    # Decode step polyline
    step_polyline_str = step.get("polyline", {}).get("encodedPolyline5", "")
    if not step_polyline_str:
        step_polyline_str = step.get("polyline", {}).get("encodedPolyline", "")

    step_coords = decode_polyline(step_polyline_str) if step_polyline_str else []

    # Decode route polyline if provided
    route_coords = None
    if route_polyline:
        route_coords = decode_polyline(route_polyline)

    # Render map
    return render_map(
        step_coords, lat, lng, heading,
        start_lat, start_lng,
        end_lat or start_lat, end_lng or start_lng,
        output_path,
        route_polyline_coords=route_coords,
        car_icon=car_icon,
        assets_dir=assets_dir,
        car_icon_scale=car_icon_scale,
        zoom=zoom, width=width, height=height,
    )


def get_available_car_icons() -> List[str]:
    """Return list of available car icon names."""
    return list(CAR_ICONS.keys())


__all__ = [
    "render_map",
    "render_map_async",
    "generate_step_map_osm",
    "get_available_car_icons",
    "CAR_ICONS",
    "CAR_ICON_HEADING_OFFSET",
    "DEFAULT_CAR_ICON_SCALE",
    "HAS_PLAYWRIGHT",
]
