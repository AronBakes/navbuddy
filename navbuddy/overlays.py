"""Navigation overlay helpers for map images.

Generates Google Maps-style navigation overlays:
- Top navigation card (distance + instruction + next turn)
- Bottom ETA card (remaining time + distance)

Can render overlays using:
- PIL (simple, fast) - overlay_nav_eta_pil()
- Playwright (styled HTML, high quality) - overlay_nav_eta_html()
"""

from __future__ import annotations

import base64
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


# Check for Playwright
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


# Maneuver icon mapping for Material Symbols
MANEUVER_ICONS = {
    "STRAIGHT": "straight",
    "CONTINUE": "straight",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "TURN_SLIGHT_LEFT": "turn_slight_left",
    "TURN_SLIGHT_RIGHT": "turn_slight_right",
    "TURN_SHARP_LEFT": "turn_sharp_left",
    "TURN_SHARP_RIGHT": "turn_sharp_right",
    "UTURN_LEFT": "u_turn_left",
    "UTURN_RIGHT": "u_turn_right",
    "KEEP_LEFT": "keep_left",
    "KEEP_RIGHT": "keep_right",
    "RAMP_LEFT": "ramp_left",
    "RAMP_RIGHT": "ramp_right",
    "MERGE": "merge",
    "FORK_LEFT": "fork_left",
    "FORK_RIGHT": "fork_right",
    "ROUNDABOUT_LEFT": "roundabout_left",
    "ROUNDABOUT_RIGHT": "roundabout_right",
    "DEPART": "navigation",
    "START": "navigation",
    "ARRIVE": "flag",
    "DESTINATION": "flag",
    "FERRY": "directions_boat",
    "FERRY_TRAIN": "train",
    "NAME_CHANGE": "signpost",
    "BECOMES": "signpost",
    "MANEUVER_UNSPECIFIED": "navigation",
    "NONE": "navigation",
}


def _load_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font, falling back to default if not available."""
    for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_metres(val: Any, step: Optional[Dict] = None) -> Optional[float]:
    """Convert various distance formats to metres."""
    if val is None and step:
        try:
            d = float(step.get("distance_m") or step.get("distanceMeters") or 0)
            c = float(step.get("current_distance_m", 0))
            return max(0.0, d - c)
        except Exception:
            return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().lower().replace("meters", "m").replace("metres", "m")
        km_match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*km\s*$", s)
        m_match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*m\s*$", s)
        if km_match:
            return float(km_match.group(1)) * 1000.0
        if m_match:
            return float(m_match.group(1))
        try:
            return float(s)
        except Exception:
            return None
    return None


def format_distance(val: Any) -> str:
    """Format distance value to human-readable string."""
    metres = _to_metres(val)
    if metres is None or not isinstance(metres, (int, float)) or math.isinf(metres) or math.isnan(metres):
        return "--"
    if metres < 1000:
        return f"{int(round(metres))} m"
    km = float(metres) / 1000.0
    if km > 1e9:
        return ">>> km"
    return f"{round(km, 1) if km < 10 else int(round(km))} km"


def format_eta_remaining(minutes_remaining: Optional[float]) -> str:
    """Format remaining time as ETA duration text.

    Examples:
      10.2 -> "10m"
      60.0 -> "1h"
      89.8 -> "1h 30m"
    """
    if minutes_remaining is None:
        return "--"
    try:
        total_minutes = max(0, int(round(float(minutes_remaining))))
    except Exception:
        return "--"
    if total_minutes < 60:
        return f"{total_minutes}m"
    hours, mins = divmod(total_minutes, 60)
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


def _extract_step_nav(step: Dict) -> Tuple[str, str, Optional[float]]:
    """Extract maneuver, instruction, and remaining distance from step."""
    nav = (step or {}).get("navigationInstruction", {}) or {}
    instruction = (
        nav.get("instruction")
        or nav.get("instructions")
        or step.get("instruction")
        or step.get("effective_instruction")
        or ""
    )
    maneuver = nav.get("maneuver") or step.get("maneuver") or "MANEUVER_UNSPECIFIED"
    remaining = (
        nav.get("remaining_distance_m")
        or step.get("remaining_distance_m")
        or step.get("distance_m")
        or step.get("distanceMeters")
    )
    metres = _to_metres(remaining, step)
    return maneuver, instruction, metres


def _get_icon_name(maneuver: str) -> str:
    """Get Material Symbols icon name for maneuver."""
    return MANEUVER_ICONS.get((maneuver or "").upper(), "navigation")


def _vstack(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """Stack two images vertically."""
    w = max(top.width, bottom.width)
    h = top.height + bottom.height
    out = Image.new("RGBA", (w, h))
    out.paste(top, (0, 0))
    out.paste(bottom, (0, top.height))
    return out


def overlay_nav_eta_pil(
    image_path: Path,
    step: Dict[str, Any],
    next_step: Optional[Dict[str, Any]] = None,
    *,
    arrival_time: str = "--:--",
    minutes_remaining: Optional[float] = None,
    distance_km: Optional[float] = None,
    out_path: Optional[Path] = None,
    pad: int = 16,
) -> Path:
    """Compose nav (top-left) and ETA (bottom-left) cards onto image using PIL.

    Args:
        image_path: Path to base image
        step: Current step dict with navigationInstruction
        next_step: Optional next step for secondary instruction
        arrival_time: ETA arrival time string
        minutes_remaining: Minutes until arrival
        distance_km: Distance remaining in km
        out_path: Output path (default: overwrite input)
        pad: Padding in pixels

    Returns:
        Path to output image
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except (OSError, SyntaxError) as e:
        print(f"[warn] Skipping overlay for corrupted image {image_path}: {e}")
        return Path(out_path or image_path)

    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Colors
    green_primary = (11, 128, 67, 230)
    green_secondary = (8, 92, 45, 230)
    white = (255, 255, 255, 240)
    text_light = (255, 255, 255, 255)
    text_dark = (20, 20, 20, 255)
    green_text = (8, 128, 67, 255)

    # Fonts
    font_large = _load_font(24)
    font_med = _load_font(18)
    font_small = _load_font(14)

    # Extract step info
    maneuver, instr, dist_m = _extract_step_nav(step)
    dist_str = format_distance(dist_m)

    # Nav card dimensions
    nav_width = int(w * 0.9)
    nav_height = int(h * 0.18)
    nav_box = Image.new("RGBA", (nav_width, nav_height), green_primary)
    nav_draw = ImageDraw.Draw(nav_box)

    y = pad
    if dist_str:
        nav_draw.text((pad, y), dist_str, fill=text_light, font=font_large)
        try:
            y += font_large.getbbox("Hg")[3] + 4
        except Exception:
            y += 28
    nav_draw.text((pad, y), instr[:180], fill=text_light, font=font_med)

    # Secondary line (next step)
    if next_step:
        _, ntext, _ = _extract_step_nav(next_step)
        if ntext:
            sec_h = int(nav_height * 0.45)
            sec_box = Image.new("RGBA", (nav_width, sec_h), green_secondary)
            sec_draw = ImageDraw.Draw(sec_box)
            sec_draw.text((pad, int(sec_h * 0.3)), ntext[:160], fill=text_light, font=font_small)
            nav_box = _vstack(nav_box, sec_box)

    # ETA card
    eta_width = int(w * 0.56)
    eta_height = int(h * 0.12)
    eta_box = Image.new("RGBA", (eta_width, eta_height), white)
    eta_draw = ImageDraw.Draw(eta_box)

    # ETA remaining + distance columns
    col_width = eta_width // 2
    eta_str = format_eta_remaining(minutes_remaining)
    eta_draw.text((pad, pad), "ETA", fill=text_dark, font=font_small)
    try:
        eta_draw.text((pad, pad * 2 + font_small.getbbox("Hg")[3]), eta_str, fill=green_text, font=font_med)
    except Exception:
        eta_draw.text((pad, pad * 2 + 16), eta_str, fill=green_text, font=font_med)

    if distance_km is not None:
        try:
            dist_eta = f"{float(distance_km):.1f} km"
            eta_draw.text((col_width + pad, pad), dist_eta, fill=text_dark, font=font_med)
            eta_draw.text((col_width + pad, pad * 2 + 16), "distance", fill=text_dark, font=font_small)
        except Exception:
            pass

    # Compose on base image
    img.paste(nav_box, (pad, pad), nav_box)
    img.paste(eta_box, (pad, h - eta_height - pad), eta_box)

    output = Path(out_path or image_path)
    img.convert("RGB").save(output, format="PNG")
    return output


def _render_nav_sign_html(
    step: Dict,
    next_step: Optional[Dict] = None,
    subtitle: Optional[str] = None,
    scale: float = 1.0,
    max_width_px: Optional[float] = None,
) -> str:
    """Generate HTML for navigation sign card."""
    maneuver, instr, metres = _extract_step_nav(step)
    dist_str = format_distance(metres)
    icon_name = _get_icon_name(maneuver)

    # Truncate instruction to prevent overflow
    max_instr_len = 60
    if len(instr) > max_instr_len:
        instr = instr[:max_instr_len - 3] + "..."

    # Constrained dimensions - fit within typical 800px map with padding
    # max_width_px is passed as (image_width - 2*padding), so use it directly
    # Apply scale factor to base max width (scale applied once only)
    base_max = 380
    sign_max_width = min(max_width_px or int(base_max * scale), int(base_max * scale))

    # Apply scale to all dimensions
    sign_min_height = int(68 * scale)
    sign_padding_vert = int(10 * scale)
    sign_padding_horiz = int(12 * scale)
    sign_border_radius = int(12 * scale)
    sign_margin_bottom = int(5 * scale)

    secondary_max_width = sign_max_width
    secondary_padding_vert = int(8 * scale)
    secondary_padding_horiz = int(12 * scale)
    secondary_min_height = int(32 * scale)
    secondary_gap = int(8 * scale)

    row_gap = int(10 * scale)
    ico_width = int(40 * scale)
    ico_height = int(40 * scale)
    ico_border_radius = 999
    ico_font_size = int(24 * scale)

    dist_font_size = int(22 * scale)
    title_font_size = int(15 * scale)
    sub_font_size = int(12 * scale)
    next_font_size = int(12 * scale)
    next_ico_font_size = int(18 * scale)
    next_ico_width = int(22 * scale)
    next_ico_height = int(22 * scale)

    parts = [
        '<div class="nb-wrap">',
        f'<div class="nb-sign" style="max-width:{sign_max_width}px; min-height:{sign_min_height}px; padding:{sign_padding_vert}px {sign_padding_horiz}px; border-radius:{sign_border_radius}px; margin-bottom:{sign_margin_bottom}px; box-sizing:border-box;">',
        f'  <div class="nb-row" style="grid-template-columns:{ico_width}px 1fr; column-gap:{row_gap}px;">',
        f'    <span class="nb-ico" style="width:{ico_width}px; height:{ico_height}px; border-radius:{ico_border_radius}px;"><span class="material-symbols-outlined" style="font-size:{ico_font_size}px;">{icon_name}</span></span>',
        '    <div style="overflow:hidden;">',
        f'      <div class="nb-dist" style="font-size:{dist_font_size}px; margin-bottom:2px;">{dist_str}</div>' if dist_str else "",
        f'      <div class="nb-title" style="font-size:{title_font_size}px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{instr}</div>',
        f'      <div class="nb-sub" style="font-size:{sub_font_size}px;">{subtitle}</div>' if subtitle else "",
        "    </div>",
        "  </div>",
        "</div>",
    ]

    if next_step is not None:
        nm, ntxt, _ = _extract_step_nav(next_step)
        if ntxt:
            # Truncate next instruction too
            if len(ntxt) > 50:
                ntxt = ntxt[:47] + "..."
            next_icon = _get_icon_name(nm)
            parts += [
                f'<div class="nb-sign secondary" style="max-width:{secondary_max_width}px; background:#085C2D; display:flex; align-items:center; gap:{secondary_gap}px; padding:{secondary_padding_vert}px {secondary_padding_horiz}px; min-height:{secondary_min_height}px; box-sizing:border-box;">',
                f'  <span class="nb-next-ico" style="width:{next_ico_width}px; height:{next_ico_height}px;"><span class="material-symbols-outlined" style="font-size:{next_ico_font_size}px;">{next_icon}</span></span>',
                f'  <span class="nb-next" style="font-size:{next_font_size}px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{ntxt}</span>',
                "</div>",
            ]
    parts.append("</div>")
    return "\n".join(parts)


def _render_eta_card_html(
    arrival_time: str,
    minutes_remaining: Optional[float],
    distance_remaining_km: Optional[float],
    scale: float = 1.0,
) -> str:
    """Generate HTML for ETA card."""
    dstr = format_distance((distance_remaining_km or 0) * 1000)
    # Extract number and unit separately so the label matches the actual unit
    if " " in dstr:
        d_num, d_unit = dstr.split(" ", 1)
    else:
        d_num, d_unit = dstr, ""
    eta_str = format_eta_remaining(minutes_remaining)

    # Compact ETA card dimensions - apply scale factor
    eta_border_radius = int(10 * scale)
    eta_padding_vert = int(8 * scale)
    eta_padding_horiz = int(12 * scale)
    eta_col_padding = int(10 * scale)
    eta_big_font_size = int(18 * scale)
    eta_sub_font_size = int(11 * scale)
    eta_min_width = int(190 * scale)
    eta_divider_width = int(max(1, scale))

    return f"""
    <div class="nb-eta" style="border-radius:{eta_border_radius}px; padding:{eta_padding_vert}px {eta_padding_horiz}px; display:inline-grid; grid-template-columns:auto auto; min-width:{eta_min_width}px;">
      <div class="nb-eta-col" style="padding:0 {eta_col_padding}px;">
        <div class="nb-eta-big nb-eta-green" style="font-size:{eta_big_font_size}px;">{eta_str}</div>
        <div class="nb-eta-sub" style="font-size:{eta_sub_font_size}px;">ETA</div>
      </div>
      <div class="nb-eta-col" style="padding:0 {eta_col_padding}px; border-left:{eta_divider_width}px solid rgba(0,0,0,.08);">
        <div class="nb-eta-big" style="font-size:{eta_big_font_size}px;">{d_num}</div>
        <div class="nb-eta-sub" style="font-size:{eta_sub_font_size}px;">{d_unit}</div>
      </div>
    </div>
    """


# CSS styles for HTML overlays
_OVERLAY_CSS = """
  <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
  <style>
    .material-symbols-outlined{
      font-family:'Material Symbols Outlined';
      font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 48;
      -webkit-font-feature-settings:'liga'; font-feature-settings:'liga';
    }
    .nb-wrap{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; width:100%;}
    .nb-sign{width:100%;
             color:#fff;background:#0B8043;border-radius:12px;box-shadow:0 4px 10px rgba(0,0,0,.25);margin-bottom:6px}
    .nb-sign.secondary{background:#085C2D;display:flex;align-items:center;gap:8px;width:100%;overflow:hidden}
    .nb-row{display:grid;grid-template-columns:44px 1fr;column-gap:10px;align-items:center}
    .nb-ico{display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:999px;background:rgba(255,255,255,.12)}
    .nb-ico>.material-symbols-outlined{font-size:28px;line-height:1}
    .nb-dist{font-weight:700;font-size:22px;margin-bottom:2px}
    .nb-title{font-weight:600;font-size:16px;white-space:normal;overflow-wrap:break-word;overflow:hidden;text-overflow:ellipsis}
    .nb-sub{opacity:.9;font-size:13px;white-space:normal;overflow:hidden;text-overflow:ellipsis}
    .nb-next{font-size:14px;font-weight:500;white-space:normal;overflow:hidden;text-overflow:ellipsis}
    .nb-next-ico{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px}

    .nb-eta{display:grid;grid-template-columns:1fr 1fr;background:#fff;color:#000;border-radius:14px;
            padding:10px 14px;box-shadow:0 4px 10px rgba(0,0,0,.18);width:max-content;
            font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
    .nb-eta-col{text-align:center;padding:0 12px}
    .nb-eta-col+.nb-eta-col{border-left:1px solid rgba(0,0,0,.08)}
    .nb-eta-big{font-weight:800;font-size:20px;line-height:1.1}
    .nb-eta-sub{font-size:12px;line-height:1.1;opacity:.75}
    .nb-eta-green{color:#188038}
  </style>
"""


def overlay_nav_eta_html(
    image_path: Path,
    step: Dict[str, Any],
    next_step: Optional[Dict[str, Any]] = None,
    *,
    arrival_time: str = "--:--",
    minutes_remaining: Optional[float] = None,
    distance_km: Optional[float] = None,
    out_path: Optional[Path] = None,
    pad: int = 16,
    device_scale_factor: int = 2,
    overlay_scale: float = 1.0,
) -> Path:
    """Compose nav and ETA cards onto image using Playwright + HTML.

    Produces high-quality styled overlays matching Google Maps navigation.

    Args:
        image_path: Path to base image
        step: Current step dict with navigationInstruction
        next_step: Optional next step for secondary instruction
        arrival_time: ETA arrival time string
        minutes_remaining: Minutes until arrival
        distance_km: Distance remaining in km
        out_path: Output path (default: overwrite input)
        pad: Padding in pixels
        device_scale_factor: Retina scale factor
        overlay_scale: Scale factor for overlay elements (1.0 = default, 1.2 = 20% larger)

    Returns:
        Path to output image
    """
    if not HAS_PLAYWRIGHT:
        raise RuntimeError("Playwright is required for HTML overlays. Run: pip install playwright && playwright install chromium")

    img = Image.open(image_path)
    w, h = img.size
    nav_max_width = max(280, int(w - pad * 2))

    nav_html = _render_nav_sign_html(step, next_step=next_step, max_width_px=nav_max_width, scale=overlay_scale)
    eta_html = _render_eta_card_html(arrival_time, minutes_remaining, distance_km, scale=overlay_scale)

    # Encode image as data URI
    mime = "image/png" if Path(image_path).suffix.lower() == ".png" else "image/jpeg"
    data_uri = f"data:{mime};base64,{base64.b64encode(Path(image_path).read_bytes()).decode('ascii')}"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  {_OVERLAY_CSS}
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; }}
    #wrap {{
      position: relative;
      width: {w}px;
      height: {h}px;
      background: url('{data_uri}') no-repeat;
      background-size: contain;
      overflow: hidden;
    }}
    .nb-overlay    {{ position:absolute; top:{pad}px; left:{pad}px; right:{pad}px; z-index:1000; pointer-events:none; }}
    .nb-overlay-bl {{ position:absolute; bottom:{pad}px; left:{pad}px; z-index:1000; pointer-events:none; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div class="nb-overlay">{nav_html}</div>
    <div class="nb-overlay-bl">{eta_html}</div>
  </div>
</body>
</html>
"""

    output = Path(out_path or image_path)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=device_scale_factor)
        page.set_content(html)
        page.evaluate("(document.fonts && document.fonts.ready) ? document.fonts.ready : Promise.resolve()")
        page.wait_for_timeout(300)
        element = page.locator("#wrap")
        element.screenshot(path=str(output), type="png")
        browser.close()

    return output


def add_overlay_to_map(
    map_path: Path,
    step: Dict[str, Any],
    next_step: Optional[Dict[str, Any]] = None,
    *,
    arrival_time: Optional[str] = None,
    minutes_remaining: Optional[float] = None,
    distance_km: Optional[float] = None,
    out_path: Optional[Path] = None,
    use_playwright: bool = True,
    overlay_scale: float = 1.0,
) -> Path:
    """Add navigation overlay to a map image.

    Convenience function that picks the best renderer available.

    Args:
        map_path: Path to map image
        step: Current step dict
        next_step: Optional next step
        arrival_time: ETA arrival time string
        minutes_remaining: Minutes until arrival
        distance_km: Distance remaining in km
        out_path: Output path (default: overwrite input)
        use_playwright: Use Playwright for high-quality HTML rendering
        overlay_scale: Scale factor for overlay elements (1.0 = default, 1.2 = 20% larger)

    Returns:
        Path to output image with overlay
    """
    if use_playwright and HAS_PLAYWRIGHT:
        return overlay_nav_eta_html(
            map_path, step, next_step,
            arrival_time=arrival_time or "--:--",
            minutes_remaining=minutes_remaining,
            distance_km=distance_km,
            out_path=out_path,
            overlay_scale=overlay_scale,
        )
    else:
        return overlay_nav_eta_pil(
            map_path, step, next_step,
            arrival_time=arrival_time or "--:--",
            minutes_remaining=minutes_remaining,
            distance_km=distance_km,
            out_path=out_path,
        )


def estimate_eta_from_sample(
    sample: Dict[str, Any],
    route_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Estimate ETA fields from a sample and optional route metadata.

    Uses route total_distance_m / total_duration_s to interpolate remaining time.
    Falls back to 40 km/h average if route metadata is unavailable.

    Args:
        sample: A samples.jsonl row with distances.remaining_distance_m populated.
        route_metadata: Optional route metadata dict (total_distance_m, total_duration_s).

    Returns:
        (arrival_time, minutes_remaining, distance_km)
        arrival_time: formatted "H:MM" string, or None if distance unavailable
        minutes_remaining: float minutes to destination, or None
        distance_km: float km remaining, or None
    """
    remaining_m = sample.get("distances", {}).get("remaining_distance_m")
    if not isinstance(remaining_m, (int, float)) or remaining_m < 0:
        return None, None, None

    distance_km = float(remaining_m) / 1000.0
    remaining_s: Optional[float] = None

    if route_metadata:
        total_m = route_metadata.get("total_distance_m")
        total_s = route_metadata.get("total_duration_s")
        if (
            isinstance(total_m, (int, float))
            and isinstance(total_s, (int, float))
            and total_m > 0
            and total_s > 0
        ):
            remaining_s = float(total_s) * (float(remaining_m) / float(total_m))

    if remaining_s is None:
        # Fallback: 40 km/h urban average
        remaining_s = distance_km / 40.0 * 3600.0

    minutes_remaining = remaining_s / 60.0
    arrival_time = (datetime.now() + timedelta(seconds=remaining_s)).strftime("%I:%M").lstrip("0")
    return arrival_time, minutes_remaining, distance_km


def build_step_payload_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Build an overlay step payload dict from a samples.jsonl row.

    Returns a dict suitable for passing as the `step` argument to
    add_overlay_to_map / overlay_nav_eta_html / overlay_nav_eta_pil.
    """
    instruction = sample.get("prior", {}).get("instruction", "")
    maneuver = sample.get("maneuver", "MANEUVER_UNSPECIFIED")
    step_distance_m = sample.get("distances", {}).get("step_distance_m", 0)
    return {
        "navigationInstruction": {
            "instruction": instruction,
            "maneuver": maneuver,
            "remaining_distance_m": step_distance_m,
        },
        "instruction": instruction,
        "maneuver": maneuver,
        "distance_m": step_distance_m,
        "current_distance_m": 0,
    }


def overlay_scale_for_map(width: int, height: int, reference_scale: float = 2.1, reference_height: int = 600) -> float:
    """Compute a proportional overlay_scale for a given map size.

    Keeps overlay elements the same fraction of map height as on the reference map.

    Args:
        width: Map width in pixels (unused, height drives scale)
        height: Map height in pixels
        reference_scale: overlay_scale used on the reference map size
        reference_height: height of the reference map

    Returns:
        overlay_scale appropriate for this map size
    """
    return reference_scale * (height / reference_height)


__all__ = [
    "overlay_nav_eta_pil",
    "overlay_nav_eta_html",
    "add_overlay_to_map",
    "estimate_eta_from_sample",
    "build_step_payload_from_sample",
    "overlay_scale_for_map",
    "format_distance",
    "MANEUVER_ICONS",
    "HAS_PLAYWRIGHT",
]
