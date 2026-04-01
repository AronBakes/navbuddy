#!/usr/bin/env python3
"""Regenerate all 100 NavBuddy-100 overhead maps using the same logic as `navbuddy generate`.

- Sedan car icon (yellow cab)
- Car positioned 20m before step end (matching frame position)
- Full route polyline shown as dimmed background
- Navigation overlays (green header + ETA footer)
- 640x400 output

Usage:
    python scripts/regenerate_maps.py [--output-dir data/maps]
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS_DIR = Path(__file__).parent.parent / "assets"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate NavBuddy-100 maps")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "maps")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(DATA_DIR / "navbuddy100_manifest.json"))

    from navbuddy.map_renderer_osm import render_map, DEFAULT_CAR_ICON_SCALE
    from navbuddy.polylines import decode_polyline, pose_from_polyline, encode_polyline
    from navbuddy.overlays import add_overlay_to_map, overlay_scale_for_map

    if not (ASSETS_DIR / "navbuddy-sedan.png").exists():
        print(f"ERROR: sedan icon not found at {ASSETS_DIR / 'navbuddy-sedan.png'}")
        sys.exit(1)

    overlay_scale = overlay_scale_for_map(640, 400) * 0.8
    total = 0
    failed = 0

    for route in manifest["routes"]:
        route_id = route["route_id"]
        steps = route["steps"]

        # Build full route polyline from all step polylines
        all_route_coords = []
        for step in steps:
            coords = decode_polyline(step["polyline"])
            all_route_coords.extend(coords)
        route_polyline_coords = all_route_coords if all_route_coords else None

        # Precompute remaining distance at each step
        remaining_per_step = []
        running = 0
        for step in reversed(steps):
            running += int(step.get("distance_m", 0))
            remaining_per_step.append(running)
        remaining_per_step.reverse()

        for idx, step in enumerate(steps):
            step_index = step["step_index"]
            map_filename = f"{route_id}_step{step_index:03d}_map.png"
            map_path = output_dir / map_filename

            step_polyline = step["polyline"]
            step_coords = decode_polyline(step_polyline)
            if not step_coords:
                print(f"  SKIP {map_filename}: empty polyline")
                continue

            step_distance_m = float(step.get("distance_m", 0) or 0)

            # Car position: 40m from step end (matching sparse4 closest frame),
            # or midpoint for short steps
            car_lat, car_lng, car_heading = None, None, None
            if step_polyline and step_distance_m > 0:
                target_from_end = min(40.0, step_distance_m / 2.0) if step_distance_m < 80.0 else 40.0
                pose = pose_from_polyline(step_polyline, target_from_end)
                if pose:
                    car_lat, car_lng, car_heading = pose

            # Fallback to step start
            if car_lat is None:
                car_lat = step["start_lat"]
                car_lng = step["start_lng"]
                car_heading = step.get("heading", 0.0)

            # Render base map
            try:
                ok = render_map(
                    step_polyline_coords=step_coords,
                    car_lat=car_lat,
                    car_lng=car_lng,
                    heading=car_heading or 0.0,
                    start_lat=step["start_lat"],
                    start_lng=step["start_lng"],
                    end_lat=step["end_lat"],
                    end_lng=step["end_lng"],
                    output_path=map_path,
                    route_polyline_coords=route_polyline_coords,
                    car_icon="sedan",
                    assets_dir=ASSETS_DIR,
                    car_icon_scale=DEFAULT_CAR_ICON_SCALE,
                )
            except Exception as e:
                print(f"  FAIL {map_filename}: render error: {e}")
                failed += 1
                continue

            if not ok:
                print(f"  FAIL {map_filename}: render returned False")
                failed += 1
                continue

            # Add navigation overlay
            try:
                distance_m = step.get("distance_m", 0)
                remaining_m = remaining_per_step[idx]
                remaining_km = remaining_m / 1000.0
                remaining_min = (remaining_m / 1000.0) / 40.0 * 60.0
                remaining_s = remaining_min * 60
                arrival_time = (datetime.now() + timedelta(seconds=remaining_s)).strftime("%I:%M")

                # Step payload matching what overlays.py expects
                step_payload = {
                    "instruction": step.get("instruction", ""),
                    "maneuver": step.get("maneuver", ""),
                    "distance_m": distance_m,
                }

                next_payload = None
                if idx + 1 < len(steps):
                    ns = steps[idx + 1]
                    next_payload = {
                        "instruction": ns.get("instruction", ""),
                        "maneuver": ns.get("maneuver", ""),
                        "distance_m": ns.get("distance_m", 0),
                    }

                add_overlay_to_map(
                    map_path,
                    step_payload,
                    next_payload,
                    arrival_time=arrival_time,
                    minutes_remaining=remaining_min,
                    distance_km=remaining_km,
                    overlay_scale=overlay_scale,
                )
            except Exception as e:
                print(f"  WARN {map_filename}: overlay failed: {e}")

            total += 1
            print(f"  [{total:3d}/100] {map_filename}")

    print(f"\nDone: {total} maps generated, {failed} failed")


if __name__ == "__main__":
    main()
