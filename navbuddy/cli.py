"""NavBuddy CLI - Generate VLM training data for road navigation."""

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load .env file before any other imports that might need env vars
load_dotenv()

app = typer.Typer(
    name="navbuddy",
    help="Generate VLM training data for road navigation",
    add_completion=False,
)

console = Console()


def parse_latlon(value: str) -> tuple[float, float]:
    """Parse a lat,lon string into a tuple."""
    cleaned = value.strip().replace(";", ",")
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in cleaned.split() if p.strip()]
    if len(parts) != 2:
        raise typer.BadParameter(f"Invalid lat/lon: {value!r}")
    return float(parts[0]), float(parts[1])


@app.callback()
def callback():
    """NavBuddy - VLM training data for road navigation."""
    pass


@app.command(rich_help_panel="Setup")
def setup(
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)"
    ),
    output_dir: Path = typer.Option(
        Path("./data"), "--output-dir", "-O", help="Output directory"
    ),
    frame_profile: str = typer.Option(
        "manifest", "--frame-profile",
        help="Download profile: manifest (1 frame/step, ~$0.70) or sparse4 (4 frames/step, ~$2.72)",
    ),
    skip_maps: bool = typer.Option(
        False, "--skip-maps", help="Skip bundled overhead maps"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt"
    ),
):
    """Download NavBuddy-100 and set up the dataset.

    Downloads 100 Street View frames, fetches pre-rendered OSM overhead maps,
    and writes samples.jsonl with ground-truth annotations.

    Examples:
        navbuddy setup
        navbuddy setup --api-key YOUR_KEY
        navbuddy setup --frame-profile sparse4
    """
    import os
    import io
    import zipfile
    import urllib.request

    MAPS_URL = "https://github.com/AronBakes/navbuddy/releases/download/v0.1.0/navbuddy100_maps.zip"

    # ── Find manifest ──
    manifest_path = Path(__file__).parent.parent / "manifests" / "navbuddy100.json"
    if not manifest_path.exists():
        manifest_path = Path("manifests/navbuddy100.json")
    if not manifest_path.exists():
        console.print("[red]Cannot find manifests/navbuddy100.json[/red]")
        console.print("Run this command from the navbuddy repo root, or install via: pip install navbuddy")
        raise typer.Exit(1)

    # ── Resolve API key (prompt if missing) ──
    key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        console.print()
        console.print("[bold]Google Maps API Key[/bold]")
        console.print("  NavBuddy downloads Street View imagery using the Google Maps API.")
        console.print("  Get a key at: [blue]https://developers.google.com/maps/documentation/streetview/get-api-key[/blue]")
        console.print("  Enable: [dim]Street View Static API, Directions API, Geocoding API[/dim]")
        console.print()
        key = typer.prompt("Enter your Google Maps API key")
        if not key or not key.strip():
            console.print("[red]No API key provided. Cannot download Street View frames.[/red]")
            raise typer.Exit(1)
        key = key.strip()

    # ── Validate API key with a test call ──
    try:
        from navbuddy.streetview_client import check_streetview_coverage
        test = check_streetview_coverage(-33.8567, 151.2153, api_key=key)
        if test.get("status") == "REQUEST_DENIED":
            console.print("[red]API key is invalid or Street View Static API is not enabled.[/red]")
            console.print(f"  Error: {test.get('error_message', 'REQUEST_DENIED')}")
            raise typer.Exit(1)
    except Exception as e:
        if "REQUEST_DENIED" in str(e):
            console.print(f"[red]API key validation failed: {e}[/red]")
            raise typer.Exit(1)

    # ── Cost estimate ──
    frames_count = 100 if frame_profile == "manifest" else 389
    cost = frames_count * 0.007
    console.print()
    console.print(f"[bold]NavBuddy-100 Setup[/bold]")
    console.print(f"  Frames: {frames_count} Street View images (~${cost:.2f})")
    if not skip_maps:
        console.print(f"  Maps: 100 pre-rendered OSM overhead maps (14MB download)")
    console.print(f"  Output: {output_dir}")
    console.print()

    if not yes:
        proceed = typer.confirm("Continue?", default=True)
        if not proceed:
            raise typer.Exit(0)

    # ── Download pre-rendered maps ──
    if not skip_maps:
        maps_dir = output_dir / "maps"
        existing_maps = len(list(maps_dir.glob("*.png"))) if maps_dir.exists() else 0
        if existing_maps >= 100:
            console.print(f"  Maps: {existing_maps} already present, skipping download")
        else:
            console.print(f"  Downloading overhead maps...", end="")
            try:
                # Try direct download first, fall back to gh CLI for private repos
                import subprocess as _sp
                try:
                    req = urllib.request.Request(MAPS_URL, headers={"User-Agent": "navbuddy"})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = resp.read()
                except urllib.request.HTTPError:
                    # Private repo — use gh CLI which has auth
                    result = _sp.run(
                        ["gh", "release", "download", "v0.1.0", "-R", MAPS_URL.split("github.com/")[1].split("/releases")[0], "-p", "navbuddy100_maps.zip", "-D", str(output_dir)],
                        capture_output=True, timeout=120,
                    )
                    if result.returncode != 0:
                        raise Exception(result.stderr.decode().strip())
                    data = (output_dir / "navbuddy100_maps.zip").read_bytes()
                    (output_dir / "navbuddy100_maps.zip").unlink()
                maps_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    extracted = 0
                    for name in zf.namelist():
                        if name.endswith(".png"):
                            dest = maps_dir / name
                            if not dest.exists():
                                dest.write_bytes(zf.read(name))
                                extracted += 1
                console.print(f" {extracted} maps extracted")
            except Exception as e:
                console.print(f"\n[yellow]  Maps download failed: {e}[/yellow]")
                console.print("[dim]  You can render maps later with: navbuddy download-manifest -m manifests/navbuddy100.json --render-maps[/dim]")

    # ── Download frames ──
    from navbuddy.manifest import download_from_manifest

    stats = download_from_manifest(
        manifest_file=manifest_path,
        output_dir=output_dir,
        api_key=key,
        frame_profile=frame_profile,
        render_maps=False,
        verbose=True,
    )

    # ── Summary ──
    console.print()
    console.print(f"[bold green]Setup complete![/bold green]")
    console.print(f"  Frames: {stats.get('downloaded', 0)} downloaded, {stats.get('skipped', 0)} skipped")
    if stats.get('failed', 0):
        console.print(f"  [yellow]Failed: {stats['failed']} (check API key / quota)[/yellow]")
    console.print(f"  Samples: {stats.get('samples_written', 0)}")
    console.print(f"  Output: {output_dir}")
    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print(f"  navbuddy stats -d {output_dir}")
    console.print(f"  navbuddy evaluate -d {output_dir}/samples.jsonl -m google/gemini-3-flash-preview -n 5")


@app.command(rich_help_panel="Utilities")
def geocode(
    address: str = typer.Argument(..., help="Address to geocode"),
):
    """Convert an address to lat,lng coordinates.

    Examples:
        navbuddy geocode "Sydney Opera House"
        navbuddy geocode "123 Queen St, Brisbane QLD"
    """
    from navbuddy.routing_client import geocode as _geocode

    try:
        lat, lng = _geocode(address)
        console.print(f"{lat},{lng}")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command("reverse-geocode", rich_help_panel="Utilities")
def reverse_geocode_cmd(
    coords: str = typer.Option(..., "--coords", "-c", help="Coordinates as 'lat,lng'"),
):
    """Convert lat,lng coordinates to a street address.

    Examples:
        navbuddy reverse-geocode -c "-27.4698,153.0251"
        navbuddy reverse-geocode -c "-33.8568,151.2153"
    """
    from navbuddy.routing_client import reverse_geocode as _reverse_geocode

    try:
        lat, lng = parse_latlon(coords)
        address = _reverse_geocode(lat, lng)
        console.print(address)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command(rich_help_panel="Data")
def generate(
    origin: str = typer.Option(..., "--origin", "-o", help="Origin as 'lat,lon' or address"),
    dest: str = typer.Option(..., "--dest", "-d", help="Destination as 'lat,lon' or address"),
    output_dir: Path = typer.Option(
        Path("./data"), "--output-dir", "-O", help="Output directory"
    ),
    city: Optional[str] = typer.Option(
        None, "--city", "-c", help="City name for route ID prefix"
    ),
    route_id: Optional[str] = typer.Option(
        None, "--route-id", help="Custom route ID (auto-generated if not provided)"
    ),
    skip_images: bool = typer.Option(
        False, "--skip-images", help="Skip downloading Street View images"
    ),
    frame_profile: str = typer.Option(
        "sparse4",
        "--frame-profile",
        help="Frame profile: 'sparse4' (100/80/60/40), 'video5m' (full step @ 5m), or 'custom'",
    ),
    sample_mode: str = typer.Option(
        "sparse",
        "--sample-mode",
        "-m",
        help="Legacy alias: sparse->sparse4, dense->video5m, custom->custom",
    ),
    spacing: float = typer.Option(
        20.0,
        "--spacing",
        "-s",
        help="Spacing between samples in meters (used by custom profile)",
    ),
    sample_start: Optional[float] = typer.Option(
        None, "--sample-start", help="Start of sampling window in meters from end of step (e.g., 150 = start 150m before end)"
    ),
    sample_end: Optional[float] = typer.Option(
        None, "--sample-end", help="End of sampling window in meters from end of step (e.g., 30 = stop 30m before end)"
    ),
    map_renderer: str = typer.Option(
        "osm", "--map-renderer", help="Map renderer: 'osm' (default, Playwright + Leaflet) or 'google' (Static Maps API)"
    ),
    car_icon: str = typer.Option(
        "cybertruck", "--car-icon", help="Car icon: 'cybertruck' (default), 'arrow', 'f1', 'model3', 'wrx'"
    ),
    car_icon_scale: float = typer.Option(
        0.025, "--car-icon-scale", help="Scale factor for car icons (default 0.025, maintains aspect ratio)"
    ),
    assets_dir: Optional[Path] = typer.Option(
        None, "--assets-dir", help="Directory containing car icon images"
    ),
    add_overlays: bool = typer.Option(
        True, "--add-overlays/--no-overlays", help="Add navigation overlays (header + ETA) to map images (default: on)"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt"
    ),
):
    """Generate a route with Street View frames and metadata.

    Origin and destination accept either 'lat,lng' coordinates or street addresses.

    Examples:
        navbuddy generate -o "Sydney Opera House" -d "Bondi Beach"
        navbuddy generate -o "-27.4698,153.0251" -d "-27.4512,153.0389" -c brisbane
        navbuddy generate -o "123 Queen St, Brisbane" -d "South Bank, Brisbane" --frame-profile sparse4
    """
    from navbuddy.generate import generate_route, preflight_route
    from navbuddy.sampling import FRAME_PROFILES, profile_from_sample_mode

    # Accept either "lat,lon" or an address string
    def resolve_location(value: str) -> tuple[float, float]:
        try:
            return parse_latlon(value)
        except (typer.BadParameter, ValueError):
            # Not coords — try geocoding as address
            from navbuddy.routing_client import geocode as _geocode
            console.print(f"  Geocoding: [dim]{value}[/dim]")
            lat, lng = _geocode(value)
            console.print(f"  Resolved:  [bold]{lat},{lng}[/bold]")
            return (lat, lng)

    try:
        origin_coords = resolve_location(origin)
        dest_coords = resolve_location(dest)
    except Exception as e:
        console.print(f"[red]Error resolving location: {e}[/red]")
        raise typer.Exit(1)

    # Check Street View coverage at origin
    if not skip_images:
        import os
        from navbuddy.streetview_client import check_streetview_coverage

        _api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        if _api_key:
            for label, coords in [("Origin", origin_coords), ("Destination", dest_coords)]:
                meta = check_streetview_coverage(coords[0], coords[1], api_key=_api_key)
                if meta.get("status") != "OK":
                    console.print(
                        f"[yellow]Warning: No Street View coverage near {label} "
                        f"({coords[0]:.4f}, {coords[1]:.4f}). "
                        f"Frames may fail to download.[/yellow]"
                    )

    # Resolve profile with backward-compatible sample_mode alias.
    profile = (frame_profile or "sparse4").strip().lower()
    if profile not in FRAME_PROFILES:
        console.print(f"[red]Error: Invalid --frame-profile '{frame_profile}'[/red]")
        console.print("Valid profiles: sparse4, video5m, custom")
        raise typer.Exit(1)

    legacy_mode = (sample_mode or "sparse").strip().lower()
    if legacy_mode not in {"sparse", "dense", "custom"}:
        console.print(f"[red]Error: Invalid --sample-mode '{sample_mode}'[/red]")
        raise typer.Exit(1)

    # If user explicitly set legacy mode away from default, it wins unless conflicting.
    legacy_profile = profile_from_sample_mode(legacy_mode)
    if legacy_mode != "sparse":
        if profile != "sparse4" and profile != legacy_profile:
            console.print(
                f"[red]Error: Conflicting options: --frame-profile {profile} and --sample-mode {legacy_mode}[/red]"
            )
            raise typer.Exit(1)
        profile = legacy_profile

    # Window/spacing options imply custom profile.
    if spacing != 20.0 or sample_start is not None or sample_end is not None:
        if profile != "custom":
            console.print(
                "[yellow]Info: sampling window options detected, switching frame profile to custom.[/yellow]"
            )
            profile = "custom"

    # ── Preflight: fetch route and estimate costs ──
    console.print("\n[bold]Fetching route from google...[/bold]")
    try:
        estimate = preflight_route(
            origin=origin_coords,
            destination=dest_coords,
            frame_profile=profile,
            spacing=spacing,
            sample_start=sample_start,
            sample_end=sample_end,
            map_renderer=map_renderer,
            skip_images=skip_images,
        )
    except Exception as e:
        console.print(f"[red]Error fetching route: {e}[/red]")
        raise typer.Exit(1)

    # Display route summary
    dist_km = estimate["total_distance_m"] / 1000.0
    dur_min = estimate["total_duration_s"] / 60.0
    api_calls = estimate["api_calls"]

    console.print(f"\n{'─' * 60}")
    console.print(f"[bold]Route Summary[/bold]")
    console.print(f"{'─' * 60}")
    console.print(f"  Steps:         {estimate['steps_count']}")
    console.print(f"  Distance:      {dist_km:.1f} km")
    console.print(f"  Duration:      {dur_min:.0f} min")
    console.print(f"  Frame profile: [cyan]{profile}[/cyan]")
    console.print(f"  Total frames:  {estimate['total_frames']}")
    console.print(f"  Overhead maps: {estimate['total_maps']}")

    console.print(f"\n[bold]Per-Step Breakdown[/bold]")
    for sd in estimate["step_details"]:
        dist_str = f"{sd['distance_m']:.0f}m"
        frames_str = f"{sd['frames']} frames"
        targets = sd["remaining_targets_m"]
        if len(targets) <= 6:
            targets_str = ", ".join(f"{d}m" for d in targets)
        else:
            targets_str = ", ".join(f"{d}m" for d in targets[:4]) + f" ... ({len(targets)} total)"
        console.print(
            f"  Step {sd['step_index']:>2}: {sd['maneuver']:<20} {dist_str:>6}  {frames_str:>10}  [{targets_str}]"
        )

    console.print(f"\n[bold]API Calls[/bold]")
    console.print(f"  Routing:              {api_calls['routing']}  (already done)")
    console.print(f"  Street View metadata: {api_calls['streetview_metadata']}  (free)")
    console.print(f"  Street View images:   {api_calls['streetview_images']}  (${estimate['cost_breakdown']['streetview']:.4f})")
    console.print(f"  OSM Overpass:         {api_calls['osm_overpass']}  (free)")
    if api_calls["playwright_maps"] > 0:
        console.print(f"  Playwright maps:      {api_calls['playwright_maps']}  (free, local)")
    if api_calls["static_maps"] > 0:
        console.print(f"  Google Static Maps:   {api_calls['static_maps']}  (${estimate['cost_breakdown']['static_maps']:.4f})")

    console.print(f"\n[bold green]  Estimated cost: ${estimate['estimated_cost_usd']:.4f}[/bold green]")
    console.print(f"{'─' * 60}")

    if not yes and not typer.confirm("\nProceed with generation?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    # ── Execute: download frames and generate maps ──
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating route...", total=None)

        # If city is provided, write into output_dir/city/ subdirectory
        effective_output_dir = output_dir / city if city else output_dir

        try:
            result = generate_route(
                origin=origin_coords,
                destination=dest_coords,
                output_dir=effective_output_dir,
                city=city,
                route_id=route_id,
                skip_images=skip_images,
                frame_profile=profile,
                sample_mode=legacy_mode,
                spacing=spacing,
                sample_start=sample_start,
                sample_end=sample_end,
                map_renderer=map_renderer,
                car_icon=car_icon,
                car_icon_scale=car_icon_scale,
                assets_dir=assets_dir,
                add_overlays=add_overlays,
                progress_callback=lambda msg: progress.update(task, description=msg),
            )

            progress.update(task, description="Done!")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"\n[green]Route generated successfully![/green]")
    console.print(f"  Route ID: {result['route_id']}")
    console.print(f"  Engine: {result.get('engine', 'unknown')}")
    console.print(f"  Steps: {result['steps_count']}")
    console.print(f"  Frames: {result['frames_count']}")
    console.print(f"  Frame profile: {profile}")
    console.print(f"  Maps: {result.get('maps_count', 0)}")
    console.print(f"  Output: {result['output_dir']}")


@app.command(rich_help_panel="Data")
def download_manifest(
    manifest: Path = typer.Option(..., "--manifest", "-m", help="Manifest JSON file"),
    output_dir: Path = typer.Option(
        Path("./data"), "--output-dir", "-O", help="Output directory"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Google Street View API key (or use env var)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum frames to download"
    ),
    frame_profile: str = typer.Option(
        "manifest",
        "--frame-profile",
        help="Download frame profile: manifest, sparse4, video5m, or custom",
    ),
    spacing: float = typer.Option(
        5.0,
        "--spacing",
        "-s",
        help="Spacing in meters when --frame-profile custom (full-step resampling)",
    ),
    sample_start: Optional[float] = typer.Option(
        None,
        "--sample-start",
        help="Custom profile window start in meters from step end",
    ),
    sample_end: Optional[float] = typer.Option(
        None,
        "--sample-end",
        help="Custom profile window end in meters from step end",
    ),
    cost_per_1000: float = typer.Option(
        7.0,
        "--cost-per-1000",
        help="Estimated Street View price in USD per 1000 requests",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
    render_maps: bool = typer.Option(
        False,
        "--render-maps",
        help="Render OSM overhead maps (requires: playwright install chromium)",
    ),
    car_icon: str = typer.Option(
        "arrow",
        "--car-icon",
        help="Car marker on overhead maps: arrow, sedan, cybertruck, f1, model3, wrx",
    ),
):
    """Download images from a manifest file.

    Users provide their own API key to download images specified in manifests.
    Also reconstructs `samples.jsonl` + `routes/*/metadata.json` in output_dir.

    Examples:
        # Download all frames
        navbuddy download-manifest -m manifest.json --api-key YOUR_KEY

        # Download NavBuddy-100 with overhead maps
        navbuddy download-manifest -m navbuddy100_manifest.json --api-key YOUR_KEY --render-maps

        # Download first 100 frames (for testing)
        navbuddy download-manifest -m manifest.json --api-key YOUR_KEY --limit 100

        # Custom output directory
        navbuddy download-manifest -m manifest.json -O ./my_data --api-key YOUR_KEY

        # Full-route dense download at 5m spacing (shows cost estimate + confirmation)
        navbuddy download-manifest -m manifest.json --frame-profile custom --spacing 5
    """
    import os
    from navbuddy.manifest import DOWNLOAD_FRAME_PROFILES, download_from_manifest, estimate_download_from_manifest

    # Try to get API key from param or environment
    key = api_key or os.environ.get("GOOGLE_STREETVIEW_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        console.print(
            "[red]Error: No API key provided. Use --api-key or set GOOGLE_MAPS_API_KEY[/red]"
        )
        raise typer.Exit(1)

    if not manifest.exists():
        console.print(f"[red]Error: Manifest not found: {manifest}[/red]")
        raise typer.Exit(1)
    frame_profile = frame_profile.strip().lower()
    if frame_profile not in DOWNLOAD_FRAME_PROFILES:
        console.print(
            f"[red]Error: --frame-profile must be one of {', '.join(sorted(DOWNLOAD_FRAME_PROFILES))}[/red]"
        )
        raise typer.Exit(1)
    if spacing <= 0:
        console.print("[red]Error: --spacing must be > 0[/red]")
        raise typer.Exit(1)
    if frame_profile != "custom" and (sample_start is not None or sample_end is not None or spacing != 5.0):
        console.print("[yellow]Info: --spacing/--sample-start/--sample-end are ignored unless --frame-profile custom[/yellow]")

    console.print(f"[bold]Download from Manifest[/bold]")
    console.print(f"  Manifest: {manifest}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Frame profile: {frame_profile}")
    if frame_profile == "custom":
        console.print(f"  Spacing: {spacing}m")
    if limit:
        console.print(f"  Limit: {limit} frames")
    console.print()

    try:
        estimate = estimate_download_from_manifest(
            manifest_file=manifest,
            output_dir=output_dir,
            frame_profile=frame_profile,
            spacing=spacing,
            sample_start=sample_start,
            sample_end=sample_end,
            limit=limit,
        )
    except Exception as e:
        console.print(f"[red]Error estimating requests: {e}[/red]")
        raise typer.Exit(1)

    estimated_requests = int(estimate["estimated_requests"])
    estimated_cost_usd = (estimated_requests / 1000.0) * float(cost_per_1000)
    console.print("[bold]Cost Estimate[/bold]")
    console.print(f"  Routes: {estimate['routes']}")
    console.print(f"  Steps: {estimate['steps']}")
    console.print(f"  Target frames: {estimate['total_targets']}")
    console.print(f"  Existing files: {estimate['existing']}")
    console.print(f"  To download: {estimate['to_download']}")
    console.print(
        f"  Estimated requests: {estimated_requests} "
        f"(~${estimated_cost_usd:.2f} at ${cost_per_1000:.2f}/1k)"
    )
    console.print()

    if frame_profile != "manifest" and estimated_requests > 0 and not yes:
        proceed = typer.confirm("Proceed with this estimated Street View cost?", default=False)
        if not proceed:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    try:
        stats = download_from_manifest(
            manifest_file=manifest,
            output_dir=output_dir,
            api_key=key,
            limit=limit,
            frame_profile=frame_profile,
            spacing=spacing,
            sample_start=sample_start,
            sample_end=sample_end,
            render_maps=render_maps,
            car_icon=car_icon,
            verbose=True,
        )

        console.print()
        console.print(f"[green]Download complete![/green]")
        console.print(f"  Downloaded: {stats['downloaded']}")
        console.print(f"  Skipped (existing): {stats['skipped']}")
        console.print(f"  Failed: {stats['failed']}")
        console.print(f"  Routes metadata: {stats.get('routes_written', 0)}")
        console.print(f"  Samples metadata: {stats.get('samples_written', 0)}")
        if render_maps:
            console.print(f"  Maps rendered: {stats.get('maps_rendered', 0)}")
        console.print(f"  Output: {output_dir}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("regenerate-frames", rich_help_panel="Data")
def regenerate_frames(
    data_root: Path = typer.Option(..., "--data-root", "-d", help="Dataset root (contains samples.jsonl)"),
    frame_profile: str = typer.Option(
        "sparse4",
        "--frame-profile",
        help="Frame profile: sparse4, video5m, custom",
    ),
    spacing: float = typer.Option(
        20.0,
        "--spacing",
        "-s",
        help="Custom profile spacing in meters",
    ),
    sample_start: Optional[float] = typer.Option(
        None,
        "--sample-start",
        help="Custom profile window start (meters from step end)",
    ),
    sample_end: Optional[float] = typer.Option(
        None,
        "--sample-end",
        help="Custom profile window end (meters from step end)",
    ),
    replace: bool = typer.Option(
        False,
        "--replace/--no-replace",
        help="Replace existing frames and remove stale per-step frame files",
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        "-j",
        help="Parallel download workers",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Google Street View API key (or use env vars)",
    ),
):
    """Regenerate Street View frames for all samples in a dataset."""
    import os
    from datetime import datetime
    from navbuddy.sampling import FRAME_PROFILES
    from navbuddy.frame_regenerator import regenerate_frames_dataset

    if not data_root.exists():
        console.print(f"[red]Error: Data root not found: {data_root}[/red]")
        raise typer.Exit(1)
    if frame_profile not in FRAME_PROFILES:
        console.print(f"[red]Error: Invalid frame profile '{frame_profile}'[/red]")
        console.print("Valid profiles: sparse4, video5m, custom")
        raise typer.Exit(1)
    if frame_profile != "custom" and (sample_start is not None or sample_end is not None or spacing != 20.0):
        console.print("[yellow]Info: custom window options ignored unless --frame-profile custom[/yellow]")

    key = api_key or os.environ.get("GOOGLE_STREETVIEW_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        console.print("[red]Error: missing API key. Use --api-key or GOOGLE_MAPS_API_KEY[/red]")
        raise typer.Exit(1)

    samples_path = data_root / "samples.jsonl"
    if not samples_path.exists():
        console.print(f"[red]Error: samples.jsonl not found: {samples_path}[/red]")
        raise typer.Exit(1)

    backup = samples_path.with_name(
        f"samples.jsonl.bak_regen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    backup.write_bytes(samples_path.read_bytes())

    console.print("[bold]Regenerate Frames[/bold]")
    console.print(f"  Data root: {data_root}")
    console.print(f"  Frame profile: {frame_profile}")
    console.print(f"  Replace: {replace}")
    console.print(f"  Concurrency: {concurrency}")
    console.print(f"  Backup: {backup}")
    console.print()

    try:
        stats = regenerate_frames_dataset(
            data_root=data_root,
            api_key=key,
            frame_profile=frame_profile,
            spacing=spacing,
            sample_start=sample_start,
            sample_end=sample_end,
            replace=replace,
            concurrency=concurrency,
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print("[green]Regeneration complete[/green]")
    console.print(f"  Samples total: {stats.samples_total}")
    console.print(f"  Samples updated: {stats.samples_updated}")
    console.print(f"  Downloaded: {stats.downloaded}")
    console.print(f"  Skipped existing: {stats.skipped_existing}")
    console.print(f"  Failed: {stats.failed}")
    console.print(f"  Removed old: {stats.removed_old}")


@app.command(rich_help_panel="Data")
def list_manifests():
    """List available sample manifests."""
    from navbuddy.utils import Config

    manifests_dir = Config.MANIFESTS_DIR

    if not manifests_dir.exists():
        console.print("[yellow]No manifests directory found.[/yellow]")
        return

    manifests = [
        m for m in manifests_dir.glob("*.json")
        if m.name != "schema.json"
    ]
    if not manifests:
        console.print("[yellow]No manifests found.[/yellow]")
        return

    console.print("[bold]Available manifests:[/bold]")
    for m in sorted(manifests):
        console.print(f"  - {m.name}")


@app.command(rich_help_panel="Evaluation")
def evaluate(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to samples.jsonl file"),
    model: str = typer.Option(..., "--model", "-m", help="Model ID (e.g., google/gemini-2.0-flash-001)"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output JSONL file (default: results_{model_id}.jsonl)"
    ),
    modality: str = typer.Option(
        "video + prior", "--modality", help="Input modality: 'video + prior' or 'prior'"
    ),
    provider: str = typer.Option(
        "openrouter", "--provider", "-p", help="API provider: 'openrouter' or 'local'"
    ),
    data_root: Optional[Path] = typer.Option(
        None, "--data-root", help="Root directory for image paths (default: dataset parent dir)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum number of samples to process"
    ),
    use_segformer_context: bool = typer.Option(
        False,
        "--use-segformer-context",
        help="Inject SegFormer-derived spatial context into prompts",
    ),
    segformer_model_id: str = typer.Option(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        "--segformer-model-id",
        help="SegFormer checkpoint ID for context extraction",
    ),
    segformer_device: str = typer.Option(
        "auto",
        "--segformer-device",
        help="SegFormer device: auto, cpu, or cuda",
    ),
    segformer_cache_dir: Optional[Path] = typer.Option(
        None,
        "--segformer-cache-dir",
        help="Optional local cache directory for SegFormer weights",
    ),
    local_device: str = typer.Option(
        "auto",
        "--local-device",
        help="Local provider device: auto, cuda, or cpu",
    ),
    local_dtype: str = typer.Option(
        "auto",
        "--local-dtype",
        help="Local provider dtype: auto, float16, bfloat16, or float32",
    ),
    local_load_in_4bit: bool = typer.Option(
        True,
        "--local-load-in-4bit/--no-local-load-in-4bit",
        help="Enable 4-bit quantization for local provider",
    ),
    local_max_new_tokens: int = typer.Option(
        256,
        "--local-max-new-tokens",
        help="Max new tokens for local provider",
    ),
    local_temperature: float = typer.Option(
        0.0,
        "--local-temperature",
        help="Sampling temperature for local provider (0.0 = greedy)",
    ),
    dedupe_frames: bool = typer.Option(
        True,
        "--dedupe-frames/--no-dedupe-frames",
        help="Drop duplicate frame images (same path/content) before model inference",
    ),
    include_arrive_steps: bool = typer.Option(
        False,
        "--include-arrive-steps/--skip-arrive-steps",
        help="Whether to include terminal ARRIVE samples in inference (default: skip)",
    ),
    augment: Optional[str] = typer.Option(
        None,
        "--augment",
        help="Image augment for video modality: fog, night, rain, motion_blur",
    ),
    variant: Optional[str] = typer.Option(
        None,
        "--variant",
        help="Optional non-augmentation variant tag stored in result rows",
    ),
    route_id: Optional[str] = typer.Option(
        None,
        "--route-id",
        help="Only process samples matching this route ID (comma-separated for multiple)",
    ),
    sample_id: Optional[str] = typer.Option(
        None,
        "--sample-id",
        help="Only process samples matching this sample ID (comma-separated for multiple)",
    ),
    icl_k: Optional[int] = typer.Option(
        None,
        "--icl-k",
        help="Number of ICL few-shot examples to include (with images). Requires --icl-examples.",
    ),
    icl_examples: Optional[str] = typer.Option(
        None,
        "--icl-examples",
        help="Comma-separated 1-based indices of ICL examples to use from icl_examples.jsonl (e.g. '1,2').",
    ),
    structured_output: bool = typer.Option(
        False,
        "--structured-output/--no-structured-output",
        help="Use JSON schema response_format to constrain output. Requires provider support (OpenAI, Google, Anthropic).",
    ),
    prompt_version: str = typer.Option(
        "v1",
        "--prompt-version",
        help="Prompt version to use: v1 (default) or v2 (tighter landmark quality criteria)",
    ),
    redis_url: Optional[str] = typer.Option(
        None,
        "--redis-url",
        help="Upstash Redis REST URL for result caching (or set UPSTASH_REDIS_REST_URL env var)",
        envvar="UPSTASH_REDIS_REST_URL",
    ),
    redis_token: Optional[str] = typer.Option(
        None,
        "--redis-token",
        help="Upstash Redis REST token for result caching (or set UPSTASH_REDIS_REST_TOKEN env var)",
        envvar="UPSTASH_REDIS_REST_TOKEN",
    ),
):
    """Run VLM inference on a dataset.

    Examples:
        # Run inference with OpenRouter-hosted model
        navbuddy evaluate --dataset ./data/samples.jsonl --model google/gemini-2.0-flash-001

        # Prior-only baseline (no images)
        navbuddy evaluate --dataset ./data/samples.jsonl --model google/gemini-2.0-flash-001 --modality prior

        # Limit to first 10 samples
        navbuddy evaluate --dataset ./data/samples.jsonl --model gpt-4o --limit 10

        # Local Qwen2.5-VL 3B (8GB-friendly with 4-bit load)
        navbuddy evaluate --provider local --model Qwen/Qwen2.5-VL-3B-Instruct --dataset ./data/samples.jsonl
    """
    from navbuddy.eval.inference import run_inference, PROMPT_VERSIONS, load_icl_examples, build_icl_messages

    if not dataset.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    # Resolve prompt version
    if prompt_version not in PROMPT_VERSIONS:
        console.print(f"[red]Unknown prompt version '{prompt_version}'. Use: {', '.join(PROMPT_VERSIONS)}[/red]")
        raise typer.Exit(1)
    _sys_prompt, _prior_sys_prompt = PROMPT_VERSIONS[prompt_version]

    # Parse comma-separated route/sample IDs
    route_ids = [r.strip() for r in route_id.split(",") if r.strip()] if route_id else None
    sample_ids = [s.strip() for s in sample_id.split(",") if s.strip()] if sample_id else None

    # Generate output filename if not provided
    if output is None:
        model_slug = model.replace("/", "_").replace(":", "_")
        modality_slug = modality.replace(" ", "_").replace("+", "")
        suffix_parts = [f"results_{model_slug}_{modality_slug}"]
        if prompt_version != "v1":
            suffix_parts.append(f"prompt_{prompt_version}")
        if variant:
            suffix_parts.append(f"variant_{variant.replace(' ', '_')}")
        if augment:
            suffix_parts.append(f"aug_{augment}")
        output = Path("results") / ("_".join(suffix_parts) + ".jsonl")

    console.print(f"[bold]NavBuddy VLM Evaluation[/bold]")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Model: {model}")
    console.print(f"  Modality: {modality}")
    console.print(f"  Provider: {provider}")
    console.print(f"  Output: {output}")
    console.print(f"  Variant: {variant or '-'}")
    console.print(f"  Augment: {augment or '-'}")
    console.print(f"  Prompt version: {prompt_version}")
    if route_ids:
        console.print(f"  Route IDs: {', '.join(route_ids)}")
    if sample_ids:
        console.print(f"  Sample IDs: {', '.join(sample_ids)}")
    console.print(f"  Dedupe frames: {dedupe_frames}")
    console.print(f"  Include ARRIVE steps: {include_arrive_steps}")
    console.print(f"  SegFormer context: {use_segformer_context}")
    if structured_output:
        console.print(f"  Structured output: [green]enabled[/green] (JSON schema constrained)")
    if provider == "local":
        console.print(f"  Local device: {local_device}")
        console.print(f"  Local dtype: {local_dtype}")
        console.print(f"  Local 4-bit: {local_load_in_4bit}")
        console.print(f"  Local max new tokens: {local_max_new_tokens}")
        console.print(f"  Local temperature: {local_temperature}")
    if limit:
        console.print(f"  Limit: {limit} samples")
    console.print()

    # Set up Redis cache if credentials provided
    _cache = None
    if redis_url and redis_token:
        try:
            from navbuddy.eval.cache import InferenceCache
            _cache = InferenceCache(url=redis_url, token=redis_token)
            console.print(f"  Redis cache: [green]enabled[/green]")
        except Exception as _e:
            console.print(f"  [yellow]Redis cache unavailable: {_e}[/yellow]")
    else:
        console.print(f"  Redis cache: disabled (pass --redis-url / --redis-token to enable)")

    # Build multimodal ICL messages if requested
    _icl_messages = None
    if icl_k is not None and icl_examples is not None:
        _icl_indices = [int(x.strip()) for x in icl_examples.split(",") if x.strip()]
        _examples_path = Path("results") / "icl_examples.jsonl"
        _icl_data_root = data_root if data_root else dataset.parent
        _resolved = load_icl_examples(_examples_path, _icl_data_root, example_indices=_icl_indices)
        _selected = _resolved[:icl_k]
        if len(_selected) < icl_k:
            console.print(f"[yellow]Warning: requested k={icl_k} but only {len(_selected)} examples resolved[/yellow]")
        _icl_messages = build_icl_messages(_selected)

        # Count images for verification
        _n_imgs = sum(
            sum(1 for c in m.get("content", []) if isinstance(c, dict) and c.get("type") == "image_url")
            for m in _icl_messages if isinstance(m.get("content"), list)
        )
        console.print(f"  ICL: k={icl_k}, examples={icl_examples} → {len(_selected)} examples, {_n_imgs} images, {len(_icl_messages)} turns")
        for i, ex in enumerate(_selected):
            console.print(f"    #{_icl_indices[i]}: {ex['sample_id']}  frame={ex['frame_path'].name if ex.get('frame_path') else '?'}")
    elif icl_k is not None or icl_examples is not None:
        console.print("[red]Error: --icl-k and --icl-examples must be used together[/red]")
        raise typer.Exit(1)

    try:
        results = run_inference(
            dataset_path=dataset,
            model_id=model,
            output_path=output,
            modality=modality,
            provider=provider,
            data_root=data_root,
            limit=limit,
            verbose=True,
            augment=augment,
            variant=variant,
            use_segformer_context=use_segformer_context,
            segformer_model_id=segformer_model_id,
            segformer_device=segformer_device,
            segformer_cache_dir=str(segformer_cache_dir) if segformer_cache_dir else None,
            local_device=local_device,
            local_dtype=local_dtype,
            local_load_in_4bit=local_load_in_4bit,
            local_max_new_tokens=local_max_new_tokens,
            local_temperature=local_temperature,
            dedupe_frames=dedupe_frames,
            include_arrive_steps=include_arrive_steps,
            route_ids=route_ids,
            sample_ids=sample_ids,
            system_prompt=_sys_prompt,
            prior_system_prompt=_prior_sys_prompt,
            cache=_cache,
            icl_messages=_icl_messages,
            structured_output=structured_output,
        )

        # Summary
        successful = sum(1 for r in results if not r.error)
        failed = sum(1 for r in results if r.error)
        avg_latency = sum(r.inference_metadata.latency_ms or 0 for r in results) / len(results) if results else 0

        console.print()
        console.print(f"[green]Evaluation complete![/green]")
        console.print(f"  Total: {len(results)}")
        console.print(f"  Successful: {successful}")
        console.print(f"  Failed: {failed}")
        console.print(f"  Avg latency: {avg_latency:.0f}ms")
        console.print(f"  Results: {output}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("evaluate-matrix", rich_help_panel="Evaluation")
def evaluate_matrix(
    config: Path = typer.Option(..., "--config", "-c", help="Matrix config YAML/JSON file"),
    missing_only: bool = typer.Option(
        False,
        "--missing-only/--no-missing-only",
        help="Only run missing sample/model/modality cells",
    ),
    sample_ids: Optional[str] = typer.Option(
        None,
        "--sample-ids",
        "-s",
        help="Comma-separated sample IDs to run (default: all samples in dataset)",
    ),
):
    """Run 3-modality matrix evaluation for configured models."""
    from navbuddy.eval.matrix_runner import run_evaluation_matrix

    if not config.exists():
        console.print(f"[red]Error: Config not found: {config}[/red]")
        raise typer.Exit(1)

    parsed_sample_ids = [s.strip() for s in sample_ids.split(",") if s.strip()] if sample_ids else None
    if parsed_sample_ids:
        console.print(f"  Filtering to {len(parsed_sample_ids)} sample(s): {', '.join(parsed_sample_ids[:3])}{'...' if len(parsed_sample_ids) > 3 else ''}")

    try:
        summary = run_evaluation_matrix(config, missing_only=missing_only, verbose=True, sample_ids=parsed_sample_ids)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print("[green]Matrix evaluation complete[/green]")
    console.print(f"  Dataset: {summary['dataset']}")
    console.print(f"  Samples: {summary['samples_total']}")
    console.print(f"  Routes: {summary['routes_total']}")
    console.print(f"  Missing-only: {summary['missing_only']}")
    for item in summary["summaries"]:
        console.print(
            "  - "
            + f"{item['output_path']}: "
            + f"attempted={item['attempted']} "
            + f"written={item['written']} "
            + f"skipped={item['skipped_existing']} "
            + f"errors={item['errors']}"
        )


@app.command("eval-assign-augments", rich_help_panel="Evaluation")
def eval_assign_augments(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to samples.jsonl"),
    output: Path = typer.Option(
        Path("config/eval/brisbane_route_augments_v1.json"),
        "--output",
        "-o",
        help="Output assignment JSON",
    ),
):
    """Create deterministic route->augment assignments."""
    from navbuddy.eval.augment_assignment import build_assignment_payload, save_assignment_file

    if not dataset.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    try:
        payload = build_assignment_payload(dataset)
        save_assignment_file(payload, output)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print("[green]Route augment assignments generated[/green]")
    console.print(f"  Routes: {payload.get('routes_total', 0)}")
    console.print(f"  Output: {output}")


@app.command("eval-coverage", rich_help_panel="Evaluation")
def eval_coverage(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to samples.jsonl"),
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir",
        "-r",
        help="Directory containing inference result JSONL files",
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models",
        help="Comma-separated model IDs to include",
    ),
    augment_assignment: Optional[Path] = typer.Option(
        None,
        "--augment-assignment",
        help="Optional route->augment assignment JSON",
    ),
    output_json: Path = typer.Option(
        Path("out/eval/coverage_brisbane.json"),
        "--output-json",
        help="Coverage JSON output path",
    ),
    output_md: Path = typer.Option(
        Path("out/eval/coverage_brisbane.md"),
        "--output-md",
        help="Coverage markdown output path",
    ),
):
    """Compute modality coverage and missing-by-route report."""
    from navbuddy.eval.coverage import compute_modality_coverage, coverage_markdown

    if not dataset.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)
    if not results_dir.exists():
        console.print(f"[red]Error: Results dir not found: {results_dir}[/red]")
        raise typer.Exit(1)

    selected_models = None
    if models:
        selected_models = [m.strip() for m in models.split(",") if m.strip()]

    try:
        report = compute_modality_coverage(
            dataset_path=dataset,
            results_dir=results_dir,
            models=selected_models,
            augment_assignment_file=augment_assignment,
        )
        md = coverage_markdown(report)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    output_md.write_text(md)

    console.print("[green]Coverage report written[/green]")
    console.print(f"  JSON: {output_json}")
    console.print(f"  Markdown: {output_md}")


@app.command(rich_help_panel="Evaluation")
def metrics(
    predictions: Path = typer.Option(..., "--predictions", "-p", help="Predictions JSONL"),
    labels: Path = typer.Option(..., "--labels", "-l", help="Labels JSONL"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output metrics JSON file"
    ),
    mode: str = typer.Option(
        "basic",
        "--mode",
        help="Metrics mode: 'basic' (maneuver only) or 'composite' (semantic stack)",
    ),
    data_root: Optional[Path] = typer.Option(
        None,
        "--data-root",
        help="Root directory for resolving label image paths (default: labels parent)",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Judge model for calibration in composite mode (e.g., openai/gpt-4o)",
    ),
    judge_subsample: int = typer.Option(
        0,
        "--judge-subsample",
        help="Number of samples to score with judge in composite mode",
    ),
):
    """Compute evaluation metrics.

    Compares predictions against ground truth labels for maneuver classification
    and instruction quality.

    Examples:
        navbuddy metrics -p results_gpt4o.jsonl -l samples.jsonl
        navbuddy metrics -p predictions.jsonl -l labels.jsonl -o metrics.json
    """
    import json
    from collections import defaultdict
    from rich.table import Table

    if not predictions.exists():
        console.print(f"[red]Error: Predictions file not found: {predictions}[/red]")
        raise typer.Exit(1)

    if not labels.exists():
        console.print(f"[red]Error: Labels file not found: {labels}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]NavBuddy Metrics[/bold]")
    console.print(f"  Predictions: {predictions}")
    console.print(f"  Labels: {labels}")
    console.print(f"  Mode: {mode}")
    console.print()

    if mode == "composite":
        from navbuddy.eval.metrics_semantic import evaluate_composite_metrics

        try:
            report = evaluate_composite_metrics(
                predictions_path=predictions,
                labels_path=labels,
                data_root=data_root or labels.parent,
                judge_model=judge_model,
                judge_subsample=judge_subsample,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        means = report.get("metrics_mean", {})
        console.print("[bold cyan]Composite Results[/bold cyan]")
        console.print(f"  Matched samples: {report.get('matched_samples', 0)}")
        console.print(f"  Composite score: {means.get('composite_score', 0.0):.4f}")
        console.print(f"  Action reward: {means.get('action_reward', 0.0):.4f}")
        console.print(f"  Lane reward: {means.get('lane_reward', 0.0):.4f}")
        console.print(f"  BERTScore: {means.get('bertscore_reward', 0.0):.4f}")
        console.print(f"  CLIPScore: {means.get('clipscore_reward', 0.0):.4f}")
        console.print(f"  CIDEr-like: {means.get('cider_reward', 0.0):.4f}")
        console.print(f"  ROUGE-L: {means.get('rouge_l_reward', 0.0):.4f}")

        calibration = report.get("judge_calibration")
        if calibration:
            console.print()
            console.print("[bold cyan]Judge Calibration[/bold cyan]")
            console.print(f"  Judge model: {calibration.get('judge_model')}")
            console.print(f"  Successful judged: {calibration.get('successful_judged', 0)}")
            console.print(f"  Spearman: {calibration.get('spearman')}")
            console.print(f"  Kendall: {calibration.get('kendall')}")

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            console.print(f"Metrics saved to: {output}")
        return

    # Load labels
    labels_dict = {}
    with open(labels, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                labels_dict[s.get("id")] = s

    # Load predictions
    predictions_list = []
    with open(predictions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions_list.append(json.loads(line))

    # Compute metrics
    total = 0
    matched = 0
    maneuver_correct = 0
    maneuver_confusion = defaultdict(lambda: defaultdict(int))

    for pred in predictions_list:
        sample_id = pred.get("sample_id") or pred.get("id")
        if sample_id not in labels_dict:
            continue

        label = labels_dict[sample_id]
        total += 1
        matched += 1

        # Maneuver accuracy (if predicted)
        pred_maneuver = pred.get("predicted_maneuver") or pred.get("maneuver") or pred.get("next_action")
        true_maneuver = label.get("maneuver", "UNKNOWN")

        if pred_maneuver:
            if str(pred_maneuver).upper() == str(true_maneuver).upper():
                maneuver_correct += 1
            maneuver_confusion[str(true_maneuver)][str(pred_maneuver).upper()] += 1

    # Summary metrics
    metrics_data = {
        "mode": "basic",
        "total_predictions": len(predictions_list),
        "matched_samples": matched,
        "unmatched_samples": len(predictions_list) - matched,
    }

    if total > 0:
        metrics_data["maneuver_accuracy"] = maneuver_correct / total if maneuver_correct else 0

    console.print("[bold cyan]Results[/bold cyan]")
    console.print(f"  Total predictions: {len(predictions_list)}")
    console.print(f"  Matched to labels: {matched}")
    if total > 0:
        acc = (maneuver_correct / total) * 100
        console.print(f"  Maneuver accuracy: {maneuver_correct}/{total} ({acc:.1f}%)")
    console.print()

    # Confusion matrix (if we have maneuver predictions)
    if maneuver_confusion:
        console.print("[bold cyan]Maneuver Confusion Matrix[/bold cyan]")
        all_maneuvers = sorted(set(maneuver_confusion.keys()) |
                               set(m for v in maneuver_confusion.values() for m in v.keys()))

        table = Table(show_header=True, header_style="bold")
        table.add_column("True \\ Pred", style="dim")
        for m in all_maneuvers:
            table.add_column(m[:8], justify="center")

        for true_m in all_maneuvers:
            row = [true_m]
            for pred_m in all_maneuvers:
                count = maneuver_confusion[true_m].get(pred_m, 0)
                if count > 0:
                    style = "green" if true_m == pred_m else "red"
                    row.append(f"[{style}]{count}[/{style}]")
                else:
                    row.append("-")
            table.add_row(*row)

        console.print(table)
        console.print()

    # Save to file if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        console.print(f"Metrics saved to: {output}")


@app.command(rich_help_panel="Data")
def augment(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Input frames directory"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output base directory"),
    augmentations: str = typer.Option(
        "night,fog,rain,motion_blur",
        "--augmentations", "-a",
        help="Comma-separated augmentation types: night,fog,rain,motion_blur"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit number of frames to process (for testing)"
    ),
):
    """Apply image augmentations to frames.

    Creates augmented versions of frames for training on adverse conditions.

    Examples:
        # All augmentations
        navbuddy augment --input ./data/brisbane/frames --output ./data/brisbane

        # Specific augmentations
        navbuddy augment --input ./data/brisbane/frames --output ./data/brisbane -a night,rain

        # Test on first 10 frames
        navbuddy augment --input ./data/brisbane/frames --output ./data/brisbane --limit 10
    """
    from navbuddy.augment import augment_dataset, AugmentationType
    import cv2

    if not input_dir.exists():
        console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    # Parse augmentation types
    aug_list = [a.strip() for a in augmentations.split(",")]
    valid_types = ["night", "fog", "rain", "motion_blur"]
    for a in aug_list:
        if a not in valid_types:
            console.print(f"[red]Error: Invalid augmentation type: {a}[/red]")
            console.print(f"Valid types: {', '.join(valid_types)}")
            raise typer.Exit(1)

    console.print(f"[bold]NavBuddy Image Augmentation[/bold]")
    console.print(f"  Input: {input_dir}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Augmentations: {', '.join(aug_list)}")
    if limit:
        console.print(f"  Limit: {limit} frames")
    console.print()

    # Get image files
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    if limit:
        image_files = image_files[:limit]

    if not image_files:
        console.print("[red]Error: No images found in input directory[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(image_files)} images to process")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        results = {}

        for aug_type in aug_list:
            task = progress.add_task(f"Applying {aug_type}...", total=None)

            # Create output directory
            aug_output_dir = output_dir / f"frames_{aug_type}"
            aug_output_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for i, img_path in enumerate(image_files):
                progress.update(task, description=f"[{aug_type}] {i+1}/{len(image_files)}")

                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Apply augmentation
                from navbuddy.augment import augment_frame
                augmented = augment_frame(img, aug_type)

                # Save
                output_path = aug_output_dir / img_path.name
                cv2.imwrite(str(output_path), augmented)
                count += 1

            results[aug_type] = count
            progress.update(task, description=f"[{aug_type}] Done: {count} frames")

    console.print()
    console.print("[green]Augmentation complete![/green]")
    for aug_type, count in results.items():
        console.print(f"  {aug_type}: {count} frames -> {output_dir}/frames_{aug_type}/")


@app.command(rich_help_panel="Evaluation")
def rank(
    inputs: Optional[str] = typer.Option(
        None, "--inputs", "-i",
        help="Comma-separated paths to inference result files. Omit to auto-discover from --results-dir."
    ),
    results_dir: Optional[Path] = typer.Option(
        None, "--results-dir", "-r",
        help="Directory to auto-discover result files from (e.g. results/). Used when --inputs is not given."
    ),
    samples: Path = typer.Option(
        ..., "--samples", "-s",
        help="Path to samples.jsonl"
    ),
    output: Path = typer.Option(
        Path("rankings.jsonl"), "--output", "-o",
        help="Output file for rankings"
    ),
    judge: str = typer.Option(
        "openai/gpt-4o", "--judge", "-j",
        help="Judge model ID (e.g., openai/gpt-4o)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n",
        help="Maximum samples to judge"
    ),
    mode: str = typer.Option(
        "pairwise", "--mode", "-m",
        help="Judge mode: 'pairwise' (rank all together) or 'pointwise' (score each independently)"
    ),
    data_root: Optional[Path] = typer.Option(
        None, "--data-root", "-d",
        help="Path to data directory (e.g., data/brisbane). When provided, the judge sees the same images the VLMs saw."
    ),
    route: Optional[str] = typer.Option(
        None, "--route",
        help="Filter to samples from this route_id only (e.g. brisbane_routex3kg7736w)."
    ),
    sample_ids: Optional[str] = typer.Option(
        None, "--sample-ids",
        help="Comma-separated sample IDs to judge (overrides --route)."
    ),
    modality_filter: Optional[str] = typer.Option(
        None, "--modality",
        help="Only include results with this modality (e.g. 'video + prior')."
    ),
):
    """Rank VLM outputs using LLM-as-judge.

    Takes multiple inference result files and ranks them using a judge model.
    Result files can be specified explicitly (--inputs) or auto-discovered from
    a directory (--results-dir).

    Modes:
        pairwise  — All outputs sent in one prompt, ranked comparatively (default).
        pointwise — Each output scored independently against a fixed rubric.
                    Eliminates positional bias and comparative anchoring.
                    Cost: N_models * N_samples calls (vs N_samples for pairwise).

    Examples:
        # Auto-discover results for a route and judge pairwise
        navbuddy rank -r results/ -s data/brisbane/samples.jsonl \\
            --route brisbane_routex3kg7736w -d data/brisbane -o rankings_route.jsonl

        # Explicit files, pointwise
        navbuddy rank -i results_gpt4o.jsonl,results_gemini.jsonl \\
            -s samples.jsonl --mode pointwise -o rankings_pointwise.jsonl
    """
    import glob as glob_mod
    from navbuddy.eval.judge import run_judge_pipeline, run_pointwise_judge_pipeline

    if not inputs and not results_dir:
        console.print("[red]Error: provide either --inputs or --results-dir[/red]")
        raise typer.Exit(1)

    # Resolve result files
    if inputs:
        input_files = [Path(p.strip()) for p in inputs.split(",")]
    else:
        pattern = str(results_dir / "results_*.jsonl")
        input_files = sorted(Path(p) for p in glob_mod.glob(pattern))
        if not input_files:
            console.print(f"[red]Error: no results_*.jsonl files found in {results_dir}[/red]")
            raise typer.Exit(1)

    # Validate files exist
    missing = [f for f in input_files if not f.exists()]
    if missing:
        console.print(f"[red]Error: {len(missing)} file(s) not found, e.g.: {missing[0]}[/red]")
        raise typer.Exit(1)

    # Apply modality filter — drop files whose first result doesn't match
    if modality_filter:
        import json as _json
        filtered = []
        for f in input_files:
            try:
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            d = _json.loads(line)
                            if d.get("modality", "") == modality_filter:
                                filtered.append(f)
                            break
            except Exception:
                pass
        input_files = filtered

    if not samples.exists():
        console.print(f"[red]Error: Samples file not found: {samples}[/red]")
        raise typer.Exit(1)

    if mode not in ("pairwise", "pointwise"):
        console.print(f"[red]Error: Invalid mode '{mode}'. Use 'pairwise' or 'pointwise'.[/red]")
        raise typer.Exit(1)

    # Resolve sample_ids filter
    import json as _json2
    filter_ids: Optional[list] = None
    if sample_ids:
        filter_ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
    elif route:
        # Load all sample IDs from samples file that belong to this route
        filter_ids = []
        with open(samples, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s = _json2.loads(line)
                if s.get("route_id") == route or s.get("id", "").startswith(route):
                    filter_ids.append(s["id"])
        if not filter_ids:
            console.print(f"[yellow]Warning: no samples found for route '{route}'[/yellow]")

    console.print(f"[bold]NavBuddy LLM Judge ({mode})[/bold]")
    console.print(f"  Samples: {samples}")
    console.print(f"  Result files: {len(input_files)}")
    if route:
        console.print(f"  Route filter: {route} ({len(filter_ids or [])} samples)")
    if sample_ids:
        console.print(f"  Sample filter: {len(filter_ids or [])} IDs")
    if modality_filter:
        console.print(f"  Modality filter: {modality_filter}")
    console.print(f"  Judge: {judge}")
    console.print(f"  Mode: {mode}")
    console.print(f"  Images: {'yes (' + str(data_root) + ')' if data_root else 'no (text-only)'}")
    console.print(f"  Output: {output}")
    if limit:
        console.print(f"  Limit: {limit} samples")
    console.print()

    try:
        if mode == "pointwise":
            results = run_pointwise_judge_pipeline(
                samples_file=samples,
                result_files=input_files,
                output_file=output,
                judge_model=judge,
                limit=limit,
                verbose=True,
                data_root=data_root,
            )
        else:
            results = run_judge_pipeline(
                samples_file=samples,
                result_files=input_files,
                output_file=output,
                judge_model=judge,
                limit=limit,
                verbose=True,
                data_root=data_root,
                sample_ids=filter_ids,
            )

        # Summary
        successful = sum(1 for r in results if not r.error)
        failed = sum(1 for r in results if r.error)

        console.print()
        console.print(f"[green]Ranking complete![/green]")
        console.print(f"  Judged: {len(results)}")
        console.print(f"  Successful: {successful}")
        console.print(f"  Failed: {failed}")
        console.print(f"  Output: {output}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _detect_modality(result_files) -> Optional[str]:
    """Detect modality bucket from result file names."""
    names = [f.name for f in result_files if f]
    if any("_augmented_" in n for n in names):
        return "augmented"
    if any("_video_prior" in n for n in names):
        return "video_prior"
    if any("_image_prior_normal" in n for n in names):
        return "image_prior"
    if any("_prior_only" in n for n in names):
        return "prior_only"
    return None


@app.command(rich_help_panel="Evaluation")
def benchmark(
    inputs: Optional[str] = typer.Option(
        None, "--inputs", "-i",
        help="Comma-separated paths to inference result files. Omit to auto-discover from --results-dir."
    ),
    results_dir: Optional[Path] = typer.Option(
        None, "--results-dir", "-r",
        help="Directory to auto-discover results_*.jsonl files from (e.g. results/)."
    ),
    route: Optional[str] = typer.Option(
        None, "--route",
        help="Filter to samples from this route_id only (e.g. brisbane_routex3kg7736w)."
    ),
    sample_ids_str: Optional[str] = typer.Option(
        None, "--sample-ids",
        help="Comma-separated sample IDs to judge (overrides --route)."
    ),
    samples: Path = typer.Option(
        ..., "--samples", "-s",
        help="Path to samples.jsonl"
    ),
    output: Path = typer.Option(
        Path("results/benchmark.jsonl"), "--output", "-o",
        help="Output JSONL file for benchmark comparisons"
    ),
    models: Optional[str] = typer.Option(
        None, "--models",
        help="Comma-separated model IDs to include (default: all from result files)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n",
        help="Max samples per pair"
    ),
    k_factor: float = typer.Option(
        32.0, "--k-factor",
        help="Elo K-factor (higher = more volatile ratings)"
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for A/B position assignment"
    ),
    data_root: Optional[Path] = typer.Option(
        None, "--data-root", "-d",
        help="Data directory for images (enables visual grounding for judges)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show pair count and cost estimate without running"
    ),
    include_human: bool = typer.Option(
        False, "--include-human",
        help="Include human custom labels as a competitor"
    ),
    human_labels: Optional[Path] = typer.Option(
        None, "--human-labels",
        help="Path to custom labels JSONL (default: results/custom_labels.jsonl)"
    ),
    annotator_name: str = typer.Option(
        "annotator", "--annotator-name",
        help="Name for the human annotator, e.g. 'aron' → model_id 'human/aron'"
    ),
    include_baseline: bool = typer.Option(
        False, "--include-baseline",
        help="Include original Maps+OSM instructions as a baseline competitor"
    ),
    max_elo_gap: int = typer.Option(
        600, "--max-elo-gap",
        help="Skip pairs with Elo gap > this (0 = no limit). Like chess matchmaking."
    ),
    gt_only: bool = typer.Option(
        False, "--gt-only",
        help="Only judge samples that have a manual GT label"
    ),
    gt_file: Optional[Path] = typer.Option(
        None, "--gt-file",
        help="Path to ground_truth.jsonl (default: results/ground_truth.jsonl next to output)"
    ),
    gt_weight: float = typer.Option(
        1.0, "--gt-weight",
        help="GT samples are this many times more likely to be sampled (e.g. 3.0)"
    ),
    total_limit: Optional[int] = typer.Option(
        None, "--total-limit",
        help="Global cap on total comparisons across all pairs"
    ),
    judge_panel: Optional[str] = typer.Option(
        None, "--judge-panel",
        help="Comma-separated judge model IDs (default: claude-sonnet-4.6, gpt-5.2, gemini-3-flash)"
    ),
):
    """Run cross-company pairwise A/B benchmark with 3-judge Elo ranking."""
    from navbuddy.eval.company_groups import (
        estimate_benchmark_cost,
        generate_cross_company_pairs,
        group_models_by_company,
    )
    from navbuddy.eval.judge import (
        baseline_instructions_to_results,
        human_labels_to_results,
        load_inference_results,
        run_benchmark_pipeline,
    )
    from navbuddy.eval.schemas import BenchmarkConfig

    import glob as _glob_mod

    try:
        if not inputs and not results_dir:
            console.print("[red]Error: provide either --inputs or --results-dir[/red]")
            raise typer.Exit(1)

        # Parse input files
        if inputs:
            result_files = [Path(p.strip()) for p in inputs.split(",")]
        else:
            pattern = str(results_dir / "results_*.jsonl")
            result_files = sorted(Path(p) for p in _glob_mod.glob(pattern))
            if not result_files:
                console.print(f"[red]Error: no results_*.jsonl files found in {results_dir}[/red]")
                raise typer.Exit(1)

        for rf in result_files:
            if not rf.exists():
                console.print(f"[red]Result file not found: {rf}[/red]")
                raise typer.Exit(1)

        if not samples.exists():
            console.print(f"[red]Samples file not found: {samples}[/red]")
            raise typer.Exit(1)

        # Resolve sample ID filter
        filter_ids: Optional[list] = None
        if sample_ids_str:
            filter_ids = [s.strip() for s in sample_ids_str.split(",") if s.strip()]
        elif route:
            filter_ids = []
            with open(samples, encoding="utf-8") as _fh:
                for _line in _fh:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _s = json.loads(_line)
                    if _s.get("route_id") == route or _s.get("id", "").startswith(route):
                        filter_ids.append(_s["id"])
            if not filter_ids:
                console.print(f"[yellow]Warning: no samples found for route '{route}'[/yellow]")

        # Discover available models
        available_models = []
        for rf in result_files:
            results = load_inference_results(rf)
            if results:
                first = next(iter(results.values()))
                model_id = first.get("model_id", rf.stem)
                if model_id not in available_models:
                    available_models.append(model_id)

        # Load human labels if requested
        human_results = None
        if include_human:
            human_path = human_labels or Path("results/custom_labels.jsonl")
            human_results = human_labels_to_results(human_path, annotator_name=annotator_name)
            if human_results:
                human_model_id = f"human/{annotator_name}"
                available_models.append(human_model_id)
                console.print(f"  Human labels: {len(human_results)} samples from {human_path} (model_id: {human_model_id})")
            else:
                console.print("[yellow]Warning: --include-human but no custom labels found[/yellow]")

        # Load baseline instructions if requested
        baseline_results = None
        if include_baseline:
            baseline_results = baseline_instructions_to_results(samples)
            if baseline_results:
                available_models.append("maps-osm/baseline")
                console.print(f"  Baseline (Maps+OSM): {len(baseline_results)} samples")
            else:
                console.print("[yellow]Warning: --include-baseline but no samples found[/yellow]")

        # Filter to requested subset
        model_ids = None
        if models:
            model_ids = [m.strip() for m in models.split(",")]
            missing = [m for m in model_ids if m not in available_models]
            if missing:
                console.print(f"[yellow]Warning: models not found in results: {missing}[/yellow]")
            available_models = [m for m in available_models if m in model_ids]

        # Group and generate pairs
        groups = group_models_by_company(available_models)
        pairs = generate_cross_company_pairs(available_models, seed=seed)

        # Count same-company pairs
        same_company = sum(
            len(models_list) * (len(models_list) - 1) // 2
            for models_list in groups.values()
        )

        # Count available samples
        import json as _json
        sample_count = 0
        with open(samples, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample_count += 1
        if limit:
            sample_count = min(sample_count, limit)

        # Check for existing comparisons (dedup)
        from navbuddy.eval.judge import load_existing_comparisons
        existing_keys = load_existing_comparisons(output)

        console.print()
        console.print("[bold]NavBuddy A/B Benchmark[/bold]")
        if route:
            console.print(f"  Route filter: {route} ({len(filter_ids or [])} samples)")
        elif filter_ids:
            console.print(f"  Sample filter: {len(filter_ids)} IDs")
        console.print(f"  Result files: {len(result_files)}")
        console.print(f"  Models: {len(available_models)} (from {len(groups)} companies)")
        for company, model_list in sorted(groups.items()):
            names = ", ".join(m.split("/")[-1] for m in model_list)
            console.print(f"    {company}: {names}")
        console.print(f"  Cross-company pairs: {len(pairs)}")
        console.print(f"  Samples per pair: up to {sample_count}")

        cost = estimate_benchmark_cost(
            n_models=len(available_models),
            n_companies=len(groups),
            n_samples=sample_count,
            same_company_pairs=same_company,
        )
        console.print(f"  Total judge calls: {cost['total_judge_calls']:,} (3 judges x {cost['total_comparisons']:,} comparisons)")

        if existing_keys:
            console.print(f"  Already judged: [green]{len(existing_keys)}[/green] (will skip)")
            remaining = cost['total_comparisons'] - len(existing_keys)
            remaining = max(remaining, 0)
            console.print(f"  New comparisons: ~{remaining:,}")
            console.print(f"  Estimated cost: ~${remaining * 3 * 0.025:.0f} (new only)")
        else:
            console.print(f"  Estimated cost: ~${cost['estimated_cost_usd']:.0f}")

        if dry_run:
            console.print("\n[yellow]Dry run — no API calls made.[/yellow]")
            return

        console.print()

        config_kwargs: dict = dict(
            k_factor=k_factor,
            seed=seed,
            include_images=data_root is not None,
            max_elo_gap=max_elo_gap,
            gt_only=gt_only,
            gt_weight=gt_weight,
            total_limit=total_limit,
        )
        if judge_panel:
            config_kwargs["judge_panel"] = [j.strip() for j in judge_panel.split(",")]
        config = BenchmarkConfig(**config_kwargs)

        # Load ground truth for gt_only filter and/or gt_weight sampling
        gt_data = None
        if gt_only or gt_weight > 1.0:
            gt_path = gt_file or (output.parent / "ground_truth.jsonl")
            if gt_path.exists():
                gt_data = {}
                with open(gt_path, encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line:
                            entry = json.loads(_line)
                            gt_data[entry["sample_id"]] = entry
                console.print(f"  GT entries loaded: {len(gt_data)}")
            else:
                console.print(f"[yellow]Warning: GT file not found at {gt_path}[/yellow]")

        extra = {}
        if human_results:
            extra[f"human/{annotator_name}"] = human_results
        if baseline_results:
            extra["maps-osm/baseline"] = baseline_results
        extra = extra or None

        detected_modality = _detect_modality(result_files)

        run = run_benchmark_pipeline(
            samples_file=samples,
            result_files=result_files,
            output_file=output,
            config=config,
            model_ids=model_ids,
            limit=limit,
            verbose=True,
            data_root=data_root,
            extra_results=extra,
            ground_truth=gt_data,
            sample_ids=filter_ids,
            modality=detected_modality,
        )

        console.print()
        console.print(f"[green]Benchmark complete![/green]")
        console.print(f"  Output: {output}")
        console.print(f"  Total comparisons in file: {len(run.comparisons)}")
        console.print(f"  New this run: {run.successful_comparisons} successful, {run.failed_comparisons} failed")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command(rich_help_panel="Data")
def play(
    route_id: str = typer.Argument(..., help="Route ID to play"),
    data_root: Path = typer.Option(
        Path("./data"), "--data-root", "-d",
        help="Data root directory"
    ),
    static: bool = typer.Option(
        False, "--static",
        help="Static display (no interactivity)"
    ),
):
    """Play a route with frames and instructions.

    Interactive TUI for previewing routes.

    Examples:
        # Interactive player
        navbuddy play 4000_4006_X3KG7736W

        # Static summary
        navbuddy play 4000_4006_X3KG7736W --static

        # Custom data directory
        navbuddy play 2000_2000_J4MNCG01R -d ./data/brisbane
    """
    from navbuddy.player import play_route, list_routes

    if not data_root.exists():
        console.print(f"[red]Error: Data root not found: {data_root}[/red]")
        raise typer.Exit(1)

    # Check if route exists
    routes_dir = data_root / "routes" / route_id
    if not routes_dir.exists():
        console.print(f"[red]Error: Route not found: {route_id}[/red]")
        console.print()
        console.print("[yellow]Available routes:[/yellow]")
        for r in list_routes(data_root)[:10]:
            console.print(f"  - {r}")
        routes = list_routes(data_root)
        if len(routes) > 10:
            console.print(f"  ... and {len(routes) - 10} more")
        raise typer.Exit(1)

    play_route(route_id, data_root, interactive=not static, console=console)


@app.command("list-routes", rich_help_panel="Data")
def list_routes_cmd(
    data_root: Path = typer.Option(
        Path("./data"), "--data-root", "-d",
        help="Data root directory"
    ),
):
    """List available routes.

    Examples:
        navbuddy list-routes
        navbuddy list-routes -d ./data/brisbane
    """
    from navbuddy.player import list_routes

    if not data_root.exists():
        console.print(f"[red]Error: Data root not found: {data_root}[/red]")
        raise typer.Exit(1)

    routes = list_routes(data_root)

    if not routes:
        console.print("[yellow]No routes found.[/yellow]")
        return

    console.print(f"[bold]Available Routes ({len(routes)}):[/bold]")
    for route in routes:
        console.print(f"  - {route}")


@app.command("purge-routes", rich_help_panel="Data")
def purge_routes(
    data_dir: Path = typer.Option(..., "--data-dir", "-d", help="City data directory (e.g. ./data/brisbane)"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by metadata.source (e.g. 'google')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without applying"),
):
    """Delete routes matching a filter from samples.jsonl, frames, maps, routes/, and results.

    Examples:
        navbuddy purge-routes -d ./data/brisbane --source google --dry-run
        navbuddy purge-routes -d ./data/brisbane --source google
    """
    import re
    import shutil

    ROOT = Path(__file__).parent.parent
    RESULTS_DIR = ROOT / "results"

    samples_path = data_dir / "samples.jsonl"
    if not samples_path.exists():
        console.print(f"[red]No samples.jsonl found at {samples_path}[/red]")
        raise typer.Exit(1)

    # Load all samples, split into keep vs purge
    all_samples = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))

    def matches(s: dict) -> bool:
        if source:
            return s.get("metadata", {}).get("source") == source
        return False

    purge_samples = [s for s in all_samples if matches(s)]
    keep_samples = [s for s in all_samples if not matches(s)]

    purge_route_ids = sorted({s["route_id"] for s in purge_samples})
    purge_sample_ids = {s["id"] for s in purge_samples}

    console.print(f"[bold]Purge summary:[/bold]")
    console.print(f"  Total samples: {len(all_samples)}")
    console.print(f"  Matching (to purge): {len(purge_samples)} samples across {len(purge_route_ids)} routes")
    console.print(f"  Keeping: {len(keep_samples)} samples")

    if not purge_route_ids:
        console.print("[yellow]Nothing to purge.[/yellow]")
        return

    if dry_run:
        console.print("\n[yellow]=== DRY RUN — no files will be modified ===[/yellow]\n")
        for rid in purge_route_ids[:5]:
            console.print(f"  route: {rid}")
        if len(purge_route_ids) > 5:
            console.print(f"  ... and {len(purge_route_ids) - 5} more")

    # 1. Rewrite samples.jsonl
    if not dry_run:
        backup = samples_path.with_suffix(".jsonl.bak_purge")
        shutil.copy2(samples_path, backup)
        samples_path.write_text("\n".join(json.dumps(s) for s in keep_samples) + "\n")
        console.print(f"[green]samples.jsonl:[/green] removed {len(purge_samples)} entries (backup: {backup.name})")
    else:
        console.print(f"  [dim]samples.jsonl: would remove {len(purge_samples)} entries[/dim]")

    # 2. Delete frame files
    frames_dir = data_dir / "frames"
    frame_count = 0
    if frames_dir.exists():
        for rid in purge_route_ids:
            for f in sorted(frames_dir.glob(f"{rid}_step*.jpg")):
                frame_count += 1
                if not dry_run:
                    f.unlink()
    if not dry_run:
        console.print(f"[green]frames/:[/green] deleted {frame_count} files")
    else:
        console.print(f"  [dim]frames/: would delete {frame_count} files[/dim]")

    # 3. Delete map files
    maps_dir = data_dir / "maps"
    map_count = 0
    if maps_dir.exists():
        for rid in purge_route_ids:
            for f in sorted(maps_dir.glob(f"{rid}_step*_map.png")):
                map_count += 1
                if not dry_run:
                    f.unlink()
    if not dry_run:
        console.print(f"[green]maps/:[/green] deleted {map_count} files")
    else:
        console.print(f"  [dim]maps/: would delete {map_count} files[/dim]")

    # 4. Delete route directories
    routes_dir = data_dir / "routes"
    route_dir_count = 0
    if routes_dir.exists():
        for rid in purge_route_ids:
            rd = routes_dir / rid
            if rd.exists():
                route_dir_count += 1
                if not dry_run:
                    shutil.rmtree(rd)
    if not dry_run:
        console.print(f"[green]routes/:[/green] deleted {route_dir_count} directories")
    else:
        console.print(f"  [dim]routes/: would delete {route_dir_count} directories[/dim]")

    # 5. Strip from results files
    step_re = re.compile(r"^(.+)_(step\d+)$")
    total_results_removed = 0
    for results_file in sorted(RESULTS_DIR.glob("results_*.jsonl")):
        lines = results_file.read_text().splitlines()
        new_lines = []
        removed = 0
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("id") in purge_sample_ids:
                removed += 1
            else:
                new_lines.append(line)
        if removed:
            total_results_removed += removed
            if not dry_run:
                results_file.write_text("\n".join(new_lines) + "\n")
                console.print(f"[green]{results_file.name}:[/green] removed {removed} entries")
            else:
                console.print(f"  [dim]{results_file.name}: would remove {removed} entries[/dim]")

    # 6. Strip from ground_truth.jsonl
    gt_path = RESULTS_DIR / "ground_truth.jsonl"
    if gt_path.exists():
        lines = gt_path.read_text().splitlines()
        new_lines = []
        gt_removed = 0
        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("sample_id") in purge_sample_ids:
                gt_removed += 1
            else:
                new_lines.append(line)
        if gt_removed:
            if not dry_run:
                shutil.copy2(gt_path, gt_path.with_suffix(".jsonl.bak_purge"))
                gt_path.write_text("\n".join(new_lines) + "\n")
                console.print(f"[green]ground_truth.jsonl:[/green] removed {gt_removed} entries")
            else:
                console.print(f"  [dim]ground_truth.jsonl: would remove {gt_removed} entries[/dim]")

    if dry_run:
        console.print("\n[yellow]=== Dry run complete. Run without --dry-run to apply. ===[/yellow]")
    else:
        console.print("\n[bold green]Purge complete.[/bold green] Restart the dashboard backend to reload data.")


@app.command("export-manifest", rich_help_panel="Data")
def export_manifest_cmd(
    data_root: Path = typer.Option(
        Path("./data"), "--data-root", "-d",
        help="Data root directory"
    ),
    output: Path = typer.Option(
        Path("manifest.json"), "--output", "-o",
        help="Output manifest file"
    ),
    name: str = typer.Option(
        "navbuddy-dataset", "--name", "-n",
        help="Dataset name"
    ),
    description: str = typer.Option(
        "Navigation VLM training dataset", "--description",
        help="Dataset description"
    ),
    manifest_frame_profile: str = typer.Option(
        "sparse4",
        "--manifest-frame-profile",
        help="Manifest frame profile: 'sparse4' (default) or 'all'",
    ),
):
    """Export dataset manifest (no images).

    Creates a manifest file that others can use to download the dataset
    with their own Google Maps API key.

    Examples:
        navbuddy export-manifest -o manifest.json
        navbuddy export-manifest -d ./data/brisbane -o navbuddy_v3.json
    """
    from navbuddy.manifest import export_manifest

    if not data_root.exists():
        console.print(f"[red]Error: Data root not found: {data_root}[/red]")
        raise typer.Exit(1)
    if manifest_frame_profile not in {"sparse4", "all"}:
        console.print("[red]Error: --manifest-frame-profile must be 'sparse4' or 'all'[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Export Dataset Manifest[/bold]")
    console.print(f"  Data root: {data_root}")
    console.print(f"  Output: {output}")
    console.print()

    try:
        manifest = export_manifest(
            data_root=data_root,
            output_file=output,
            name=name,
            description=description,
            manifest_frame_profile=manifest_frame_profile,
        )

        console.print(f"[green]Manifest exported![/green]")
        console.print(f"  Routes: {manifest.routes_count}")
        console.print(f"  Samples: {manifest.samples_count}")
        console.print(f"  Frames: {manifest.total_frames}")
        console.print(f"  Frame profile: {manifest_frame_profile}")
        console.print(f"  Output: {output}")
        console.print()
        console.print("[dim]Share this manifest file. Users can download images with:[/dim]")
        console.print(f"[cyan]  navbuddy download-manifest -m {output} --api-key YOUR_KEY[/cyan]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("prepare-training", rich_help_panel="Training (coming soon)")
def prepare_training():
    """Prepare SFT/DPO training data from judge rankings."""
    console.print("[yellow]Training commands are coming soon.[/yellow]")
    raise typer.Exit(0)


@app.command(rich_help_panel="Training (coming soon)")
def train():
    """Fine-tune a VLM with SFT, DPO, or GRPO."""
    console.print("[yellow]Training commands are coming soon.[/yellow]")
    raise typer.Exit(0)


@app.command(rich_help_panel="Data")
def stats(
    data_root: Path = typer.Option(
        Path("./data"), "--data-root", "-d",
        help="Data root directory"
    ),
):
    """Show dataset statistics.

    Examples:
        navbuddy stats
        navbuddy stats -d ./data/brisbane
    """
    import json
    from rich.table import Table

    if not data_root.exists():
        console.print(f"[red]Error: Data root not found: {data_root}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Dataset Statistics: {data_root}[/bold]")
    console.print()

    # Count routes
    routes_dir = data_root / "routes"
    routes = []
    if routes_dir.exists():
        routes = [d for d in routes_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]

    # Count frames
    frames_dir = data_root / "frames"
    frames_count = 0
    if frames_dir.exists():
        frames_count = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))

    # Count augmented frames (check both 'frames_{type}' and 'augments/{type}' dirs)
    aug_counts = {}
    for aug_type in ["night", "fog", "rain", "motion_blur"]:
        count = 0
        for pattern in [f"frames_{aug_type}", f"augments/{aug_type}"]:
            aug_dir = data_root / pattern
            if aug_dir.exists():
                count += len(list(aug_dir.glob("*.jpg"))) + len(list(aug_dir.glob("*.png")))
        if count > 0:
            aug_counts[aug_type] = count

    # Count OSM maps (could be in 'maps' or 'osm_maps' directory)
    osm_count = 0
    for map_dir_name in ["maps", "osm_maps"]:
        osm_dir = data_root / map_dir_name
        if osm_dir.exists():
            osm_count += len(list(osm_dir.glob("*.png")))

    # Load samples
    samples_file = data_root / "samples.jsonl"
    samples = []
    maneuvers = {}
    if samples_file.exists():
        with open(samples_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    samples.append(s)
                    m = s.get("maneuver", "UNKNOWN")
                    maneuvers[m] = maneuvers.get(m, 0) + 1

    # Overall summary
    console.print("[bold cyan]Overview[/bold cyan]")
    console.print(f"  Routes: {len(routes)}")
    console.print(f"  Samples: {len(samples)}")
    console.print(f"  Frames: {frames_count}")
    console.print(f"  OSM Maps: {osm_count}")
    if aug_counts:
        console.print(f"  Augmented frames: {sum(aug_counts.values())}")
    console.print()

    # Maneuver distribution
    if maneuvers:
        console.print("[bold cyan]Maneuvers[/bold cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Maneuver", style="yellow")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")

        total = sum(maneuvers.values())
        for m, count in sorted(maneuvers.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100
            table.add_row(m, str(count), f"{pct:.1f}%")

        console.print(table)
        console.print()

    # Augmentation breakdown
    if aug_counts:
        console.print("[bold cyan]Augmentations[/bold cyan]")
        for aug_type, count in sorted(aug_counts.items()):
            console.print(f"  {aug_type}: {count} frames")
        console.print()

    # Check for result files
    result_files = list(data_root.parent.glob("results_*.jsonl")) + list(Path(".").glob("results_*.jsonl"))
    if result_files:
        console.print("[bold cyan]Inference Results[/bold cyan]")
        for rf in result_files:
            count = sum(1 for _ in open(rf, encoding="utf-8"))
            console.print(f"  {rf.name}: {count} results")
        console.print()

    # Check for rankings
    ranking_files = list(data_root.parent.glob("rankings*.jsonl")) + list(Path(".").glob("rankings*.jsonl"))
    if ranking_files:
        console.print("[bold cyan]Rankings[/bold cyan]")
        for rf in ranking_files:
            count = sum(1 for _ in open(rf, encoding="utf-8"))
            console.print(f"  {rf.name}: {count} ranked samples")
        console.print()


        raise typer.Exit(1)



@app.command("assign-splits", rich_help_panel="Training (coming soon)")
def assign_splits():
    """Assign train/val/test splits to samples."""
    console.print("[yellow]Training commands are coming soon.[/yellow]")
    raise typer.Exit(0)


@app.command("metric-eval", rich_help_panel="Evaluation")
def metric_eval(
    dataset: Path = typer.Option(
        ..., "-d", "--dataset", help="Path to samples.jsonl"
    ),
    results_dir: Path = typer.Option(
        Path("results"), "--results-dir", "-r", help="Directory with result JSONL files"
    ),
    ground_truth: Path = typer.Option(
        Path("results/ground_truth.jsonl"), "--gt", help="Path to ground_truth.jsonl"
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output JSONL path for detailed scores"
    ),
):
    """Score model results against human ground-truth labels (no AI calls)."""
    from navbuddy.eval.metric_eval import run_metric_eval

    if not ground_truth.exists():
        console.print(f"[red]Ground truth file not found:[/red] {ground_truth}")
        raise typer.Exit(1)
    if not results_dir.exists():
        console.print(f"[red]Results directory not found:[/red] {results_dir}")
        raise typer.Exit(1)

    eval_results = run_metric_eval(
        results_dir=results_dir,
        ground_truth_path=ground_truth,
        samples_path=dataset,
        output_path=output,
        verbose=True,
    )

    console.print(f"\n[green]Done.[/green] Scored {len(eval_results)} model×sample pairs.")


@app.command("index-embeddings", rich_help_panel="Evaluation")
def index_embeddings(
    results: List[Path] = typer.Argument(
        ..., help="One or more results JSONL files to index"
    ),
    namespace: str = typer.Option(
        "instructions",
        "--namespace",
        help="Upstash Vector namespace to index into",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="After indexing, run this similarity query and print top results",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Number of results to return for --query",
    ),
):
    """Index instruction embeddings from results JSONL files into Upstash Vector.

    Indexes enhanced_instruction fields for semantic search and DPO hard negative mining.
    Requires UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN environment variables.

    Examples:
        navbuddy index-embeddings results/results_*.jsonl
        navbuddy index-embeddings results/results_gemini.jsonl --query "turn left at lights"
    """
    from navbuddy.eval.embeddings import InstructionIndex

    idx = InstructionIndex.from_env(namespace=namespace)

    total = 0
    for path in results:
        if not path.exists():
            console.print(f"[yellow]Skipping missing file: {path}[/yellow]")
            continue
        n = idx.index_jsonl(str(path))
        console.print(f"  Indexed {n} entries from {path.name}")
        total += n

    console.print(f"\n[green]Done.[/green] Total indexed: {total}, collection size: {idx.count()}")

    if query:
        console.print(f"\nQuery: [bold]{query}[/bold]")
        hits = idx.find_similar(query, n=top_k)
        for i, hit in enumerate(hits):
            text_preview = (hit.metadata or {}).get("text", "")[:120]
            console.print(f"  [{i+1}] score={hit.score:.3f}  id={hit.id}")
            console.print(f"        {text_preview}")



def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
