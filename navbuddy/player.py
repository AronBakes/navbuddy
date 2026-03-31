"""Route Player TUI - Preview routes with frames and instructions.

A simple terminal-based route player for quick validation.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Try to import for image display (optional)
try:
    from rich_pixels import Pixels
    HAS_PIXELS = True
except ImportError:
    HAS_PIXELS = False


class RoutePlayer:
    """Terminal-based route player."""

    def __init__(
        self,
        route_id: str,
        data_root: Path,
        console: Optional[Console] = None,
    ):
        self.route_id = route_id
        self.data_root = Path(data_root)
        self.console = console or Console()

        # Load route data
        self.samples = self._load_samples()
        self.route_metadata = self._load_route_metadata()

        # Player state
        self.current_step = 0
        self.playing = False
        self.playback_speed = 1.0  # seconds per frame

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples for this route."""
        samples_file = self.data_root / "samples.jsonl"
        samples = []

        if samples_file.exists():
            with open(samples_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line)
                    if s.get("route_id") == self.route_id:
                        samples.append(s)

        # Sort by step index
        samples.sort(key=lambda x: x.get("step_index", 0))
        return samples

    def _load_route_metadata(self) -> Dict[str, Any]:
        """Load route metadata."""
        metadata_file = self.data_root / "routes" / self.route_id / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _get_frame_paths(self, sample: Dict[str, Any]) -> List[Path]:
        """Get absolute paths to frame images."""
        frames = sample.get("images", {}).get("frames", [])
        return [self.data_root / f for f in frames if f]

    def _get_map_path(self, sample: Dict[str, Any]) -> Optional[Path]:
        """Get absolute path to overhead map."""
        overhead = sample.get("images", {}).get("overhead")
        if overhead:
            return self.data_root / overhead
        return None

    def _build_step_panel(self, sample: Dict[str, Any]) -> Panel:
        """Build the step info panel."""
        step_idx = sample.get("step_index", 0)
        total_steps = len(self.samples)
        maneuver = sample.get("maneuver", "UNKNOWN")
        instruction = sample.get("prior", {}).get("instruction", "No instruction")

        # Distances
        distances = sample.get("distances", {})
        step_distance = distances.get("step_distance_m", 0)
        remaining = distances.get("remaining_distance_m", 0)

        # OSM road info
        osm = sample.get("osm_road", {}) or {}
        road_name = osm.get("name") or "Unknown road"
        highway = osm.get("highway") or ""
        lanes = osm.get("lanes")
        maxspeed = osm.get("maxspeed")

        # Build content
        content = Text()
        content.append(f"Step {step_idx + 1}/{total_steps}\n", style="bold cyan")
        content.append(f"Maneuver: ", style="dim")
        content.append(f"{maneuver}\n", style="bold yellow")
        content.append("\n")
        content.append(f"📍 {instruction}\n", style="bold white")
        content.append("\n")
        content.append(f"Road: {road_name}", style="dim")
        if highway:
            content.append(f" ({highway})", style="dim")
        content.append("\n")

        if lanes or maxspeed:
            content.append(f"Lanes: {lanes or '?'}", style="dim")
            if maxspeed:
                content.append(f" | Speed limit: {maxspeed} km/h", style="dim")
            content.append("\n")

        content.append("\n")
        content.append(f"Step distance: {step_distance}m\n", style="dim")
        content.append(f"Remaining: {remaining}m\n", style="dim")

        return Panel(content, title=f"[bold]{self.route_id}[/bold]", border_style="blue")

    def _build_frames_panel(self, sample: Dict[str, Any]) -> Panel:
        """Build the frames info panel."""
        frames = self._get_frame_paths(sample)
        map_path = self._get_map_path(sample)

        content = Text()
        content.append("Frames:\n", style="bold")

        if frames:
            for i, fp in enumerate(frames):
                exists = "✓" if fp.exists() else "✗"
                style = "green" if fp.exists() else "red"
                content.append(f"  {exists} ", style=style)
                content.append(f"{fp.name}\n", style="dim")
        else:
            content.append("  No frames\n", style="dim")

        content.append("\nOverhead Map:\n", style="bold")
        if map_path:
            exists = "✓" if map_path.exists() else "✗"
            style = "green" if map_path.exists() else "red"
            content.append(f"  {exists} ", style=style)
            content.append(f"{map_path.name}\n", style="dim")
        else:
            content.append("  No map\n", style="dim")

        return Panel(content, title="[bold]Assets[/bold]", border_style="green")

    def _build_controls_panel(self) -> Panel:
        """Build the controls help panel."""
        content = Text()
        content.append("Controls:\n", style="bold")
        content.append("  ← / →  ", style="cyan")
        content.append("Previous / Next step\n")
        content.append("  Space  ", style="cyan")
        content.append("Play / Pause\n")
        content.append("  +/-    ", style="cyan")
        content.append("Speed up / slow down\n")
        content.append("  o      ", style="cyan")
        content.append("Open frame in viewer\n")
        content.append("  m      ", style="cyan")
        content.append("Open map in viewer\n")
        content.append("  q      ", style="cyan")
        content.append("Quit\n")

        return Panel(content, title="[bold]Controls[/bold]", border_style="magenta")

    def _build_layout(self) -> Layout:
        """Build the full layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="main", ratio=3),
            Layout(name="footer", ratio=1),
        )

        layout["main"].split_row(
            Layout(name="step_info", ratio=2),
            Layout(name="assets", ratio=1),
        )

        return layout

    def render(self) -> Layout:
        """Render the current state."""
        if not self.samples:
            return Panel("No samples found for this route", style="red")

        sample = self.samples[self.current_step]
        layout = self._build_layout()

        layout["step_info"].update(self._build_step_panel(sample))
        layout["assets"].update(self._build_frames_panel(sample))
        layout["footer"].update(self._build_controls_panel())

        return layout

    def next_step(self):
        """Move to next step."""
        if self.current_step < len(self.samples) - 1:
            self.current_step += 1

    def prev_step(self):
        """Move to previous step."""
        if self.current_step > 0:
            self.current_step -= 1

    def open_frame(self):
        """Open current frame in system viewer."""
        if not self.samples:
            return

        sample = self.samples[self.current_step]
        frames = self._get_frame_paths(sample)

        if frames and frames[-1].exists():
            # Use system default image viewer
            import subprocess
            if sys.platform == "darwin":
                subprocess.run(["open", str(frames[-1])])
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(frames[-1])])
            elif sys.platform == "win32":
                os.startfile(str(frames[-1]))

    def open_map(self):
        """Open current map in system viewer."""
        if not self.samples:
            return

        sample = self.samples[self.current_step]
        map_path = self._get_map_path(sample)

        if map_path and map_path.exists():
            import subprocess
            if sys.platform == "darwin":
                subprocess.run(["open", str(map_path)])
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(map_path)])
            elif sys.platform == "win32":
                os.startfile(str(map_path))

    def run_interactive(self):
        """Run interactive TUI."""
        import select
        import termios
        import tty

        if not self.samples:
            self.console.print(f"[red]No samples found for route: {self.route_id}[/red]")
            return

        self.console.print(f"[bold]Route Player: {self.route_id}[/bold]")
        self.console.print(f"Found {len(self.samples)} steps")
        self.console.print()

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to raw mode
            tty.setcbreak(sys.stdin.fileno())

            with Live(self.render(), console=self.console, refresh_per_second=4) as live:
                last_auto_advance = time.time()

                while True:
                    # Check for input
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)

                        if key == "q":
                            break
                        elif key == " ":
                            self.playing = not self.playing
                        elif key in ("\x1b", "["):  # Arrow key prefix
                            # Read rest of escape sequence
                            if key == "\x1b":
                                key += sys.stdin.read(2)
                            if key == "\x1b[C":  # Right arrow
                                self.next_step()
                            elif key == "\x1b[D":  # Left arrow
                                self.prev_step()
                        elif key == "+":
                            self.playback_speed = max(0.1, self.playback_speed - 0.2)
                        elif key == "-":
                            self.playback_speed = min(5.0, self.playback_speed + 0.2)
                        elif key == "o":
                            self.open_frame()
                        elif key == "m":
                            self.open_map()

                    # Auto-advance if playing
                    if self.playing:
                        now = time.time()
                        if now - last_auto_advance >= self.playback_speed:
                            self.next_step()
                            last_auto_advance = now
                            if self.current_step >= len(self.samples) - 1:
                                self.playing = False

                    live.update(self.render())

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def run_static(self):
        """Run static display (no interactivity)."""
        if not self.samples:
            self.console.print(f"[red]No samples found for route: {self.route_id}[/red]")
            return

        self.console.print(f"[bold]Route: {self.route_id}[/bold]")
        self.console.print(f"Steps: {len(self.samples)}")

        if self.route_metadata:
            origin = self.route_metadata.get("origin", {})
            dest = self.route_metadata.get("destination", {})
            distance = self.route_metadata.get("total_distance_m", 0)
            self.console.print(f"Distance: {distance}m")
            self.console.print(f"Origin: {origin.get('lat', 0):.5f}, {origin.get('lng', 0):.5f}")
            self.console.print(f"Destination: {dest.get('lat', 0):.5f}, {dest.get('lng', 0):.5f}")

        self.console.print()

        # Print step summary table
        table = Table(title="Steps")
        table.add_column("#", style="dim")
        table.add_column("Maneuver", style="yellow")
        table.add_column("Instruction")
        table.add_column("Distance", justify="right")
        table.add_column("Frames", justify="right")

        for i, sample in enumerate(self.samples):
            maneuver = sample.get("maneuver", "?")
            instruction = sample.get("prior", {}).get("instruction", "")
            if len(instruction) > 40:
                instruction = instruction[:37] + "..."
            distance = sample.get("distances", {}).get("step_distance_m", 0)
            frames = len(sample.get("images", {}).get("frames", []))

            table.add_row(
                str(i),
                maneuver,
                instruction,
                f"{distance}m",
                str(frames),
            )

        self.console.print(table)


def list_routes(data_root: Path, console: Optional[Console] = None) -> List[str]:
    """List all available routes."""
    console = console or Console()
    routes_dir = data_root / "routes"

    if not routes_dir.exists():
        console.print("[red]No routes directory found[/red]")
        return []

    routes = []
    for d in sorted(routes_dir.iterdir()):
        if d.is_dir() and (d / "metadata.json").exists():
            routes.append(d.name)

    return routes


def play_route(
    route_id: str,
    data_root: Path,
    interactive: bool = True,
    console: Optional[Console] = None,
):
    """Play a route."""
    player = RoutePlayer(route_id, data_root, console)

    if interactive:
        player.run_interactive()
    else:
        player.run_static()


__all__ = [
    "RoutePlayer",
    "list_routes",
    "play_route",
]
