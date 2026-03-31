"""Canonical frame sampling for Street View generation.

All distances are *remaining metres to the maneuver point* (step end).
Two primitives cover every use case:

  clamp_targets  — fixed list of desired distances, clamped to step length.
                   Use for sparse4 ([100, 80, 60, 40]) or single ([40]).

  spaced_targets — walk from start_m down to end_m at a fixed spacing.
                   Use for video5m (every 5 m) or custom windowed sampling.

profile_distances() is the named dispatcher for CLI / config use.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

FRAME_PROFILES = {"single", "sparse4", "video5m", "custom"}
SPARSE4_TARGETS_REMAINING_M = (100, 80, 60, 40)
VIDEO5M_DEFAULT_SPACING_M = 5.0
VIDEO5M_DEFAULT_END_FLOOR_M = 5.0


def clamp_targets(step_distance_m: float, targets: Sequence[int]) -> List[int]:
    """Clamp a list of desired remaining-distance targets to the step length.

    Any target exceeding the step length is clamped down to it. Duplicates
    and non-positive values are dropped. Pass targets in descending order to
    get a descending result.

    Examples:
        clamp_targets(180, [100, 80, 60, 40]) -> [100, 80, 60, 40]
        clamp_targets(70,  [100, 80, 60, 40]) -> [70, 60, 40]
        clamp_targets(23,  [100, 80, 60, 40]) -> [23]
    """
    step_len = float(step_distance_m or 0.0)
    if step_len <= 0:
        return []
    clamped = [min(step_len, float(t)) for t in targets]
    return _dedupe_positive_desc(clamped)


def spaced_targets(
    step_distance_m: float,
    *,
    spacing_m: float = VIDEO5M_DEFAULT_SPACING_M,
    start_m: Optional[float] = None,
    end_m: float = VIDEO5M_DEFAULT_END_FLOOR_M,
) -> List[int]:
    """Generate targets by stepping from start_m down to end_m at spacing_m.

    start_m defaults to the full step length.
    end_m is the floor — don't sample closer than this to the maneuver.

    If the entire step is shorter than end_m, one frame is returned at the
    step length (so very short steps always get at least one frame).

    Examples:
        spaced_targets(23, spacing_m=5, end_m=5) -> [23, 18, 13, 8, 5]
        spaced_targets(4,  spacing_m=5, end_m=5) -> [4]   # step < floor
    """
    step_len = float(step_distance_m or 0.0)
    if step_len <= 0:
        return []

    spacing = max(0.5, float(spacing_m))
    end = max(0.0, float(end_m))
    start = min(float(start_m), step_len) if start_m is not None else step_len

    # Step is entirely within the end floor — return one frame at step length.
    if step_len <= end:
        return [max(1, int(round(step_len)))]

    if start <= end:
        return _dedupe_positive_desc([end] if end <= step_len else [])

    distances: List[float] = []
    current = start
    while current >= end:
        if current <= step_len:
            distances.append(current)
        current -= spacing

    if distances and distances[-1] > end and end <= step_len:
        distances.append(end)

    return _dedupe_positive_desc(distances)


def profile_distances(
    *,
    step_distance_m: float,
    frame_profile: str = "sparse4",
    spacing_m: float = 20.0,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    video_end_floor_m: float = VIDEO5M_DEFAULT_END_FLOOR_M,
) -> List[int]:
    """Return sampling distances for a named frame profile.

    Profiles:
        sparse4  — [100, 80, 60, 40] m remaining, clamped (default)
        single   — [40] m remaining, clamped (1 frame per step)
        video5m  — full step at 5 m spacing with end floor
        custom   — user-defined spacing / window
    """
    profile = (frame_profile or "sparse4").strip().lower()
    if profile not in FRAME_PROFILES:
        raise ValueError(f"Unknown frame profile: {frame_profile}")

    if profile == "single":
        return clamp_targets(step_distance_m, [40])
    if profile == "sparse4":
        return clamp_targets(step_distance_m, SPARSE4_TARGETS_REMAINING_M)
    if profile == "video5m":
        return spaced_targets(
            step_distance_m,
            spacing_m=VIDEO5M_DEFAULT_SPACING_M,
            end_m=video_end_floor_m,
        )
    # custom
    return spaced_targets(
        step_distance_m,
        spacing_m=spacing_m,
        start_m=sample_start,
        end_m=sample_end or 0.0,
    )


def profile_from_sample_mode(mode: str) -> str:
    """Map legacy sample modes to canonical frame profiles."""
    normalized = (mode or "").strip().lower()
    if normalized == "dense":
        return "video5m"
    if normalized == "custom":
        return "custom"
    return "sparse4"


def _dedupe_positive_desc(values: Iterable[float]) -> List[int]:
    seen = set()
    out: List[int] = []
    for value in values:
        v = int(round(float(value)))
        if v <= 0 or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


__all__ = [
    "FRAME_PROFILES",
    "SPARSE4_TARGETS_REMAINING_M",
    "VIDEO5M_DEFAULT_SPACING_M",
    "VIDEO5M_DEFAULT_END_FLOOR_M",
    "clamp_targets",
    "spaced_targets",
    "profile_distances",
    "profile_from_sample_mode",
]
