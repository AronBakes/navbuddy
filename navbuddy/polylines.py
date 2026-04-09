"""Geometry helpers for working with navigation polylines."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

LatLng = Tuple[float, float]

EARTH_RADIUS_M = 6_371_000.0

__all__ = [
    "decode_polyline",
    "encode_polyline",
    "haversine_m",
    "bearing_deg",
    "pose_at_remaining_m",
    "pose_from_polyline",
]


def decode_polyline(encoded: str, precision: int = 5) -> List[LatLng]:
    """Decode a Google-encoded polyline string into a list of coordinates.

    Args:
        encoded: The encoded polyline string.
        precision: Decimal precision used during encoding. Google Maps uses 5
            (default); Valhalla uses 6.

    Raises:
        ValueError: If the encoded string is malformed.
    """
    scale = 10 ** precision
    points: List[LatLng] = []
    index = 0
    lat = 0
    lon = 0
    while index < len(encoded):
        dlat, index = _decode_coord(encoded, index)
        dlon, index = _decode_coord(encoded, index)
        lat += dlat
        lon += dlon
        points.append((lat / scale, lon / scale))
    return points


def encode_polyline(points: Sequence[LatLng], precision: int = 5) -> str:
    """Encode a list of (lat, lon) pairs into a Google polyline string.

    Args:
        points: Sequence of (lat, lon) tuples.
        precision: Decimal precision to encode at. Google Maps uses 5 (default);
            Valhalla uses 6. Must match the precision used when decoding.
    """
    if not points:
        return ""

    scale = 10 ** precision
    last_lat = 0
    last_lon = 0
    result = []

    for lat, lon in points:
        ilat = int(round(lat * scale))
        ilon = int(round(lon * scale))

        dlat = ilat - last_lat
        dlon = ilon - last_lon
        last_lat, last_lon = ilat, ilon

        result.append(_encode_coord(dlat))
        result.append(_encode_coord(dlon))

    return "".join(result)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two points in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the compass bearing from point 1 to point 2 in degrees."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def pose_at_remaining_m(
    points: Sequence[LatLng],
    step_length_m: float,
    remaining_m: float,
) -> Tuple[float, float, float]:
    """Return (lat, lon, heading) at exactly *remaining_m* before the step end.

    Mirrors the frame-sampling convention: remaining_m is the distance from the
    current position to the maneuver point (step end).  Use sparse4_distances()
    to get the canonical remaining values; they are already clamped to the step
    length, so remaining_m <= step_length_m is guaranteed.

    Args:
        points: Decoded polyline as list of (lat, lon) tuples.
        step_length_m: Total step length in metres (used as authoritative length).
        remaining_m: Distance from position to step end, in metres.

    Returns:
        (lat, lon, heading_deg) — heading is forward along the polyline.
    """
    if not points:
        return 0.0, 0.0, 0.0
    if len(points) == 1:
        return points[0][0], points[0][1], 0.0

    # Haversine distance (metres) for each consecutive pair of polyline points.
    # One element shorter than points — 5 points → 4 segment lengths.
    segment_lengths = [
        haversine_m(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i in range(len(points) - 1)
    ]
    cumulative = _cumulative_lengths(segment_lengths)
    total = float(step_length_m) if step_length_m and step_length_m > 0 else cumulative[-1]

    # Distance from start = total - remaining (clamped to valid range)
    s = max(0.0, min(float(total), float(total) - float(remaining_m)))
    lat, lon = _interp_at_arc_length(points, segment_lengths, cumulative, s)

    # Heading: direction of the segment at position s
    index = _find_segment_index(cumulative, s)
    lat_a, lon_a = points[index]
    lat_b, lon_b = points[index + 1] if index + 1 < len(points) else points[index]
    hdg = bearing_deg(lat_a, lon_a, lat_b, lon_b) if (lat_a, lon_a) != (lat_b, lon_b) else 0.0
    return lat, lon, hdg


def pose_from_polyline(
    encoded_polyline: str,
    remaining_m: float,
) -> Optional[Tuple[float, float, float]]:
    """Decode an encoded polyline and return (lat, lon, heading) at remaining_m from end.

    Convenience wrapper for callers that have an encoded polyline string rather
    than already-decoded coords. Returns None if the polyline is empty or invalid.
    step_length_m is derived from the decoded coords (no authoritative length needed).
    """
    if not encoded_polyline:
        return None
    try:
        points = decode_polyline(encoded_polyline)
    except Exception:
        return None
    if len(points) < 2:
        return None
    return pose_at_remaining_m(points, 0.0, remaining_m)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Inverse pair: _encode_coord encodes one integer delta → string chunks;
# _decode_coord decodes string chunks → integer delta.

def _encode_coord(value: int) -> str:
    """Encode one integer coordinate delta into Google polyline character chunks."""
    value = ~(value << 1) if value < 0 else value << 1
    output = ""
    while value >= 0x20:
        output += chr((0x20 | (value & 0x1F)) + 63)
        value >>= 5
    output += chr(value + 63)
    return output


def _decode_coord(encoded: str, index: int) -> Tuple[int, int]:
    """Decode one integer coordinate delta from Google polyline character chunks.

    Returns:
        (delta, new_index) — the decoded integer delta and the updated cursor.
    """
    result = 0
    shift = 0
    while True:
        if index >= len(encoded):
            raise ValueError("Invalid encoded polyline")
        b = ord(encoded[index]) - 63
        index += 1
        result |= (b & 0x1F) << shift
        shift += 5
        if b < 0x20:
            break
    delta = ~(result >> 1) if (result & 1) else (result >> 1)
    return delta, index



def _cumulative_lengths(segment_lengths: Sequence[float]) -> List[float]:
    cumulative = [0.0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
    return cumulative


def _interp_at_arc_length(
    points: Sequence[LatLng], segment_lengths: Sequence[float], cumulative: Sequence[float], s: float
) -> LatLng:
    if s <= 0.0:
        return points[0]
    if s >= cumulative[-1]:
        return points[-1]

    index = _find_segment_index(cumulative, s)
    seg_len = segment_lengths[index] or 1.0
    # Fraction of the way through this segment (0=start, 1=end).
    t = (s - cumulative[index]) / seg_len
    # Linear interpolation: a + (b - a) * t, applied separately to lat and lon.
    lat = points[index][0] + (points[index + 1][0] - points[index][0]) * t
    lon = points[index][1] + (points[index + 1][1] - points[index][1]) * t
    return lat, lon


def _find_segment_index(cumulative: Sequence[float], s: float) -> int:
    lo, hi = 0, len(cumulative) - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if cumulative[mid] <= s <= cumulative[mid + 1]:
            return mid
        if s < cumulative[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return max(0, min(len(cumulative) - 2, lo))

