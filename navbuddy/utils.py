"""NavBuddy utilities - ID generation, N+1 logic, config, and helpers."""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

LatLon = Tuple[float, float]

# Crockford's Base32 alphabet (excludes I, L, O, U to avoid confusion)
ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


__all__ = [
    "Config",
    "get_api_key",
    "short_ulid",
    "generate_route_id",
    "generate_sample_id",
    "slugify",
    "resolve_effective_instructions",
    "generate_frame_filename",
    "generate_map_filename",
]


# =============================================================================
# Configuration
# =============================================================================


class Config:
    """Application configuration from environment variables."""

    # Google Maps API (for Directions + Street View)
    GOOGLE_MAPS_API_KEY: Optional[str] = os.getenv("GOOGLE_MAPS_API_KEY")

    # Legacy: separate Street View key (falls back to GOOGLE_MAPS_API_KEY)
    GOOGLE_STREETVIEW_API_KEY: Optional[str] = (
        os.getenv("GOOGLE_STREETVIEW_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
    )

    # Street View settings
    STREETVIEW_DEFAULT_SIZE: str = os.getenv("STREETVIEW_DEFAULT_SIZE", "640x400")
    STREETVIEW_DEFAULT_FOV: int = int(os.getenv("STREETVIEW_DEFAULT_FOV", "90"))

    # Evaluation / LLM
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    GOOGLE_AI_API_KEY: Optional[str] = os.getenv("GOOGLE_AI_API_KEY")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    LLM_JUDGE_MODEL: str = os.getenv("LLM_JUDGE_MODEL", "gpt-4o")

    # Mapillary (optional fallback for street imagery)
    MAPILLARY_ACCESS_TOKEN: Optional[str] = os.getenv("MAPILLARY_ACCESS_TOKEN")

    # Paths
    DATA_DIR: Path = Path(os.getenv("NAVBUDDY_DATA_DIR", "./data"))
    MANIFESTS_DIR: Path = Path(os.getenv("NAVBUDDY_MANIFESTS_DIR", "./manifests"))


def get_api_key(name: str) -> Optional[str]:
    """Get API key from environment with fallbacks.

    For GOOGLE_STREETVIEW_API_KEY, falls back to GOOGLE_MAPS_API_KEY.
    """
    key = os.getenv(name)

    # Fallback: STREETVIEW key can use MAPS key
    if not key and name == "GOOGLE_STREETVIEW_API_KEY":
        key = os.getenv("GOOGLE_MAPS_API_KEY")

    return key


# =============================================================================
# ID Generation
# =============================================================================


def encode_base32(value: int, length: int) -> str:
    """Encode integer to Crockford's Base32."""
    result = []
    for _ in range(length):
        result.append(ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(result))


def short_ulid(length: int = 10) -> str:
    """Generate a short time-sortable ULID.

    Args:
        length: Output length (default 10 chars = 50 bits)

    Returns:
        Lowercase ULID string
    """
    timestamp_ms = int(time.time() * 1000)
    random_bits = int.from_bytes(os.urandom(8), "big")
    combined = (timestamp_ms << 16) | (random_bits & 0xFFFF)
    return encode_base32(combined, length).lower()


def slugify(text: str, max_len: int = 40) -> str:
    """Convert text to URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    slug = slug.strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_len]


def generate_route_id(
    city: Optional[str] = None,
    prefix: str = "route",
) -> str:
    """Generate a unique route ID.

    Args:
        city: Optional city name for prefix
        prefix: ID prefix (default "route")

    Returns:
        Route ID like "brisbane_route001" or "route_01abc123"
    """
    ulid = short_ulid(8)
    if city:
        return f"{slugify(city)}_{prefix}{ulid}"
    return f"{prefix}_{ulid}"


def generate_sample_id(
    route_id: str,
    step_index: int,
) -> str:
    """Generate sample ID for a route step.

    Args:
        route_id: Parent route ID
        step_index: Step index (0-based)

    Returns:
        Sample ID like "brisbane_route001_step000"
    """
    return f"{route_id}_step{step_index:03d}"


# =============================================================================
# N+1 Instruction Resolution
# =============================================================================


def get_instruction_text(step: Dict[str, Any]) -> str:
    """Extract instruction text from a step."""
    nav = step.get("navigationInstruction", {})
    return nav.get("instruction", "") or nav.get("verbal_pre_transition_instruction", "")


def resolve_effective_instructions(
    steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve effective instructions for all steps using N+1 policy.

    N+1 Policy (applied to ALL non-last steps):
    - Every step stores the NEXT step's instruction (what the driver will do at
      the end of this segment), so the model learns to predict the upcoming maneuver.
    - Last step (arrive): stores its own instruction. This step is skipped during
      inference.

    Args:
        steps: List of route steps

    Returns:
        Steps with 'effective_instruction' and 'instruction_policy' added
    """
    result = []

    for idx, step in enumerate(steps):
        step = dict(step)  # Don't mutate original

        current_instruction = get_instruction_text(step)

        if idx == len(steps) - 1:
            # Last step (arrive): use arrival instruction
            step["effective_instruction"] = current_instruction or "Arrive at destination"
            step["instruction_policy"] = "arrival"

        else:
            # All non-last steps: use next step's instruction (n+1 policy)
            next_instruction = get_instruction_text(steps[idx + 1])
            step["effective_instruction"] = next_instruction or current_instruction
            step["instruction_policy"] = "n_plus_one"

        # Store next instruction for overlay displays
        if idx < len(steps) - 1:
            step["next_instruction"] = get_instruction_text(steps[idx + 1])

        result.append(step)

    return result


# =============================================================================
# Frame Sampling
# =============================================================================



def generate_frame_filename(
    route_id: str,
    step_index: int,
    distance_from_end_m: int,
    step_distance_m: int,
    augmentation: Optional[str] = None,
) -> str:
    """Generate filename for a frame image.

    Args:
        route_id: Route ID (e.g., "brisbane_route001")
        step_index: Step index (0-based)
        distance_from_end_m: Distance from end of step (maneuver) in meters
        step_distance_m: Total step distance in meters
        augmentation: Optional augmentation type (e.g., "motionblur")

    Returns:
        Filename like "brisbane_route001_step000_020m_080m.jpg"
        Format: step<N>_<meters_into_step>_<meters_remaining>.jpg
        This ensures files sort in driving order.
    """
    meters_into_step = max(0, step_distance_m - distance_from_end_m)
    meters_remaining = distance_from_end_m
    base = f"{route_id}_step{step_index:03d}_{meters_into_step:03d}m_{meters_remaining:03d}m"
    if augmentation:
        return f"{base}_{augmentation}.jpg"
    return f"{base}.jpg"


def generate_map_filename(
    route_id: str,
    step_index: int,
) -> str:
    """Generate filename for an overhead map.

    Args:
        route_id: Route ID
        step_index: Step index (0-based)

    Returns:
        Filename like "brisbane_route001_step000_map.png"
    """
    return f"{route_id}_step{step_index:03d}_map.png"
