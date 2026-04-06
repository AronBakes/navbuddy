"""Fuzzy landmark matching for evaluation.

Provides normalization and flexible matching so that minor spelling
differences, plurals, and common abbreviations don't cause false
negatives when checking whether a model mentioned a ground-truth landmark.
"""

from __future__ import annotations

import re
from typing import Optional

# Common AU/NZ abbreviations and colloquialisms
ABBREVIATIONS: dict[str, str] = {
    "uni": "university",
    "servo": "service station",
    "maccas": "mcdonalds",
    "bp": "bp petrol station",
    "shell": "shell petrol station",
    "caltex": "caltex petrol station",
    "7-eleven": "seven eleven",
    "7 eleven": "seven eleven",
    "traffic light": "traffic lights",
    "roundabout": "roundabouts",
    "pedestrian crossing": "pedestrian crossings",
    "ped crossing": "pedestrian crossing",
    "t-intersection": "t intersection",
    "servo": "service station",
    "petrol station": "service station",
    "gas station": "service station",
    "bottle shop": "bottle-o",
    "bottleshop": "bottle shop",
}


def normalize_landmark(name: str) -> str:
    """Normalize a landmark name for comparison.

    Lowercases, strips whitespace, removes trailing 's' for simple
    plural handling, and collapses multiple spaces.
    """
    name = name.lower().strip()
    # Normalize underscores to spaces, then collapse whitespace
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    # Strip trailing possessive 's
    name = re.sub(r"'s$", "", name)
    # Simple plural: strip trailing 's' if word is > 3 chars
    # (avoids mangling short words like "bus")
    if len(name) > 3 and name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]
    return name


def _expand_abbreviations(name: str) -> set[str]:
    """Return the original name plus any abbreviation expansions."""
    variants = {name}
    lower = name.lower().strip()
    # Check if the full name matches an abbreviation key
    if lower in ABBREVIATIONS:
        variants.add(ABBREVIATIONS[lower])
    # Check reverse: if name matches an abbreviation value
    for abbr, full in ABBREVIATIONS.items():
        if lower == full:
            variants.add(abbr)
    return variants


def landmarks_match(a: str, b: str) -> bool:
    """Return True if landmark names *a* and *b* likely refer to the same thing.

    Checks (in order):
      1. Exact normalized match
      2. Singular/plural normalization
      3. Abbreviation expansion
      4. Substring containment (either direction)
    """
    if not a or not b:
        return False

    norm_a = normalize_landmark(a)
    norm_b = normalize_landmark(b)

    # 1. Exact match after normalization
    if norm_a == norm_b:
        return True

    # 2. Already handled by normalize_landmark (strips trailing 's')

    # 3. Abbreviation expansion
    variants_a = _expand_abbreviations(a)
    variants_b = _expand_abbreviations(b)
    for va in variants_a:
        for vb in variants_b:
            if normalize_landmark(va) == normalize_landmark(vb):
                return True

    # 4. Substring containment (either direction, on normalized forms)
    if len(norm_a) >= 3 and len(norm_b) >= 3:
        if norm_a in norm_b or norm_b in norm_a:
            return True

    return False


def find_matching_landmark(
    needle: str,
    haystack: list[str],
) -> Optional[str]:
    """Find the first landmark in *haystack* that matches *needle*.

    Returns the matching haystack entry, or None if no match.
    """
    for candidate in haystack:
        if landmarks_match(needle, candidate):
            return candidate
    return None
