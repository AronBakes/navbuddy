"""VLM inference module for NavBuddy evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import cv2
import numpy as np

from navbuddy.augment import augment_frame
from navbuddy.eval.schemas import (
    InferenceMetadata,
    InferenceResult,
    PromptMeta,
    SampleMetadata,
    VALID_AUGMENTS,
    VLMOutput,
)

__all__ = [
    "OpenRouterClient",
    "LocalTransformersClient",
    "build_rag_prompt",
    "build_icl_prompt",
    "run_inference",
    "load_samples",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_V2",
    "SYSTEM_PROMPT_V3",
    "SYSTEM_PROMPT_V4",
    "PRIOR_SYSTEM_PROMPT",
    "PRIOR_SYSTEM_PROMPT_V2",
    "PRIOR_SYSTEM_PROMPT_V3",
    "PRIOR_SYSTEM_PROMPT_V4",
    "PROMPT_VERSIONS",
    "DRIVING_SYSTEM_PROMPT",
    "DRIVING_SYSTEM_PROMPT_BRIEF",
    "STRUCTURED_OUTPUT_SCHEMA",
]


# -- Structured output JSON schema for OpenRouter response_format ----------
# Forces the model to output valid JSON matching this exact schema at the
# token level. Supported by OpenAI, Google, Anthropic, and Fireworks providers.
# Use via --structured-output flag on CLI or structured_output=True in code.
STRUCTURED_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "navigation_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "enhanced_instruction": {
                    "type": "string",
                    "description": "Concise enhanced navigation instruction (<=16 words)",
                },
                "lane_change_required": {
                    "type": "string",
                    "enum": ["yes", "no", "unknown"],
                },
                "lanes_count": {
                    "type": ["integer", "null"],
                    "description": "Number of lanes visible or expected",
                },
                "next_action": {
                    "type": "string",
                    "enum": [
                        "turn_left", "turn_right", "straight",
                        "slight_left", "slight_right",
                        "sharp_left", "sharp_right",
                        "merge_left", "merge_right", "merge",
                        "keep_left", "keep_right",
                        "fork_left", "fork_right",
                        "ramp_left", "ramp_right",
                        "roundabout", "uturn",
                    ],
                },
                "relevant_landmarks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Visible landmarks referenced in the instruction",
                },
                "potential_hazards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hazards visible in the scene",
                },
            },
            "required": [
                "enhanced_instruction",
                "lane_change_required",
                "next_action",
                "relevant_landmarks",
                "potential_hazards",
            ],
            "additionalProperties": False,
        },
    },
}


SYSTEM_PROMPT = """You are a driving instruction assistant. You will receive:
1. Multiple dashcam frames approaching a maneuver (chronological order, last is closest)
2. An overhead map showing the route
3. The original navigation instruction

Your task is to enhance the instruction with visible landmarks and lane guidance.

Output JSON with these fields:
- "enhanced_instruction": One concise sentence (<=16 words)
- "lane_change_required": "yes" or "no"
- "lanes_count": Number of lanes visible (integer)
- "next_action": The maneuver type (turn_left, turn_right, straight, merge_left, merge_right, uturn, roundabout, fork_left, fork_right, keep_left, keep_right)
- "relevant_landmarks": Array of visible landmarks used (e.g., ["traffic_light", "stop_sign", "petrol_station"])
- "potential_hazards": Array of hazards (e.g., ["pedestrians", "cyclists", "roadworks"])
- "reasoning": Brief explanation of your analysis

Rules:
- Only reference VISIBLE landmarks from the images
- Only reference landmarks on the correct side relative to the maneuver (e.g. for a right turn, prefer landmarks on the left/ahead that confirm the junction — not landmarks on the right which the driver is turning toward)
- Keep instructions concise and actionable
- Use Australian English
- If unsure about a landmark, omit it
- Lane guidance should be included when relevant
- Output valid JSON only, no markdown formatting
"""

SYSTEM_PROMPT_V2 = """You are a driving instruction assistant. You will receive:
1. Multiple dashcam frames approaching a maneuver (chronological order, last is closest)
2. An overhead map showing the route
3. The original navigation instruction

Your task is to enhance the instruction with visible landmarks and lane guidance.

Output JSON with these fields:
- "enhanced_instruction": One concise sentence (<=16 words)
- "lane_change_required": "yes" or "no"
- "lanes_count": Number of lanes visible (integer)
- "next_action": The maneuver type (turn_left, turn_right, straight, merge_left, merge_right, uturn, roundabout, fork_left, fork_right, keep_left, keep_right)
- "relevant_landmarks": Array of visible landmarks used (e.g., ["traffic_light", "stop_sign", "petrol_station"])
- "potential_hazards": Array of hazards (e.g., ["pedestrians", "cyclists", "roadworks"])
- "reasoning": Brief explanation of your analysis

Rules:
- Only reference VISIBLE landmarks from the images
- Only include a landmark if it is UNMISTAKABLY identifiable from a moving vehicle without prior brand knowledge (e.g. "traffic light", "roundabout", "large Shell sign", "stop sign") — NOT road markings, cycle lanes, or business names that require reading small text
- Prefer landmarks that give clear spatial cues: "traffic light on the left", "T-intersection ahead", "pedestrian crossing immediately before turn"
- Reference landmarks on the side that confirms the junction approach: for a right turn, favour landmarks on the left/ahead; for a left turn, favour landmarks on the right/ahead. Avoid referencing a landmark purely because it is on the side you are turning toward.
- Do NOT reference lane markings (cycle lanes, bus lanes, painted arrows) as landmarks
- Keep instructions concise and actionable
- Use Australian English
- If unsure about a landmark, omit it
- Lane guidance should be included when relevant
- Output valid JSON only, no markdown formatting
"""

PRIOR_SYSTEM_PROMPT = """You are a driving instruction assistant. You will receive:
1. An overhead map showing the route
2. The original navigation instruction

No dashcam footage is provided. Use the map and instruction to enhance the navigation guidance.

Output JSON with these fields:
- "enhanced_instruction": One concise sentence (<=16 words)
- "lane_change_required": "yes" or "no"
- "lanes_count": Number of lanes you expect based on the road type (integer)
- "next_action": The maneuver type (turn_left, turn_right, straight, merge_left, merge_right, uturn, roundabout, fork_left, fork_right, keep_left, keep_right)
- "relevant_landmarks": Array of landmarks visible on the map (e.g., ["intersection", "roundabout", "overpass"])
- "potential_hazards": Array of likely hazards (e.g., ["pedestrians", "intersection_traffic"])
- "reasoning": Brief explanation of your analysis

Rules:
- Only reference features visible on the map or inferable from the road context
- Keep instructions concise and actionable
- Use Australian English
- Output valid JSON only, no markdown formatting
"""

PRIOR_SYSTEM_PROMPT_V2 = """You are a driving instruction assistant. You will receive:
1. An overhead map showing the route
2. The original navigation instruction

No dashcam footage is provided. Use the map and instruction to enhance the navigation guidance.

Output JSON with these fields:
- "enhanced_instruction": One concise sentence (<=16 words)
- "lane_change_required": "yes" or "no"
- "lanes_count": Number of lanes you expect based on the road type (integer)
- "next_action": The maneuver type (turn_left, turn_right, straight, merge_left, merge_right, uturn, roundabout, fork_left, fork_right, keep_left, keep_right)
- "relevant_landmarks": Array of landmarks visible on the map (e.g., ["intersection", "roundabout", "overpass"])
- "potential_hazards": Array of likely hazards (e.g., ["pedestrians", "intersection_traffic"])
- "reasoning": Brief explanation of your analysis

Rules:
- Only reference features visible on the map or inferable from the road context
- Do NOT invent landmarks that are not explicitly labelled on the overhead map
- Keep instructions concise and actionable
- Use Australian English
- Output valid JSON only, no markdown formatting
"""

SYSTEM_PROMPT_V3 = """You are a driving instruction assistant. You will receive:
1. Multiple dashcam frames approaching a maneuver (chronological order, last is closest)
2. An overhead map showing the route
3. The original navigation instruction

Your task is to enhance the instruction with visible landmarks and lane guidance.

Output JSON with these fields:
- "enhanced_instruction": One concise sentence (<=16 words)
- "lane_change_required": "yes" or "no"
- "lanes_count": Number of lanes visible (integer)
- "next_action": The maneuver type (turn_left, turn_right, straight, merge_left, merge_right, uturn, roundabout, fork_left, fork_right, keep_left, keep_right)
- "spatial_landmarks": Object with landmarks classified by position:
  - "left": Array of landmarks on the driver's left (e.g., ["Shell petrol station", "bus stop"])
  - "center": Array of landmarks ahead/center (e.g., ["traffic light", "pedestrian crossing"])
  - "right": Array of landmarks on the driver's right (e.g., ["McDonalds", "park entrance"])
- "potential_hazards": Array of hazards (e.g., ["pedestrians", "cyclists", "roadworks"])
- "reasoning": Brief explanation of your analysis

Rules:
- Classify each landmark by its position relative to the driver: left, center (ahead), or right
- Only reference landmarks on the APPROACH side of your maneuver in your enhanced_instruction:
  - For a right turn: reference landmarks on the left or ahead (the driver sees these before turning)
  - For a left turn: reference landmarks on the right or ahead
  - For straight/continue: center landmarks preferred, both sides acceptable
  - For merges: reference landmarks on the opposite side from the merge direction
- Only reference VISIBLE landmarks from the images
- Only include a landmark if it is UNMISTAKABLY identifiable from a moving vehicle without prior brand knowledge
- Do NOT reference lane markings (cycle lanes, bus lanes, painted arrows) as landmarks
- Keep instructions concise and actionable
- Use Australian English
- If unsure about a landmark, omit it
- Lane guidance should be included when relevant
- Output valid JSON only, no markdown formatting
"""

PRIOR_SYSTEM_PROMPT_V3 = """You are a driving instruction assistant. You will receive:
1. An overhead map showing the route
2. The original navigation instruction

No dashcam footage is provided. Use the map and instruction to enhance the navigation guidance.

Output JSON with these fields:
- "enhanced_instruction": One concise sentence (<=16 words)
- "lane_change_required": "yes" or "no"
- "lanes_count": Number of lanes you expect based on the road type (integer)
- "next_action": The maneuver type (turn_left, turn_right, straight, merge_left, merge_right, uturn, roundabout, fork_left, fork_right, keep_left, keep_right)
- "spatial_landmarks": Object with landmarks classified by position:
  - "left": Array of landmarks on the driver's left
  - "center": Array of landmarks ahead/center
  - "right": Array of landmarks on the driver's right
- "potential_hazards": Array of likely hazards (e.g., ["pedestrians", "intersection_traffic"])
- "reasoning": Brief explanation of your analysis

Rules:
- Classify each landmark by its position relative to the driver: left, center (ahead), or right
- Only reference landmarks on the APPROACH side of your maneuver in your enhanced_instruction
- Only reference features visible on the map or inferable from the road context
- Do NOT invent landmarks that are not explicitly labelled on the overhead map
- Keep instructions concise and actionable
- Use Australian English
- Output valid JSON only, no markdown formatting
"""

SYSTEM_PROMPT_V4 = """You are a driving instruction assistant. You will receive:
1. Multiple dashcam frames approaching a maneuver (chronological order, last is closest)
2. An overhead map showing the route
3. The original navigation instruction

Your task is to enhance the instruction with visible landmarks and lane guidance.

Output JSON with these fields:
- "enhanced_instruction": One concise sentence (<=16 words)
- "lane_change_required": "yes" or "no"
- "lanes_count": Number of lanes visible (integer)
- "next_action": The maneuver type (turn_left, turn_right, straight, merge_left, merge_right, uturn, roundabout, fork_left, fork_right, keep_left, keep_right)
- "relevant_landmarks": Array of landmarks ordered by relevance (most useful first), each with position:
  [{"name": "traffic light", "position": "center"}, {"name": "Shell station", "position": "left"}, ...]
  Position is "left", "center", or "right" relative to the driver.
- "potential_hazards": Array of hazards (e.g., ["pedestrians", "cyclists", "roadworks"])
- "reasoning": Brief explanation of your analysis

Rules:
- List landmarks in order of RELEVANCE to the driver (most helpful for navigation first)
- Tag each landmark with its position: "left", "center" (ahead), or "right" relative to the driver
- Prefer approach-side landmarks in your enhanced_instruction:
  - Right turn: left/center landmarks confirm the junction (driver sees them before turning)
  - Left turn: right/center landmarks confirm the junction
  - Straight: center landmarks preferred, both sides acceptable
  - Merges: landmarks on the opposite side from the merge direction
- Only reference VISIBLE landmarks from the images
- Only include a landmark if it is UNMISTAKABLY identifiable from a moving vehicle
- Do NOT reference lane markings (cycle lanes, bus lanes, painted arrows) as landmarks
- Keep instructions concise and actionable
- Use Australian English
- If unsure about a landmark, omit it
- Lane guidance should be included when relevant
- Output valid JSON only, no markdown formatting
"""

PRIOR_SYSTEM_PROMPT_V4 = """You are a driving instruction assistant. You will receive:
1. An overhead map showing the route
2. The original navigation instruction

No dashcam footage is provided. Use the map and instruction to enhance the navigation guidance.

Output JSON with these fields:
- "enhanced_instruction": One concise sentence (<=16 words)
- "lane_change_required": "yes" or "no"
- "lanes_count": Number of lanes you expect based on the road type (integer)
- "next_action": The maneuver type (turn_left, turn_right, straight, merge_left, merge_right, uturn, roundabout, fork_left, fork_right, keep_left, keep_right)
- "relevant_landmarks": Array of landmarks ordered by relevance (most useful first), each with position:
  [{"name": "intersection", "position": "center"}, {"name": "roundabout", "position": "center"}, ...]
  Position is "left", "center", or "right" relative to the driver.
- "potential_hazards": Array of likely hazards (e.g., ["pedestrians", "intersection_traffic"])
- "reasoning": Brief explanation of your analysis

Rules:
- List landmarks in order of RELEVANCE to the driver (most helpful for navigation first)
- Tag each landmark with its position: "left", "center" (ahead), or "right" relative to the driver
- Prefer approach-side landmarks in your enhanced_instruction
- Only reference features visible on the map or inferable from the road context
- Do NOT invent landmarks that are not explicitly labelled on the overhead map
- Keep instructions concise and actionable
- Use Australian English
- Output valid JSON only, no markdown formatting
"""

DRIVING_SYSTEM_PROMPT = """You are a calm, helpful driving co-pilot speaking to the driver through the car's audio system.

You will receive dashcam frames and a navigation instruction. Give the driver clear, spoken guidance.

Rules:
- Keep instructions under 20 words
- Speak naturally as if sitting in the passenger seat
- Lead with any safety hazards: pedestrians, cyclists, roadworks, merging traffic
- Reference visible landmarks only: traffic lights, signs, buildings, intersections
- Use left/right relative to the driver (Australian: driver on right, drives on left)
- Use distance cues: "in about 50 metres", "just ahead", "after the lights"
- If the road ahead is clear and simple, be brief: "Stay in your lane, straight ahead"
- Never output JSON, markdown, bullet points, or field labels
- For lane changes: tell them which lane, when, and why
- Tone: calm, reassuring, like a friend who knows the route well

If the driver asks a question, answer conversationally and concisely.

Examples of good output:
- "Traffic light ahead, get in the right lane for your turn onto Queen Street."
- "Roundabout coming up, take the second exit. Watch for the cyclist on the left."
- "Stay straight here, you're doing great. Your turn is in about 200 metres."
- "Slight left after the Shell station, then keep left."
"""

DRIVING_SYSTEM_PROMPT_BRIEF = """You are a driving co-pilot. Give the driver one short spoken sentence about their next maneuver. Mention hazards first, then the instruction. Under 15 words. No JSON. Australian English. Calm tone.

Example: "Watch the pedestrian, then turn left at the lights ahead."
"""

# Maps version tag to (system_prompt, prior_system_prompt)
PROMPT_VERSIONS: dict[str, tuple[str, str]] = {
    "v1": (SYSTEM_PROMPT, PRIOR_SYSTEM_PROMPT),
    "v2": (SYSTEM_PROMPT_V2, PRIOR_SYSTEM_PROMPT_V2),
    "v3": (SYSTEM_PROMPT_V3, PRIOR_SYSTEM_PROMPT_V3),
    "v4": (SYSTEM_PROMPT_V4, PRIOR_SYSTEM_PROMPT_V4),
    "driving": (DRIVING_SYSTEM_PROMPT, DRIVING_SYSTEM_PROMPT),
    "driving_brief": (DRIVING_SYSTEM_PROMPT_BRIEF, DRIVING_SYSTEM_PROMPT_BRIEF),
}


def build_rag_prompt(system_prompt: str, examples: list[dict]) -> str:
    """Augment system prompt with few-shot examples from ChromaDB.

    Args:
        system_prompt: The base system prompt
        examples: List of dicts with keys: instruction, enhanced_output, maneuver

    Returns:
        Augmented system prompt with examples appended
    """
    if not examples:
        return system_prompt

    parts = [system_prompt, "\n\nHere are examples of high-quality navigation instructions for similar scenarios:\n"]
    for i, ex in enumerate(examples, 1):
        maneuver = ex.get("maneuver", "unknown")
        instruction = ex.get("instruction", "")
        output = ex.get("enhanced_output", "")
        parts.append(f"\nExample {i} (maneuver: {maneuver}):")
        parts.append(f"Instruction: {instruction}")
        parts.append(f"Enhanced output: {output}")

    parts.append("\nNow enhance the following instruction using the same format and quality level:\n")
    return "\n".join(parts)


def build_icl_prompt(system_prompt: str, examples: list[dict], k: int | None = None) -> str:
    """Augment system prompt with k structured few-shot examples.

    Each example dict should have keys:
        instruction, maneuver, enhanced_instruction,
        lane_change_required, lanes_count, next_action,
        relevant_landmarks, potential_hazards

    Args:
        system_prompt: Base system prompt.
        examples: List of example dicts (from /api/icl-examples/formatted).
        k: Use only the first k examples (default: all).

    Returns:
        Augmented system prompt with full structured examples appended.
    """
    import json as _json

    if not examples:
        return system_prompt

    selected = examples[:k] if k is not None else examples
    if not selected:
        return system_prompt

    parts = [
        system_prompt,
        "\n\n---\nHere are examples of high-quality enhanced navigation outputs for similar scenarios:\n",
    ]

    for i, ex in enumerate(selected, 1):
        maneuver = ex.get("maneuver", "unknown")
        instruction = ex.get("instruction", "")
        lane_chg = ex.get("lane_change_required")
        parts.append(f"\nExample {i} (maneuver: {maneuver}):")
        parts.append(f"Input instruction: {instruction}")
        parts.append("Output:")
        output = {
            "enhanced_instruction": ex.get("enhanced_instruction", ""),
            "lane_change_required": "yes" if lane_chg is True else "no" if lane_chg is False else "unknown",
            "lanes_count": ex.get("lanes_count"),
            "next_action": ex.get("next_action", ""),
            "relevant_landmarks": ex.get("relevant_landmarks") or [],
            "potential_hazards": ex.get("potential_hazards") or [],
        }
        parts.append(_json.dumps(output, ensure_ascii=False))

    parts.append("\n---\nNow produce the output for the following input using the same format and quality:\n")
    return "\n".join(parts)


# -- Multimodal ICL helpers ---------------------------------------------------

# Override: some ICL examples use a non-default representative frame.
# Map sample_id → frame index (default is -1, i.e. last/closest frame).
ICL_FRAME_OVERRIDE: Dict[str, int] = {
    # GT instruction references BP petrol station visible only from 100m
    "brisbane_route0bmpfsqys_step007": 0,
}


def load_icl_examples(
    examples_path: Path,
    data_root: Path,
    example_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Load and resolve ICL examples with frames and GT fields.

    Args:
        examples_path: Path to icl_examples.jsonl (list of sample_ids).
        data_root: Root data directory containing city subdirs.
        example_indices: 1-based indices of examples to use (e.g. [1,2]).
            If None, returns all.

    Returns:
        List of resolved example dicts with keys: sample_id, instruction,
        gt_instruction, gt_fields, frame_path, overhead_path.
    """
    import json as _json

    # Load example entries
    entries: List[Dict[str, Any]] = []
    with open(examples_path) as f:
        for line in f:
            if line.strip():
                entries.append(_json.loads(line))

    # Filter by indices if specified
    if example_indices:
        entries = [entries[i - 1] for i in example_indices if 0 < i <= len(entries)]

    # Load all samples across cities
    samples_by_id: Dict[str, Dict[str, Any]] = {}
    for city_dir in data_root.iterdir():
        samples_file = city_dir / "samples.jsonl"
        if samples_file.is_file():
            for line in open(samples_file):
                if line.strip():
                    s = _json.loads(line)
                    s["_city"] = city_dir.name
                    samples_by_id[s["id"]] = s
    # Also check gt_split
    gt_split = data_root / "gt_split_samples.jsonl"
    if gt_split.exists():
        for line in open(gt_split):
            if line.strip():
                s = _json.loads(line)
                if s["id"] not in samples_by_id:
                    s["_city"] = s["id"].split("_route")[0]
                    samples_by_id[s["id"]] = s

    # Load GT instructions
    gt_instrs: Dict[str, str] = {}
    gt_file = data_root.parent / "results" / "ground_truth.jsonl"
    if not gt_file.exists():
        gt_file = Path("results") / "ground_truth.jsonl"
    if gt_file.exists():
        for line in open(gt_file):
            if line.strip():
                r = _json.loads(line)
                if not r.get("is_auto"):
                    gt_instrs[r["sample_id"]] = r.get("instruction", "")

    # Load custom labels for structured GT fields
    custom_labels: Dict[str, Dict[str, Any]] = {}
    cl_file = data_root.parent / "results" / "custom_labels.jsonl"
    if not cl_file.exists():
        cl_file = Path("results") / "custom_labels.jsonl"
    if cl_file.exists():
        for line in open(cl_file):
            if line.strip():
                cl = _json.loads(line)
                if cl.get("sample_id"):
                    custom_labels[cl["sample_id"]] = cl

    resolved: List[Dict[str, Any]] = []
    for entry in entries:
        sid = entry["sample_id"]
        sample = samples_by_id.get(sid)
        if not sample:
            continue

        city = sample.get("_city", sid.split("_route")[0])
        frames = (sample.get("images") or {}).get("frames", [])
        overhead = (sample.get("images") or {}).get("overhead")
        instruction = (sample.get("prior") or {}).get("instruction", "")
        gt_instr = gt_instrs.get(sid, "")
        cl = custom_labels.get(sid, {})

        if not gt_instr:
            continue

        # Select representative frame
        frame_idx = ICL_FRAME_OVERRIDE.get(sid, -1)
        frame_rel = frames[frame_idx] if frames else None
        frame_path = data_root / city / frame_rel if frame_rel else None
        overhead_path = data_root / city / overhead if overhead else None

        resolved.append({
            "sample_id": sid,
            "instruction": instruction,
            "gt_instruction": gt_instr,
            "gt_fields": {
                "lane_change_required": cl.get("lane_change_required"),
                "lanes_count": cl.get("lanes_count"),
                "next_action": cl.get("next_action", ""),
                "relevant_landmarks": cl.get("relevant_landmarks") or [],
                "potential_hazards": cl.get("potential_hazards") or [],
            },
            "frame_path": frame_path,
            "overhead_path": overhead_path,
        })

    return resolved


def build_icl_messages(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build multi-turn ICL messages with images for k examples.

    Each example produces a (user, assistant) turn pair. The user turn
    contains the instruction text + dashcam frame + overhead map as images.
    The assistant turn contains the GT output as JSON.

    Args:
        examples: Resolved example dicts from load_icl_examples().

    Returns:
        List of message dicts ready for OpenRouterClient.infer(icl_messages=...).
    """
    import json as _json

    messages: List[Dict[str, Any]] = []
    for example in examples:
        user_content: List[Dict[str, Any]] = []
        user_content.append({"type": "text", "text": f"Instruction: {example['instruction']}"})

        if example["frame_path"] and example["frame_path"].exists():
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(example["frame_path"])},
            })

        if example["overhead_path"] and example["overhead_path"].exists():
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(example["overhead_path"])},
            })

        gt_output = {
            "enhanced_instruction": example["gt_instruction"],
            "lane_change_required": (
                "yes" if example["gt_fields"].get("lane_change_required") is True
                else "no" if example["gt_fields"].get("lane_change_required") is False
                else "unknown"
            ),
            "lanes_count": example["gt_fields"].get("lanes_count"),
            "next_action": example["gt_fields"].get("next_action", ""),
            "relevant_landmarks": example["gt_fields"].get("relevant_landmarks") or [],
            "potential_hazards": example["gt_fields"].get("potential_hazards") or [],
        }

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": _json.dumps(gt_output, ensure_ascii=False)})

    return messages


def image_to_data_url(path: Path, augment: Optional[str] = None) -> str:
    """Convert image file to base64 data URL, optionally applying augmentation.

    Args:
        path: Path to image file
        augment: Optional augmentation type (fog, rain, night, motion_blur)

    Returns:
        Base64 data URL
    """
    # Bedrock-backed providers may enforce a 5MB limit on BASE64 payload size.
    # Base64 expands binary payload by ~33%, so cap raw bytes conservatively.
    max_bytes = int(os.environ.get("OPENROUTER_MAX_IMAGE_BYTES", "3800000"))

    def _encode_jpeg_under_limit(img: np.ndarray, byte_limit: int) -> bytes:
        """JPEG-encode image under byte limit via quality + scale backoff."""
        best: Optional[bytes] = None
        scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        qualities = [90, 85, 80, 75, 70, 65, 60, 55, 50]

        for scale in scales:
            if scale < 1.0:
                h, w = img.shape[:2]
                resized = cv2.resize(
                    img,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                resized = img

            for quality in qualities:
                ok, buffer = cv2.imencode(
                    ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if not ok:
                    continue
                data = buffer.tobytes()
                if best is None or len(data) < len(best):
                    best = data
                if len(data) <= byte_limit:
                    return data

        if best is None:
            raise ValueError(f"Failed to encode image: {path}")
        return best

    if augment:
        # Load image with OpenCV, apply augmentation, encode to bytes
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Apply augmentation
        augmented = augment_frame(img, augment)

        # Encode to JPEG bytes under provider size limits.
        data = _encode_jpeg_under_limit(augmented, max_bytes)
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    else:
        # Cap large images (especially overhead maps) to max 800px on longest side
        max_dim = int(os.environ.get("OPENROUTER_MAX_IMAGE_DIM", "800"))
        data = path.read_bytes()
        img_cv = cv2.imread(str(path))
        needs_resize = False
        if img_cv is not None:
            h, w = img_cv.shape[:2]
            needs_resize = max(h, w) > max_dim or len(data) > max_bytes
        if needs_resize and img_cv is not None:
            h, w = img_cv.shape[:2]
            scale = min(max_dim / max(h, w), 1.0)
            if scale < 1.0:
                img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            data = _encode_jpeg_under_limit(img_cv, max_bytes)
            encoded = base64.b64encode(data).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        encoded = base64.b64encode(data).decode("ascii")
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64,{encoded}"


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from model response, handling markdown code blocks and thinking models."""
    text = text.strip()

    # Strip <think>...</think> blocks (reasoning/thinking models like Qwen3-Thinking)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL).strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try to find JSON object in text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


class OpenRouterClient:
    """Client for OpenRouter API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_order: Optional[List[str]] = None,
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            base_url: API base URL (default: https://openrouter.ai/api/v1)
            provider_order: Preferred provider order (e.g. ["DeepInfra", "Together"])
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.provider_order = provider_order

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

        self._client = None
        self.last_raw_response: Optional[str] = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai package required. Install with: pip install openai"
                ) from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def infer(
        self,
        model_id: str,
        instruction: str,
        frame_paths: Sequence[Path],
        overhead_path: Optional[Path] = None,
        system_prompt: str = SYSTEM_PROMPT,
        augment: Optional[str] = None,
        context_block: Optional[str] = None,
        icl_messages: Optional[List[Dict[str, Any]]] = None,
        structured_output: bool = False,
    ) -> tuple[VLMOutput, InferenceMetadata]:
        """Run inference on a single sample.

        Args:
            model_id: Model identifier (e.g., "google/gemini-2.0-flash-001")
            instruction: Original navigation instruction
            frame_paths: Paths to dashcam frames (chronological order)
            overhead_path: Path to overhead map image
            system_prompt: System prompt for the model
            augment: Optional augmentation type (fog, rain, night, motion_blur)
            context_block: Optional SegFormer context text
            icl_messages: Optional list of pre-built ICL messages (user/assistant turns)
                to insert before the actual sample. Each is a dict with "role" and "content".
            structured_output: If True, use JSON schema response_format to constrain
                output at the token level. Requires provider support (OpenAI, Google,
                Anthropic, Fireworks). Falls back gracefully if unsupported.

        Returns:
            Tuple of (VLMOutput, InferenceMetadata)
        """
        # Build user message content
        contents: List[Dict[str, Any]] = []

        # Add text prompt
        prompt = f"Instruction: {instruction}"
        if context_block:
            prompt = f"{prompt}\n{context_block}"
        contents.append({"type": "text", "text": prompt})

        # Add frame images (chronological order)
        for frame_path in frame_paths:
            if Path(frame_path).exists():
                data_url = image_to_data_url(Path(frame_path), augment=augment)
                contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })

        # Add overhead map (no augmentation on overhead maps)
        if overhead_path and Path(overhead_path).exists():
            data_url = image_to_data_url(Path(overhead_path))
            contents.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })

        # Build messages: system + optional ICL turns + actual sample
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if icl_messages:
            messages.extend(icl_messages)
        messages.append({"role": "user", "content": contents})

        # Call API
        start_time = time.perf_counter()
        kwargs: Dict[str, Any] = dict(
            model=model_id,
            messages=messages,
            max_tokens=4096,
        )
        if structured_output:
            kwargs["response_format"] = STRUCTURED_OUTPUT_SCHEMA
        extra_body: Dict[str, Any] = {}
        if self.provider_order:
            extra_body["provider"] = {
                "order": self.provider_order,
                "allow_fallbacks": True,
                "require_parameters": True,
            }
        if structured_output and not self.provider_order:
            # When using structured output without explicit provider order,
            # require the provider to support response_format parameters
            extra_body["provider"] = {
                "allow_fallbacks": True,
                "require_parameters": True,
            }
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Parse response
        if not response.choices or not response.choices[0].message:
            raise ValueError(
                f"Empty response from {model_id} — provider may not support multi-image input"
            )
        raw_content = response.choices[0].message.content or ""
        # Capture reasoning_content (separate field used by OpenAI o-series, xAI Grok thinking)
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning_content:
            self.last_raw_response = f"<think>{reasoning_content}</think>\n{raw_content}"
        else:
            self.last_raw_response = raw_content
        parsed = parse_json_response(raw_content)

        # Build VLMOutput
        output = VLMOutput(
            enhanced_instruction=parsed.get("enhanced_instruction", instruction),
            lane_change_required=parsed.get("lane_change_required", "no"),
            lanes_count=parsed.get("lanes_count"),
            next_action=parsed.get("next_action", "unknown"),
            relevant_landmarks=parsed.get("relevant_landmarks", []),
            potential_hazards=parsed.get("potential_hazards", []),
            reasoning=parsed.get("reasoning"),
        )

        # Build metadata
        usage = response.usage
        included_frames = [str(p) for p in frame_paths if Path(p).exists()]
        included_overhead = str(overhead_path) if overhead_path and Path(overhead_path).exists() else None
        prompt_text = f"Instruction: {instruction}"
        if context_block:
            prompt_text = f"{prompt_text}\n{context_block}"
        # Extract reasoning tokens if available
        reasoning_tokens = None
        if usage:
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd:
                reasoning_tokens = getattr(ctd, "reasoning_tokens", None)

        # Count ICL images
        icl_image_count = 0
        if icl_messages:
            for m in icl_messages:
                if isinstance(m.get("content"), list):
                    icl_image_count += sum(
                        1 for c in m["content"]
                        if isinstance(c, dict) and c.get("type") == "image_url"
                    )
        sample_image_count = len(included_frames) + (1 if included_overhead else 0)

        metadata = InferenceMetadata(
            latency_ms=latency_ms,
            tokens_in=usage.prompt_tokens if usage else None,
            tokens_out=usage.completion_tokens if usage else None,
            tokens_reasoning=reasoning_tokens,
            prompt_meta=PromptMeta(
                text_prompt=prompt_text,
                frame_paths=included_frames,
                overhead_path=included_overhead,
                num_images_sent=sample_image_count + icl_image_count,
                system_prompt=system_prompt,
            ),
        )

        return output, metadata

    def infer_prior_only(
        self,
        model_id: str,
        instruction: str,
        overhead_path: Optional[Path] = None,
        system_prompt: str = PRIOR_SYSTEM_PROMPT,
        context_block: Optional[str] = None,
    ) -> tuple[VLMOutput, InferenceMetadata]:
        """Run inference with map + instruction only, no dashcam frames.

        Args:
            model_id: Model identifier
            instruction: Original navigation instruction
            overhead_path: Path to overhead map image
            system_prompt: System prompt

        Returns:
            Tuple of (VLMOutput, InferenceMetadata)
        """
        contents: List[Dict[str, Any]] = []
        prompt = f"Instruction: {instruction}"
        if context_block:
            prompt = f"{prompt}\n{context_block}"
        contents.append({"type": "text", "text": prompt})

        # Add overhead map (the key visual input for prior-only)
        if overhead_path and Path(overhead_path).exists():
            data_url = image_to_data_url(Path(overhead_path))
            contents.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })

        start_time = time.perf_counter()
        kwargs: Dict[str, Any] = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": contents},
            ],
            max_tokens=4096,
        )
        if self.provider_order:
            kwargs["extra_body"] = {
                "provider": {
                    "order": self.provider_order,
                    "allow_fallbacks": True,
                    "require_parameters": True,
                }
            }
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if not response.choices or not response.choices[0].message:
            raise ValueError(
                f"Empty response from {model_id} — provider may not support this input format"
            )
        raw_content = response.choices[0].message.content or ""
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning_content:
            self.last_raw_response = f"<think>{reasoning_content}</think>\n{raw_content}"
        else:
            self.last_raw_response = raw_content
        parsed = parse_json_response(raw_content)

        output = VLMOutput(
            enhanced_instruction=parsed.get("enhanced_instruction", instruction),
            lane_change_required=parsed.get("lane_change_required", "no"),
            lanes_count=parsed.get("lanes_count"),
            next_action=parsed.get("next_action", "unknown"),
            relevant_landmarks=parsed.get("relevant_landmarks", []),
            potential_hazards=parsed.get("potential_hazards", []),
            reasoning=parsed.get("reasoning"),
        )

        usage = response.usage
        reasoning_tokens = None
        if usage:
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd:
                reasoning_tokens = getattr(ctd, "reasoning_tokens", None)

        included_overhead = str(overhead_path) if overhead_path and Path(overhead_path).exists() else None
        prompt_text = f"Instruction: {instruction}"
        if context_block:
            prompt_text = f"{prompt_text}\n{context_block}"
        metadata = InferenceMetadata(
            latency_ms=latency_ms,
            tokens_in=usage.prompt_tokens if usage else None,
            tokens_out=usage.completion_tokens if usage else None,
            tokens_reasoning=reasoning_tokens,
            prompt_meta=PromptMeta(
                text_prompt=prompt_text,
                frame_paths=[],
                overhead_path=included_overhead,
                num_images_sent=1 if included_overhead else 0,
                system_prompt=system_prompt,
            ),
        )

        return output, metadata

    def infer_with_rag(
        self,
        model_id: str,
        instruction: str,
        frame_paths: list,
        overhead_path: str | None,
        experience_store,
        k: int = 3,
        min_reward: float = 0.6,
        maneuver: str | None = None,
    ) -> tuple[VLMOutput, InferenceMetadata]:
        """Inference with RAG: retrieve similar examples from ChromaDB as few-shot demos.

        Args:
            model_id: Model to use
            instruction: Navigation instruction
            frame_paths: List of frame image paths
            overhead_path: Overhead map path
            experience_store: ExperienceStore instance
            k: Number of examples to retrieve
            min_reward: Minimum reward threshold for retrieved examples
            maneuver: Expected maneuver type (for diversity filtering)

        Returns:
            Tuple of (VLMOutput, InferenceMetadata)
        """
        from navbuddy.architectures.navclip.model import text_to_embedding, image_to_embedding, _truncate

        # Compute query embedding from last frame + instruction
        image_emb = [0.0] * 256
        if frame_paths:
            last_frame = str(frame_paths[-1])
            image_emb = image_to_embedding(last_frame, dim=256)
        text_emb = text_to_embedding(instruction, dim=256)
        segformer_part = [0.0] * 384  # Skip SegFormer at query time for speed
        query_emb = image_emb + text_emb + segformer_part

        # Query ChromaDB for similar high-reward examples
        where_filter = {"reward_composite": {"$gte": min_reward}}
        results = experience_store.query_similar(
            embedding=query_emb,
            n_results=k * 3,  # over-fetch for filtering
            where=where_filter,
        )

        # Diversity: pick up to 2 same-maneuver + 1 different
        examples = []
        same_maneuver = []
        diff_maneuver = []
        for r in results:
            meta = r.get("metadata", {})
            doc = r.get("document", "")
            entry = {
                "instruction": meta.get("instruction", ""),
                "enhanced_output": doc,
                "maneuver": meta.get("maneuver", "unknown"),
            }
            if maneuver and meta.get("maneuver", "").upper() == maneuver.upper():
                same_maneuver.append(entry)
            else:
                diff_maneuver.append(entry)

        examples = same_maneuver[:2] + diff_maneuver[:1]
        if len(examples) < k:
            # Fill remaining from whichever pool has more
            remaining = (same_maneuver[2:] + diff_maneuver[1:])[:k - len(examples)]
            examples.extend(remaining)
        examples = examples[:k]

        # Build augmented prompt and call existing infer
        augmented_prompt = build_rag_prompt(SYSTEM_PROMPT, examples)

        # Use the existing infer method but with modified system prompt
        return self.infer(
            model_id=model_id,
            instruction=instruction,
            frame_paths=frame_paths,
            overhead_path=Path(overhead_path) if overhead_path else None,
            system_prompt=augmented_prompt,
        )


def _load_pil_image(path: Path, augment: Optional[str] = None):
    """Load an image as RGB PIL.Image, optionally applying augmentation."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for local inference. Install with: pip install Pillow") from exc

    if augment:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        augmented = augment_frame(img, augment)
        rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    return Image.open(path).convert("RGB")


class LocalTransformersClient:
    """Local Hugging Face Transformers client for vision-language inference."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_raw_response: Optional[str] = None

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Local inference requires transformers + torch. Install with: "
                "pip install transformers torch accelerate"
            ) from exc

        self._torch = torch
        self._processor_cls = AutoProcessor

        # Detect model architecture to use correct class (e.g. Qwen3VL vs Qwen2.5VL)
        from transformers import AutoConfig
        try:
            _cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            _arch = getattr(_cfg, "architectures", [None])[0]
        except Exception:
            _arch = None

        if _arch and _arch != "AutoModelForImageTextToText":
            try:
                import importlib
                _mod = importlib.import_module("transformers")
                self._model_cls = getattr(_mod, _arch, AutoModelForImageTextToText)
            except Exception:
                self._model_cls = AutoModelForImageTextToText
        else:
            self._model_cls = AutoModelForImageTextToText

        self._resolved_device = self._resolve_device(device=device, torch_mod=torch)
        self._resolved_dtype = self._resolve_dtype(dtype=dtype, device=self._resolved_device, torch_mod=torch)

        quantization_config = None
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }

        if self.load_in_4bit and self._resolved_device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self._resolved_dtype or torch.float16,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            except Exception:
                warnings.warn(
                    "bitsandbytes is not available; local model will load without 4-bit quantization.",
                    stacklevel=2,
                )

        if quantization_config is None and self._resolved_device == "cuda":
            model_kwargs["device_map"] = "auto"

        if self._resolved_dtype is not None:
            model_kwargs["torch_dtype"] = self._resolved_dtype

        self.processor = self._processor_cls.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = self._model_cls.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

        if quantization_config is None and self._resolved_device == "cuda":
            self.model.to("cuda")

        self._input_device = self._detect_input_device()

    @staticmethod
    def _resolve_device(*, device: str, torch_mod) -> str:
        if device == "auto":
            return "cuda" if torch_mod.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("local_device must be one of: auto, cpu, cuda")
        if device == "cuda" and not torch_mod.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU.", stacklevel=2)
            return "cpu"
        return device

    @staticmethod
    def _resolve_dtype(*, dtype: str, device: str, torch_mod):
        if dtype == "auto":
            if device == "cuda":
                return torch_mod.float16
            return torch_mod.float32
        mapping = {
            "float16": torch_mod.float16,
            "bfloat16": torch_mod.bfloat16,
            "float32": torch_mod.float32,
        }
        if dtype not in mapping:
            raise ValueError("local_dtype must be one of: auto, float16, bfloat16, float32")
        return mapping[dtype]

    def _detect_input_device(self):
        model_device = getattr(self.model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return model_device
        try:
            return next(self.model.parameters()).device
        except Exception:
            return None

    def _build_messages(
        self,
        *,
        instruction: str,
        system_prompt: str,
        context_block: Optional[str],
        image_count: int,
    ) -> List[Dict[str, Any]]:
        prompt = f"Instruction: {instruction}"
        if context_block:
            prompt = f"{prompt}\n{context_block}"

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        user_content.extend({"type": "image"} for _ in range(image_count))

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def _generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        images: List[Any],
        fallback_instruction: str,
    ) -> tuple[VLMOutput, InferenceMetadata]:
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if images:
            inputs = self.processor(text=[prompt_text], images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=[prompt_text], return_tensors="pt")

        tokens_in = None
        if "input_ids" in inputs and inputs["input_ids"] is not None:
            tokens_in = int(inputs["input_ids"].shape[-1])

        if self._input_device is not None:
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    inputs[key] = value.to(self._input_device)

        generate_kwargs: Dict[str, Any] = {"max_new_tokens": int(self.max_new_tokens)}
        if self.temperature is None or self.temperature <= 0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = float(self.temperature)
            generate_kwargs["top_p"] = 0.95

        start_time = time.perf_counter()
        with self._torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        input_ids = inputs.get("input_ids")
        trimmed_ids = []
        if input_ids is not None:
            for in_ids, out_ids in zip(input_ids, generated_ids):
                trimmed_ids.append(out_ids[in_ids.shape[-1] :])
        else:
            trimmed_ids = [generated_ids[0]]

        raw_content = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self.last_raw_response = raw_content
        parsed = parse_json_response(raw_content)

        output = VLMOutput(
            enhanced_instruction=parsed.get("enhanced_instruction", fallback_instruction),
            lane_change_required=parsed.get("lane_change_required", "no"),
            lanes_count=parsed.get("lanes_count"),
            next_action=parsed.get("next_action", "unknown"),
            relevant_landmarks=parsed.get("relevant_landmarks", []),
            potential_hazards=parsed.get("potential_hazards", []),
            reasoning=parsed.get("reasoning"),
        )

        tokens_out = None
        if trimmed_ids:
            tokens_out = int(trimmed_ids[0].shape[-1])

        metadata = InferenceMetadata(
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return output, metadata

    def infer(
        self,
        model_id: str,
        instruction: str,
        frame_paths: Sequence[Path],
        overhead_path: Optional[Path] = None,
        system_prompt: str = SYSTEM_PROMPT,
        augment: Optional[str] = None,
        context_block: Optional[str] = None,
    ) -> tuple[VLMOutput, InferenceMetadata]:
        del model_id

        images = []
        for frame_path in frame_paths:
            frame = Path(frame_path)
            if frame.exists():
                images.append(_load_pil_image(frame, augment=augment))
        if overhead_path and Path(overhead_path).exists():
            images.append(_load_pil_image(Path(overhead_path), augment=None))

        messages = self._build_messages(
            instruction=instruction,
            system_prompt=system_prompt,
            context_block=context_block,
            image_count=len(images),
        )

        return self._generate(messages=messages, images=images, fallback_instruction=instruction)

    def infer_prior_only(
        self,
        model_id: str,
        instruction: str,
        overhead_path: Optional[Path] = None,
        system_prompt: str = PRIOR_SYSTEM_PROMPT,
        context_block: Optional[str] = None,
    ) -> tuple[VLMOutput, InferenceMetadata]:
        del model_id

        images = []
        if overhead_path and Path(overhead_path).exists():
            images.append(_load_pil_image(Path(overhead_path), augment=None))

        messages = self._build_messages(
            instruction=instruction,
            system_prompt=system_prompt,
            context_block=context_block,
            image_count=len(images),
        )

        return self._generate(messages=messages, images=images, fallback_instruction=instruction)


def load_samples(dataset_path: Path) -> Iterator[SampleMetadata]:
    """Load samples from JSONL dataset file.

    Args:
        dataset_path: Path to samples.jsonl file

    Yields:
        SampleMetadata objects
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield SampleMetadata(**data)


def _dedupe_frame_paths(frame_paths: Sequence[Path]) -> List[Path]:
    """Remove duplicate frame inputs by path and file-content hash.

    Keeps first occurrence order to preserve chronology as much as possible.
    """
    deduped: List[Path] = []
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()

    for frame_path in frame_paths:
        p = Path(frame_path)
        if p in seen_paths:
            continue
        seen_paths.add(p)

        try:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
        except Exception:
            # If a file can't be read here, defer failure to normal inference path.
            deduped.append(p)
            continue

        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        deduped.append(p)

    return deduped


def run_inference(
    dataset_path: Path,
    model_id: str,
    output_path: Path,
    *,
    modality: str = "video + prior",
    provider: str = "openrouter",
    api_key: Optional[str] = None,
    data_root: Optional[Path] = None,
    limit: Optional[int] = None,
    verbose: bool = True,
    augment: Optional[str] = None,
    variant: Optional[str] = None,
    use_segformer_context: bool = False,
    segformer_model_id: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    segformer_device: str = "auto",
    segformer_cache_dir: Optional[str] = None,
    local_device: str = "auto",
    local_dtype: str = "auto",
    local_load_in_4bit: bool = True,
    local_max_new_tokens: int = 256,
    local_temperature: float = 0.0,
    dedupe_frames: bool = True,
    include_arrive_steps: bool = False,
    provider_order: Optional[List[str]] = None,
    route_ids: Optional[List[str]] = None,
    sample_ids: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    prior_system_prompt: Optional[str] = None,
    cache: Optional[Any] = None,
    icl_messages: Optional[List[Dict[str, Any]]] = None,
    structured_output: bool = False,
) -> List[InferenceResult]:
    """Run inference on a dataset.

    Args:
        dataset_path: Path to samples.jsonl
        model_id: Model identifier (e.g., "google/gemini-2.0-flash-001")
        output_path: Path to write results JSONL
        modality: "video + prior" or "prior"
        provider: "openrouter" or "local"
        api_key: API key (or use env var)
        data_root: Root directory for resolving image paths
        limit: Maximum number of samples to process
        verbose: Print progress
        augment: Optional augmentation type (fog, rain, night, motion_blur)
        variant: Optional non-augmentation experiment tag for result identity
        use_segformer_context: Inject SegFormer-derived context into prompt
        segformer_model_id: SegFormer checkpoint ID
        segformer_device: Device override for context extraction
        segformer_cache_dir: Optional model cache path
        local_device: Local provider device selection (auto/cuda/cpu)
        local_dtype: Local provider dtype (auto/float16/bfloat16/float32)
        local_load_in_4bit: Enable 4-bit quantization when running local model
        local_max_new_tokens: Max completion tokens for local model
        local_temperature: Local sampling temperature (0.0 = greedy)
        dedupe_frames: Remove duplicate frames by path/content hash before inference
        include_arrive_steps: If False, skip terminal ARRIVE steps entirely
        route_ids: If provided, only process samples matching these route IDs
        sample_ids: If provided, only process samples matching these sample IDs
        system_prompt: Override system prompt for video/image modalities (default: SYSTEM_PROMPT)
        prior_system_prompt: Override system prompt for prior-only modality (default: PRIOR_SYSTEM_PROMPT)
        cache: Optional InferenceCache instance. When provided, results are looked up before
            calling the API and stored after successful inference.

    Returns:
        List of InferenceResult objects
    """
    if augment and augment not in VALID_AUGMENTS:
        raise ValueError(
            f"Invalid augment '{augment}'. Allowed: {sorted(VALID_AUGMENTS)}"
        )
    if modality == "prior" and augment:
        raise ValueError("augment is not allowed when modality is 'prior'")

    if provider == "openrouter":
        client = OpenRouterClient(api_key=api_key, provider_order=provider_order)
    elif provider == "local":
        client = LocalTransformersClient(
            model_id=model_id,
            device=local_device,
            dtype=local_dtype,
            load_in_4bit=local_load_in_4bit,
            max_new_tokens=local_max_new_tokens,
            temperature=local_temperature,
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: 'openrouter', 'local'."
        )
    context_extractor = None
    if use_segformer_context:
        from navbuddy.vision.segformer_context import SegFormerContextConfig, SegFormerContextExtractor

        context_extractor = SegFormerContextExtractor(
            SegFormerContextConfig(
                model_id=segformer_model_id,
                device=segformer_device,
                cache_dir=segformer_cache_dir,
            )
        )

    # Resolve data root
    if data_root is None:
        data_root = dataset_path.parent

    def _resolve_path(rel: str, city: str | None) -> Path:
        """Resolve a relative path, falling back to data_root/{city}/rel for multi-city datasets."""
        p = data_root / rel
        if not p.exists() and city:
            city_p = data_root / city / rel
            if city_p.exists():
                return city_p
        return p

    results: List[InferenceResult] = []

    # Open output file — resume from existing results if present
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set = set()
    error_ids: set = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        _obj = json.loads(_line)
                        _id = _obj["id"]
                        if _obj.get("error") or _obj.get("error_message"):
                            error_ids.add(_id)
                        else:
                            done_ids.add(_id)
                    except (json.JSONDecodeError, KeyError):
                        pass
        if (done_ids or error_ids) and verbose:
            print(f"Resuming: {len(done_ids)} done, {len(error_ids)} errors to retry in {output_path.name}")

    # If there are errors to retry, rewrite the file without error lines
    if error_ids and done_ids:
        _keep_lines = []
        with open(output_path, encoding="utf-8") as _f:
            for _line in _f:
                _line_s = _line.strip()
                if _line_s:
                    try:
                        _obj = json.loads(_line_s)
                        if _obj["id"] not in error_ids:
                            _keep_lines.append(_line)
                    except (json.JSONDecodeError, KeyError):
                        _keep_lines.append(_line)
        with open(output_path, "w", encoding="utf-8") as _f:
            _f.writelines(_keep_lines)

    file_mode = "a" if done_ids else "w"

    with open(output_path, file_mode) as out_f:
        # Build filter sets if provided
        _route_id_set = set(route_ids) if route_ids else None
        _sample_id_set = set(sample_ids) if sample_ids else None

        matched = 0
        skipped_existing = 0
        for i, sample in enumerate(load_samples(dataset_path)):
            # Apply route/sample ID filters before counting toward limit
            if _route_id_set and sample.route_id not in _route_id_set:
                continue
            if _sample_id_set and sample.id not in _sample_id_set:
                continue

            matched += 1
            if limit and matched > limit:
                break

            if not include_arrive_steps and str(sample.maneuver).upper() == "ARRIVE":
                if verbose:
                    print(f"[{matched}] Skipping {sample.id} (ARRIVE step)")
                continue

            # Skip already-processed samples (resume support)
            if sample.id in done_ids:
                skipped_existing += 1
                continue

            # Check Redis cache before calling API
            if cache is not None:
                cached = cache.get(sample.id, model_id, modality, augment or "")
                if cached is not None:
                    try:
                        result = InferenceResult.model_validate(cached)
                        results.append(result)
                        out_f.write(result.model_dump_json() + "\n")
                        out_f.flush()
                        if verbose:
                            print(f"[{matched}] {sample.id} (cache hit)")
                        continue
                    except Exception:
                        pass  # Fall through to API call if deserialization fails

            if verbose:
                aug_label = f" [{augment}]" if augment else ""
                print(f"[{matched}] Processing {sample.id}{aug_label}...")

            try:
                _sys_prompt = system_prompt or SYSTEM_PROMPT
                _prior_sys_prompt = prior_system_prompt or PRIOR_SYSTEM_PROMPT

                if modality == "prior":
                    # Prior only (map + instruction, no dashcam frames)
                    _city = getattr(sample, "_city", None)
                    overhead_path = _resolve_path(sample.images.overhead, _city) if sample.images.overhead else None
                    # Validate: need the overhead map on disk
                    if not overhead_path or not overhead_path.exists():
                        if verbose:
                            print(f"  [SKIP] {sample.id}: overhead map not found ({overhead_path}) — modality=prior requires it")
                        skipped_existing += 1
                        continue
                    context_block = None
                    if context_extractor is not None:
                        context_block = context_extractor.build_context_block(
                            frame_paths=[],
                            overhead_path=overhead_path,
                        )
                    output, meta = client.infer_prior_only(
                        model_id=model_id,
                        instruction=sample.prior.instruction,
                        overhead_path=overhead_path,
                        system_prompt=_prior_sys_prompt,
                        context_block=context_block,
                    )
                else:
                    # Video + prior (all frames), image + prior (last frame only),
                    # or augmented (last frame only, to reduce cost)
                    _city = getattr(sample, "_city", None)
                    frame_paths = [
                        _resolve_path(p, _city) for p in sample.images.frames
                    ]
                    if (modality == "image + prior" or augment) and len(frame_paths) > 1:
                        frame_paths = frame_paths[-1:]
                    elif dedupe_frames and len(frame_paths) > 1:
                        raw_count = len(frame_paths)
                        frame_paths = _dedupe_frame_paths(frame_paths)
                        if verbose and len(frame_paths) != raw_count:
                            print(
                                f"  Deduped frames: {raw_count} -> {len(frame_paths)}"
                            )
                    overhead_path = _resolve_path(sample.images.overhead, _city) if sample.images.overhead else None

                    # Validate image availability before sending to API
                    existing_frames = [p for p in frame_paths if p.exists()]
                    has_overhead = overhead_path is not None and overhead_path.exists()
                    min_frames = 2 if (modality == "video + prior" and not augment) else 1
                    if len(existing_frames) < min_frames or not has_overhead:
                        missing_parts = []
                        if len(existing_frames) < min_frames:
                            missing_parts.append(
                                f"dashcam frames: {len(existing_frames)}/{min_frames} found under {data_root}"
                            )
                        if not has_overhead:
                            missing_parts.append(f"overhead map not found ({overhead_path})")
                        if verbose:
                            print(f"  [SKIP] {sample.id}: {'; '.join(missing_parts)} — modality={modality} requires them")
                        skipped_existing += 1
                        continue

                    context_block = None
                    if context_extractor is not None:
                        context_block = context_extractor.build_context_block(
                            frame_paths=frame_paths,
                            overhead_path=overhead_path,
                        )

                    output, meta = client.infer(
                        model_id=model_id,
                        instruction=sample.prior.instruction,
                        frame_paths=frame_paths,
                        overhead_path=overhead_path,
                        system_prompt=_sys_prompt,
                        augment=augment,
                        context_block=context_block,
                        icl_messages=icl_messages,
                        structured_output=structured_output,
                    )

                result = InferenceResult.from_vlm_output(
                    sample_id=sample.id,
                    model_id=model_id,
                    output=output,
                    modality=modality,
                    inference_metadata=meta,
                    augment=augment,
                    variant=variant,
                    provider=provider,
                    raw_response=getattr(client, "last_raw_response", None),
                )

            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                result = InferenceResult.from_error(
                    sample_id=sample.id,
                    model_id=model_id,
                    error=str(e),
                    modality=modality,
                    augment=augment,
                    variant=variant,
                    provider=provider,
                    raw_response=getattr(client, "last_raw_response", None),
                )

            results.append(result)

            # Write to output file
            out_f.write(result.model_dump_json() + "\n")
            out_f.flush()

            # Store in Redis cache on success
            if cache is not None and not result.error:
                cache.set(sample.id, model_id, modality, augment or "", result.model_dump())

            if verbose:
                print(f"  -> {result.enhanced_instruction[:60]}...")

    if verbose:
        if skipped_existing:
            print(f"\nDone. {len(results)} new results written ({skipped_existing} skipped, {len(done_ids) + len(results)} total in {output_path.name})")
        else:
            print(f"\nDone. {len(results)} results written to {output_path}")

    return results
