"""Pydantic schemas for VLM inference input/output."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import json as _json

from pydantic import BaseModel, Field, field_validator, model_validator


VALID_AUGMENTS = {"fog", "night", "rain", "motion_blur"}
VALID_MODALITIES = {"video + prior", "image + prior", "prior", "augment + prior", "cot"}


class SpatialLandmarks(BaseModel):
    """Landmarks classified by position relative to the driver."""

    left: List[str] = Field(default_factory=list, description="Landmarks on the driver's left")
    center: List[str] = Field(default_factory=list, description="Landmarks ahead/center")
    right: List[str] = Field(default_factory=list, description="Landmarks on the driver's right")


class AnnotatedLandmark(BaseModel):
    """A landmark with spatial position, used in ground truth labeling.

    Used for both positive landmarks (should be referenced) and negative
    landmarks (visible but should NOT be referenced in instructions).
    """

    name: str = Field(..., description="Landmark name (e.g. 'BP petrol station', 'red awning')")
    position: Optional[Literal["left", "center", "right"]] = Field(
        None, description="Position relative to the driver. Null = not classified."
    )


class Prior(BaseModel):
    """Prior information sent to VLM (instruction only)."""

    instruction: str = Field(..., description="Original navigation instruction")


class Images(BaseModel):
    """Image paths for a sample."""

    overhead: Optional[str] = Field(None, description="Path to overhead map image")
    frames: List[str] = Field(
        default_factory=list,
        description="Paths to dashcam frames (chronological, last is closest to maneuver)"
    )
    dashcam: Optional[str] = Field(None, description="Path to single dashcam image (legacy)")


class Geometry(BaseModel):
    """Geometry information for a step."""

    step_polyline: str = Field(..., description="Encoded polyline for the step")
    start_lat: float = Field(..., description="Start latitude")
    start_lng: float = Field(..., description="Start longitude")
    end_lat: float = Field(..., description="End latitude")
    end_lng: float = Field(..., description="End longitude")
    heading: Optional[float] = Field(None, description="Heading in degrees (0=North)")


class Distances(BaseModel):
    """Distance information for a step."""

    step_distance_m: float = Field(..., description="Total step distance in meters")
    remaining_distance_m: float = Field(..., description="Remaining distance to maneuver")


class OSMRoad(BaseModel):
    """OSM road metadata."""

    highway: Optional[str] = Field(None, description="OSM highway type")
    name: Optional[str] = Field(None, description="Road name")
    ref: Optional[str] = Field(None, description="Road reference number")
    maxspeed: Optional[str] = Field(None, description="Speed limit")
    lanes: Optional[int] = Field(None, description="Number of lanes")
    surface: Optional[str] = Field(None, description="Road surface type")
    oneway: Optional[bool] = Field(None, description="One-way street")
    lit: Optional[bool] = Field(None, description="Street lighting")
    bridge: Optional[bool] = Field(None, description="Is a bridge")
    tunnel: Optional[bool] = Field(None, description="Is a tunnel")
    toll: Optional[bool] = Field(None, description="Toll road")
    street_names: Optional[List[str]] = Field(None, description="Street names from routing")
    distance_m: Optional[float] = Field(None, description="Distance to nearest road")

    class Config:
        extra = "allow"  # Allow additional fields from OSM


class Metadata(BaseModel):
    """Sample metadata."""

    source: str = Field("google", description="Routing source")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="Creation timestamp"
    )


class SampleMetadata(BaseModel):
    """Complete metadata for a single sample (per-step)."""

    id: str = Field(..., description="Unique sample ID (route_id + step_index)")
    route_id: str = Field(..., description="Route identifier")
    step_index: int = Field(..., description="Step index (0-based)")
    dataset_version: str = Field("v1.0", description="Dataset schema version")
    split: Optional[Literal["train", "val", "test"]] = Field(None, description="Dataset split")
    maneuver: str = Field(..., description="Maneuver type (TURN_LEFT, TURN_RIGHT, etc.)")

    prior: Prior = Field(..., description="Prior information (instruction)")
    images: Images = Field(..., description="Image paths")
    geometry: Geometry = Field(..., description="Geometry information")
    distances: Distances = Field(..., description="Distance information")
    osm_road: Optional[OSMRoad] = Field(None, description="OSM road metadata")
    metadata: Metadata = Field(default_factory=Metadata, description="Sample metadata")

    class Config:
        extra = "allow"  # Allow additional fields


class VLMInput(BaseModel):
    """Input sent to VLM for inference."""

    instruction: str = Field(..., description="Original navigation instruction")
    frame_paths: List[str] = Field(
        default_factory=list,
        description="Paths to dashcam frames"
    )
    overhead_path: Optional[str] = Field(None, description="Path to overhead map")


class VLMOutput(BaseModel):
    """Structured output from VLM."""

    enhanced_instruction: str = Field(..., description="Enhanced navigation instruction")
    lane_change_required: Union[bool, Literal["yes", "no"], None] = Field(
        "no", description="Whether lane change is required"
    )
    lanes_count: Optional[int] = Field(None, description="Number of visible lanes")
    next_action: str = Field(
        ...,
        description="Maneuver type (turn_left, turn_right, straight, merge_left, etc.)"
    )
    relevant_landmarks: List[str] = Field(
        default_factory=list,
        description="Visible landmarks referenced in instruction (flat fallback)"
    )
    spatial_landmarks: Optional[SpatialLandmarks] = Field(
        None, description="Landmarks classified by position: left, center, right"
    )
    potential_hazards: List[str] = Field(
        default_factory=list,
        description="Detected hazards (pedestrians, cyclists, roadworks)"
    )
    reasoning: Optional[str] = Field(None, description="Chain-of-thought explanation")

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, v: Any) -> Optional[str]:
        """Coerce dict/list reasoning (returned by some models) to JSON string."""
        if isinstance(v, (dict, list)):
            return _json.dumps(v)
        return v

    lane_hint: Optional[Literal["keep_left", "keep_right", "stay_middle"]] = Field(
        None, description="Lane positioning hint (v2)"
    )
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score 0-1 (v2)")
    fallback_reason: Optional[str] = Field(None, description="Reason if low confidence (v2)")

    class Config:
        extra = "allow"  # Allow additional fields from model


class PromptMeta(BaseModel):
    """Summary of what was actually sent to the model (no base64 data)."""

    text_prompt: str = Field(..., description="Text portion of the user message")
    frame_paths: List[str] = Field(default_factory=list, description="Frame image paths that existed and were included")
    overhead_path: Optional[str] = Field(None, description="Overhead map path if included")
    num_images_sent: int = Field(0, description="Total number of images sent (frames + overhead)")
    system_prompt: Optional[str] = Field(None, description="System prompt used for this inference (includes ICL examples if applicable)")


class InferenceMetadata(BaseModel):
    """Metadata about the inference run."""

    latency_ms: Optional[int] = Field(None, description="Inference latency in milliseconds")
    tokens_in: Optional[int] = Field(None, description="Input token count")
    tokens_out: Optional[int] = Field(None, description="Output token count")
    tokens_reasoning: Optional[int] = Field(None, description="Reasoning/thinking token count (included in tokens_out)")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="Inference timestamp"
    )
    prompt_meta: Optional[PromptMeta] = Field(None, description="Summary of what was sent to the model")


class InferenceResult(BaseModel):
    """Complete result of VLM inference on a sample."""

    id: str = Field(..., description="Sample ID")
    model_id: str = Field(..., description="Model identifier")
    modality: Literal["video + prior", "image + prior", "prior", "augment + prior", "cot"] = Field(
        "video + prior", description="Input modality"
    )
    variant: Optional[str] = Field(
        None,
        description="Optional non-augmentation experiment variant tag",
    )
    augment: Optional[str] = Field(
        None,
        description="Image augmentation applied (fog, rain, night, motion_blur)",
    )
    provider: str = Field(
        "openrouter",
        description="Inference provider (openrouter, local)",
    )
    label_version: str = Field("v1", description="Output schema version")

    # VLM output fields
    enhanced_instruction: str = Field(..., description="Enhanced navigation instruction")
    lane_change_required: Union[bool, Literal["yes", "no"], None] = Field("no")
    lanes_count: Optional[int] = Field(None)
    next_action: str = Field(...)
    relevant_landmarks: List[str] = Field(default_factory=list)
    spatial_landmarks: Optional[SpatialLandmarks] = Field(None)
    potential_hazards: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = Field(None)
    lane_hint: Optional[Literal["keep_left", "keep_right", "stay_middle"]] = Field(None)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    fallback_reason: Optional[str] = Field(None)

    # Metadata
    inference_metadata: InferenceMetadata = Field(
        default_factory=InferenceMetadata,
        description="Inference run metadata"
    )

    # Error handling
    error: Optional[str] = Field(None, description="Error message if inference failed")
    raw_response: Optional[str] = Field(None, description="Raw model response")

    class Config:
        extra = "allow"

    @model_validator(mode="after")
    def _validate_modality_and_augment(self) -> "InferenceResult":
        if self.augment is not None and self.augment not in VALID_AUGMENTS:
            raise ValueError(
                f"Invalid augment '{self.augment}'. Allowed: {sorted(VALID_AUGMENTS)}"
            )
        if self.modality == "prior" and self.augment is not None:
            raise ValueError("augment is not allowed when modality is 'prior'")
        return self

    @classmethod
    def from_vlm_output(
        cls,
        sample_id: str,
        model_id: str,
        output: VLMOutput,
        modality: str = "video + prior",
        inference_metadata: Optional[InferenceMetadata] = None,
        augment: Optional[str] = None,
        variant: Optional[str] = None,
        provider: str = "openrouter",
        raw_response: Optional[str] = None,
    ) -> "InferenceResult":
        """Create InferenceResult from VLMOutput."""
        return cls(
            id=sample_id,
            model_id=model_id,
            modality=modality,
            variant=variant,
            augment=augment,
            provider=provider,
            enhanced_instruction=output.enhanced_instruction,
            lane_change_required=output.lane_change_required,
            lanes_count=output.lanes_count,
            next_action=output.next_action,
            relevant_landmarks=output.relevant_landmarks,
            spatial_landmarks=output.spatial_landmarks,
            potential_hazards=output.potential_hazards,
            reasoning=output.reasoning,
            lane_hint=output.lane_hint,
            confidence=output.confidence,
            fallback_reason=output.fallback_reason,
            inference_metadata=inference_metadata or InferenceMetadata(),
            raw_response=raw_response,
        )

    @classmethod
    def from_error(
        cls,
        sample_id: str,
        model_id: str,
        error: str,
        modality: str = "video + prior",
        augment: Optional[str] = None,
        variant: Optional[str] = None,
        provider: str = "openrouter",
        raw_response: Optional[str] = None,
    ) -> "InferenceResult":
        """Create InferenceResult from an error."""
        return cls(
            id=sample_id,
            model_id=model_id,
            modality=modality,
            variant=variant,
            augment=augment,
            provider=provider,
            enhanced_instruction="",
            next_action="unknown",
            error=error,
            inference_metadata=InferenceMetadata(),
            raw_response=raw_response,
        )


# ── Benchmark schemas (pairwise A/B with Elo) ────────────────────────


class JudgeVote(BaseModel):
    """A single judge's vote on a pairwise A/B comparison."""

    judge_model: str = Field(..., description="Judge model ID")
    winner: Literal["A", "B", "tie"] = Field(
        ..., description="Which response won"
    )
    justification: str = Field("", description="Brief explanation")
    latency_ms: Optional[int] = Field(None, description="Judge call latency")
    tokens_in: Optional[int] = Field(None, description="Prompt tokens used")
    tokens_out: Optional[int] = Field(None, description="Completion tokens used")
    error: Optional[str] = Field(None, description="Error if judge call failed")


class PairwiseComparison(BaseModel):
    """Result of a single A/B comparison for one sample."""

    sample_id: str
    model_a: str = Field(..., description="Model ID assigned to position A")
    model_b: str = Field(..., description="Model ID assigned to position B")
    company_a: str = Field(..., description="Company of model A")
    company_b: str = Field(..., description="Company of model B")
    judge_votes: List[JudgeVote] = Field(default_factory=list)
    majority_winner: Optional[str] = Field(
        None, description="model_id of winner, or None for tie"
    )
    is_tie: bool = Field(False, description="True if no 2/3 majority")
    modality: Optional[str] = Field(None, description="Input modality bucket: video_prior, image_prior, prior_only, augmented, or None for unknown")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class EloRating(BaseModel):
    """Elo rating for a single model."""

    model_id: str
    company: str
    rating: float = 1000.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_comparisons: int = 0


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    judge_panel: List[str] = Field(
        default_factory=lambda: [
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.2",
            "google/gemini-3-flash-preview",
        ]
    )
    k_factor: float = Field(12.0, description="Elo K-factor")
    starting_elo: float = Field(1000.0, description="Starting Elo rating")
    seed: int = Field(42, description="Random seed for A/B assignment")
    include_images: bool = Field(
        True, description="Include images in judge prompt"
    )
    max_elo_gap: int = Field(
        600,
        description="Skip pairs with Elo gap > this (0 = no limit). "
        "Like chess matchmaking: close-rated pairs are more informative.",
    )
    gt_only: bool = Field(
        False, description="Only judge samples that have a manual GT label"
    )
    gt_weight: float = Field(
        1.0,
        description="How many times more likely a GT sample is to be chosen "
        "when sampling comparisons. 3.0 = GT samples are 3x more likely.",
    )
    total_limit: Optional[int] = Field(
        None,
        description="Global cap on total comparisons across all pairs. "
        "Overrides per-pair limit when set.",
    )


class BenchmarkRun(BaseModel):
    """Complete benchmark run result."""

    run_id: str
    config: BenchmarkConfig
    models: List[str] = Field(
        default_factory=list, description="All competing models"
    )
    comparisons: List[PairwiseComparison] = Field(default_factory=list)
    elo_ratings: Dict[str, EloRating] = Field(default_factory=dict)
    total_judge_calls: int = 0
    successful_comparisons: int = 0
    failed_comparisons: int = 0
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
