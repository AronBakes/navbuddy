"""NavBuddy evaluation module for VLM inference and metrics."""

from navbuddy.eval.schemas import (
    SampleMetadata,
    VLMInput,
    VLMOutput,
    InferenceResult,
)
from navbuddy.eval.inference import (
    run_inference,
    OpenRouterClient,
    LocalTransformersClient,
)
from navbuddy.eval.augment_assignment import (
    assign_route_augments,
    build_assignment_payload,
    load_assignment_file,
)
from navbuddy.eval.matrix_runner import run_evaluation_matrix
from navbuddy.eval.coverage import compute_modality_coverage, coverage_markdown
from navbuddy.eval.metrics_semantic import (
    compute_composite_score,
    evaluate_composite_metrics,
)
def compute_navclip_score(*args, **kwargs):
    """Lazy wrapper — imports torch/navclip only when called."""
    from navbuddy.eval.navclip_score import compute_navclip_score as _fn
    return _fn(*args, **kwargs)

__all__ = [
    "SampleMetadata",
    "VLMInput",
    "VLMOutput",
    "InferenceResult",
    "run_inference",
    "OpenRouterClient",
    "LocalTransformersClient",
    "assign_route_augments",
    "build_assignment_payload",
    "load_assignment_file",
    "run_evaluation_matrix",
    "compute_modality_coverage",
    "coverage_markdown",
    "compute_composite_score",
    "evaluate_composite_metrics",
    "compute_navclip_score",
]
