"""Company grouping and cross-company pair generation for benchmarks.

Extracts the company/provider from model IDs and generates all valid
cross-company A/B pairs for pairwise evaluation.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Dict, List, Tuple

# Canonical company lookup. Keys are lowercase prefixes from model IDs.
COMPANY_ALIASES: Dict[str, str] = {
    "openai": "openai",
    "google": "google",
    "anthropic": "anthropic",
    "qwen": "qwen",
    "x-ai": "xai",
    "xai": "xai",
    "meta-llama": "meta",
    "mistralai": "mistral",
    "allenai": "allenai",
    "bytedance-seed": "bytedance",
    "nvidia": "nvidia",
    "zhipu-ai": "zhipu",
    "deepseek": "deepseek",
    "cohere": "cohere",
    "minimax": "minimax",
    "moonshot": "moonshot",
    "moonshotai": "moonshot",
    "z-ai": "zhipu",
    "stepfun": "stepfun",
    "baidu": "baidu",
    "xiaomi": "xiaomi",
    "human": "human",
    "maps-osm": "maps-osm",
}

# Company colors (hex). Served to dashboard via /api/company-colors.
COMPANY_COLORS: Dict[str, str] = {
    "openai": "#10a37f",
    "anthropic": "#e67e22",
    "google": "#4285f4",
    "bytedance": "#00bcd4",
    "deepseek": "#5b5fc7",
    "minimax": "#ec4899",
    "moonshot": "#a855f7",
    "qwen": "#0ea5e9",
    "xai": "#ef4444",
    "xiaomi": "#f59e0b",
    "zhipu": "#64748b",
    "baidu": "#dc2626",
    "mistral": "#84cc16",
    "human": "#22c55e",
    "navbuddy": "#06b6d4",
    "meta": "#1877f2",
    "nvidia": "#76b900",
    "allenai": "#2563eb",
    "cohere": "#d946ef",
    "stepfun": "#f97316",
    "maps-osm": "#78716c",
    "unknown": "#9ca3af",
}

# Display names for leaderboard / dashboard.
COMPANY_DISPLAY_NAMES: Dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "bytedance": "ByteDance",
    "deepseek": "DeepSeek",
    "minimax": "MiniMax",
    "moonshot": "Moonshot AI",
    "qwen": "Qwen",
    "xai": "xAI",
    "xiaomi": "Xiaomi",
    "zhipu": "Zhipu AI",
    "baidu": "Baidu",
    "mistral": "Mistral",
    "human": "Human",
    "navbuddy": "NavBuddy",
    "meta": "Meta",
    "nvidia": "NVIDIA",
    "allenai": "Allen AI",
    "cohere": "Cohere",
    "stepfun": "StepFun",
    "maps-osm": "Maps + OSM",
    "unknown": "Unknown",
}

# Fallback patterns for local/HuggingFace-style model IDs (e.g. "Qwen/Qwen2.5-VL-3B")
LOCAL_MODEL_PATTERNS: Dict[str, str] = {
    "qwen": "qwen",
    "llama": "meta",
    "claude": "anthropic",
    "gemini": "google",
    "gpt": "openai",
    "grok": "xai",
    "mistral": "mistral",
    "molmo": "allenai",
    "pixtral": "mistral",
    "seed": "bytedance",
    "kimi": "moonshot",
    "ernie": "baidu",
}

# Special prefix for NavBuddy fine-tuned models
NAVBUDDY_PREFIX = "navbuddy"


def extract_company(model_id: str) -> str:
    """Extract the company/provider from a model ID.

    Handles three naming conventions:
    - OpenRouter style: "provider/model-name" (e.g. "google/gemini-3-flash-preview")
    - HuggingFace style: "Org/Model-Name" (e.g. "Qwen/Qwen2.5-VL-3B-Instruct")
    - NavBuddy fine-tuned: "navbuddy/adapter-name"

    Returns:
        Lowercase company name, or "unknown" if not recognized.
    """
    model_id_lower = model_id.lower()

    # Check for navbuddy fine-tuned models
    if model_id_lower.startswith(NAVBUDDY_PREFIX):
        return NAVBUDDY_PREFIX

    # Split on "/" and check the provider prefix
    if "/" in model_id:
        prefix = model_id.split("/")[0].lower()
        if prefix in COMPANY_ALIASES:
            return COMPANY_ALIASES[prefix]

    # Fallback: scan the full model ID for known patterns
    for pattern, company in LOCAL_MODEL_PATTERNS.items():
        if pattern in model_id_lower:
            return company

    return "unknown"


def group_models_by_company(model_ids: List[str]) -> Dict[str, List[str]]:
    """Group model IDs by their company/provider.

    Returns:
        Dict mapping company name to list of model IDs.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for model_id in model_ids:
        company = extract_company(model_id)
        groups[company].append(model_id)
    return dict(groups)


def generate_cross_company_pairs(
    model_ids: List[str],
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Generate all cross-company A/B pairs with randomized position assignment.

    For each pair of models from different companies, randomly assigns
    which model is "A" and which is "B" to eliminate position bias.

    Args:
        model_ids: List of model IDs to generate pairs from.
        seed: Random seed for reproducible A/B assignment.

    Returns:
        List of (model_a, model_b) tuples where model_a and model_b
        are always from different companies.
    """
    rng = random.Random(seed)

    # Generate all C(N,2) pairs
    all_pairs = list(itertools.combinations(sorted(model_ids), 2))

    # Filter to cross-company only
    cross_company_pairs = [
        (a, b) for a, b in all_pairs
        if extract_company(a) != extract_company(b)
    ]

    # Randomize A/B assignment
    randomized = []
    for a, b in cross_company_pairs:
        if rng.random() < 0.5:
            randomized.append((b, a))
        else:
            randomized.append((a, b))

    return randomized


def get_company_color(model_id: str) -> str:
    """Return the hex color for the company that owns model_id."""
    company = extract_company(model_id)
    return COMPANY_COLORS.get(company, COMPANY_COLORS["unknown"])


def estimate_benchmark_cost(
    n_models: int,
    n_companies: int,
    n_samples: int,
    n_judges: int = 3,
    cost_per_call: float = 0.025,
    same_company_pairs: int = 0,
) -> Dict[str, float]:
    """Estimate the cost of a benchmark run.

    Args:
        n_models: Total number of models.
        n_companies: Number of distinct companies.
        n_samples: Number of samples to evaluate per pair.
        n_judges: Number of judges in the panel.
        cost_per_call: Estimated cost per judge API call (USD).
        same_company_pairs: Number of same-company pairs to subtract.

    Returns:
        Dict with cost breakdown.
    """
    total_pairs = n_models * (n_models - 1) // 2
    cross_company_pairs = total_pairs - same_company_pairs
    total_comparisons = cross_company_pairs * n_samples
    total_judge_calls = total_comparisons * n_judges
    total_cost = total_judge_calls * cost_per_call

    return {
        "total_pairs": total_pairs,
        "cross_company_pairs": cross_company_pairs,
        "same_company_pairs": same_company_pairs,
        "total_comparisons": total_comparisons,
        "total_judge_calls": total_judge_calls,
        "estimated_cost_usd": round(total_cost, 2),
    }
