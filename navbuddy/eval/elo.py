"""Elo rating computation for pairwise benchmark results.

Pure math module with no API calls. Can be used to compute ratings
from saved comparison data or to re-compute with different K-factors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from navbuddy.eval.schemas import BenchmarkConfig, EloRating, PairwiseComparison


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for player A.

    Returns the probability that A wins against B given their ratings.
    """
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 12.0,
) -> Tuple[float, float]:
    """Update Elo ratings for both players after a match.

    Args:
        rating_a: Current rating of player A.
        rating_b: Current rating of player B.
        score_a: Actual outcome for A. 1.0 = win, 0.0 = loss, 0.5 = tie.
        k: K-factor controlling rating volatility.

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea

    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * (score_b - eb)

    return new_a, new_b


def _base_model_id(model_id: str) -> str:
    """Strip result-file suffix from a model ID.

    Benchmark comparisons store the full result key as model_id
    (e.g. ``google/gemini-3-flash-preview::results_brisbane_...``).
    For Elo purposes we treat all variants of the same model as one.
    """
    return model_id.split("::")[0]


def compute_elo_ratings(
    comparisons: list,
    config: Optional[object] = None,
    k_factor: float = 12.0,
    starting_elo: float = 1500.0,
    max_matches_per_model: Optional[int] = None,
) -> Dict[str, object]:
    """Compute Elo ratings from a list of pairwise comparisons.

    Processes comparisons in their given order (should be deterministic
    for reproducibility — sorted by sample_id, model_a, model_b).

    Args:
        comparisons: List of PairwiseComparison objects.
        config: Optional BenchmarkConfig (overrides k_factor/starting_elo).
        k_factor: Elo K-factor (used if config is None).
        starting_elo: Starting rating for all models (used if config is None).
        max_matches_per_model: If set, each model participates in at most this
            many matches (comparisons where either model has hit the cap are
            skipped). This limits path-dependency from heavily-matched models.

    Returns:
        Dict mapping model_id to EloRating object.
    """
    from navbuddy.eval.company_groups import extract_company
    from navbuddy.eval.schemas import EloRating

    if config is not None:
        k_factor = config.k_factor
        starting_elo = config.starting_elo

    # Collect all model IDs (normalized — strip ::result-file suffix)
    all_models = set()
    for comp in comparisons:
        all_models.add(_base_model_id(comp.model_a))
        all_models.add(_base_model_id(comp.model_b))

    # Initialize ratings
    ratings: Dict[str, float] = {m: starting_elo for m in all_models}
    stats: Dict[str, Dict[str, int]] = {
        m: {"wins": 0, "losses": 0, "ties": 0, "total": 0}
        for m in all_models
    }

    # Track match counts for capping
    match_counts: Dict[str, int] = {m: 0 for m in all_models}

    # Process each comparison
    for comp in comparisons:
        if comp.majority_winner is None and not comp.is_tie:
            continue  # Skip invalid comparisons

        a, b = _base_model_id(comp.model_a), _base_model_id(comp.model_b)

        # Skip if either model has hit the match cap
        if max_matches_per_model is not None:
            if match_counts.get(a, 0) >= max_matches_per_model or match_counts.get(b, 0) >= max_matches_per_model:
                continue

        raw_winner = _base_model_id(comp.majority_winner) if comp.majority_winner else None

        if comp.is_tie or raw_winner is None:
            score_a = 0.5
            stats[a]["ties"] += 1
            stats[b]["ties"] += 1
        elif raw_winner == a:
            score_a = 1.0
            stats[a]["wins"] += 1
            stats[b]["losses"] += 1
        elif raw_winner == b:
            score_a = 0.0
            stats[a]["losses"] += 1
            stats[b]["wins"] += 1
        else:
            continue  # Winner doesn't match either model

        stats[a]["total"] += 1
        stats[b]["total"] += 1
        match_counts[a] = match_counts.get(a, 0) + 1
        match_counts[b] = match_counts.get(b, 0) + 1

        ratings[a], ratings[b] = update_elo(ratings[a], ratings[b], score_a, k_factor)

    # Build EloRating objects
    result = {}
    for model_id in sorted(all_models):
        s = stats[model_id]
        result[model_id] = EloRating(
            model_id=model_id,
            company=extract_company(model_id),
            rating=round(ratings[model_id], 1),
            wins=s["wins"],
            losses=s["losses"],
            ties=s["ties"],
            total_comparisons=s["total"],
        )

    return result


def compute_win_matrix(
    comparisons: list,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Build a model-vs-model win matrix from pairwise comparisons.

    Returns:
        Nested dict: {model_a: {model_b: {"wins": N, "losses": N, "ties": N}}}
    """
    matrix: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    )

    for comp in comparisons:
        a, b = _base_model_id(comp.model_a), _base_model_id(comp.model_b)
        raw_winner = _base_model_id(comp.majority_winner) if comp.majority_winner else None

        if comp.is_tie or raw_winner is None:
            matrix[a][b]["ties"] += 1
            matrix[b][a]["ties"] += 1
        elif raw_winner == a:
            matrix[a][b]["wins"] += 1
            matrix[b][a]["losses"] += 1
        elif raw_winner == b:
            matrix[a][b]["losses"] += 1
            matrix[b][a]["wins"] += 1

    # Convert defaultdicts to regular dicts for serialization
    return {k: dict(v) for k, v in matrix.items()}


def elo_confidence_interval(
    n_comparisons: int,
    k: float = 32.0,
    confidence: float = 0.95,
) -> float:
    """Approximate half-width of an Elo confidence interval.

    Uses a simple approximation: CI ≈ ±1.96 * (400 / sqrt(n)).
    More comparisons = tighter interval.

    Args:
        n_comparisons: Number of comparisons the model participated in.
        k: K-factor (not directly used, kept for API consistency).
        confidence: Confidence level (default 0.95).

    Returns:
        Half-width of the confidence interval in Elo points.
    """
    if n_comparisons <= 0:
        return float("inf")

    # z-score for common confidence levels
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    return round(z * 400.0 / math.sqrt(n_comparisons), 1)
