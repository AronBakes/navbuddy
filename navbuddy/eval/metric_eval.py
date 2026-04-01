"""Programmatic metric evaluation against ground-truth labels.

Scores every model result against the human-labelled GT on structured fields
(next_action, lane_change, lanes_count, landmarks, hazards) and instruction
text similarity.  No AI calls — everything runs locally.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Action direction groups ──────────────────────────────────────

DIRECTION_GROUPS: Dict[str, set] = {
    "left": {"turn_left", "fork_left", "merge_left", "keep_left"},
    "right": {"turn_right", "fork_right", "merge_right", "keep_right"},
    "straight": {"straight"},
    "uturn": {"uturn"},
    "roundabout": {"roundabout"},
}

# Reverse lookup: action → group name
_ACTION_TO_GROUP: Dict[str, str] = {}
for _group, _actions in DIRECTION_GROUPS.items():
    for _a in _actions:
        _ACTION_TO_GROUP[_a] = _group


# ── BERTScore (cached scorer) ────────────────────────────────────

_bertscore_scorer = None


def _bertscore_f1(pred: str, gt: str) -> float:
    """Compute BERTScore F1 with a cached scorer to avoid reloading roberta-large."""
    global _bertscore_scorer
    if _bertscore_scorer is None:
        from bert_score import BERTScorer
        _bertscore_scorer = BERTScorer(
            lang="en", rescale_with_baseline=True, model_type="roberta-large",
        )
    P, R, F1 = _bertscore_scorer.score([pred], [gt])
    return float(F1.item())


# ── Scoring functions ────────────────────────────────────────────


def score_action(
    pred: str,
    gt: str,
    acceptable_actions: Optional[List[str]] = None,
) -> float:
    """Score next_action with direction-group partial credit.

    Exact match with GT = 1.0.
    Match with any acceptable alternative = 1.0.
    Same direction group as GT = 0.75.
    Wrong = 0.0.
    """
    pred = (pred or "").strip().lower()
    gt = (gt or "").strip().lower()
    if pred == gt:
        return 1.0
    # Check acceptable alternatives (eval-only overrides for ambiguous geometry)
    if acceptable_actions:
        acceptable = {a.strip().lower() for a in acceptable_actions}
        if pred in acceptable:
            return 1.0
    pred_group = _ACTION_TO_GROUP.get(pred)
    gt_group = _ACTION_TO_GROUP.get(gt)
    if pred_group and gt_group and pred_group == gt_group:
        return 0.75
    return 0.0


def _normalize_lane_change(val: Any) -> Optional[bool]:
    """Normalize lane_change_required to bool."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "yes"
    return bool(val)


def score_lane_change(pred: Any, gt: Any) -> Optional[float]:
    """Exact match after normalizing yes/no/bool.

    Returns None when GT is None (ambiguous annotation) — the sample should be
    excluded from this metric's average entirely (no reward, no penalty).
    Returns 0.0 when GT is clear but pred is missing/unrecognised.
    """
    g = _normalize_lane_change(gt)
    if g is None:
        return None  # ambiguous GT — skip this sample for this metric
    p = _normalize_lane_change(pred)
    if p is None:
        return 0.0  # model failed to produce a value, GT is clear
    return 1.0 if p == g else 0.0


def score_lanes_count(pred: Optional[int], gt: Optional[int]) -> Optional[float]:
    """Exact = 1.0, off-by-1 = 0.5, else 0.0.

    Returns None when GT is None — exclude from metric average.
    """
    if gt is None:
        return None  # ambiguous GT — skip
    if pred is None:
        return 0.0
    diff = abs(int(pred) - int(gt))
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0


def score_set_overlap(pred: List[str], gt: List[str]) -> float:
    """Jaccard similarity on lowercase string sets.

    Empty pred & empty gt = 1.0 (both correctly identified nothing).
    """
    p = {s.strip().lower() for s in (pred or [])}
    g = {s.strip().lower() for s in (gt or [])}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


def score_set_precision_recall_f1(
    pred: List[str],
    gt: List[str],
    fuzzy: bool = True,
) -> Dict[str, float]:
    """Compute precision, recall, F1 for set-based fields.

    Uses fuzzy matching via landmark_matcher if fuzzy=True.
    Returns {"precision": ..., "recall": ..., "f1": ...}.
    """
    pred_items = [s.strip().lower() for s in (pred or []) if s.strip()]
    gt_items = [s.strip().lower() for s in (gt or []) if s.strip()]

    if not pred_items and not gt_items:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_items:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    if not gt_items:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    if fuzzy:
        from navbuddy.eval.landmark_matcher import find_matching_landmark

        matched_gt = set()
        true_positives = 0
        for p in pred_items:
            match = find_matching_landmark(p, gt_items)
            if match and match not in matched_gt:
                true_positives += 1
                matched_gt.add(match)
    else:
        gt_set = set(gt_items)
        pred_set = set(pred_items)
        true_positives = len(pred_set & gt_set)

    precision = true_positives / len(pred_items) if pred_items else 0.0
    recall = true_positives / len(gt_items) if gt_items else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ── BLEU score ──────────────────────────────────────────────────


def _count_ngrams(tokens: List[str], n: int) -> Dict[str, int]:
    """Count n-grams in a token list."""
    counts: Dict[str, int] = {}
    for i in range(len(tokens) - n + 1):
        gram = " ".join(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def bleu_score(pred: str, gt: str, max_n: int = 4) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-max_n (pure Python).

    Returns dict with keys "bleu_1", "bleu_2", ..., "bleu_{max_n}".
    Uses modified precision with brevity penalty.
    """
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    results: Dict[str, float] = {}

    if not pred_tokens or not gt_tokens:
        for n in range(1, max_n + 1):
            results[f"bleu_{n}"] = 0.0
        return results

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(gt_tokens) / len(pred_tokens))) if len(pred_tokens) > 0 else 0.0

    log_avg = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams = _count_ngrams(pred_tokens, n)
        gt_ngrams = _count_ngrams(gt_tokens, n)

        if not pred_ngrams:
            # No n-grams possible at this n
            for nn in range(n, max_n + 1):
                results[f"bleu_{nn}"] = 0.0
            break

        clipped = 0
        total = 0
        for gram, count in pred_ngrams.items():
            clipped += min(count, gt_ngrams.get(gram, 0))
            total += count

        precision_n = clipped / total if total > 0 else 0.0

        if precision_n == 0:
            # If any precision is 0, all higher BLEU scores are 0
            for nn in range(n, max_n + 1):
                results[f"bleu_{nn}"] = 0.0
            break

        log_avg += math.log(precision_n) / n
        results[f"bleu_{n}"] = round(bp * math.exp(log_avg), 4)

    return results


def token_f1(pred: str, gt: str) -> float:
    """Bag-of-words token-level F1.

    Treats each word as a token. Computes precision (% of pred tokens in GT),
    recall (% of GT tokens in pred), and their harmonic mean.
    """
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_bag = {}
    for t in pred_tokens:
        pred_bag[t] = pred_bag.get(t, 0) + 1
    gt_bag = {}
    for t in gt_tokens:
        gt_bag[t] = gt_bag.get(t, 0) + 1

    # Count matches (min of counts for each token)
    matches = sum(min(pred_bag.get(t, 0), gt_bag.get(t, 0)) for t in set(pred_tokens) | set(gt_tokens))

    precision = matches / len(pred_tokens) if pred_tokens else 0.0
    recall = matches / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ── Instruction similarity ──────────────────────────────────────

_st_model = None
_st_available: Optional[bool] = None


def _get_st_model():
    """Lazy-load sentence-transformers model."""
    global _st_model, _st_available
    if _st_available is False:
        return None
    if _st_model is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        _st_available = True
        return _st_model
    except ImportError:
        _st_available = False
        return None


def _rouge_n_f1(pred: str, gt: str, n: int) -> float:
    """Compute ROUGE-N F1 (unigram n=1, bigram n=2, etc.)."""
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    pred_ngrams = _count_ngrams(pred_tokens, n)
    gt_ngrams = _count_ngrams(gt_tokens, n)
    matches = sum(min(pred_ngrams.get(g, 0), gt_ngrams.get(g, 0)) for g in pred_ngrams)
    total_pred = sum(pred_ngrams.values())
    total_gt = sum(gt_ngrams.values())
    precision = matches / total_pred if total_pred else 0.0
    recall = matches / total_gt if total_gt else 0.0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def score_instruction_metrics(pred: str, gt: str, use_semantic: bool = True) -> Dict[str, float]:
    """Compute all lexicographic + semantic instruction metrics against a GT string.

    Returns:
        rouge_1, rouge_2, rouge_l, bleu_1, bleu_2, bleu_3, bleu_4,
        token_f1, semantic_similarity (sentence-BERT, falls back to rouge_l)
    """
    pred = (pred or "").strip()
    gt = (gt or "").strip()
    if not pred or not gt:
        zero = {k: 0.0 for k in [
            "rouge_1", "rouge_2", "rouge_l",
            "bleu_1", "bleu_2", "bleu_3", "bleu_4",
            "token_f1", "semantic_similarity", "bertscore_f1",
        ]}
        return zero

    scores: Dict[str, float] = {}
    scores["rouge_1"] = _rouge_n_f1(pred, gt, 1)
    scores["rouge_2"] = _rouge_n_f1(pred, gt, 2)
    scores["rouge_l"] = round(_rouge_l_f1(pred, gt), 4)
    scores.update(bleu_score(pred, gt, max_n=4))
    scores["token_f1"] = token_f1(pred, gt)

    if use_semantic:
        model = _get_st_model()
        if model is not None:
            embeddings = model.encode([pred, gt], normalize_embeddings=True)
            sim = float(embeddings[0] @ embeddings[1])
            scores["semantic_similarity"] = round(max(0.0, min(1.0, sim)), 4)
        else:
            scores["semantic_similarity"] = scores["rouge_l"]

        # BERTScore F1
        try:
            scores["bertscore_f1"] = round(max(0.0, min(1.0, _bertscore_f1(pred, gt))), 4)
        except Exception:
            scores["bertscore_f1"] = scores["rouge_l"]
    else:
        scores["semantic_similarity"] = scores["rouge_l"]
        scores["bertscore_f1"] = scores["rouge_l"]

    return scores


def _rouge_l_f1(pred: str, gt: str) -> float:
    """Compute ROUGE-L F1 score (pure Python, no dependencies)."""
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0

    # Longest common subsequence
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_instruction(pred: str, gt: str) -> float:
    """Score instruction similarity.

    Uses sentence-transformer cosine similarity if available,
    otherwise falls back to ROUGE-L F1.
    """
    pred = (pred or "").strip()
    gt = (gt or "").strip()
    if not pred or not gt:
        return 0.0
    if pred.lower() == gt.lower():
        return 1.0

    model = _get_st_model()
    if model is not None:
        embeddings = model.encode([pred, gt], normalize_embeddings=True)
        similarity = float(embeddings[0] @ embeddings[1])
        return max(0.0, min(1.0, similarity))

    # Fallback: ROUGE-L
    return _rouge_l_f1(pred, gt)


# ── Composite scoring ────────────────────────────────────────────

METRIC_WEIGHTS = {
    "bertscore_f1": 0.35,
    "semantic_similarity": 0.05,
    "lane_change_required": 0.25,
    "next_action": 0.20,
    "relevant_landmarks": 0.10,
    "potential_hazards": 0.05,
}


def score_result(
    pred: Dict[str, Any],
    gt: Dict[str, Any],
    acceptable_actions: Optional[List[str]] = None,
    detailed: bool = False,
    use_semantic: bool = True,
    image_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Score all fields of a prediction against GT.

    Returns dict with individual field scores and weighted total.
    If detailed=True, includes expanded metrics (BLEU, P/R/F1, token F1, ROUGE-L, CLIPScore).
    If use_semantic=False, uses ROUGE-L instead of sentence-transformers for instruction scoring.
    If image_path is provided, computes CLIPScore (text-image alignment) in detailed mode.
    """
    pred_instr = (pred.get("enhanced_instruction") or "").strip()
    gt_instr = (gt.get("enhanced_instruction") or "").strip()
    if use_semantic:
        instr_score = score_instruction(pred_instr, gt_instr)
    else:
        instr_score = _rouge_l_f1(pred_instr, gt_instr)

    scores: Dict[str, Any] = {
        "next_action": score_action(
            pred.get("next_action", ""),
            gt.get("next_action", ""),
            acceptable_actions=acceptable_actions,
        ),
        "lane_change_required": score_lane_change(
            pred.get("lane_change_required"),
            gt.get("lane_change_required"),
        ),
        "lanes_count": score_lanes_count(
            pred.get("lanes_count"),
            gt.get("lanes_count"),
        ),
        "relevant_landmarks": score_set_overlap(
            pred.get("relevant_landmarks", []),
            gt.get("relevant_landmarks", []),
        ),
        "potential_hazards": score_set_overlap(
            pred.get("potential_hazards", []),
            gt.get("potential_hazards", []),
        ),
        "enhanced_instruction": instr_score,
        # semantic_similarity used by composite — falls back to instruction score if not overridden
        "semantic_similarity": instr_score,
    }

    if detailed:
        instr_metrics = score_instruction_metrics(pred_instr, gt_instr, use_semantic=use_semantic)
        scores.update(instr_metrics)

        # Semantic similarity (same as enhanced_instruction but surfaced separately)
        scores["semantic_similarity"] = scores.get("enhanced_instruction")

        # Landmark P/R/F1
        lm_prf = score_set_precision_recall_f1(
            pred.get("relevant_landmarks", []),
            gt.get("relevant_landmarks", []),
            fuzzy=True,
        )
        scores["landmark_precision"] = lm_prf["precision"]
        scores["landmark_recall"] = lm_prf["recall"]
        scores["landmark_f1"] = lm_prf["f1"]

        # Hazard P/R/F1
        hz_prf = score_set_precision_recall_f1(
            pred.get("potential_hazards", []),
            gt.get("potential_hazards", []),
            fuzzy=True,
        )
        scores["hazard_precision"] = hz_prf["precision"]
        scores["hazard_recall"] = hz_prf["recall"]
        scores["hazard_f1"] = hz_prf["f1"]

        # CLIPScore: text-image alignment (hallucination detection)
        if image_path is not None:
            from navbuddy.eval.metrics_semantic import _clipscore_reward
            scores["clipscore"] = round(_clipscore_reward(pred_instr, image_path), 4)

        # Raw lane_change pred/gt booleans — needed for P/R/F1 aggregation
        scores["_lane_change_pred"] = _normalize_lane_change(pred.get("lane_change_required"))
        scores["_lane_change_gt"] = _normalize_lane_change(gt.get("lane_change_required"))

        # Strict next_action: exact match only (no direction-family partial credit)
        pred_action = (pred.get("next_action") or "").strip().lower()
        gt_action = (gt.get("next_action") or "").strip().lower()
        next_action_strict = 1.0 if pred_action == gt_action else 0.0
        if next_action_strict == 0.0 and acceptable_actions:
            if pred_action in {a.strip().lower() for a in acceptable_actions}:
                next_action_strict = 1.0
        scores["next_action_strict"] = next_action_strict

        # Strict lanes_count: exact only (no off-by-one partial credit) + abs error for MAE
        pred_ln = pred.get("lanes_count")
        gt_ln = gt.get("lanes_count")
        if gt_ln is None:
            scores["lanes_count_exact"] = None  # ambiguous GT — skip
        elif pred_ln is None:
            scores["lanes_count_exact"] = 0.0
        else:
            try:
                pred_int, gt_int = int(pred_ln), int(gt_ln)
                scores["lanes_count_exact"] = 1.0 if pred_int == gt_int else 0.0
                scores["lanes_count_abs_err"] = float(abs(pred_int - gt_int))
            except (TypeError, ValueError):
                scores["lanes_count_exact"] = 0.0

    # Weighted total — exclude metrics where GT was None (ambiguous)
    active = {k: w for k, w in METRIC_WEIGHTS.items() if scores.get(k) is not None}
    if active:
        total_w = sum(active.values())
        total = sum(scores[k] * w / total_w for k, w in active.items())
    else:
        total = 0.0
    scores["total"] = round(total, 4)

    # Round individual scores
    for k in list(scores):
        if isinstance(scores[k], float):
            scores[k] = round(scores[k], 4)

    return scores


# ── Main eval runner ─────────────────────────────────────────────


def run_metric_eval(
    results_dir: Path,
    ground_truth_path: Path,
    samples_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run metric eval for all model results against manual GT.

    Args:
        results_dir: Directory containing result JSONL files.
        ground_truth_path: Path to ground_truth.jsonl.
        samples_path: Optional path to samples.jsonl (unused for now).
        output_path: Optional path to write detailed JSONL output.
        verbose: Print summary table.

    Returns:
        List of per-sample per-model score dicts.
    """
    # 1. Load ground truth (manual only)
    gt_entries: Dict[str, Dict] = {}
    with open(ground_truth_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("is_auto"):
                continue
            gt_entries[entry["sample_id"]] = entry

    if verbose:
        print(f"Loaded {len(gt_entries)} manual GT entries")

    # 2. Load all results, indexed by sample_id → model_id → result dict
    all_results: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    result_files = sorted(results_dir.glob("results_*.jsonl"))
    for rf in result_files:
        with open(rf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = r.get("id") or r.get("sample_id")
                mid = r.get("model_id", "")
                if sid and mid and not r.get("error"):
                    # Keep first result per sample+model (avoid duplicates)
                    if mid not in all_results[sid]:
                        all_results[sid][mid] = r

    if verbose:
        total_results = sum(len(v) for v in all_results.values())
        print(f"Loaded {total_results} results across {len(all_results)} samples")

    # 3. For each GT sample, score all non-GT model results
    eval_results: List[Dict[str, Any]] = []

    for sample_id, gt_entry in gt_entries.items():
        gt_key = gt_entry.get("result_key") or ""
        gt_model = gt_key.split("::")[0] if "::" in gt_key else gt_key

        # Get GT structured fields from the original model result
        sample_results = all_results.get(sample_id, {})
        gt_result = sample_results.get(gt_model)

        if gt_result is None:
            # Try matching by result_key prefix
            for mid, res in sample_results.items():
                if mid == gt_model:
                    gt_result = res
                    break
            if gt_result is None:
                # Fall back: use GT instruction only, structured fields unavailable
                gt_result = {
                    "enhanced_instruction": gt_entry.get("instruction", ""),
                }

        # Eval-only overrides for ambiguous geometry
        acceptable = gt_entry.get("acceptable_actions")

        # Score every other model against this GT
        for model_id, pred_result in sample_results.items():
            if model_id == gt_model:
                continue  # Don't score GT against itself

            scores = score_result(pred_result, gt_result, acceptable_actions=acceptable, detailed=True, use_semantic=False)
            eval_results.append({
                "sample_id": sample_id,
                "model_id": model_id,
                "gt_model": gt_model,
                "scores": scores,
            })

    # 4. Write output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in eval_results:
                f.write(json.dumps(entry) + "\n")
        if verbose:
            print(f"\nWrote {len(eval_results)} scores to {output_path}")

    # 5. Print summary table
    if verbose and eval_results:
        _print_summary(eval_results)

    return eval_results


def _print_summary(eval_results: List[Dict[str, Any]]) -> None:
    """Print a model-level summary table of average scores."""
    # Aggregate by model
    model_scores: Dict[str, List[Dict]] = defaultdict(list)
    for entry in eval_results:
        model_scores[entry["model_id"]].append(entry["scores"])

    fields = ["action", "act_strict", "lane_chg", "lanes_cnt", "ln_exact", "ln_mae", "landmarks", "hazards", "instr", "total"]
    field_keys = [
        "next_action", "next_action_strict", "lane_change_required", "lanes_count",
        "lanes_count_exact", "lanes_count_abs_err",
        "relevant_landmarks", "potential_hazards", "enhanced_instruction", "total",
    ]

    # Header
    header = f"{'Model':<45}" + "".join(f"{f:>12}" for f in fields)
    print(f"\n{header}")
    print("-" * len(header))

    # Sort by total descending
    sorted_models = sorted(
        model_scores.items(),
        key=lambda kv: sum(s["total"] for s in kv[1]) / len(kv[1]),
        reverse=True,
    )

    for model_id, scores_list in sorted_models:
        row = f"{model_id:<45}"
        for fk in field_keys:
            vals = [s[fk] for s in scores_list if fk in s]
            avg = sum(vals) / len(vals) if vals else 0.0
            row += f"{avg:>12.3f}"
        n = len(scores_list)
        row += f"  (n={n})"
        print(row)
