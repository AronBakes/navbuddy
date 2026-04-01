"""Composite semantic metrics for navigation outputs."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_COMPOSITE_WEIGHTS = {
    "action_reward": 0.20,
    "lane_reward": 0.10,
    "format_reward": 0.10,
    "length_reward": 0.05,
    "bertscore_reward": 0.15,
    "clipscore_reward": 0.15,
    "cider_reward": 0.15,
    "rouge_l_reward": 0.10,
}

# V2 weights include latency (renormalized proportionally, sums to 1.0)
DEFAULT_COMPOSITE_WEIGHTS_V2 = {
    "action_reward": 0.18,
    "lane_reward": 0.09,
    "format_reward": 0.09,
    "length_reward": 0.05,
    "bertscore_reward": 0.14,
    "clipscore_reward": 0.14,
    "cider_reward": 0.13,
    "rouge_l_reward": 0.09,
    "latency_reward": 0.09,
}

_REQUIRED_FIELDS = {
    "enhanced_instruction",
    "lane_change_required",
    "next_action",
}

_MANEUVER_TO_ACTION = {
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "LEFT": "turn_left",
    "RIGHT": "turn_right",
    "SLIGHT_LEFT": "turn_left",
    "SLIGHT_RIGHT": "turn_right",
    "SHARP_LEFT": "turn_left",
    "SHARP_RIGHT": "turn_right",
    "STRAIGHT": "continue",
    "CONTINUE": "continue",
    "NAME_CHANGE": "continue",
    "DEPART": "continue",
    "MERGE": "merge",
    "MERGE_LEFT": "merge_left",
    "MERGE_RIGHT": "merge_right",
    "RAMP_LEFT": "merge_left",
    "RAMP_RIGHT": "merge_right",
    "ROUNDABOUT_LEFT": "roundabout",
    "ROUNDABOUT_RIGHT": "roundabout",
    "UTURN": "uturn",
    "FORK_LEFT": "fork_left",
    "FORK_RIGHT": "fork_right",
    "KEEP_LEFT": "keep_left",
    "KEEP_RIGHT": "keep_right",
}


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    text = _normalize_text(text)
    return [t for t in re.split(r"[^a-z0-9_]+", text) if t]


def _extract_sample_id(pred: Dict[str, Any]) -> Optional[str]:
    return pred.get("sample_id") or pred.get("id")


def _extract_instruction(pred: Dict[str, Any]) -> str:
    return pred.get("enhanced_instruction") or pred.get("response") or ""


def _expected_action(ref: Dict[str, Any]) -> str:
    maneuver = str(ref.get("maneuver", "")).upper().strip()
    if not maneuver:
        return ""
    return _MANEUVER_TO_ACTION.get(maneuver, maneuver.lower().replace(" ", "_"))


def _same_direction_family(a: str, b: str) -> bool:
    left = {"turn_left", "merge_left", "fork_left", "keep_left", "ramp_left"}
    right = {"turn_right", "merge_right", "fork_right", "keep_right", "ramp_right"}
    straight = {"continue", "straight", "merge"}
    roundabout = {"roundabout", "roundabout_left", "roundabout_right"}
    for family in (left, right, straight, roundabout):
        if a in family and b in family:
            return True
    return False


def _format_reward(pred: Dict[str, Any]) -> float:
    keys = set(pred.keys())
    if _REQUIRED_FIELDS.issubset(keys):
        return 1.0
    if "enhanced_instruction" in keys:
        return 0.5
    return 0.0


def _length_reward(pred: Dict[str, Any]) -> float:
    instruction = _extract_instruction(pred)
    wc = len(_tokenize(instruction))
    if 4 <= wc <= 16:
        return 1.0
    if 1 <= wc <= 20:
        return 0.5
    return 0.0


def _action_reward(pred: Dict[str, Any], ref: Dict[str, Any]) -> float:
    predicted = str(pred.get("next_action", "")).lower().strip()
    expected = _expected_action(ref)
    if not predicted or not expected:
        return 0.0
    if predicted == expected:
        return 1.0
    if _same_direction_family(predicted, expected):
        return 0.5
    return 0.0


def _lane_reward(pred: Dict[str, Any], ref: Dict[str, Any]) -> float:
    lane_change_pred = str(pred.get("lane_change_required", "")).lower().strip()
    lanes_pred = pred.get("lanes_count")

    score = 0.0
    if lane_change_pred in ("yes", "no"):
        if "lane_change_required" in ref:
            score += 0.6 if lane_change_pred == str(ref.get("lane_change_required", "")).lower().strip() else 0.2
        else:
            score += 0.5

    if isinstance(lanes_pred, int) and 1 <= lanes_pred <= 8:
        if isinstance(ref.get("lanes_count"), int):
            diff = abs(int(ref["lanes_count"]) - lanes_pred)
            score += 0.4 if diff == 0 else (0.2 if diff <= 1 else 0.0)
        else:
            score += 0.5

    return max(0.0, min(1.0, score))


def _latency_reward(latency_ms: Optional[float]) -> float:
    """Compute latency reward from inference latency in milliseconds.

    Navigation is a low-latency task — responses must arrive quickly.

    Scoring tiers:
        <5000ms  -> 1.0  (5/5)
        <10000ms -> 0.8  (4/5)
        <15000ms -> 0.6  (3/5)
        <20000ms -> 0.4  (2/5)
        <25000ms -> 0.2  (1/5)
        >=25000ms -> 0.0  (0/5)
    """
    if latency_ms is None:
        return 0.0
    ms = float(latency_ms)
    if ms < 5000:
        return 1.0
    if ms < 10000:
        return 0.8
    if ms < 15000:
        return 0.6
    if ms < 20000:
        return 0.4
    if ms < 25000:
        return 0.2
    return 0.0


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def _rouge_l_f1(pred_text: str, ref_text: str) -> float:
    p_toks = _tokenize(pred_text)
    r_toks = _tokenize(ref_text)
    if not p_toks or not r_toks:
        return 0.0
    lcs = _lcs_length(p_toks, r_toks)
    precision = lcs / len(p_toks)
    recall = lcs / len(r_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _cider_like(pred_text: str, ref_text: str) -> float:
    pred_tokens = _tokenize(pred_text)
    ref_tokens = _tokenize(ref_text)
    if not pred_tokens or not ref_tokens:
        return 0.0

    score = 0.0
    weights = [0.4, 0.3, 0.2, 0.1]
    for n, w in zip((1, 2, 3, 4), weights):
        p = _ngrams(pred_tokens, n)
        r = _ngrams(ref_tokens, n)
        if not p or not r:
            continue
        overlap = sum(min(c, r[g]) for g, c in p.items())
        precision = overlap / max(1, sum(p.values()))
        recall = overlap / max(1, sum(r.values()))
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        score += w * f1
    return min(1.0, score)


def _token_f1(pred_text: str, ref_text: str) -> float:
    p = Counter(_tokenize(pred_text))
    r = Counter(_tokenize(ref_text))
    if not p or not r:
        return 0.0
    overlap = sum(min(c, r[t]) for t, c in p.items())
    precision = overlap / max(1, sum(p.values()))
    recall = overlap / max(1, sum(r.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _bertscore_reward(pred_text: str, ref_text: str) -> float:
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score([pred_text], [ref_text], lang="en", verbose=False)
        return float(max(0.0, min(1.0, f1.mean().item())))
    except Exception:
        return _token_f1(pred_text, ref_text)


_CLIP_CACHE: Dict[str, Any] = {}


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"  # 768-dim, matches Upstash index


def _load_clip() -> bool:
    """Lazy-load CLIP model into _CLIP_CACHE. Returns True on success."""
    if _CLIP_CACHE:
        return True
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_CACHE["processor"] = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _CLIP_CACHE["model"] = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _CLIP_CACHE["model"].eval()
        _CLIP_CACHE["torch"] = torch
        return True
    except Exception:
        return False


def _clip_encode_image(image: Path) -> Optional[List[float]]:
    """Return L2-normalised 768-dim CLIP image embedding, or None on failure."""
    if not image.exists() or not _load_clip():
        return None
    try:
        from PIL import Image as PILImage
        model = _CLIP_CACHE["model"]
        processor = _CLIP_CACHE["processor"]
        torch = _CLIP_CACHE["torch"]
        pil = PILImage.open(image).convert("RGB")
        inputs = processor(images=[pil], return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].tolist()
    except Exception:
        return None


def _clip_encode_text(text: str) -> Optional[List[float]]:
    """Return L2-normalised 768-dim CLIP text embedding, or None on failure."""
    result = _clip_encode_text_batch([text])
    return result[0] if result else None


def _clip_encode_text_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Encode a batch of texts in one CLIP forward pass.

    Returns list of L2-normalised 768-dim embeddings (one per text), or None on failure.
    Significantly faster than calling _clip_encode_text N times — use this for
    per-landmark batch scoring.
    """
    if not texts or not _load_clip():
        return None
    try:
        model = _CLIP_CACHE["model"]
        processor = _CLIP_CACHE["processor"]
        torch = _CLIP_CACHE["torch"]
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.tolist()
    except Exception:
        return None


def _clipscore_reward(pred_text: str, image: Optional[Path]) -> float:
    """Cosine similarity between CLIP text and image embeddings, mapped to [0, 1]."""
    if image is None:
        return 0.0
    text_emb = _clip_encode_text(pred_text)
    image_emb = _clip_encode_image(image)
    if text_emb is None or image_emb is None:
        return 0.0
    try:
        torch = _CLIP_CACHE["torch"]
        t = torch.tensor(text_emb)
        i = torch.tensor(image_emb)
        cosine = float((t * i).sum().item())
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
    except Exception:
        return 0.0


def _resolve_reference_text(ref: Dict[str, Any]) -> str:
    if isinstance(ref.get("label"), dict):
        lbl = ref["label"]
        if isinstance(lbl.get("enhanced_instruction"), str):
            return lbl["enhanced_instruction"]
    if isinstance(ref.get("enhanced_instruction"), str) and ref.get("enhanced_instruction"):
        return str(ref["enhanced_instruction"])
    prior = ref.get("prior") or {}
    if isinstance(prior, dict) and prior.get("instruction"):
        return str(prior["instruction"])
    return str(ref.get("instruction", ""))


def compute_composite_score(
    pred: Dict[str, Any],
    ref: Dict[str, Any],
    image: Optional[Path] = None,
    meta: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute composite score and per-metric sub-scores for one sample.

    The ``meta`` dict can carry ``latency_ms`` (int|float) to enable the
    ``latency_reward`` metric when using ``DEFAULT_COMPOSITE_WEIGHTS_V2``.
    """
    weights = weights or DEFAULT_COMPOSITE_WEIGHTS
    pred_text = _extract_instruction(pred)
    ref_text = _resolve_reference_text(ref)

    metric_values = {
        "action_reward": _action_reward(pred, ref),
        "lane_reward": _lane_reward(pred, ref),
        "format_reward": _format_reward(pred),
        "length_reward": _length_reward(pred),
        "bertscore_reward": _bertscore_reward(pred_text, ref_text),
        "clipscore_reward": _clipscore_reward(pred_text, image),
        "cider_reward": _cider_like(pred_text, ref_text),
        "rouge_l_reward": _rouge_l_f1(pred_text, ref_text),
    }

    # Compute latency reward if the weight dict includes it
    if "latency_reward" in weights:
        latency_ms = None
        if meta and "latency_ms" in meta:
            latency_ms = meta["latency_ms"]
        metric_values["latency_reward"] = _latency_reward(latency_ms)

    composite = 0.0
    for key, weight in weights.items():
        composite += float(weight) * float(metric_values.get(key, 0.0))

    metric_values["composite_score"] = round(max(0.0, min(1.0, composite)), 6)
    return metric_values


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _resolve_image_path(ref: Dict[str, Any], data_root: Optional[Path]) -> Optional[Path]:
    images = ref.get("images") or {}
    frame = None
    if isinstance(images, dict):
        frames = images.get("frames") or []
        if frames:
            frame = frames[-1]
    if not frame:
        return None

    candidate = Path(frame)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if data_root:
        resolved = data_root / frame
        if resolved.exists():
            return resolved
    return candidate if candidate.exists() else None


def _build_judge_outputs(pred: Dict[str, Any], ref: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    pred_payload = {
        "enhanced_instruction": _extract_instruction(pred),
        "relevant_landmarks": pred.get("relevant_landmarks", []) or [],
        "potential_hazards": pred.get("potential_hazards", []) or [],
        "reasoning": pred.get("reasoning") or "",
    }
    ref_payload = {
        "enhanced_instruction": _resolve_reference_text(ref),
        "relevant_landmarks": ref.get("relevant_landmarks", []) or [],
        "potential_hazards": ref.get("potential_hazards", []) or [],
        "reasoning": "Reference label",
    }
    return [("prediction", pred_payload), ("reference", ref_payload)]


def _judge_score_for_sample(sample_id: str, pred: Dict[str, Any], ref: Dict[str, Any], judge_model: str) -> Optional[float]:
    from navbuddy.eval.judge import judge_sample

    sample_stub = {
        "id": sample_id,
        "prior": {"instruction": (_resolve_reference_text(ref) or "")},
        "maneuver": ref.get("maneuver", "UNKNOWN"),
    }
    result = judge_sample(
        sample_id=sample_id,
        sample=sample_stub,
        outputs=_build_judge_outputs(pred, ref),
        judge_model=judge_model,
    )
    if result.error:
        return None
    for ranking in result.rankings:
        if ranking.model == "prediction":
            return float(ranking.total_score)
    return None


def evaluate_composite_metrics(
    predictions_path: Path,
    labels_path: Path,
    data_root: Optional[Path] = None,
    judge_model: Optional[str] = None,
    judge_subsample: int = 0,
    weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate predictions with composite metrics and optional judge calibration."""
    weights = weights or DEFAULT_COMPOSITE_WEIGHTS

    labels: Dict[str, Dict[str, Any]] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("id")
            if sid:
                labels[sid] = row

    predictions: List[Dict[str, Any]] = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            predictions.append(json.loads(line))

    per_sample: List[Dict[str, Any]] = []
    for pred in predictions:
        sid = _extract_sample_id(pred)
        if not sid or sid not in labels:
            continue

        ref = labels[sid]
        image_path = _resolve_image_path(ref, data_root)

        # Extract latency from inference metadata if available
        latency_ms = None
        inference_meta = pred.get("inference_metadata")
        if isinstance(inference_meta, dict):
            latency_ms = inference_meta.get("latency_ms")

        scores = compute_composite_score(
            pred=pred,
            ref=ref,
            image=image_path,
            meta={"sample_id": sid, "latency_ms": latency_ms},
            weights=weights,
        )
        per_sample.append(
            {
                "sample_id": sid,
                **scores,
            }
        )

    metric_keys = [
        "action_reward",
        "lane_reward",
        "format_reward",
        "length_reward",
        "bertscore_reward",
        "clipscore_reward",
        "cider_reward",
        "rouge_l_reward",
    ]
    if "latency_reward" in weights:
        metric_keys.append("latency_reward")
    metric_keys.append("composite_score")

    means = {key: round(_mean(row[key] for row in per_sample), 6) for key in metric_keys}

    report: Dict[str, Any] = {
        "mode": "composite",
        "weights": weights,
        "total_predictions": len(predictions),
        "matched_samples": len(per_sample),
        "unmatched_samples": len(predictions) - len(per_sample),
        "metrics_mean": means,
    }

    if judge_model and judge_subsample > 0 and per_sample:
        rng = random.Random(seed)
        sampled = per_sample if judge_subsample >= len(per_sample) else rng.sample(per_sample, judge_subsample)

        composite_vals: List[float] = []
        judge_vals: List[float] = []
        judged_ids: List[str] = []

        pred_lookup = {(_extract_sample_id(p) or ""): p for p in predictions}

        for row in sampled:
            sid = row["sample_id"]
            pred = pred_lookup.get(sid)
            ref = labels.get(sid)
            if pred is None or ref is None:
                continue

            jscore = _judge_score_for_sample(sid, pred, ref, judge_model=judge_model)
            if jscore is None:
                continue

            composite_vals.append(float(row["composite_score"]))
            judge_vals.append(float(jscore))
            judged_ids.append(sid)

        calibration = {
            "judge_model": judge_model,
            "requested_subsample": int(judge_subsample),
            "successful_judged": len(judge_vals),
            "sample_ids": judged_ids,
            "spearman": round(_spearman(composite_vals, judge_vals), 6) if judge_vals else None,
            "kendall": round(_kendall_tau(composite_vals, judge_vals), 6) if judge_vals else None,
            "refit_threshold": 0.35,
        }
        spearman_value = calibration.get("spearman")
        calibration["needs_weight_refit"] = (
            spearman_value is None or float(spearman_value) < float(calibration["refit_threshold"])
        )
        report["judge_calibration"] = calibration

    return report


__all__ = [
    "DEFAULT_COMPOSITE_WEIGHTS",
    "DEFAULT_COMPOSITE_WEIGHTS_V2",
    "_latency_reward",
    "compute_composite_score",
    "evaluate_composite_metrics",
]
