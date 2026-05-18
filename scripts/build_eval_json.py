#!/usr/bin/env python3
"""Build data/eval.json from data/results/*.jsonl and data/canonical_gt.jsonl.

Computes per-model, per-modality metrics:
  - BERTScore F1 (ip, vp, p, ap)
  - Action Accuracy (ip_act, vp_act, ...)
  - Lane Change F1 (ip_lc, vp_lc, ...)
  - Lane Count MAE (ip_ln, vp_ln, ...)
  - Response Length in words (ip_len, vp_len, ...)
  - Overall average length (len)

Usage:
    python scripts/build_eval_json.py
    python scripts/build_eval_json.py --data-dir data --output data/eval.json
"""

import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path

MODALITY_MAP = {
    "image + prior": "ip",
    "video + prior": "vp",
    "prior": "p",
    "augment + prior": "ap",
}

# Display name mapping (model_id -> short display name)
DISPLAY_NAMES = {
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "openai/gpt-4.1": "GPT-4.1",
    "openai/gpt-4.1-mini": "GPT-4.1-mini",
    "openai/gpt-4.1-nano": "GPT-4.1-nano",
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "GPT-5.4-mini",
    "openai/gpt-5.4-nano": "GPT-5.4-nano",
    "openai/gpt-5.2-codex": "GPT-5.2-codex",
    "openai/gpt-5.1-codex-mini": "GPT-5.1-codex-mini",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "anthropic/claude-3-haiku": "Claude 3 Haiku",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "anthropic/claude-opus-4": "Claude Opus 4",
    "anthropic/claude-opus-4.1": "Claude Opus 4.1",
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "x-ai/grok-4.1-fast": "Grok 4.1 Fast",
    "x-ai/grok-4-fast": "Grok 4 Fast",
    "x-ai/grok-4.20-beta": "Grok 4.20",
    "qwen/qwen3-vl-8b-instruct": "Qwen3-VL-8B",
    "qwen/qwen3-vl-32b-instruct": "Qwen3-VL-32B",
    "qwen/qwen2.5-vl-72b-instruct": "Qwen2.5-VL-72B",
    "qwen/qwen2.5-vl-3b-instruct": "Qwen2.5-VL-3B",
    "qwen/qwen3.5-flash": "Qwen3.5-Flash",
}


def display_name(model_id: str) -> str:
    return DISPLAY_NAMES.get(model_id.lower(), model_id.split("/")[-1])


def main():
    parser = argparse.ArgumentParser(description="Build eval.json from results")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output", default="data/eval.json", help="Output path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    gt_path = data_dir / "canonical_gt.jsonl"

    # Load ground truth
    gt = {}
    if gt_path.exists():
        with open(gt_path) as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    gt[g["sample_id"]] = g

    # Collect results per model per modality per sample
    # model_id -> modality_key -> sample_id -> { prediction lists }
    per_sample = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "actions": [],
        "lc_preds": [],
        "lane_counts": [],
        "word_counts": [],
    })))

    latencies = defaultdict(list)
    params_map = {}

    for result_file in sorted(glob.glob(str(data_dir / "results" / "*.jsonl"))):
        with open(result_file) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed rows rather than crashing on a single bad line.
                    print(f"  warn: skipping malformed line {line_no} in {Path(result_file).name}")
                    continue
                if r.get("error"):
                    continue

                model_id = r.get("model_id", "").lower()
                modality = r.get("modality", "")
                mod_key = MODALITY_MAP.get(modality)
                if not mod_key:
                    continue

                sample_id = r.get("id", "")
                bucket = per_sample[model_id][mod_key][sample_id]

                # Collect predictions per sample
                pred_action = r.get("next_action", "")
                if pred_action:
                    bucket["actions"].append(pred_action)

                pred_lc = r.get("lane_change_required", "")
                if pred_lc != "":
                    pred_bool = pred_lc if isinstance(pred_lc, bool) else str(pred_lc).lower() in ("yes", "true")
                    bucket["lc_preds"].append(pred_bool)

                pred_lanes = r.get("lanes_count")
                if pred_lanes is not None:
                    bucket["lane_counts"].append(pred_lanes)

                inst = r.get("enhanced_instruction", "")
                if inst:
                    bucket["word_counts"].append(len(inst.split()))

                # Latency
                meta = r.get("inference_metadata", {})
                if meta and meta.get("latency_ms"):
                    latencies[model_id].append(meta["latency_ms"] / 1000.0)

    # Build output
    def _majority(items):
        """Return the most common item (majority vote)."""
        from collections import Counter
        if not items:
            return None
        return Counter(items).most_common(1)[0][0]

    models_out = []
    for model_id, mod_data in sorted(per_sample.items()):
        entry = {"m": display_name(model_id)}

        # Latency
        if latencies[model_id]:
            entry["lat"] = round(sum(latencies[model_id]) / len(latencies[model_id]), 2)

        # Per-modality metrics — aggregate per sample first, then compute overall
        all_word_counts = []
        for mod_key in ["ip", "vp", "p", "ap"]:
            samples_data = mod_data.get(mod_key)
            if not samples_data:
                continue

            action_scores = []
            lc_tp = lc_fp = lc_fn = lc_tn = 0
            lane_diffs = []
            word_counts = []

            for sample_id, preds in samples_data.items():
                g = gt.get(sample_id, {})

                # Action accuracy — majority vote across runs for this sample
                if preds["actions"]:
                    gt_action = g.get("next_action", "")
                    acceptable = g.get("acceptable_actions", [gt_action]) if gt_action else []
                    if gt_action:
                        voted_action = _majority(preds["actions"])
                        action_scores.append(1 if voted_action in acceptable else 0)

                # Lane change F1 — majority vote → confusion matrix
                if preds["lc_preds"]:
                    gt_lc = g.get("lane_change_required")
                    if gt_lc is not None:
                        voted_lc = sum(preds["lc_preds"]) > len(preds["lc_preds"]) / 2
                        if voted_lc and gt_lc:
                            lc_tp += 1
                        elif voted_lc and not gt_lc:
                            lc_fp += 1
                        elif not voted_lc and gt_lc:
                            lc_fn += 1
                        else:
                            lc_tn += 1

                # Lane count MAE — mean across runs, then compare to GT
                if preds["lane_counts"]:
                    gt_lanes = g.get("lanes_count")
                    if gt_lanes is not None:
                        avg_lanes = round(sum(preds["lane_counts"]) / len(preds["lane_counts"]))
                        lane_diffs.append(abs(avg_lanes - gt_lanes))

                # Response length — mean across runs for this sample
                if preds["word_counts"]:
                    avg_wc = sum(preds["word_counts"]) / len(preds["word_counts"])
                    word_counts.append(avg_wc)

            if action_scores:
                entry[f"{mod_key}_act"] = round(sum(action_scores) / len(action_scores), 4)
            # LC F1 (positive class = "lane change required"). 0 when no positive predictions or no positives in GT.
            if (lc_tp + lc_fp + lc_fn + lc_tn) > 0:
                precision = lc_tp / (lc_tp + lc_fp) if (lc_tp + lc_fp) else 0.0
                recall = lc_tp / (lc_tp + lc_fn) if (lc_tp + lc_fn) else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                entry[f"{mod_key}_lc"] = round(f1, 4)
            if lane_diffs:
                entry[f"{mod_key}_ln"] = round(sum(lane_diffs) / len(lane_diffs), 4)
            if word_counts:
                entry[f"{mod_key}_len"] = round(sum(word_counts) / len(word_counts), 1)
                all_word_counts.extend(word_counts)

        # Overall avg length
        if all_word_counts:
            entry["len"] = round(sum(all_word_counts) / len(all_word_counts), 1)

        models_out.append(entry)

    # Preserve existing BERTScore values from current eval.json if available
    existing = {}
    existing_path = Path(args.output)
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        existing_models = existing.get("models", existing if isinstance(existing, list) else [])
        existing_by_name = {m["m"]: m for m in existing_models}
        for entry in models_out:
            old = existing_by_name.get(entry["m"], {})
            for key in ["ip", "vp", "p", "ap", "params"]:
                if key in old and key not in entry:
                    entry[key] = old[key]

    # Canonical metric definitions. These describe exactly what the code above computes —
    # update both together if the methodology changes.
    output = {
        "_comment": existing.get("_comment",
            "Per-model NavBuddy-100 leaderboard. Per-sample aggregation = majority vote "
            "(LC, action) or mean of rounded predictions (lanes). See docs/scoring.md."),
        "dataset": existing.get("dataset", "NavBuddy-100"),
        "gt_source": existing.get("gt_source", "data/canonical_gt.jsonl"),
        "results_source": existing.get("results_source", "data/results/*.jsonl"),
        "metrics": {
            "bertscore": "BERTScore F1 (rescale_with_baseline=True, model=roberta-large). "
                         "Computed per sample, then averaged. Not regenerated by this script — "
                         "values are preserved from prior runs.",
            "action": "Next action accuracy. Per sample: majority vote across all runs in the "
                      "(model, modality) bucket; score = 1.0 if vote is in GT's acceptable_actions, "
                      "else 0.0. NO direction-group fallback at leaderboard level (only the "
                      "per-sample scorer in metric_eval.py uses left/right/straight grouping).",
            "lane_change": "Lane change F1. Per sample: majority vote across runs to get one yes/no "
                           "prediction. Across samples: precision/recall/F1 with positive class = "
                           "'lane change required'. Samples with GT=null are excluded.",
            "lanes_count": "Lanes count MAE. Per sample: mean of predicted lane counts (rounded to "
                           "nearest int), then absolute diff to GT. Across samples: arithmetic mean. "
                           "Lower is better; 0 = exact. NOT the partial-credit 1.0/0.5/0.0 scheme "
                           "used by metric_eval.score_lanes_count."
        },
        "modalities": existing.get("modalities", ["image + prior", "video + prior", "prior", "augment + prior"]),
        "models": models_out,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(models_out)} models to {args.output}")
    for m in models_out[:3]:
        print(f"  {m['m']}: ip_len={m.get('ip_len')}, vp_len={m.get('vp_len')}, p_len={m.get('p_len')}, ap_len={m.get('ap_len')}")


if __name__ == "__main__":
    main()
