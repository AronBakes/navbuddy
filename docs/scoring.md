# NavBuddy Evaluation Scoring

Reference documentation for how NavBuddy scores model predictions against
ground-truth (GT) labels.

There are **two distinct scoring layers**, and they do not agree on every metric:

1. **Per-sample scorer** (`navbuddy/eval/metric_eval.py`) — used by tools that need
   a 0-1 score per `(sample, model, modality)`, e.g. building leaderboards in
   ad-hoc notebooks, exporting per-sample diffs to the dashboard. Lenient where
   it can be (direction groups, partial credit on lanes).
2. **Leaderboard aggregator** (`scripts/build_eval_json.py`) — generates
   `data/eval.json` consumed by the public leaderboard on the GitHub Pages site.
   Stricter (no direction group fallback, lanes uses MAE).

Sections 1–5 describe the **per-sample scorer**. Section 6 describes how
duplicates are handled. Section 7 describes the **leaderboard aggregator** and
where its metrics differ from the per-sample scorer — this is the authoritative
description for what `eval.json` and the published leaderboard actually contain.

---

## 1. Action Scoring (`score_action`)

Each prediction's `next_action` is scored on **direction accuracy** — did
the model get the correct direction (left, right, or straight)?

| Tier | Condition | Score |
|------|-----------|-------|
| Exact match | `pred == gt` | **1.0** |
| Acceptable alternative | `pred` is in the GT sample's `acceptable_actions` list | **1.0** |
| Correct direction | `pred` and `gt` belong to the same direction group | **1.0** |
| Wrong direction | None of the above | **0.0** |

Tiers are evaluated top-to-bottom; the first match wins.

---

## 2. Direction Groups

All actions map to one of three direction groups. Two actions in the same
group are considered equivalent for scoring purposes.

| Group | Members |
|-------|---------|
| `left` | `turn_left`, `fork_left`, `merge_left`, `keep_left`, `slight_left`, `sharp_left`, `uturn_left`, `roundabout_left`, `ramp_left` |
| `right` | `turn_right`, `fork_right`, `merge_right`, `keep_right`, `slight_right`, `sharp_right`, `uturn_right`, `roundabout_right`, `ramp_right` |
| `straight` | `straight`, `continue`, `keep_straight` |

Actions not present in any group (e.g., bare `merge`, `roundabout`) have no
group membership and can only score via exact match or `acceptable_actions`.

---

## 3. Acceptable Actions

Each GT sample may include an `acceptable_actions` array listing every valid
action for that scenario.  This handles ambiguous road geometry -- for example,
a gentle curve that could reasonably be labelled `keep_left` or `straight`.

When `acceptable_actions` is present and the prediction matches any entry in the
list, the prediction receives full credit (1.0), even if it does not match the
primary GT action.

---

## 4. Lane Change Scoring (`score_lane_change`)

Binary match after normalizing the value to a boolean (`"yes"` / `true` -> True,
etc.):

| Condition | Score |
|-----------|-------|
| GT is `None` (ambiguous annotation) | **excluded** -- sample is skipped entirely for this metric |
| Prediction matches GT | **1.0** |
| Prediction does not match GT | **0.0** |
| Prediction is missing but GT is clear | **0.0** |

---

## 5. Lane Count Scoring (`score_lanes_count`)

Normalized lane count accuracy on a 0-1 scale. Linear decay from 1.0 (exact)
to 0.0 (off by 4 or more lanes).

| Condition | Score |
|-----------|-------|
| GT is `None` | **excluded** |
| Prediction is missing | **excluded** |
| Exact match | **1.0** |
| Off by 1 | **0.75** |
| Off by 2 | **0.50** |
| Off by 3 | **0.25** |
| Off by 4+ | **0.0** |

Formula: `max(0, 1 - abs(pred - gt) / 4)`

---

## 6. Handling Duplicates (per-sample scorer)

A model may produce multiple result entries per `(sample, modality)` — for
example, the matrix runner generates 5 image-prior rows per sample (clean +
fog + rain + night + motion_blur, all tagged as `modality = "image + prior"`
with the augmentation in a separate `augment` field). The per-sample scorer
treats every row independently. Callers are responsible for filtering by
`augment` if they want a clean-only view.

For the leaderboard aggregator's duplicate handling, see section 7.

---

## 7. Leaderboard Aggregator (`scripts/build_eval_json.py`)

Reads `data/results/*.jsonl`, aggregates into per-`(model, modality)` rows,
writes `data/eval.json`. This is what the github.io leaderboard renders.

### Per-sample aggregation (within each `(model, modality, sample)` bucket)

The matrix runner produces multiple rows per sample for the same `modality`
(one clean run + one per augmentation). The aggregator collapses them per
sample as follows:

| Field | Collapse rule |
|-------|---------------|
| `next_action` | **Majority vote** (most frequent action string) |
| `lane_change_required` | **Majority vote** (boolean) |
| `lanes_count` | **Mean rounded to nearest int** |

Each sample then contributes one prediction to the final metric.

### Final metrics (across samples)

| Output key | Metric | Definition |
|------------|--------|------------|
| `{mod}_act` | **Next action accuracy** | Fraction of samples where the majority-vote prediction is in `gt.acceptable_actions`. **No direction-group fallback** — strict match only. Range: 0–1, higher is better. |
| `{mod}_lc`  | **Lane change F1** | Treats "lane change required = true" as the positive class. Builds TP/FP/TN/FN from majority-vote predictions vs GT, computes `2 · P · R / (P + R)`. Samples with `gt.lane_change_required = null` are excluded. Range: 0–1, higher is better. |
| `{mod}_ln`  | **Lanes count MAE** | Arithmetic mean of `\|round(mean(predictions)) − gt_lanes\|` across samples. **Lower is better; 0 = exact.** This is mean absolute error, *not* the partial-credit scheme used by `score_lanes_count`. |
| `{mod}_len` | Mean response word count (information only, not a quality metric) |

Modality keys: `ip` = `image + prior`, `vp` = `video + prior`, `p` = `prior`, `ap` = `augment + prior`.

### Differences vs. the per-sample scorer

| Metric | `metric_eval.py` (per-sample) | `build_eval_json.py` (leaderboard) |
|--------|-------------------------------|-------------------------------------|
| Action | exact OR acceptable OR same direction group → 1.0; else 0.0 | exact OR acceptable → 1.0; else 0.0. **No direction-group fallback.** |
| Lane change | per-sample 0/1 match, then mean = accuracy | per-sample majority vote, then **F1 across samples** |
| Lanes count | partial credit `max(0, 1 − \|err\|/4)` | **MAE** (raw absolute error) |

If you want to reproduce the leaderboard exactly, use `build_eval_json.py`.
If you want lenient per-sample scoring (e.g., to debug a single model on a
single sample), use `metric_eval.score_result`.

---

## 8. Metric Choices (rationale)

| Metric | What it measures | Why we use it |
|--------|-----------------|---------------|
| **Next action accuracy** (strict) | Did the model produce an action that matches GT or a GT-approved alternative? | Strict accuracy keeps the metric interpretable. `acceptable_actions` already handles the genuinely ambiguous cases (≈17% of GT samples); direction-group fallback would give partial credit for wrong-but-same-side answers, which we decided is not what the leaderboard should reward. |
| **Lane change F1** | Did the model correctly identify whether a lane change is needed? | The class is imbalanced (≈40% yes / 60% no on labelled GT-100). F1 punishes models that always-say-no in a way that raw accuracy does not. |
| **Lanes count MAE** | How far off is the predicted lane count from GT? | MAE is more informative than exact-match and more honest than a custom partial-credit scheme. Easy to interpret: 0.4 means models are off by less than half a lane on average. |

---

# Experimental Metrics

The metrics below are under evaluation and may change. They measure instruction
text quality rather than structured navigation accuracy.

## 8. Set-Based Scoring (Landmarks and Hazards)

`relevant_landmarks` and `potential_hazards` are scored with **Jaccard
similarity** (intersection over union of lowercased string sets). Both empty
sets = 1.0 (correctly identified nothing).

In detailed mode, **precision / recall / F1** are also computed. When
`fuzzy=True` (default), landmark matching uses fuzzy string matching via
`navbuddy.eval.landmark_matcher` to handle paraphrases and abbreviations.

---

## 9. Text Similarity Metrics

The `enhanced_instruction` field is scored using several complementary metrics.

### Lexical metrics (always computed)

| Metric | Description |
|--------|-------------|
| `rouge_1` | ROUGE-1 F1 (unigram overlap) |
| `rouge_2` | ROUGE-2 F1 (bigram overlap) |
| `rouge_l` | ROUGE-L F1 (longest common subsequence) |
| `bleu_1` .. `bleu_4` | BLEU-1 through BLEU-4 with brevity penalty |
| `token_f1` | Bag-of-words token-level F1 |

### Semantic metrics (when `use_semantic=True`)

| Metric | Description |
|--------|-------------|
| `semantic_similarity` | Cosine similarity of sentence embeddings using `all-MiniLM-L6-v2` (sentence-transformers). Falls back to `rouge_l` if not installed. |
| `bertscore_f1` | BERTScore F1 using `roberta-large` with baseline rescaling. Falls back to `rouge_l` on error or missing dependency. |

### Image-text alignment (when `image_path` is provided in detailed mode)

| Metric | Description |
|--------|-------------|
| `clipscore` | CLIPScore reward measuring text-image alignment (hallucination detection). |

---

## 10. Aggregation

All metrics use **equal sample weighting**: per-sample scores are computed
first, then averaged across samples. This ensures each navigation scenario
contributes equally to the final score regardless of the number of inference
entries per sample.
