# NavBuddy Evaluation Scoring

Reference documentation for how NavBuddy scores model predictions against
ground-truth (GT) labels.  All scoring logic lives in
`navbuddy/eval/metric_eval.py`.

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

## 6. Handling Duplicates

A model may produce multiple result entries per sample (e.g., different prompt
versions, modalities, or retries). NavBuddy handles this differently depending
on the metric type:

**Binary metrics (lane change F1):** Majority vote across entries for each
sample. The sample contributes a single prediction — whichever value appears
most often. This ensures each sample is weighted equally (weight = 1)
regardless of how many entries it has.

**Accuracy metrics (next action, lane count MAE):** Per-sample scores are
computed first by averaging all entries within each sample, then averaged across
samples. This gives each navigation scenario equal weight in the final score,
even if some samples have more inference runs than others.

---

## 7. Metric Choices

| Metric | What it measures | Why we use it |
|--------|-----------------|---------------|
| **Direction accuracy** | Did the model predict the correct direction (left/right/straight)? | The core navigation task — a wrong direction is a wrong turn. Direction groups absorb specificity differences (e.g., `fork_left` vs `turn_left`) that don't affect the driver's decision. |
| **Lane change F1** | Did the model correctly identify whether a lane change is needed? | Lane changes are safety-critical. F1 balances false positives (unnecessary lane changes) and false negatives (missed lane changes). |
| **Lane count MAE** | How far off is the predicted lane count from ground truth? | Lane awareness matters for lane-change decisions. MAE is more informative than binary exact-match — being off by 1 is less wrong than being off by 3. |

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
