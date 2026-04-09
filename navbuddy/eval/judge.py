"""LLM-as-Judge for ranking VLM outputs.

Uses a frontier model to score and rank multiple outputs for the same sample.
Supports batch processing with retry logic and rate limiting.
"""

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx
from pydantic import BaseModel, Field

# Rate limiting
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Exponential backoff base


# Judge criteria for scoring
JUDGE_CRITERIA = {
    "accuracy": "Does the instruction accurately describe what's visible in the scene and match the intended maneuver?",
    "specificity": "Does it reference specific, visible landmarks (traffic lights, signs, buildings, intersections)?",
    "actionability": "Is it clear what lane to use, when to act, and what to look for?",
    "safety": "Does it mention relevant hazards, conditions, or safety considerations when appropriate?",
}

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating navigation instructions for autonomous vehicles.

Your task is to score and rank multiple navigation instruction outputs for the same driving scenario.

## Scoring Criteria (1-5 scale each):

1. **Accuracy** (1-5): Does the instruction accurately describe what's visible and match the maneuver?
   - 5: Perfectly accurate, matches scene exactly
   - 3: Mostly accurate with minor issues
   - 1: Incorrect or misleading

2. **Specificity** (1-5): Does it reference specific, visible landmarks?
   - 5: Multiple specific landmarks (e.g., "at the red traffic light before the Shell station")
   - 3: Some landmarks mentioned
   - 1: Generic instruction with no landmarks

3. **Actionability** (1-5): Is it clear what to do and when?
   - 5: Clear lane guidance, timing, and visual cues
   - 3: Somewhat clear but missing details
   - 1: Vague or confusing

4. **Safety** (1-5): Does it mention relevant hazards or conditions?
   - 5: Appropriately mentions hazards when visible
   - 3: Basic safety awareness
   - 1: Ignores obvious hazards (or N/A if none visible)

## Output Format

Return a JSON object with this exact structure:
{
  "rankings": [
    {
      "model": "<model_id>",
      "rank": 1,
      "scores": {
        "accuracy": 5,
        "specificity": 4,
        "actionability": 5,
        "safety": 4
      },
      "total_score": 18,
      "reasoning": "Brief explanation of ranking"
    }
  ],
  "best_output": "<model_id of rank 1>",
  "notes": "Any additional observations"
}

Rank from 1 (best) to N (worst). Higher total scores should have lower (better) ranks."""


POINTWISE_JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating a single navigation instruction for an autonomous vehicle.

Your task is to score this navigation instruction on a fixed rubric. Evaluate it independently — do NOT compare it to any other output.

## Scoring Criteria (1-5 scale each):

1. **Accuracy** (1-5): Does the instruction accurately describe what's visible and match the maneuver?
   - 5: Perfectly accurate, matches scene exactly
   - 3: Mostly accurate with minor issues
   - 1: Incorrect or misleading

2. **Specificity** (1-5): Does it reference specific, visible landmarks?
   - 5: Multiple specific landmarks (e.g., "at the red traffic light before the Shell station")
   - 3: Some landmarks mentioned
   - 1: Generic instruction with no landmarks

3. **Actionability** (1-5): Is it clear what to do and when?
   - 5: Clear lane guidance, timing, and visual cues
   - 3: Somewhat clear but missing details
   - 1: Vague or confusing

4. **Safety** (1-5): Does it mention relevant hazards or conditions?
   - 5: Appropriately mentions hazards when visible
   - 3: Basic safety awareness
   - 1: Ignores obvious hazards (or N/A if none visible)

## Output Format

Return a JSON object with this exact structure:
{
  "scores": {
    "accuracy": 4,
    "specificity": 3,
    "actionability": 5,
    "safety": 4
  },
  "total_score": 16,
  "reasoning": "Brief explanation of the scores"
}

Be consistent and calibrated. A score of 3 should always mean the same quality level regardless of which output you are evaluating."""


def _compress_image(path: Path, max_bytes: int = 200_000) -> str:
    """Compress an image to JPEG under max_bytes and return a data URL.

    Large PNGs (e.g. map tiles) can be 4+ MB base64, which causes
    timeouts/empty responses from some providers via OpenRouter.
    """
    import base64
    from io import BytesIO
    from PIL import Image

    img = Image.open(path)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Try progressively lower quality
    for quality in (85, 60, 40, 25):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes:
            break

    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    return f"data:image/jpeg;base64,{b64}"


def _build_image_content(
    sample: Dict[str, Any],
    data_root: Optional[Path],
) -> List[Dict[str, Any]]:
    """Build image content blocks for the judge prompt.

    Encodes the same dashcam frames + overhead map that the VLM saw during
    inference so the judge can verify visual accuracy claims.

    Returns a list of OpenAI-format content blocks (image_url type).
    Returns empty list if data_root is None or images can't be found.
    """
    if data_root is None:
        return []

    from navbuddy.eval.inference import image_to_data_url

    blocks: List[Dict[str, Any]] = []
    images = sample.get("images", {})

    # Dashcam frames (chronological order)
    for frame_path in images.get("frames", []):
        full_path = data_root / frame_path
        if full_path.exists():
            data_url = image_to_data_url(full_path)
            blocks.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

    # Overhead map — compress large PNGs to avoid provider payload limits
    overhead = images.get("overhead")
    if overhead:
        full_path = data_root / overhead
        if full_path.exists():
            if full_path.stat().st_size > 200_000:
                data_url = _compress_image(full_path)
            else:
                data_url = image_to_data_url(full_path)
            blocks.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

    return blocks


class JudgeScore(BaseModel):
    """Scores for a single output."""
    accuracy: int = Field(..., ge=1, le=5)
    specificity: int = Field(..., ge=1, le=5)
    actionability: int = Field(..., ge=1, le=5)
    safety: int = Field(..., ge=1, le=5)


class JudgeRanking(BaseModel):
    """Ranking for a single model output."""
    model: str
    rank: int
    scores: JudgeScore
    total_score: int
    reasoning: str
    enhanced_instruction: str = ""  # The actual output text for training
    latency_score: Optional[int] = None  # Algorithmic 0-5, not LLM-judged
    total_score_with_latency: Optional[int] = None  # total_score + latency_score


class JudgeResult(BaseModel):
    """Complete judge result for a sample."""
    sample_id: str
    judge_model: str
    mode: str = "pairwise"  # "pairwise", "pairwise_blind", "pointwise_blind"
    images_included: bool = False
    rankings: List[JudgeRanking]
    best_output: str
    notes: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None


def build_judge_prompt(
    sample: Dict[str, Any],
    outputs: List[Tuple[str, Dict[str, Any]]],  # List of (model_id, output_dict)
) -> str:
    """Build the prompt for the judge model."""

    # Original instruction
    prior_instruction = sample.get("prior", {}).get("instruction", "Unknown")
    maneuver = sample.get("maneuver", "Unknown")

    # Build outputs section
    outputs_text = []
    for model_id, output in outputs:
        enhanced = output.get("enhanced_instruction", "N/A")
        landmarks = output.get("relevant_landmarks", [])
        hazards = output.get("potential_hazards", [])
        reasoning = output.get("reasoning") or ""

        outputs_text.append(f"""
### Model: {model_id}
**Enhanced Instruction:** {enhanced}
**Landmarks:** {', '.join(landmarks) if landmarks else 'None mentioned'}
**Hazards:** {', '.join(hazards) if hazards else 'None mentioned'}
**Reasoning:** {reasoning[:200] + '...' if len(reasoning) > 200 else reasoning}
""")

    prompt = f"""## Driving Scenario

**Original Navigation Instruction:** {prior_instruction}
**Maneuver Type:** {maneuver}

## Outputs to Rank

{chr(10).join(outputs_text)}

## Task

Score and rank these outputs according to the criteria. Return your evaluation as JSON.
"""
    return prompt


def build_blind_judge_prompt(
    sample: Dict[str, Any],
    outputs: List[Tuple[str, Dict[str, Any]]],  # List of (model_id, output_dict)
    data_root: Optional[Path] = None,
) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, str]]:
    """Build an anonymised prompt for the judge model.

    Replaces model IDs with "Response A", "Response B", etc. so the judge
    cannot be biased by knowing which model produced which output.

    When *data_root* is provided, the dashcam frames and overhead map are
    included as images so the judge can verify visual accuracy claims.

    Args:
        sample: The sample dict (must contain ``prior.instruction`` and ``maneuver``).
        outputs: List of ``(model_id, output_dict)`` tuples.
        data_root: Optional path to data directory for resolving image paths.

    Returns:
        A tuple of ``(prompt_content, label_map)`` where *prompt_content* is
        either a plain string (no images) or a list of multimodal content
        blocks, and *label_map* maps each anonymous label back to its model ID.
    """

    # Original instruction
    prior_instruction = sample.get("prior", {}).get("instruction", "Unknown")
    maneuver = sample.get("maneuver", "Unknown")

    # Build label_map and anonymised outputs section
    label_map: Dict[str, str] = {}
    outputs_text: List[str] = []

    for i, (model_id, output) in enumerate(outputs):
        label = f"Response {chr(65 + i)}"
        label_map[label] = model_id

        enhanced = output.get("enhanced_instruction", "N/A")
        landmarks = output.get("relevant_landmarks", [])
        hazards = output.get("potential_hazards", [])
        reasoning = output.get("reasoning") or ""

        outputs_text.append(f"""
### {label}
**Enhanced Instruction:** {enhanced}
**Landmarks:** {', '.join(landmarks) if landmarks else 'None mentioned'}
**Hazards:** {', '.join(hazards) if hazards else 'None mentioned'}
**Reasoning:** {reasoning[:200] + '...' if len(reasoning) > 200 else reasoning}
""")

    # Build list of anonymous labels for the instruction text
    label_list = ", ".join(f'"{label}"' for label in label_map)

    text = f"""## Driving Scenario

**Original Navigation Instruction:** {prior_instruction}
**Maneuver Type:** {maneuver}

The dashcam frames and overhead map below show the driving scene that each model was given. Use them to verify whether the outputs accurately describe what is visible.

## Outputs to Rank

{chr(10).join(outputs_text)}

## Task

Score and rank these outputs according to the criteria. Use the anonymous labels ({label_list}) as the "model" field in your JSON response. Return your evaluation as JSON.
"""

    # Build multimodal content if images available
    image_blocks = _build_image_content(sample, data_root)
    if image_blocks:
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(image_blocks)
        return content, label_map

    return text, label_map


def build_pointwise_prompt(
    sample: Dict[str, Any],
    output: Dict[str, Any],
    data_root: Optional[Path] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """Build a prompt for pointwise (single-output) judging.

    Evaluates one model output in isolation against the driving scenario,
    eliminating both positional bias and comparative anchoring.

    When *data_root* is provided, the dashcam frames and overhead map are
    included as images so the judge can verify visual accuracy claims.

    Args:
        sample: The sample dict (must contain ``prior.instruction`` and ``maneuver``).
        output: A single model's output dict.
        data_root: Optional path to data directory for resolving image paths.

    Returns:
        Either a plain text string or a list of multimodal content blocks.
    """
    prior_instruction = sample.get("prior", {}).get("instruction", "Unknown")
    maneuver = sample.get("maneuver", "Unknown")

    enhanced = output.get("enhanced_instruction", "N/A")
    landmarks = output.get("relevant_landmarks", [])
    hazards = output.get("potential_hazards", [])
    reasoning = output.get("reasoning") or ""

    text = f"""## Driving Scenario

**Original Navigation Instruction:** {prior_instruction}
**Maneuver Type:** {maneuver}

The dashcam frames and overhead map below show the driving scene that this model was given. Use them to verify whether the output accurately describes what is visible.

## Output to Evaluate

**Enhanced Instruction:** {enhanced}
**Landmarks:** {', '.join(landmarks) if landmarks else 'None mentioned'}
**Hazards:** {', '.join(hazards) if hazards else 'None mentioned'}
**Reasoning:** {reasoning[:200] + '...' if len(reasoning) > 200 else reasoning}

## Task

Score this output according to the criteria. Evaluate it on its own merits — do not assume or compare against other possible outputs. Return your evaluation as JSON.
"""

    # Build multimodal content if images available
    image_blocks = _build_image_content(sample, data_root)
    if image_blocks:
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(image_blocks)
        return content

    return text


def call_judge_model(
    prompt: Any,
    judge_model: str = "openai/gpt-4o",
    api_key: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
    system_prompt: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[str], int, int, int]:
    """Call the judge model via OpenRouter with retry logic.

    Args:
        prompt: Either a plain text string or a list of content blocks
               (multimodal) for the user message.
        judge_model: Model ID on OpenRouter.
        api_key: OpenRouter API key.
        max_retries: Number of retries on transient failures.
        system_prompt: Override default judge system prompt.

    Returns: (result_dict, error_message, latency_ms, tokens_in, tokens_out)
    """
    if system_prompt is None:
        system_prompt = JUDGE_SYSTEM_PROMPT

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None, "No OPENROUTER_API_KEY found", 0, 0, 0

    start_time = time.time()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/navbuddy",
                },
                json={
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,  # Low temperature for consistent judging
                    "max_tokens": 8192,  # Avoid 65k default; Grok can emit ~6k tokens
                    # Anthropic models via OpenRouter often return empty with json_object mode
                    **({} if "anthropic/" in judge_model else {"response_format": {"type": "json_object"}}),
                },
                timeout=90.0,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Rate limit handling
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY_BASE * (2 ** attempt)))
                time.sleep(retry_after)
                continue

            if response.status_code != 200:
                last_error = f"API error: {response.status_code} - {response.text[:200]}"
                if response.status_code >= 500:  # Server error, retry
                    time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                    continue
                return None, last_error, latency_ms, 0, 0

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)

            # Parse JSON response — handle models that wrap JSON in prose
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON object from text (e.g. Claude returns reasoning + JSON)
                import re
                m = re.search(r'\{[^{}]*"winner"\s*:\s*"[^"]*"[^{}]*\}', content)
                if m:
                    result = json.loads(m.group())
                else:
                    raise
            return result, None, latency_ms, tokens_in, tokens_out

        except httpx.TimeoutException:
            last_error = "Request timeout"
            time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
            continue
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
            continue
        except Exception as e:
            last_error = f"Error: {e}"
            time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
            continue

    latency_ms = int((time.time() - start_time) * 1000)
    return None, f"Failed after {max_retries} retries: {last_error}", latency_ms, 0, 0


def judge_sample(
    sample_id: str,
    sample: Dict[str, Any],
    outputs: List[Tuple[str, Dict[str, Any]]],
    judge_model: str = "openai/gpt-4o",
    api_key: Optional[str] = None,
    latencies: Optional[Dict[str, float]] = None,
    data_root: Optional[Path] = None,
) -> JudgeResult:
    """Judge and rank outputs for a single sample."""

    has_images = data_root is not None

    if len(outputs) < 2:
        return JudgeResult(
            sample_id=sample_id,
            judge_model=judge_model,
            mode="pairwise_blind",
            images_included=has_images,
            rankings=[],
            best_output=outputs[0][0] if outputs else "",
            error="Need at least 2 outputs to rank",
        )

    # Build model -> output mapping for enhanced_instruction lookup
    output_map = {model_id: output for model_id, output in outputs}

    # Use blind (anonymous) judging to prevent bias from model names
    prompt, label_map = build_blind_judge_prompt(sample, outputs, data_root=data_root)
    # Reverse map: anonymous label -> original model_id
    reverse_map = {label: model_id for label, model_id in label_map.items()}

    result, error, latency_ms, _tok_in, _tok_out = call_judge_model(prompt, judge_model, api_key)

    if error:
        return JudgeResult(
            sample_id=sample_id,
            judge_model=judge_model,
            mode="pairwise_blind",
            images_included=has_images,
            rankings=[],
            best_output="",
            error=error,
            latency_ms=latency_ms,
        )

    # Parse rankings and map anonymous labels back to model IDs
    # Handle models returning a list directly instead of {"rankings": [...]}
    raw_rankings = result if isinstance(result, list) else result.get("rankings", [])
    rankings = []
    for r in raw_rankings:
        try:
            label = r["model"]
            # Map anonymous label back to real model_id
            model_id = reverse_map.get(label, label)
            # Get enhanced_instruction from original output
            original_output = output_map.get(model_id, {})
            enhanced_instruction = original_output.get("enhanced_instruction", "")

            rankings.append(JudgeRanking(
                model=model_id,
                rank=r["rank"],
                scores=JudgeScore(**r["scores"]),
                total_score=r["total_score"],
                reasoning=r.get("reasoning", ""),
                enhanced_instruction=enhanced_instruction,
            ))
        except Exception:
            continue

    # Sort by rank
    rankings.sort(key=lambda x: x.rank)

    # Attach algorithmic latency scores if latency data provided
    if latencies:
        from navbuddy.eval.metrics_semantic import _latency_reward

        for ranking in rankings:
            ms = latencies.get(ranking.model)
            if ms is not None:
                reward = _latency_reward(ms)
                ranking.latency_score = int(round(reward * 5))
                ranking.total_score_with_latency = ranking.total_score + ranking.latency_score

    return JudgeResult(
        sample_id=sample_id,
        judge_model=judge_model,
        mode="pairwise_blind",
        images_included=has_images,
        rankings=rankings,
        best_output=reverse_map.get(
            result.get("best_output", "") if isinstance(result, dict) else "",
            rankings[0].model if rankings else "",
        ),
        notes=result.get("notes") if isinstance(result, dict) else None,
        latency_ms=latency_ms,
    )


def pointwise_judge_sample(
    sample_id: str,
    sample: Dict[str, Any],
    model_id: str,
    output: Dict[str, Any],
    judge_model: str = "openai/gpt-4o",
    api_key: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> JudgeRanking:
    """Score a single model output for a single sample using pointwise evaluation.

    Unlike pairwise judging, this evaluates one output at a time against a fixed
    rubric, eliminating both positional bias and comparative anchoring.

    Returns a JudgeRanking (with rank=0 as a placeholder — ranks are computed
    after all pointwise scores are collected).
    """
    prompt = build_pointwise_prompt(sample, output, data_root=data_root)
    result, error, latency_ms, _tok_in, _tok_out = call_judge_model(
        prompt, judge_model, api_key,
        system_prompt=POINTWISE_JUDGE_SYSTEM_PROMPT,
    )

    if error:
        return JudgeRanking(
            model=model_id,
            rank=0,
            scores=JudgeScore(accuracy=1, specificity=1, actionability=1, safety=1),
            total_score=0,
            reasoning=f"Error: {error}",
            enhanced_instruction=output.get("enhanced_instruction", ""),
        )

    try:
        scores = JudgeScore(**result["scores"])
        return JudgeRanking(
            model=model_id,
            rank=0,  # assigned later
            scores=scores,
            total_score=result.get("total_score", sum(result["scores"].values())),
            reasoning=result.get("reasoning", ""),
            enhanced_instruction=output.get("enhanced_instruction", ""),
        )
    except Exception as e:
        return JudgeRanking(
            model=model_id,
            rank=0,
            scores=JudgeScore(accuracy=1, specificity=1, actionability=1, safety=1),
            total_score=0,
            reasoning=f"Parse error: {e}",
            enhanced_instruction=output.get("enhanced_instruction", ""),
        )


def run_pointwise_judge_pipeline(
    samples_file: Path,
    result_files: List[Path],
    output_file: Path,
    judge_model: str = "openai/gpt-4o",
    limit: Optional[int] = None,
    verbose: bool = True,
    data_root: Optional[Path] = None,
) -> List[JudgeResult]:
    """Run the pointwise judge pipeline on multiple result files.

    Each model's output is scored independently (one judge call per model per
    sample). Rankings are computed afterwards by sorting on total_score.

    Cost: N_models * N_samples judge calls (vs N_samples for pairwise).
    Each call is smaller (1 output vs N_models outputs), but image tokens
    are repeated per call so total cost is higher when images are included.

    Args:
        samples_file: Path to samples.jsonl
        result_files: List of paths to inference result files
        output_file: Where to save judge results
        judge_model: Model to use as judge
        limit: Max samples to judge
        verbose: Print progress
        data_root: Path to data directory for resolving image paths. When
                   provided, the judge sees the same images the VLMs saw.

    Returns:
        List of JudgeResult objects (one per sample, with rankings derived from
        pointwise scores).
    """
    # Load samples
    samples = {}
    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                samples[s["id"]] = s

    # Load all result files
    # Use a composite key (model_id:modality:augment) when multiple files
    # share the same model_id (e.g. different modalities/augmentations).
    all_results: Dict[str, Dict[str, Dict]] = {}
    for rf in result_files:
        results = load_inference_results(rf)
        if results:
            first_result = next(iter(results.values()))
            base_model_id = first_result.get("model_id", rf.stem)
            modality = first_result.get("modality", "")
            augment = first_result.get("augment") or ""
            variant = first_result.get("variant") or ""
            # Build a unique key to avoid collisions
            key = base_model_id
            suffixes = []
            if modality:
                suffixes.append(modality.replace(" + ", "+"))
            if augment:
                suffixes.append(augment)
            if variant:
                suffixes.append(variant)
            if suffixes:
                key = f"{base_model_id}:{':'.join(suffixes)}"
            # If key still collides, fall back to filename stem
            if key in all_results:
                key = f"{base_model_id}:{rf.stem}"
            all_results[key] = results

    if verbose:
        print(f"Loaded results from {len(all_results)} models")
        for key, results in all_results.items():
            print(f"  {key}: {len(results)} results")

    # Find common samples (use union for pointwise since each is scored independently)
    all_sample_ids: set = set()
    for results in all_results.values():
        all_sample_ids |= set(results.keys())
    # Only include samples that exist in the samples file
    common_sample_ids = all_sample_ids & set(samples.keys())

    common_sample_ids = sorted(common_sample_ids)
    if limit:
        common_sample_ids = common_sample_ids[:limit]

    n_models = len(all_results)
    n_samples = len(common_sample_ids)

    if verbose:
        print(f"Pointwise judging: {n_samples} samples x {n_models} models")

    # Rate limiting
    min_delay = 60.0 / REQUESTS_PER_MINUTE

    judge_results: List[JudgeResult] = []
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-judged samples
    done_judge_ids: set = set()
    if output_file.exists() and output_file.stat().st_size > 0:
        with open(output_file, encoding="utf-8") as _rf:
            for _line in _rf:
                _line = _line.strip()
                if _line:
                    try:
                        done_judge_ids.add(json.loads(_line)["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        if done_judge_ids and verbose:
            print(f"Resuming: {len(done_judge_ids)} samples already judged, skipping")

    pending_sample_ids = [s for s in common_sample_ids if s not in done_judge_ids]
    total_calls = sum(
        1 for sid in pending_sample_ids
        for results in all_results.values()
        if sid in results
    )
    if done_judge_ids and verbose:
        print(f"  {len(done_judge_ids)} already done, {len(pending_sample_ids)} remaining ({total_calls} judge calls)")
    elif verbose:
        print(f"  {total_calls} judge calls")
    file_mode = "a" if done_judge_ids else "w"

    with open(output_file, file_mode) as f:
        last_request_time = 0.0
        call_idx = 0

        for sample_id in pending_sample_ids:
            sample = samples[sample_id]
            sample_rankings: List[JudgeRanking] = []

            for model_id, results in all_results.items():
                if sample_id not in results:
                    continue
                output = results[sample_id]
                call_idx += 1

                # Rate limiting
                elapsed = time.time() - last_request_time
                if elapsed < min_delay:
                    time.sleep(min_delay - elapsed)

                if verbose:
                    print(
                        f"[{call_idx}/{total_calls}] {sample_id} / {model_id.split('/')[-1]}...",
                        end=" ",
                        flush=True,
                    )

                last_request_time = time.time()
                ranking = pointwise_judge_sample(
                    sample_id, sample, model_id, output, judge_model,
                    data_root=data_root,
                )
                sample_rankings.append(ranking)

                if verbose:
                    if ranking.total_score == 0 and ranking.reasoning.startswith("Error"):
                        print(f"ERROR")
                    else:
                        print(f"{ranking.total_score}/20")

            # Assign ranks by total_score descending
            sample_rankings.sort(key=lambda r: r.total_score, reverse=True)
            for rank_idx, ranking in enumerate(sample_rankings, start=1):
                ranking.rank = rank_idx

            best_model = sample_rankings[0].model if sample_rankings else ""

            result = JudgeResult(
                sample_id=sample_id,
                judge_model=judge_model,
                mode="pointwise_blind",
                images_included=data_root is not None,
                rankings=sample_rankings,
                best_output=best_model,
                notes="pointwise",
            )
            judge_results.append(result)

            f.write(result.model_dump_json() + "\n")
            f.flush()

            if verbose:
                scores_summary = ", ".join(
                    f"{r.model.split('/')[-1]}={r.total_score}/20"
                    for r in sample_rankings
                )
                print(f"  -> ranks: {scores_summary} | best: {best_model.split('/')[-1]}")

    # Summary
    if verbose:
        successful = sum(1 for r in judge_results if not r.error)
        print(f"\nPointwise judging complete: {successful}/{len(judge_results)} samples")

        # Aggregate win counts
        win_counts: Dict[str, int] = {}
        score_totals: Dict[str, List[int]] = {}
        for r in judge_results:
            if r.rankings:
                winner = r.rankings[0].model
                win_counts[winner] = win_counts.get(winner, 0) + 1
            for ranking in r.rankings:
                score_totals.setdefault(ranking.model, []).append(ranking.total_score)

        print("\nWin counts:")
        for model, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
            avg = sum(score_totals[model]) / len(score_totals[model])
            print(f"  {model}: {wins} wins (avg {avg:.1f}/20)")

    return judge_results


def load_inference_results(results_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load inference results from JSONL file.

    Returns dict mapping sample_id to result.
    """
    results = {}
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("sample_id") or r.get("id")
            if sid:
                results[sid] = r
    return results


def human_labels_to_results(
    custom_labels_path: Path,
    annotator_name: str = "annotator",
) -> Dict[str, Dict[str, Any]]:
    """Load custom human labels as pseudo-inference results for benchmark.

    Converts each custom label entry into the same output dict format
    that build_ab_judge_prompt() expects. For samples with multiple
    labels, the last entry wins (most recent edit).

    Args:
        custom_labels_path: Path to custom_labels.jsonl.

    Returns:
        Dict mapping sample_id to output dict keyed like inference results.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not custom_labels_path.exists():
        return results

    with open(custom_labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sample_id = entry.get("sample_id")
            if not sample_id:
                continue
            # Last entry wins (most recent edit)
            # Support both custom_labels.jsonl format (enhanced_instruction)
            # and canonical_gt.jsonl format (instruction field)
            instruction = entry.get("enhanced_instruction") or entry.get("instruction", "")
            results[sample_id] = {
                "id": sample_id,
                "model_id": f"human/{annotator_name}",
                "enhanced_instruction": instruction,
                "next_action": entry.get("next_action", "unknown"),
                "relevant_landmarks": entry.get("relevant_landmarks", []),
                "potential_hazards": entry.get("potential_hazards", []),
                "reasoning": entry.get("reasoning", ""),
                "lane_hint": entry.get("lane_hint"),
                "lanes_count": entry.get("lanes_count"),
                "confidence": entry.get("confidence"),
            }
    return results


def baseline_instructions_to_results(
    samples_file: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load original Google Maps + OSM instructions as pseudo-inference results.

    Wraps each sample's prior.instruction (plus OSM road data) as a
    pseudo-result under model_id="maps-osm/baseline", so it can compete
    in A/B benchmarks against VLM-enhanced versions.

    Args:
        samples_file: Path to samples.jsonl.

    Returns:
        Dict mapping sample_id to output dict keyed like inference results.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not samples_file.exists():
        return results

    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample_id = sample.get("id")
            if not sample_id:
                continue
            instruction = sample.get("prior", {}).get("instruction", "")
            if not instruction:
                continue
            # Enrich with OSM data if available
            osm = sample.get("osm_road", {})
            osm_parts = []
            if osm.get("lanes"):
                osm_parts.append(f"{osm['lanes']} lanes")
            if osm.get("name"):
                osm_parts.append(f"on {osm['name']}")
            if osm.get("maxspeed"):
                osm_parts.append(f"speed limit {osm['maxspeed']}")
            osm_note = "; ".join(osm_parts)

            results[sample_id] = {
                "id": sample_id,
                "model_id": "maps-osm/baseline",
                "enhanced_instruction": instruction,
                "next_action": sample.get("prior", {}).get("instruction", "unknown")[:50],
                "relevant_landmarks": [],
                "potential_hazards": [],
                "reasoning": f"Original navigation instruction from Google Maps.{f' OSM: {osm_note}.' if osm_note else ''}",
            }
    return results


def run_judge_pipeline(
    samples_file: Path,
    result_files: List[Path],
    output_file: Path,
    judge_model: str = "openai/gpt-4o",
    limit: Optional[int] = None,
    verbose: bool = True,
    parallel: int = 1,
    data_root: Optional[Path] = None,
    sample_ids: Optional[List[str]] = None,
) -> List[JudgeResult]:
    """Run the judge pipeline on multiple result files.

    Args:
        samples_file: Path to samples.jsonl
        result_files: List of paths to inference result files
        output_file: Where to save judge results
        judge_model: Model to use as judge
        limit: Max samples to judge
        verbose: Print progress
        parallel: Number of parallel requests (default 1 for rate limiting)
        data_root: Path to data directory for resolving image paths. When
                   provided, the judge sees the same images the VLMs saw.

    Returns:
        List of JudgeResult objects
    """
    # Load samples
    samples = {}
    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                samples[s["id"]] = s

    # Load all result files
    # Use a composite key (model_id:modality:augment) when multiple files
    # share the same model_id (e.g. different modalities/augmentations).
    all_results: Dict[str, Dict[str, Dict]] = {}  # unique_key -> {sample_id -> result}
    for rf in result_files:
        results = load_inference_results(rf)
        # Infer model_id from first result
        if results:
            first_result = next(iter(results.values()))
            base_model_id = first_result.get("model_id", rf.stem)
            modality = first_result.get("modality", "")
            augment = first_result.get("augment") or ""
            variant = first_result.get("variant") or ""
            key = base_model_id
            suffixes = []
            if modality:
                suffixes.append(modality.replace(" + ", "+"))
            if augment:
                suffixes.append(augment)
            if variant:
                suffixes.append(variant)
            if suffixes:
                key = f"{base_model_id}:{':'.join(suffixes)}"
            if key in all_results:
                key = f"{base_model_id}:{rf.stem}"
            all_results[key] = results

    if verbose:
        print(f"Loaded results from {len(all_results)} models")
        for key, results in all_results.items():
            print(f"  {key}: {len(results)} results")

    # Find common samples (present in all result files)
    common_sample_ids = set(samples.keys())
    for results in all_results.values():
        common_sample_ids &= set(results.keys())

    common_sample_ids = sorted(common_sample_ids)
    if sample_ids is not None:
        keep = set(sample_ids)
        common_sample_ids = [s for s in common_sample_ids if s in keep]
    if limit:
        common_sample_ids = common_sample_ids[:limit]

    if verbose:
        print(f"Judging {len(common_sample_ids)} common samples")

    # Prepare tasks (with per-model latency data)
    tasks = []
    for sample_id in common_sample_ids:
        sample = samples[sample_id]
        outputs = [
            (model_id, results[sample_id])
            for model_id, results in all_results.items()
        ]
        # Extract latencies from inference_metadata
        sample_latencies: Dict[str, float] = {}
        for model_id, results in all_results.items():
            result = results.get(sample_id, {})
            meta = result.get("inference_metadata") or {}
            if isinstance(meta, dict) and meta.get("latency_ms") is not None:
                sample_latencies[model_id] = float(meta["latency_ms"])
        tasks.append((sample_id, sample, outputs, sample_latencies or None))

    # Resume support: collect already-judged sample_ids from existing output
    judge_results = []
    output_file.parent.mkdir(parents=True, exist_ok=True)

    done_judge_ids: set = set()
    if output_file.exists() and output_file.stat().st_size > 0:
        with open(output_file, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        done_judge_ids.add(json.loads(_line)["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        if done_judge_ids and verbose:
            print(f"Resuming: {len(done_judge_ids)} samples already judged in {output_file.name}, skipping")

    pending_tasks = [(sid, s, outs, lats) for sid, s, outs, lats in tasks if sid not in done_judge_ids]
    if len(pending_tasks) < len(tasks) and verbose:
        print(f"  {len(tasks) - len(pending_tasks)} skipped, {len(pending_tasks)} remaining")

    # Rate limiting: minimum delay between requests
    min_delay = 60.0 / REQUESTS_PER_MINUTE

    file_mode = "a" if done_judge_ids else "w"

    if parallel > 1:
        # Parallel processing with rate limiting
        with open(output_file, file_mode) as f:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {}
                for i, (sample_id, sample, outputs, sample_latencies) in enumerate(pending_tasks):
                    future = executor.submit(judge_sample, sample_id, sample, outputs, judge_model, None, sample_latencies, data_root)
                    futures[future] = (i, sample_id)

                for future in as_completed(futures):
                    i, sample_id = futures[future]
                    result = future.result()
                    judge_results.append(result)

                    f.write(result.model_dump_json() + "\n")
                    f.flush()

                    if verbose:
                        if result.error:
                            print(f"[{len(judge_results)}/{len(pending_tasks)}] {sample_id}: ERROR - {result.error}")
                        else:
                            print(f"[{len(judge_results)}/{len(pending_tasks)}] {sample_id}: Best={result.best_output} ({result.latency_ms}ms)")
    else:
        # Sequential processing with rate limiting
        with open(output_file, file_mode) as f:
            last_request_time = 0

            for i, (sample_id, sample, outputs, sample_latencies) in enumerate(pending_tasks):
                elapsed = time.time() - last_request_time
                if elapsed < min_delay:
                    time.sleep(min_delay - elapsed)

                if verbose:
                    print(f"[{i+1}/{len(pending_tasks)}] Judging {sample_id}...", end=" ", flush=True)

                last_request_time = time.time()
                result = judge_sample(sample_id, sample, outputs, judge_model, latencies=sample_latencies, data_root=data_root)
                judge_results.append(result)

                f.write(result.model_dump_json() + "\n")
                f.flush()

                if verbose:
                    if result.error:
                        print(f"ERROR: {result.error}")
                    else:
                        print(f"Best: {result.best_output} ({result.latency_ms}ms)")

    # Summary
    if verbose:
        successful = sum(1 for r in judge_results if not r.error)
        failed = sum(1 for r in judge_results if r.error)
        print(f"\nJudging complete: {successful} successful, {failed} failed")

    return judge_results


def compute_agreement_stats(results: List[JudgeResult]) -> Dict[str, Any]:
    """Compute statistics about judge rankings.

    Returns dict with:
        - model_win_rates: How often each model was ranked #1
        - avg_scores: Average scores per model
        - score_distribution: Score distribution per criterion
    """
    model_wins = {}
    model_scores = {}
    criterion_scores = {"accuracy": [], "specificity": [], "actionability": [], "safety": []}

    for result in results:
        if result.error or not result.rankings:
            continue

        # Count wins
        best = result.rankings[0].model
        model_wins[best] = model_wins.get(best, 0) + 1

        # Aggregate scores
        for ranking in result.rankings:
            if ranking.model not in model_scores:
                model_scores[ranking.model] = []
            model_scores[ranking.model].append(ranking.total_score)

            criterion_scores["accuracy"].append(ranking.scores.accuracy)
            criterion_scores["specificity"].append(ranking.scores.specificity)
            criterion_scores["actionability"].append(ranking.scores.actionability)
            criterion_scores["safety"].append(ranking.scores.safety)

    # Compute averages
    total_samples = sum(model_wins.values())
    win_rates = {m: w / total_samples for m, w in model_wins.items()} if total_samples > 0 else {}
    avg_scores = {m: sum(s) / len(s) for m, s in model_scores.items() if s}
    avg_criterion = {c: sum(s) / len(s) for c, s in criterion_scores.items() if s}

    return {
        "model_win_rates": win_rates,
        "avg_scores": avg_scores,
        "avg_criterion_scores": avg_criterion,
        "total_judged": total_samples,
    }


# ── Pairwise A/B Benchmark (cross-company, 3-judge panel, Elo) ────────


AB_JUDGE_SYSTEM_PROMPT = """You are an expert driving instructor evaluating navigation instructions for a human driver who is looking at the road ahead.

You will see a driving scenario with dashcam frames and an overhead map, followed by two responses (Response A and Response B). Decide which response would be more helpful to a driver in real time.

The golden rule: **if the original instruction is already clear enough for the maneuver, the best response is one that stays close to it.** Not every instruction needs enhancement. Adding detail to a simple maneuver is a fault, not a virtue.

Evaluation criteria (in priority order):

1. **Correctness & visual grounding**: The instruction must match the actual maneuver AND the dashcam images. **Critically penalise hallucinated landmarks** — if a response names a business, sign, or feature that is NOT visible in the provided dashcam frames, treat it as a factual error even if the landmark plausibly exists nearby. Only landmarks the driver can actually SEE in the images should count as helpful. Also penalise wrong side landmarks, incorrect lane references, and wrong street names. A landmark referenced on the wrong side of the road for the given maneuver is a correctness error (e.g. "petrol station on the left" when turning left — the landmark is on the side you are turning toward, not a useful approach cue). Conversely, referencing a real object visible in the dashcam (a specific car, road marking, building facade) is valid even if it is temporary — the driver sees it RIGHT NOW.

2. **No unnecessary additions**: Every word beyond the core maneuver must earn its place. Heavily penalise:
   - Stating what the driver can already see (e.g. "from the right-turn lane" when they're already in it)
   - Lane guidance when the driver is already in the correct lane or there is only one option
   - Landmarks that don't help locate the maneuver point (e.g. naming a building the driver has already passed)
   - Speed/hazard/safety advice that isn't warranted by the scene
   - Restating the obvious (e.g. "continue driving" on a straight road)
   A shorter response that says the same thing is ALWAYS better.

3. **Helpful additions only when needed**: Extra detail is justified ONLY when the scene is genuinely complex:
   - A lane change is required and it's not obvious which lane
   - The turn is easy to miss and a landmark helps locate it
   - There is a real hazard the driver might not see
   If the maneuver is simple (single right turn, continue straight, obvious roundabout exit), the original instruction is probably sufficient.

4. **Actionability**: When guidance IS needed, is it clear and correctly timed?

Pick the better response. Prefer the more concise response unless the longer one adds information that genuinely helps. **Default to tie** when both responses make the same error (e.g. both reference a landmark on the wrong side), when both are equivalent rewordings of the original, or when any perceived difference is too marginal to matter to a real driver.

Return a JSON object with exactly these fields:
{
  "winner": "A" or "B" or "tie",
  "justification": "Brief explanation (1-2 sentences)"
}"""


# Primary panel: cheap judges used for all comparisons
DEFAULT_JUDGE_PANEL = [
    "google/gemini-3-flash-preview",
    "x-ai/grok-4.1-fast",
    "qwen/qwen3-vl-8b-instruct",
]

# Cheap backup judge when exclusion leaves too few
_BACKUP_JUDGE = "openai/gpt-4o-mini"

# Expensive tiebreaker — only called when primary judges disagree
TIEBREAKER_JUDGE = "anthropic/claude-sonnet-4.6"


def select_judges_for_pair(
    model_a: str,
    model_b: str,
    panel: list[str] | None = None,
    min_judges: int = 3,
) -> list[str]:
    """Select judges for an A/B pair, excluding same-company judges.

    If a judge's company matches either contestant's company, it's excluded.
    If fewer than min_judges remain, adds gpt-4o-mini as cheap backup.

    Returns list of judge model IDs.
    """
    from navbuddy.eval.company_groups import extract_company

    if panel is None:
        panel = DEFAULT_JUDGE_PANEL

    company_a = extract_company(model_a).lower()
    company_b = extract_company(model_b).lower()

    eligible = [
        j for j in panel
        if extract_company(j).lower() not in (company_a, company_b)
    ]

    # Add backup if we don't have enough judges
    while len(eligible) < min_judges:
        if _BACKUP_JUDGE not in eligible:
            backup_company = extract_company(_BACKUP_JUDGE).lower()
            if backup_company not in (company_a, company_b):
                eligible.append(_BACKUP_JUDGE)
                continue
        # Can't get enough unbiased judges — use what we have
        break

    return eligible[:min_judges]


def build_ab_judge_prompt(
    sample: Dict[str, Any],
    output_a: Dict[str, Any],
    output_b: Dict[str, Any],
    data_root: Optional[Path] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """Build a blind A/B comparison prompt for exactly two model outputs.

    Model IDs are never revealed. Reasoning fields are omitted to prevent
    the judge from evaluating chain-of-thought quality instead of the
    actual instruction.

    Args:
        sample: Sample dict with prior.instruction and maneuver.
        output_a: First model's output dict.
        output_b: Second model's output dict.
        data_root: Path to data directory for resolving image paths.

    Returns:
        Either a plain text string or a list of multimodal content blocks.
    """
    prior_instruction = sample.get("prior", {}).get("instruction", "Unknown")
    maneuver = sample.get("maneuver", "Unknown")

    def _format_output(label: str, output: Dict[str, Any]) -> str:
        enhanced = output.get("enhanced_instruction", "N/A")
        return f"""### {label}
{enhanced}"""

    text = f"""## Driving Scenario

**Original Navigation Instruction:** {prior_instruction}
**Maneuver Type:** {maneuver}

The dashcam frames and overhead map below show the driving scene. Use them to verify whether the responses accurately describe what is visible.

## Responses

{_format_output("Response A", output_a)}

{_format_output("Response B", output_b)}

## Task

Which response is better? Return JSON with "winner" ("A", "B", or "tie") and "justification".
"""

    image_blocks = _build_image_content(sample, data_root)
    if image_blocks:
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(image_blocks)
        return content

    return text


def call_ab_judge(
    sample: Dict[str, Any],
    output_a: Dict[str, Any],
    output_b: Dict[str, Any],
    judge_model: str,
    api_key: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> "JudgeVote":
    """Call a single judge for an A/B comparison.

    Returns a JudgeVote with the winner and justification.
    """
    from navbuddy.eval.schemas import JudgeVote

    prompt = build_ab_judge_prompt(sample, output_a, output_b, data_root)
    result, error, latency_ms, tokens_in, tokens_out = call_judge_model(
        prompt, judge_model, api_key,
        system_prompt=AB_JUDGE_SYSTEM_PROMPT,
    )

    if error:
        return JudgeVote(
            judge_model=judge_model,
            winner="tie",
            justification="",
            latency_ms=latency_ms,
            tokens_in=tokens_in or None,
            tokens_out=tokens_out or None,
            error=error,
        )

    try:
        winner = result.get("winner", "tie").upper()
        if winner not in ("A", "B", "TIE"):
            winner = "tie"
        else:
            winner = winner.lower() if winner == "TIE" else winner
        justification = result.get("justification", "")
        return JudgeVote(
            judge_model=judge_model,
            winner=winner if winner in ("A", "B") else "tie",
            justification=justification,
            latency_ms=latency_ms,
            tokens_in=tokens_in or None,
            tokens_out=tokens_out or None,
        )
    except Exception as e:
        return JudgeVote(
            judge_model=judge_model,
            winner="tie",
            justification="",
            latency_ms=latency_ms,
            tokens_in=tokens_in or None,
            tokens_out=tokens_out or None,
            error=f"Parse error: {e}",
        )


def run_ab_panel(
    sample: Dict[str, Any],
    output_a: Dict[str, Any],
    output_b: Dict[str, Any],
    judge_panel: List[str],
    api_key: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> List["JudgeVote"]:
    """Run an A/B comparison across the full judge panel in parallel.

    Each judge in the panel is a different model/provider, so they can
    be called concurrently without rate limit interference.

    Args:
        sample: Sample dict.
        output_a: First model's output.
        output_b: Second model's output.
        judge_panel: List of judge model IDs.
        api_key: OpenRouter API key.
        data_root: Path to data directory.

    Returns:
        List of JudgeVote objects (one per judge).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    votes = []
    with ThreadPoolExecutor(max_workers=len(judge_panel)) as executor:
        futures = {
            executor.submit(
                call_ab_judge, sample, output_a, output_b, judge_model, api_key, data_root
            ): judge_model
            for judge_model in judge_panel
        }
        for future in as_completed(futures):
            votes.append(future.result())

    return votes


def majority_vote(
    votes: List["JudgeVote"],
    model_a: str,
    model_b: str,
) -> Tuple[Optional[str], bool]:
    """Determine the winner from judge panel votes via majority rule.

    Args:
        votes: List of JudgeVote objects.
        model_a: Model ID assigned to position A.
        model_b: Model ID assigned to position B.

    Returns:
        Tuple of (winner_model_id_or_None, is_tie).
        - If 2+ judges agree on A or B: (winner_model_id, False)
        - If no majority or majority is tie: (None, True)
    """
    counts = {"A": 0, "B": 0, "tie": 0}
    for vote in votes:
        if vote.error:
            counts["tie"] += 1  # Errors count as ties
        elif vote.winner in counts:
            counts[vote.winner] += 1
        else:
            counts["tie"] += 1

    # Check for majority (2/3 or more)
    threshold = len(votes) / 2.0
    if counts["A"] > threshold:
        return model_a, False
    elif counts["B"] > threshold:
        return model_b, False
    else:
        return None, True


def _comparison_key(sample_id: str, model_a: str, model_b: str) -> Tuple[str, str, str]:
    """Canonical order-independent key for a pairwise comparison.

    Returns (sample_id, model_lo, model_hi) where model_lo < model_hi
    lexicographically.  This ensures A-vs-B and B-vs-A hash to the same key.
    """
    lo, hi = (model_a, model_b) if model_a <= model_b else (model_b, model_a)
    return (sample_id, lo, hi)


def load_existing_comparisons(path: Path) -> Set[Tuple[str, str, str]]:
    """Load already-judged comparison keys from a benchmark JSONL file.

    Returns a set of (sample_id, model_lo, model_hi) tuples.
    """
    seen: Set[Tuple[str, str, str]] = set()
    if not path.exists():
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = _comparison_key(
                    rec["sample_id"], rec["model_a"], rec["model_b"]
                )
                seen.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def run_benchmark_pipeline(
    samples_file: Path,
    result_files: List[Path],
    output_file: Path,
    config: Optional[Any] = None,
    model_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    verbose: bool = True,
    data_root: Optional[Path] = None,
    extra_results: Optional[Dict[str, Dict[str, Dict]]] = None,
    ground_truth: Optional[Dict[str, dict]] = None,
    sample_ids: Optional[List[str]] = None,
    modality: Optional[str] = None,
) -> Any:
    """Run the cross-company pairwise A/B benchmark.

    For each sample, runs all cross-company pairs through the 3-judge panel
    and computes Elo ratings from the results.

    Args:
        samples_file: Path to samples.jsonl.
        result_files: List of inference result JSONL files.
        output_file: Output JSONL for PairwiseComparison records.
        config: BenchmarkConfig (uses defaults if None).
        model_ids: Subset of model IDs to include (None = all).
        limit: Max samples per pair.
        verbose: Print progress.
        data_root: Path to data directory for images.
        extra_results: Additional result dicts to merge (e.g. human labels).

    Returns:
        BenchmarkRun with all comparisons and Elo ratings.
    """
    import uuid
    from datetime import datetime

    from navbuddy.eval.company_groups import (
        extract_company,
        generate_cross_company_pairs,
        group_models_by_company,
    )
    from navbuddy.eval.elo import compute_elo_ratings
    from navbuddy.eval.schemas import (
        BenchmarkConfig,
        BenchmarkRun,
        PairwiseComparison,
    )

    if config is None:
        config = BenchmarkConfig()

    # ── Load samples ──────────────────────────────────────────────────
    samples = {}
    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                samples[s["id"]] = s

    # ── Sample ID filter (e.g. route filter) ──────────────────────────
    if sample_ids is not None:
        keep = set(sample_ids)
        samples = {sid: s for sid, s in samples.items() if sid in keep}
        if verbose:
            print(f"Sample filter: {len(samples)} samples")

    # ── GT-only filter ────────────────────────────────────────────────
    if config.gt_only and ground_truth:
        gt_sample_ids = {
            sid for sid, gt in ground_truth.items() if not gt.get("is_auto")
        }
        before = len(samples)
        samples = {sid: s for sid, s in samples.items() if sid in gt_sample_ids}
        if verbose:
            print(f"GT-only filter: {len(samples)}/{before} samples have manual GT")
    elif config.gt_only and not ground_truth:
        if verbose:
            print("Warning: gt_only=True but no ground_truth data provided — no filter applied")

    # ── Load inference results ────────────────────────────────────────
    all_results: Dict[str, Dict[str, Dict]] = {}
    for rf in result_files:
        results = load_inference_results(rf)
        if results:
            first_result = next(iter(results.values()))
            base_model_id = first_result.get("model_id", rf.stem)
            modality = first_result.get("modality", "")
            augment = first_result.get("augment") or ""
            # Build a unique key
            key = base_model_id
            if augment:
                key = f"{base_model_id}::{augment}"
            if key in all_results:
                key = f"{base_model_id}::{rf.stem}"
            all_results[key] = results

    # ── Merge extra result sources (e.g. human labels) ────────────────
    if extra_results:
        for key, results in extra_results.items():
            if key not in all_results:
                all_results[key] = results

    # ── Determine competing models ────────────────────────────────────
    available_models = list(all_results.keys())
    if model_ids:
        available_models = [m for m in available_models if m in model_ids]

    if verbose:
        groups = group_models_by_company(available_models)
        print(f"Models: {len(available_models)} from {len(groups)} companies")
        for company, models in sorted(groups.items()):
            print(f"  {company}: {', '.join(m.split('/')[-1] for m in models)}")

    # ── Generate cross-company pairs ──────────────────────────────────
    pairs = generate_cross_company_pairs(available_models, seed=config.seed)

    if verbose:
        print(f"Cross-company pairs: {len(pairs)}")

    # ── Load existing comparisons for per-judge dedup ────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # judged_by_key: {(sample_id, lo, hi) → set of judge_model ids that voted}
    judged_by_key: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    # existing_by_key: full PairwiseComparison objects keyed by pair key
    existing_by_key: Dict[Tuple[str, str, str], PairwiseComparison] = {}
    if output_file.exists():
        with open(output_file, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    comp = PairwiseComparison.model_validate_json(_line)
                    key = _comparison_key(comp.sample_id, comp.model_a, comp.model_b)
                    existing_by_key[key] = comp
                    for vote in comp.judge_votes:
                        if not vote.error:
                            judged_by_key[key].add(vote.judge_model)
                except Exception:
                    pass
    prior_comparisons: List[PairwiseComparison] = list(existing_by_key.values())

    if verbose and existing_by_key:
        total_votes = sum(len(v) for v in judged_by_key.values())
        print(f"Loaded {len(existing_by_key)} existing comparisons ({total_votes} judge votes)")

    # ── Elo-aware pair ordering ──────────────────────────────────────
    # Like chess matchmaking: prioritize pairs with close Elo ratings.
    # Mismatched pairs (large Elo gap) produce predictable outcomes and
    # tiny rating changes, wasting API budget.
    current_elo: Dict[str, float] = {}
    if prior_comparisons:
        try:
            prior_ratings = compute_elo_ratings(prior_comparisons, config)
            current_elo = {m: r.rating for m, r in prior_ratings.items()}
        except Exception:
            pass
    starting_elo = getattr(config, "starting_elo", 1500.0)

    # Sort pairs by Elo gap (closest first), break ties deterministically
    def pair_elo_gap(pair: Tuple[str, str]) -> float:
        a_elo = current_elo.get(pair[0], starting_elo)
        b_elo = current_elo.get(pair[1], starting_elo)
        return abs(a_elo - b_elo)

    pairs_sorted = sorted(pairs, key=lambda p: (pair_elo_gap(p), p))

    # Skip pairs with Elo gap > threshold (configurable, default 300)
    max_elo_gap = getattr(config, "max_elo_gap", 300)
    if max_elo_gap and current_elo:
        before = len(pairs_sorted)
        pairs_sorted = [p for p in pairs_sorted if pair_elo_gap(p) <= max_elo_gap]
        skipped_pairs = before - len(pairs_sorted)
        if verbose and skipped_pairs:
            print(f"Skipped {skipped_pairs} pairs with Elo gap > {max_elo_gap}")

    if verbose and current_elo:
        print(f"Pair order (by Elo proximity): {len(pairs_sorted)} pairs")
        for p in pairs_sorted[:5]:
            a_short = p[0].split("/")[-1][:20]
            b_short = p[1].split("/")[-1][:20]
            gap = pair_elo_gap(p)
            print(f"  {a_short} vs {b_short} (gap: {gap:.0f})")
        if len(pairs_sorted) > 5:
            print(f"  ... and {len(pairs_sorted) - 5} more")

    # ── GT-weighted sample selection ──────────────────────────────────
    import random as _random
    gt_weight = getattr(config, "gt_weight", 1.0)
    total_limit = getattr(config, "total_limit", None)
    gt_ids: set = set(ground_truth.keys()) if ground_truth else set()
    _rng = _random.Random(config.seed)

    def _weighted_sample(population: List[str], n: int) -> List[str]:
        """Sample n items without replacement, GT samples gt_weight-times more likely."""
        if gt_weight <= 1.0 or not gt_ids:
            return population[:n]
        weights = [gt_weight if sid in gt_ids else 1.0 for sid in population]
        result: List[str] = []
        remaining = list(zip(population, weights))
        for _ in range(min(n, len(remaining))):
            total_w = sum(w for _, w in remaining)
            r = _rng.uniform(0, total_w)
            cumulative = 0.0
            for i, (sid, w) in enumerate(remaining):
                cumulative += w
                if r <= cumulative:
                    result.append(sid)
                    remaining.pop(i)
                    break
        return result

    # ── Build work items interleaved: sample-first, then pairs ───────
    # This ensures diverse pair coverage even in partial runs, so every
    # model gets matchups against many opponents rather than exhausting
    # all samples for one pair before moving to the next.
    work_items: List[Tuple[str, str, str]] = []  # (sample_id, model_a, model_b)
    pair_common: Dict[Tuple[str, str], List[str]] = {}
    for model_a, model_b in pairs_sorted:
        results_a = all_results.get(model_a, {})
        results_b = all_results.get(model_b, {})
        common = sorted(set(results_a.keys()) & set(results_b.keys()) & set(samples.keys()))
        n = limit if limit else len(common)
        common = _weighted_sample(common, n)
        pair_common[(model_a, model_b)] = common

    # Round-robin: for each sample index, iterate all pairs
    max_samples = max((len(c) for c in pair_common.values()), default=0)
    for sample_idx in range(max_samples):
        for (model_a, model_b), common in pair_common.items():
            if sample_idx < len(common):
                work_items.append((common[sample_idx], model_a, model_b))

    if total_limit:
        work_items = work_items[:total_limit]

    # ── Run comparisons ───────────────────────────────────────────────
    new_comparisons: List[PairwiseComparison] = []
    updated_keys: Set[Tuple[str, str, str]] = set()  # keys that got new votes
    total_judge_calls = 0
    successful = 0
    failed = 0
    skipped = 0

    min_delay = 60.0 / REQUESTS_PER_MINUTE
    comparison_idx = 0
    last_request_time = 0.0

    for sample_id, model_a, model_b in work_items:
        key = _comparison_key(sample_id, model_a, model_b)

        # ── Per-judge dedup: only call judges that haven't voted yet ──
        judges_done = judged_by_key.get(key, set())
        judges_needed = [j for j in config.judge_panel if j not in judges_done]

        if not judges_needed:
            skipped += 1
            continue

        comparison_idx += 1
        sample = samples[sample_id]
        output_a = all_results[model_a][sample_id]
        output_b = all_results[model_b][sample_id]

        # Rate limiting (sequential comparisons)
        elapsed = time.time() - last_request_time
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)

        existing_comp = existing_by_key.get(key)
        n_existing_votes = len(existing_comp.judge_votes) if existing_comp else 0

        if verbose:
            a_short = model_a.split("/")[-1][:20]
            b_short = model_b.split("/")[-1][:20]
            judge_info = f"+{len(judges_needed)} judge(s)" if n_existing_votes else f"{len(judges_needed)} judge(s)"
            print(
                f"[{comparison_idx}] {sample_id} | {a_short} vs {b_short} [{judge_info}]...",
                end=" ",
                flush=True,
            )

        last_request_time = time.time()
        new_votes = run_ab_panel(
            sample, output_a, output_b,
            judges_needed,
            data_root=data_root if config.include_images else None,
        )
        total_judge_calls += len(new_votes)

        # Merge with any existing votes (drop old error votes that are being retried)
        retried_judges = {v.judge_model for v in new_votes}
        prior_votes = [
            v for v in (existing_comp.judge_votes if existing_comp else [])
            if not (v.error and v.judge_model in retried_judges)
        ]
        all_votes = prior_votes + new_votes
        winner, is_tie = majority_vote(all_votes, model_a, model_b)

        comp_kwargs: Dict[str, Any] = dict(
            sample_id=sample_id,
            model_a=model_a,
            model_b=model_b,
            company_a=extract_company(model_a),
            company_b=extract_company(model_b),
            judge_votes=all_votes,
            majority_winner=winner,
            is_tie=is_tie,
            modality=modality,
        )
        if existing_comp:
            comp_kwargs["timestamp"] = existing_comp.timestamp
        comp = PairwiseComparison(**comp_kwargs)

        # Update in-memory state
        existing_by_key[key] = comp
        for vote in new_votes:
            judged_by_key[key].add(vote.judge_model)
        updated_keys.add(key)

        errors_in_new = sum(1 for v in new_votes if v.error)
        if errors_in_new == len(new_votes):
            failed += 1
        else:
            successful += 1
            new_comparisons.append(comp)

            if verbose:
                vote_summary = "/".join(v.winner for v in all_votes)
                if is_tie:
                    print(f"TIE ({vote_summary})")
                else:
                    winner_short = winner.split("/")[-1][:20] if winner else "?"
                    print(f"{winner_short} ({vote_summary})")

        # ── Incremental atomic write after each comparison ────────────
        # Allows safe interruption at any point without losing progress.
        tmp = output_file.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for comp in existing_by_key.values():
                f.write(comp.model_dump_json() + "\n")
        os.replace(tmp, output_file)

    # ── Final write (no-op if already written incrementally) ──────────
    if updated_keys or not output_file.exists():
        tmp = output_file.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for comp in existing_by_key.values():
                f.write(comp.model_dump_json() + "\n")
        os.replace(tmp, output_file)

    comparisons: List[PairwiseComparison] = list(existing_by_key.values())

    # ── Compute Elo ratings ───────────────────────────────────────────
    elo_ratings = compute_elo_ratings(comparisons, config)

    run = BenchmarkRun(
        run_id=str(uuid.uuid4())[:8],
        config=config,
        models=available_models,
        comparisons=comparisons,
        elo_ratings=elo_ratings,
        total_judge_calls=total_judge_calls,
        successful_comparisons=successful,
        failed_comparisons=failed,
    )

    # ── Print leaderboard ─────────────────────────────────────────────
    if verbose:
        print(f"\nBenchmark complete:")
        print(f"  New comparisons: {successful} successful, {failed} failed")
        if skipped:
            print(f"  Skipped (already judged): {skipped}")
        updated_existing = sum(1 for k in updated_keys if k in {_comparison_key(c.sample_id, c.model_a, c.model_b) for c in prior_comparisons})
        print(f"  Total in file: {len(comparisons)} ({len(prior_comparisons)} prior, {len(updated_keys) - updated_existing} new, {updated_existing} updated)")
        print(f"  Judge calls (this run): {total_judge_calls}")

        # Token/cost summary for new comparisons
        total_tokens_in = 0
        total_tokens_out = 0
        for comp in new_comparisons:
            for v in comp.judge_votes:
                total_tokens_in += v.tokens_in or 0
                total_tokens_out += v.tokens_out or 0
        total_tokens = total_tokens_in + total_tokens_out
        # Rough cost estimate ($0.30/M input, $1.20/M output avg across judge models)
        est_cost = (total_tokens_in * 0.30 + total_tokens_out * 1.20) / 1_000_000
        if total_tokens > 0:
            print(f"  Tokens: {total_tokens_in:,} in + {total_tokens_out:,} out = {total_tokens:,} total")
            print(f"  Estimated cost: ~${est_cost:.2f}")

        print(f"\nElo Leaderboard:")
        sorted_ratings = sorted(
            elo_ratings.values(), key=lambda r: r.rating, reverse=True
        )
        for rank, r in enumerate(sorted_ratings, 1):
            name = r.model_id.split("/")[-1][:30]
            print(
                f"  {rank:>2}. {name:<32s} {r.company:<12s} "
                f"Elo {r.rating:>6.1f}  W/L/T {r.wins}/{r.losses}/{r.ties}"
            )

    return run


__all__ = [
    "JudgeScore",
    "JudgeRanking",
    "JudgeResult",
    "build_blind_judge_prompt",
    "build_pointwise_prompt",
    "judge_sample",
    "pointwise_judge_sample",
    "run_judge_pipeline",
    "run_pointwise_judge_pipeline",
    "compute_agreement_stats",
    "JUDGE_CRITERIA",
    # Pairwise A/B benchmark
    "AB_JUDGE_SYSTEM_PROMPT",
    "DEFAULT_JUDGE_PANEL",
    "build_ab_judge_prompt",
    "call_ab_judge",
    "run_ab_panel",
    "majority_vote",
    "run_benchmark_pipeline",
    "load_existing_comparisons",
    "_comparison_key",
    "human_labels_to_results",
    "baseline_instructions_to_results",
]
