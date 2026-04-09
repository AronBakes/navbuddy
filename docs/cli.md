# CLI Reference

Complete reference for all NavBuddy CLI commands.

---

## Setup

### `navbuddy setup`

Download NavBuddy-100 and set up the dataset. Downloads 100 Street View frames, fetches pre-rendered OSM overhead maps, and writes `samples.jsonl` with ground-truth annotations. Interactively prompts for API keys (Google Maps, OpenRouter, HuggingFace) if not already configured.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--api-key` | `str` | `None` | Google Maps API key (or set `GOOGLE_MAPS_API_KEY` env var) |
| `--output-dir`, `-O` | `Path` | `./data` | Output directory |
| `--profile` | `str` | `None` | Download profile: `manifest` (1 frame/step, ~$0.70) or `sparse4` (4 frames/step, ~$2.72). Prompted interactively if omitted. |
| `--skip-maps` | `bool` | `False` | Skip bundled overhead maps |
| `--yes`, `-y` | `bool` | `False` | Skip confirmation prompt |

**Examples:**
```bash
# Interactive setup (prompts for keys and profile)
navbuddy setup

# Non-interactive with API key
navbuddy setup --api-key YOUR_KEY --yes

# Download with sparse4 profile (4 frames per step)
navbuddy setup --profile sparse4 -O ./my_data
```

---

### `navbuddy browse`

Launch the NavBuddy-100 sample viewer. Starts a local API server and opens the Next.js dashboard in your browser. If the dashboard frontend is not available, runs in API-only mode.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data-dir`, `-d` | `Path` | `./data` | Data directory |
| `--port`, `-p` | `int` | `8765` | API server port |
| `--no-open` | `bool` | `False` | Don't open browser automatically |

**Examples:**
```bash
# Launch with defaults
navbuddy browse

# Custom data directory and port
navbuddy browse -d ./data -p 9000

# Start server without opening browser
navbuddy browse --no-open
```

---

## Data

### `navbuddy route`

Fetch a route from Google Directions API and save route metadata. Downloads only the route data (no Street View frames or maps). Saves `route.json`, `metadata.json`, `guidance.json`, and `polyline.json` to `data/routes/{route_id}/`. Cost: ~$0.005 per route (1 Directions API call).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--origin`, `-o` | `str` | *required* | Origin as `lat,lon` or address |
| `--dest`, `-d` | `str` | *required* | Destination as `lat,lon` or address |
| `--output-dir`, `-O` | `Path` | `./data` | Output directory |
| `--city`, `-c` | `str` | `None` | City name for route ID prefix |
| `--route-id` | `str` | `None` | Custom route ID (auto-generated if not provided) |

**Examples:**
```bash
# Route between two addresses
navbuddy route -o "Sydney Opera House" -d "Bondi Beach"

# Route with coordinates and city prefix
navbuddy route -o "-27.4698,153.0251" -d "-27.4512,153.0389" -c brisbane

# Custom output directory
navbuddy route -o "Central Station, Sydney" -d "Manly Beach" -O ./my_routes
```

---

### `navbuddy generate`

Generate a route with Street View frames and OSM overhead maps. Origin and destination accept either `lat,lng` coordinates or street addresses. Shows a cost estimate and per-step breakdown before downloading.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--origin`, `-o` | `str` | *required* | Origin as `lat,lon` or address |
| `--dest`, `-d` | `str` | *required* | Destination as `lat,lon` or address |
| `--output-dir`, `-O` | `Path` | `./data` | Output directory |
| `--city`, `-c` | `str` | `None` | City name for route ID prefix |
| `--route-id` | `str` | `None` | Custom route ID (auto-generated if not provided) |
| `--skip-images` | `bool` | `False` | Skip downloading Street View images |
| `--profile` | `str` | `manifest` | Frame profile: `manifest` (1 frame @ 40m), `sparse4` (4 frames), `dense` (every 5m), or `custom` |
| `--spacing`, `-s` | `float` | `None` | Frame spacing in meters (implies custom profile, e.g. `-s 20`) |
| `--sample-start` | `float` | `None` | Start of sampling window in meters from end of step |
| `--sample-end` | `float` | `None` | End of sampling window in meters from end of step |
| `--map-renderer` | `str` | `osm` | Map renderer: `osm` (Playwright + Leaflet) or `google` (Static Maps API) |
| `--car-icon` | `str` | `sedan` | Car icon: `sedan`, `arrow`, `cybertruck`, `f1`, `model3`, `wrx` |
| `--car-icon-scale` | `float` | `0.025` | Scale factor for car icons (maintains aspect ratio) |
| `--assets-dir` | `Path` | `None` | Directory containing car icon images |
| `--add-overlays/--no-overlays` | `bool` | `True` | Add navigation overlays (header + ETA) to map images |
| `--yes`, `-y` | `bool` | `False` | Skip confirmation prompt |

**Examples:**
```bash
# Generate from addresses
navbuddy generate -o "Sydney Opera House" -d "Bondi Beach"

# Generate with coordinates and city tag
navbuddy generate -o "-27.4698,153.0251" -d "-27.4512,153.0389" -c brisbane

# Sparse4 profile with custom car icon
navbuddy generate -o "123 Queen St, Brisbane" -d "South Bank, Brisbane" --profile sparse4 --car-icon cybertruck
```

---

### `navbuddy download-manifest`

Download Street View images from a manifest file. Users provide their own API key. Also reconstructs `samples.jsonl` and `routes/*/metadata.json` in the output directory.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--manifest`, `-m` | `Path` | *required* | Manifest JSON file |
| `--output-dir`, `-O` | `Path` | `./data` | Output directory |
| `--api-key` | `str` | `None` | Google Street View API key (or use env var) |
| `--limit`, `-n` | `int` | `None` | Maximum frames to download |
| `--profile` | `str` | `manifest` | Download frame profile: `manifest`, `sparse4`, `video5m`, or `custom` |
| `--spacing`, `-s` | `float` | `5.0` | Spacing in meters when `--profile custom` |
| `--sample-start` | `float` | `None` | Custom profile window start in meters from step end |
| `--sample-end` | `float` | `None` | Custom profile window end in meters from step end |
| `--cost-per-1000` | `float` | `7.0` | Estimated Street View price in USD per 1000 requests |
| `--yes`, `-y` | `bool` | `False` | Skip confirmation prompt |
| `--render-maps` | `bool` | `False` | Render OSM overhead maps (requires: `playwright install chromium`) |
| `--car-icon` | `str` | `arrow` | Car marker on overhead maps: `arrow`, `sedan`, `cybertruck`, `f1`, `model3`, `wrx` |

**Examples:**
```bash
# Download all frames from manifest
navbuddy download-manifest -m data/navbuddy100_manifest.json --api-key YOUR_KEY

# Download with overhead maps
navbuddy download-manifest -m navbuddy100_manifest.json --api-key YOUR_KEY --render-maps

# Download first 100 frames only
navbuddy download-manifest -m manifest.json --api-key YOUR_KEY --limit 100
```

---

### `navbuddy stats`

Show dataset statistics. Displays route count, sample count, frame count, maneuver distribution, augmentation breakdown, and any inference result files found.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data-root`, `-d` | `Path` | `./data` | Data directory |

**Examples:**
```bash
# Stats for default data directory
navbuddy stats

# Stats for a specific city dataset
navbuddy stats -d ./data/brisbane
```

---

### `navbuddy list-routes`

List all available routes in the dataset. Searches the data directory and its city subdirectories for route metadata.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data-root`, `-d` | `Path` | `./data` | Data directory |

**Examples:**
```bash
# List routes in default data directory
navbuddy list-routes

# List routes in a specific directory
navbuddy list-routes -d ./data/brisbane
```

---

### `navbuddy play`

Play a route with frames and instructions. Interactive TUI for previewing routes step by step.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `ROUTE_ID` (argument) | `str` | *required* | Route ID to play |
| `--data-root`, `-d` | `Path` | `./data` | Data directory |
| `--static` | `bool` | `False` | Static display (no interactivity) |

**Examples:**
```bash
# Interactive player
navbuddy play 4000_4006_X3KG7736W

# Static summary
navbuddy play 4000_4006_X3KG7736W --static

# Custom data directory
navbuddy play 2000_2000_J4MNCG01R -d ./data/brisbane
```

---

## Evaluation

### `navbuddy evaluate`

Run VLM inference on a dataset. Supports OpenRouter-hosted models and local models (e.g. Qwen2.5-VL). Outputs results as JSONL with predicted maneuvers, instructions, and inference metadata.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset`, `-d` | `Path` | *required* | Path to `samples.jsonl` file |
| `--model`, `-m` | `str` | *required* | Model ID (e.g. `google/gemini-2.0-flash-001`) |
| `--output`, `-o` | `Path` | auto-generated | Output JSONL file (default: `results_{model_id}.jsonl`) |
| `--modality` | `str` | `video + prior` | Input modality: `video + prior` or `prior` |
| `--provider`, `-p` | `str` | `openrouter` | API provider: `openrouter` or `local` |
| `--data-root` | `Path` | `None` | Root directory for image paths (default: dataset parent dir) |
| `--limit`, `-n` | `int` | `None` | Maximum number of samples to process |
| `--frames` | `str` | `None` | Comma-separated remaining distances to send (e.g. `200,60,40`). +/-10m tolerance. |
| `--dedupe-frames/--no-dedupe-frames` | `bool` | `True` | Drop duplicate frame images before model inference |
| `--include-arrive-steps/--skip-arrive-steps` | `bool` | `False` | Whether to include terminal ARRIVE samples |
| `--augment` | `str` | `None` | Image augment for video modality: `fog`, `night`, `rain`, `motion_blur` |
| `--variant` | `str` | `None` | Optional non-augmentation variant tag stored in result rows |
| `--route-id` | `str` | `None` | Only process samples matching this route ID (comma-separated for multiple) |
| `--sample-id` | `str` | `None` | Only process samples matching this sample ID (comma-separated for multiple) |
| `--prompt-version` | `str` | `v1` | Prompt version: `v1` (default) or `v2` (tighter landmark quality criteria) |
| `--structured-output/--no-structured-output` | `bool` | `False` | Use JSON schema `response_format` to constrain output |
| `--use-segformer-context` | `bool` | `False` | Inject SegFormer-derived spatial context into prompts |
| `--segformer-model-id` | `str` | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | SegFormer checkpoint ID |
| `--segformer-device` | `str` | `auto` | SegFormer device: `auto`, `cpu`, or `cuda` |
| `--segformer-cache-dir` | `Path` | `None` | Local cache directory for SegFormer weights |
| `--local-device` | `str` | `auto` | Local provider device: `auto`, `cuda`, or `cpu` |
| `--local-dtype` | `str` | `auto` | Local provider dtype: `auto`, `float16`, `bfloat16`, or `float32` |
| `--local-load-in-4bit/--no-local-load-in-4bit` | `bool` | `True` | Enable 4-bit quantization for local provider |
| `--local-max-new-tokens` | `int` | `256` | Max new tokens for local provider |
| `--local-temperature` | `float` | `0.0` | Sampling temperature for local provider (0.0 = greedy) |
| `--icl-k` | `int` | `None` | Number of ICL few-shot examples to include (with images) |
| `--icl-examples` | `str` | `None` | Comma-separated 1-based indices of ICL examples from `icl_examples.jsonl` |
| `--redis-url` | `str` | `None` | Upstash Redis REST URL for result caching (or set `UPSTASH_REDIS_REST_URL` env var) |
| `--redis-token` | `str` | `None` | Upstash Redis REST token for result caching (or set `UPSTASH_REDIS_REST_TOKEN` env var) |

**Examples:**
```bash
# Run inference with OpenRouter-hosted model
navbuddy evaluate -d ./data/samples.jsonl -m google/gemini-2.0-flash-001

# Prior-only baseline (no images)
navbuddy evaluate -d ./data/samples.jsonl -m google/gemini-2.0-flash-001 --modality prior

# Local Qwen2.5-VL 3B (8GB-friendly with 4-bit quantization)
navbuddy evaluate --provider local -m Qwen/Qwen2.5-VL-3B-Instruct -d ./data/samples.jsonl -n 10
```

---

### `navbuddy metric-eval`

Score model results against human ground-truth labels using deterministic metrics (no AI calls). Compares all result files in a directory against a ground-truth JSONL file.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset`, `-d` | `Path` | *required* | Path to `samples.jsonl` |
| `--results-dir`, `-r` | `Path` | `results` | Directory with result JSONL files |
| `--gt` | `Path` | `data/canonical_gt.jsonl` | Path to `canonical_gt.jsonl` |
| `--output`, `-o` | `Path` | `None` | Output JSONL path for detailed scores |

**Examples:**
```bash
# Score all results in the default directory
navbuddy metric-eval -d data/samples.jsonl

# Custom results directory and ground truth
navbuddy metric-eval -d data/samples.jsonl -r ./my_results --gt ./my_gt.jsonl

# Save detailed scores to file
navbuddy metric-eval -d data/samples.jsonl -o scores.jsonl
```

---

## Utilities

### `navbuddy geocode`

Convert an address to lat,lng coordinates using the Google Geocoding API. Prompts for confirmation on partial matches.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `ADDRESS` (argument) | `str` | *required* | Address to geocode |

**Examples:**
```bash
navbuddy geocode "Sydney Opera House"

navbuddy geocode "123 Queen St, Brisbane QLD"
```

---

### `navbuddy reverse-geocode`

Convert lat,lng coordinates to a street address using the Google Geocoding API.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--coords`, `-c` | `str` | *required* | Coordinates as `lat,lng` |

**Examples:**
```bash
navbuddy reverse-geocode -c "-27.4698,153.0251"

navbuddy reverse-geocode -c "-33.8568,151.2153"
```
