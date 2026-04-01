# NavBuddy

A VLM benchmarking toolkit for autonomous navigation. NavBuddy generates street-level imagery datasets paired with turn-by-turn instructions, benchmarks frontier and open-weight vision-language models on navigation tasks, and provides ground-truth labels for action prediction, lane-change detection, and lane counting.

**[Benchmark leaderboard](https://aronbakes.github.io/navbuddy/)**

## What it does

- **Dataset generation** — Given an origin and destination, NavBuddy routes via Google Maps, downloads Street View frames at configurable distances from each maneuver, renders OSM overhead maps, and enriches samples with road metadata from OpenStreetMap.
- **VLM benchmarking** — Run any model available on OpenRouter (Gemini, GPT, Claude, Grok, Qwen, etc.) against the dataset and score outputs on BERTScore, action accuracy, lane-change F1, and lane count MAE.
- **NavBuddy-100** — A 100-sample held-out evaluation split across Brisbane, Sydney, Melbourne, and Canberra with human-verified ground-truth labels. One command to download.

---

## Quick start

```bash
git clone https://github.com/AronBakes/navbuddy.git
cd navbuddy
pip install uv    # if you don't have uv
uv sync
source .venv/bin/activate
```

### Download NavBuddy-100

Requires a [Google Maps API key](https://console.cloud.google.com/apis/credentials) with the following APIs enabled:

| API | Required for | Enable at |
|-----|-------------|-----------|
| **Street View Static API** | `navbuddy setup`, frame download | [Enable](https://console.cloud.google.com/apis/library/street-view-image-backend.googleapis.com) |
| **Directions API** | `navbuddy generate` (new routes) | [Enable](https://console.cloud.google.com/apis/library/directions-backend.googleapis.com) |
| **Geocoding API** | `navbuddy geocode` (address lookup) | [Enable](https://console.cloud.google.com/apis/library/geocoding-backend.googleapis.com) |

For **NavBuddy-100 only** (no route generation), you just need **Street View Static API**.

```bash
navbuddy setup
```

Prompts for your API key, downloads 100 Street View frames (~$0.70) and pre-rendered overhead maps. For multi-frame sequences (4 per step, ~$2.72): `navbuddy setup --frame-profile sparse4`.

Save your key for other commands:

```bash
echo 'GOOGLE_MAPS_API_KEY=your_key' >> .env
```

### Run inference

```bash
echo 'OPENROUTER_API_KEY=your_key' >> .env
navbuddy evaluate -d ./data/samples.jsonl -m google/gemini-3-flash-preview --data-root ./data -n 5
```

---

## CLI usage

### Generate routes

```bash
# From addresses (auto-geocoded)
navbuddy generate -o "Sydney Opera House" -d "Bondi Beach" -c sydney

# From coordinates
navbuddy generate -o "-27.4698,153.0251" -d "-27.4512,153.0389" -c brisbane

# Dense sampling: every 10m along the route
navbuddy generate -o "QUT, Brisbane" -d "South Bank, Brisbane" --spacing 10

# Custom window: 10m spacing, 200m to 20m before each maneuver
navbuddy generate -o "QUT, Brisbane" -d "South Bank, Brisbane" --spacing 10 --sample-start 200 --sample-end 20
```

To render overhead maps for custom routes, install the render extra:

```bash
pip install navbuddy[render] && playwright install chromium
```

### Evaluate and score

```bash
navbuddy evaluate -d ./data/samples.jsonl -m google/gemini-3-flash-preview -o results/gemini.jsonl
navbuddy metrics -p results/gemini.jsonl -l data/ground_truth.jsonl
```

### Explore data

```bash
navbuddy stats -d ./data
navbuddy list-routes -d ./data
navbuddy play <route_id> -d ./data
```

### Utilities

```bash
navbuddy geocode "Sydney Opera House"
navbuddy reverse-geocode -c "-27.47,153.02"
```

---

## N+1 instruction offset

Google Maps labels each instruction on the segment it *produces*, not the segment where the driver is approaching the action. NavBuddy corrects for this: every sample's `prior.instruction` is set to the **next step's** instruction — the upcoming maneuver the model needs to predict.

```text
Google Maps step N:  "Turn left onto Breakfast Creek Rd"   <- describes entry into step N
NavBuddy step N:     prior.instruction = step N+1          <- approaching the next turn
```

---

## Environment variables

Copy `.env.example` to `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_MAPS_API_KEY` | Yes | Routing, Street View, geocoding |
| `OPENROUTER_API_KEY` | For inference | VLM inference via [OpenRouter](https://openrouter.ai/keys) |

---

## Citation

```bibtex
@inproceedings{bakes2026navbuddy,
  author    = {Bakes, Aron and Nguyen, Tony and Elhenawy, Mohammed and Rakotonirainy, Andry},
  title     = {NavBuddy: An AI-Augmented Navigation Assistant for Context-Aware Route Guidance},
  booktitle = {2026 IEEE International Conference on Computing and Machine Intelligence (ICMI)},
  year      = {2026},
  note      = {to appear}
}
```
