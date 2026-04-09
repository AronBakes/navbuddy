# NavBuddy Data Formats

Reference for all data files produced and consumed by NavBuddy.

---

## 1. `canonical_gt.jsonl` -- Ground Truth

One JSON object per line. Each object is a human-verified ground-truth label for a single navigation sample.

| Field | Type | Description |
|---|---|---|
| `sample_id` | `string` | Unique sample identifier (`{city}_{route_id}_step{NNN}`) |
| `enhanced_instruction` | `string` | Human-written navigation instruction with landmarks |
| `source` | `string` | Label origin, e.g. `"custom_label"` |
| `next_action` | `string` | Canonical maneuver type (e.g. `"turn_left"`, `"ramp_left"`, `"turn_right"`) |
| `next_action_human` | `string` | Human-judged action -- may differ from `next_action` when the Google maneuver name is misleading (e.g. `"ramp_left"` mapped to `"straight"`) |
| `google_maneuver` | `string` | Raw Google Directions maneuver enum (e.g. `"TURN_LEFT"`, `"RAMP_LEFT"`) |
| `next_action_source` | `string` | How `next_action` was determined, e.g. `"google_maneuver"` |
| `lane_change_required` | `bool \| null` | Whether a lane change is needed before the maneuver. `null` = unknown |
| `lanes_count` | `int` | Number of visible lanes |
| `relevant_landmarks` | `string[]` | Landmarks visible in the frame (e.g. `["traffic_light", "bridge"]`) |
| `potential_hazards` | `string[]` | Hazards present (e.g. `["pedestrians", "roadworks", "merging_traffic"]`) |
| `acceptable_actions` | `string[]` | All maneuver types considered correct for evaluation |

```json
{
  "sample_id": "brisbane_route0bmpfsqys_step001",
  "enhanced_instruction": "Turn left onto Herschel St",
  "source": "custom_label",
  "next_action": "turn_left",
  "next_action_human": "turn_left",
  "google_maneuver": "TURN_LEFT",
  "next_action_source": "google_maneuver",
  "lane_change_required": true,
  "lanes_count": 3,
  "relevant_landmarks": [],
  "potential_hazards": ["pedestrians", "cyclists"],
  "acceptable_actions": ["turn_left"]
}
```

---

## 2. `samples.jsonl` -- Sample Metadata

Created by `navbuddy setup` or `navbuddy generate`. One JSON object per line. Describes a single evaluation sample including its images, geometry, and context.

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Sample identifier (matches `sample_id` in ground truth) |
| `route_id` | `string` | Parent route identifier |
| `step_index` | `int` | 1-based step index within the route |
| `dataset_version` | `string` | Schema version, e.g. `"v1.0"` |
| `split` | `string` | Dataset split: `"train"`, `"val"`, or `"test"` |
| `maneuver` | `string` | Google maneuver type for this step |
| `prior` | `object` | Navigation context (see below) |
| `images` | `object` | Paths to associated image files (see below) |
| `geometry` | `object` | Spatial data for the step (see below) |
| `distances` | `object` | Distance info (see below) |
| `osm_road` | `object` | OpenStreetMap road attributes (see below) |
| `metadata` | `object` | Provenance info (see below) |

### Nested objects

**`prior`**

| Field | Type | Description |
|---|---|---|
| `instruction` | `string` | Original navigation instruction text |

**`images`**

| Field | Type | Description |
|---|---|---|
| `overhead` | `string \| null` | Relative path to the OSM overhead map image |
| `frames` | `string[]` | Relative paths to dashcam frame images |

**`geometry`**

| Field | Type | Description |
|---|---|---|
| `step_polyline` | `string` | Encoded polyline for this step |
| `start_lat` | `float` | Latitude at step start |
| `start_lng` | `float` | Longitude at step start |
| `end_lat` | `float` | Latitude at step end |
| `end_lng` | `float` | Longitude at step end |
| `heading` | `float` | Camera heading in degrees |

**`distances`**

| Field | Type | Description |
|---|---|---|
| `step_distance_m` | `int` | Total distance of this step in metres |
| `remaining_distance_m` | `int` | Distance remaining in the route after this step |

**`osm_road`** -- see [OSM Road Object](#osm-road-object) below.

**`metadata`**

| Field | Type | Description |
|---|---|---|
| `source` | `string` | Routing engine used, e.g. `"google"` |
| `created_at` | `string` | ISO 8601 timestamp |

```json
{
  "id": "brisbane_route0bmpfsqys_step001",
  "route_id": "brisbane_route0bmpfsqys",
  "step_index": 1,
  "dataset_version": "v1.0",
  "split": "train",
  "maneuver": "TURN_LEFT",
  "prior": { "instruction": "Turn left onto Herschel St" },
  "images": {
    "overhead": "maps/brisbane_route0bmpfsqys_step001_map.png",
    "frames": ["frames/brisbane_route0bmpfsqys_step001_118m_100m.jpg"]
  },
  "geometry": {
    "step_polyline": "f~sfDk~}d\\oClCGDmBlBYTGH",
    "start_lat": -27.468683,
    "start_lng": 153.0213364,
    "end_lat": -27.4671968,
    "end_lng": 153.0198945,
    "heading": 318.42
  },
  "distances": { "step_distance_m": 218, "remaining_distance_m": 17866 },
  "osm_road": { "highway": "primary", "name": "Turbot Street", "lanes": 3 },
  "metadata": { "source": "google", "created_at": "2026-03-30T07:00:00+00:00" }
}
```

---

## 3. `results/*.jsonl` -- Inference Results

One file per model (e.g. `claude_opus_4.jsonl`). Each line is a JSON object with the model's prediction for one sample.

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Sample identifier (matches ground truth `sample_id`) |
| `model_id` | `string` | Provider-qualified model name, e.g. `"anthropic/claude-opus-4"` |
| `modality` | `string` | Input modality used, e.g. `"image + prior"` |
| `variant` | `string \| null` | Prompt variant identifier |
| `augment` | `string \| null` | Augmentation applied, if any |
| `provider` | `string` | API provider, e.g. `"openrouter"` |
| `label_version` | `string` | Prompt/label schema version, e.g. `"v1"` |
| `enhanced_instruction` | `string` | Model-generated navigation instruction |
| `lane_change_required` | `string` | `"yes"` or `"no"` |
| `lanes_count` | `int` | Number of lanes predicted |
| `next_action` | `string` | Predicted maneuver type |
| `relevant_landmarks` | `string[]` | Landmarks identified by the model |
| `spatial_landmarks` | `any \| null` | Spatially-located landmarks (if supported) |
| `potential_hazards` | `string[]` | Hazards identified by the model |
| `reasoning` | `string` | Model's chain-of-thought explanation |
| `lane_hint` | `string \| null` | Lane positioning advice |
| `confidence` | `float \| null` | Model self-reported confidence |
| `fallback_reason` | `string \| null` | Reason if fallback logic was triggered |
| `inference_metadata` | `object` | Timing and token usage (see below) |
| `error` | `string \| null` | Error message if inference failed |

### `inference_metadata`

| Field | Type | Description |
|---|---|---|
| `latency_ms` | `int` | End-to-end inference latency in milliseconds |
| `tokens_in` | `int` | Input token count |
| `tokens_out` | `int` | Output token count |
| `tokens_reasoning` | `int` | Reasoning/thinking tokens (0 if not applicable) |
| `timestamp` | `string` | ISO 8601 inference timestamp |
| `prompt_meta` | `object` | Prompt construction details (see below) |

### `prompt_meta`

| Field | Type | Description |
|---|---|---|
| `text_prompt` | `string` | Text portion of the prompt sent to the model |
| `frame_paths` | `string[]` | Dashcam frame paths sent |
| `overhead_path` | `string` | Overhead map path sent |
| `num_images_sent` | `int` | Total number of images in the request |
| `system_prompt` | `string` | System prompt used |

---

## 4. `routes/{route_id}/` -- Route Data

Each route directory contains three files (four if created via `navbuddy route` or `navbuddy generate`).

### `route.json` (optional)

Full normalized Google Directions API response. Only present for routes created via `navbuddy route` or `navbuddy generate` -- not included in NavBuddy-100 downloads.

### `metadata.json`

Route-level summary.

| Field | Type | Description |
|---|---|---|
| `route_id` | `string` | Unique route identifier |
| `origin` | `{lat, lng}` | Route start coordinates |
| `destination` | `{lat, lng}` | Route end coordinates |
| `total_distance_m` | `int` | Total route distance in metres |
| `total_duration_s` | `int` | Estimated driving time in seconds |
| `steps_count` | `int` | Number of navigation steps |
| `city` | `string` | City name (e.g. `"brisbane"`) |
| `routing_engine` | `string` | Routing engine used, e.g. `"google"` |

```json
{
  "route_id": "brisbane_route0bmpfsqys",
  "origin": { "lat": -27.4687864, "lng": 153.0211966 },
  "destination": { "lat": -27.5772596, "lng": 153.0089299 },
  "total_distance_m": 18084,
  "total_duration_s": 1475,
  "steps_count": 7,
  "city": "brisbane",
  "routing_engine": "google"
}
```

### `guidance.json`

Step-by-step maneuver list. Contains a top-level `steps` array.

| Field | Type | Description |
|---|---|---|
| `maneuverIndex` | `int` | 1-based step index |
| `distanceMeters` | `int` | Step distance in metres |
| `polyline.encodedPolyline` | `string` | Google-encoded polyline for this step |
| `polyline.encodedPolyline5` | `string` | Polyline (precision 5) |
| `startLocation.latLng.latitude` | `float` | Step start latitude |
| `startLocation.latLng.longitude` | `float` | Step start longitude |
| `endLocation.latLng.latitude` | `float` | Step end latitude |
| `endLocation.latLng.longitude` | `float` | Step end longitude |
| `navigationInstruction.maneuver` | `string` | Google maneuver enum (e.g. `"TURN_LEFT"`) |
| `navigationInstruction.instruction` | `string` | Human-readable turn instruction |

```json
{
  "steps": [
    {
      "maneuverIndex": 1,
      "distanceMeters": 218,
      "polyline": {
        "encodedPolyline": "f~sfDk~}d\\oClCGDmBlBYTGH",
        "encodedPolyline5": "f~sfDk~}d\\oClCGDmBlBYTGH"
      },
      "startLocation": { "latLng": { "latitude": -27.468683, "longitude": 153.0213364 } },
      "endLocation": { "latLng": { "latitude": -27.4671968, "longitude": 153.0198945 } },
      "navigationInstruction": {
        "maneuver": "TURN_LEFT",
        "instruction": "Turn left onto Herschel St"
      }
    }
  ]
}
```

### `polyline.json`

Full route polyline.

| Field | Type | Description |
|---|---|---|
| `encoded` | `string` | Google-encoded polyline (full route) |
| `encoded5` | `string` | Same polyline at precision 5 |

---

## 5. `navbuddy100_manifest.json` -- Benchmark Manifest

Top-level manifest for the NavBuddy-100 evaluation benchmark. Used by `navbuddy download-manifest` to fetch Street View imagery.

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Benchmark name (`"navbuddy-100"`) |
| `version` | `string` | Manifest version |
| `created_at` | `string` | ISO 8601 creation timestamp |
| `license` | `string` | Data license (e.g. `"CC-BY-NC-4.0"`) |
| `description` | `string` | Human-readable description |
| `routes_count` | `int` | Number of routes in the benchmark |
| `samples_count` | `int` | Total number of evaluation samples |
| `total_frames.single` | `int` | Frame count for single-frame profile |
| `total_frames.sparse4` | `int` | Frame count for sparse4 profile |
| `download_instructions` | `string` | Setup instructions for users |
| `routes` | `array` | Array of route objects (see below) |

### Route object (within `routes[]`)

| Field | Type | Description |
|---|---|---|
| `route_id` | `string` | Route identifier |
| `city` | `string` | City name |
| `origin` | `{lat, lng}` | Route start coordinates |
| `destination` | `{lat, lng}` | Route end coordinates |
| `total_distance_m` | `int` | Route distance in metres |
| `total_duration_s` | `int` | Estimated duration in seconds |
| `steps_count` | `int` | Number of steps |
| `routing_engine` | `string` | Routing engine used |
| `steps` | `array` | Array of step objects (see below) |

### Step object (within `routes[].steps[]`)

| Field | Type | Description |
|---|---|---|
| `step_index` | `int` | 1-based step index |
| `maneuver` | `string` | Google maneuver enum |
| `instruction` | `string` | Navigation instruction text |
| `polyline` | `string` | Encoded polyline for this step |
| `distance_m` | `int` | Step distance in metres |
| `start_lat` | `float` | Step start latitude |
| `start_lng` | `float` | Step start longitude |
| `end_lat` | `float` | Step end latitude |
| `end_lng` | `float` | Step end longitude |
| `heading` | `float` | Driving heading in degrees |
| `frame` | `object` | Single-frame download params (see below) |
| `sparse4_frames` | `array` | Array of frame objects for sparse4 profile |
| `osm_road` | `object` | OSM road attributes (see below) |

### Frame object (within `frame` or `sparse4_frames[]`)

| Field | Type | Description |
|---|---|---|
| `filename` | `string` | Target filename for the downloaded image |
| `distance_into_step_m` | `int` | Metres from step start to frame location |
| `remaining_m` | `int` | Metres from frame location to step end |
| `pano_id` | `string \| null` | Google Street View panorama ID (if pinned) |
| `lat` | `float` | Frame latitude |
| `lng` | `float` | Frame longitude |
| `heading` | `float` | Camera heading in degrees |
| `pitch` | `int` | Camera pitch (typically `0`) |
| `fov` | `int` | Field of view in degrees (typically `90`) |
| `size` | `string` | Image dimensions, e.g. `"640x400"` |

### OSM Road Object

Attached to each step in the manifest and optionally in `samples.jsonl`.

| Field | Type | Description |
|---|---|---|
| `highway` | `string` | OSM highway classification (e.g. `"primary"`, `"motorway"`) |
| `name` | `string` | Road name |
| `ref` | `string \| null` | Road reference number |
| `maxspeed` | `string` | Speed limit (e.g. `"60"`) |
| `lanes` | `int` | Number of lanes from OSM data |
| `surface` | `string` | Road surface (e.g. `"asphalt"`) |
| `oneway` | `bool` | Whether the road is one-way |
| `lit` | `bool` | Whether the road is lit |
| `bridge` | `bool` | Whether on a bridge |
| `tunnel` | `bool` | Whether in a tunnel |
| `toll` | `bool` | Whether a toll road |
| `street_names` | `string[]` | Additional street names at this location |

---

## 6. Frame Filenames

Dashcam frame images follow this naming convention:

```
{route_id}_step{NNN}_{into}m_{remaining}m.jpg
```

| Component | Description |
|---|---|
| `route_id` | Route identifier (e.g. `brisbane_route0bmpfsqys`) |
| `NNN` | Zero-padded 3-digit step index (e.g. `001`) |
| `into` | Distance into the step in metres (e.g. `118`) |
| `remaining` | Distance remaining to step end in metres (e.g. `100`) |

**Examples:**

- `brisbane_route0bmpfsqys_step001_118m_100m.jpg` -- step 1, 118 m into step, 100 m remaining
- `brisbane_route0bmpfsqys_step004_7010m_040m.jpg` -- step 4, 7010 m into step, 40 m remaining
