# NavBuddy-100 Dataset

The quickest way to get started:

```bash
navbuddy setup --api-key YOUR_GOOGLE_MAPS_KEY
```

Or download manually:

```bash
navbuddy download-manifest \
  --manifest data/navbuddy100_manifest.json \
  --api-key YOUR_GOOGLE_MAPS_KEY \
  --output-dir ./data \
  -y
```

After download, the structure will be:

```
data/
├── samples.jsonl                              # 100 sample metadata
├── frames/                                    # Street View dashcam images
│   └── {route_id}_step{N}_{into}m_{rem}m.jpg
├── maps/                                      # OSM overhead maps (pre-rendered, from GitHub release)
│   └── {route_id}_step{N}_map.png
├── routes/{route_id}/
│   └── metadata.json                          # Route origin/destination/distance
│
├── navbuddy100_manifest.json   # Pre-included: manifest for download
├── gt_split_samples.jsonl      # Pre-included: NavBuddy-100 sample definitions
├── gt_split_config.json        # Pre-included: train/val/test split (70/15/15)
├── canonical_gt.jsonl          # Pre-included: ground truth labels (action, LC, lanes)
├── models.json                 # Pre-included: model registry (OpenRouter IDs, params, costs)
└── results/                    # Pre-included: benchmark results (29 models × 4 modalities)
    └── {model}.jsonl
```

**Cost**: ~$0.70 for 100 Street View frames (single frame per sample). ~$2.73 for sparse4 (4 frames per sample).
**Requirements**: Google Maps API key with Street View Static API enabled.
