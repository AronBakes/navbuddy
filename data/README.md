# NavBuddy-100 Dataset

This directory is populated by downloading the NavBuddy-100 manifest:

```bash
navbuddy download-manifest \
  --manifest manifests/navbuddy100.json \
  --api-key YOUR_GOOGLE_MAPS_KEY \
  --render-maps \
  --output-dir ./data \
  -y
```

After download, the structure will be:

```
data/
├── samples.jsonl                              # 100 sample metadata
├── frames/                                    # Street View dashcam images
│   └── {route_id}_step{N}_{into}m_{rem}m.jpg
├── maps/                                      # OSM overhead maps (rendered locally)
│   └── {route_id}_step{N}_map.png
├── routes/{route_id}/
│   └── metadata.json                          # Route origin/destination/distance
│
├── gt_split_samples.jsonl      # Pre-included: NavBuddy-100 sample definitions
├── gt_split_config.json        # Pre-included: split configuration
├── canonical_gt.jsonl          # Pre-included: ground truth labels (action, LC, lanes)
├── custom_labels.jsonl         # Pre-included: human annotations
├── ground_truth.jsonl          # Pre-included: starred GT instructions
└── navbuddy100_manifest.json   # Pre-included: manifest for download
```

**Cost**: ~$0.70 for 100 Street View frames. Maps are rendered locally (free).
**Requirements**: Google Maps API key with Street View Static API enabled, Playwright (`playwright install chromium`).
