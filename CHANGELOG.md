# Changelog

## v0.3.0 (2026-04-09)

- **`navbuddy route` command** — fetch route data from Google Directions API without frames or maps
- **Route metadata in setup** — `navbuddy setup` now generates `data/routes/{route_id}/` with metadata, guidance, and polyline files
- **Config-driven rendering** — all map rendering defaults (zoom, car icon, overlay scale, polyline colors) read from `config/osm_map.yaml`
- **Fixed overlay icons** — maneuver icons (turn left, turn right, etc.) now render correctly on overhead maps
- **Augmented frames** — `navbuddy setup` optionally generates night, fog, rain, and motion blur variants
- **Evaluation scoring fixes** — 3 direction groups (left/right/straight), normalized lane count scoring (0-1), no composite weights
- **Documentation** — added `docs/cli.md`, `docs/scoring.md`, `docs/data-format.md`, `docs/api-keys.md`
- **Benchmark page** — compare section loads from `data/compare.json`, Material Symbols action icons, metric explainer

## v0.2.0 (2026-03-30)

- Benchmark data for 29 models across 4 modalities
- Sample viewer dashboard (Next.js)
- Eval metrics: BERTScore, action accuracy, lane-change F1, lane count
- Pre-rendered OSM overhead maps (GitHub release)

## v0.1.0 (2026-03-25)

- Initial release
- Route generation via Google Directions API
- Street View frame download
- OSM overhead map rendering with Playwright + Leaflet
- N+1 instruction offset correction
- NavBuddy-100 manifest and ground truth
