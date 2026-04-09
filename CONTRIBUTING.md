# Contributing to NavBuddy

Thank you for your interest in contributing to NavBuddy. This guide covers
development setup, code style, and how to submit new model results.

## Development Setup

```bash
git clone https://github.com/AronBakes/navbuddy.git
cd navbuddy
pip install uv
uv sync --all-extras
source .venv/bin/activate
```

For map rendering support:

```bash
pip install navbuddy[render] && playwright install chromium
```

## Code Style

We use [black](https://black.readthedocs.io/) for formatting and [ruff](https://docs.astral.sh/ruff/) for linting (configured in `pyproject.toml`). CI checks these automatically, but you can run them locally:

```bash
black .
ruff check .
```

## Adding a New Model

Run the evaluation harness against any OpenRouter-compatible model:

```bash
navbuddy evaluate -d data/samples.jsonl -m <openrouter-model-id> --data-root data
```

Results are saved to `data/results/` as a `.jsonl` file named after the model.

## Submitting Results

1. Fork the repository and create a feature branch.
2. Run the evaluation as described above.
3. Open a pull request that includes the new `.jsonl` file in `data/results/`.
4. Ensure your branch passes linting before requesting review.

## Project Structure

- **navbuddy/** -- Core Python package: route generation, map rendering, overlays, evaluation logic, and CLI entry points.
- **data/** -- Benchmark samples, evaluation manifests, and per-model result files (`.jsonl`).
- **config/** -- YAML configuration files for map rendering and route generation parameters.
- **scripts/** -- Standalone utility scripts for map regeneration, eval-set construction, and similar one-off tasks.
- **dashboard/** -- Next.js web application for browsing samples, viewing per-model results, and comparing metrics.
- **docs/** -- Project documentation, architecture diagrams, and supplementary references.

## Questions

Open an issue on GitHub if you have questions or run into problems.
