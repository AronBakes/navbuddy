#!/usr/bin/env python3
"""Validate result files for schema, vocabulary, and structural integrity.

Walks data/results/*.jsonl and flags rows that:
  - Are not valid JSON.
  - Are missing required fields (id, model_id, modality, next_action).
  - Use a next_action outside VALID_ACTIONS (e.g. 'take_the_Kings_Ave_ramp').
  - Have a modality outside the 4 canonical values.
  - Have a lane_change_required value outside {true, false, yes, no, null}.

Usage:
    python scripts/validate_results.py
    python scripts/validate_results.py --fix      # quarantine bad rows
    python scripts/validate_results.py --verbose  # print every bad row
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make navbuddy importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navbuddy.eval.metric_eval import VALID_ACTIONS, normalize_action  # noqa

VALID_MODALITIES = {"image + prior", "video + prior", "prior", "augment + prior", "cot"}
VALID_LC = {True, False, None, "yes", "no", "true", "false", ""}
REQUIRED_FIELDS = ("id", "model_id", "modality")


def classify(row):
    """Return a list of issue strings, empty if the row is clean."""
    issues = []
    for f in REQUIRED_FIELDS:
        if f not in row or row[f] in (None, ""):
            issues.append(f"missing:{f}")
    # next_action — soft: only flag if it's a non-empty value outside vocab
    na = row.get("next_action")
    if na not in (None, ""):
        norm = normalize_action(na)
        if norm not in VALID_ACTIONS:
            issues.append(f"oov_action:{norm}")
    # modality
    mod = row.get("modality")
    if mod and mod not in VALID_MODALITIES:
        issues.append(f"oov_modality:{mod}")
    # lane_change_required
    lc = row.get("lane_change_required")
    if isinstance(lc, str):
        lc_n = lc.strip().lower()
    else:
        lc_n = lc
    if lc_n not in VALID_LC:
        issues.append(f"oov_lc:{lc!r}")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results", type=Path)
    ap.add_argument("--fix", action="store_true",
                    help="Move bad rows to <file>.quarantine.jsonl, leave clean rows in place")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        sys.exit(f"results dir not found: {args.results_dir}")

    per_file_stats = []
    reason_totals = Counter()
    oov_action_examples = defaultdict(int)
    grand_total = 0
    grand_bad = 0
    grand_malformed = 0

    for path in sorted(args.results_dir.glob("*.jsonl")):
        clean_rows = []
        bad_rows = []
        n = 0
        n_bad = 0
        n_malformed = 0
        per_file_reasons = Counter()
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                n += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_malformed += 1
                    per_file_reasons["malformed_json"] += 1
                    reason_totals["malformed_json"] += 1
                    bad_rows.append(line)
                    continue
                issues = classify(row)
                if issues:
                    n_bad += 1
                    for i in issues:
                        per_file_reasons[i.split(":")[0]] += 1
                        reason_totals[i.split(":")[0]] += 1
                        if i.startswith("oov_action:"):
                            oov_action_examples[i.split(":", 1)[1]] += 1
                    bad_rows.append(line)
                    if args.verbose:
                        print(f"  {path.name}:{line_no}  {issues}")
                else:
                    clean_rows.append(line)
        grand_total += n
        grand_bad += n_bad
        grand_malformed += n_malformed
        per_file_stats.append((path.name, n, n_malformed, n_bad, per_file_reasons))

        if args.fix and bad_rows:
            quarantine = path.with_suffix(".quarantine.jsonl")
            with open(quarantine, "w") as f:
                f.writelines(bad_rows)
            with open(path, "w") as f:
                f.writelines(clean_rows)
            print(f"  fixed: {path.name}  →  {quarantine.name} ({len(bad_rows)} rows)")

    # Summary table
    print()
    print(f"{'file':<55} {'rows':>6} {'bad':>5} {'malformed':>10}")
    print("-" * 80)
    for fname, n, nm, nb, reasons in per_file_stats:
        if nb or nm:
            print(f"{fname:<55} {n:>6} {nb:>5} {nm:>10}  {dict(reasons)}")
    print()
    print(f"TOTAL: {grand_total} rows, {grand_bad} with issues, {grand_malformed} malformed")
    print(f"Issue breakdown: {dict(reason_totals)}")
    if oov_action_examples:
        print("\nOut-of-vocabulary actions seen (count):")
        for a, c in sorted(oov_action_examples.items(), key=lambda x: -x[1])[:20]:
            print(f"  {a!r:<40} {c}")
    return 1 if (grand_bad or grand_malformed) and not args.fix else 0


if __name__ == "__main__":
    sys.exit(main())
