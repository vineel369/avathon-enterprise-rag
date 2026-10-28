"""Summarize chunk diagnostics and emit staged experiments, never invent scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.chunking.build_chunks import ROOT, STRATEGY_VERSIONS


def experiment_plan() -> dict:
    return json.loads((ROOT / "config/chunking_experiments.json").read_text(encoding="utf-8"))


def classify_runs(runs):
    active, superseded = [], []
    for run in runs:
        config = run["config"]
        expected = STRATEGY_VERSIONS[config["strategy"]]
        (active if config["strategy_version"] == expected else superseded).append(run)
    return active, superseded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, default=ROOT / "data/processed/techqa/chunks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase3/comparison.json")
    args = parser.parse_args()
    runs = []
    for report in sorted(args.chunks_dir.glob("*/report.json")):
        if report.parent.name.endswith(".partial"):
            continue
        config = json.loads((report.parent / "config.json").read_text(encoding="utf-8"))
        runs.append({"config": config, "diagnostics": json.loads(report.read_text(encoding="utf-8"))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    active, superseded = classify_runs(runs)
    result = {"experiment_plan": experiment_plan(), "chunk_runs": active,
              "superseded_chunk_runs": superseded,
              "note": "Chunk diagnostics only. Retrieval scores and final selection require later phases."}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(active)} active and {len(superseded)} superseded chunk runs to {args.output}")


if __name__ == "__main__":
    main()
