"""Audit structure-chunk sizes and retain source examples without loading QA."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.chunking.build_chunks import ROOT


def inspect(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    sizes = []
    examples = []
    tiny_sources = Counter()
    for line in (directory / "chunks.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        size = row["token_count"]
        sizes.append(size)
        if size < 16:
            tiny_sources[row["source_filename"]] += 1
            if len(examples) < 16:
                examples.append({key: row[key] for key in
                                 ("source_filename", "chunk_id", "text", "section_path", "token_count")})
    sizes.sort()
    return {
        "configuration_id": directory.name, "strategy_version": config["strategy_version"],
        "chunks": len(sizes), "average_tokens": sum(sizes) / len(sizes),
        "median_tokens": sizes[len(sizes) // 2],
        "under_threshold": {str(cut): sum(size < cut for size in sizes) for cut in (8, 16, 32, 64, 128)},
        "under_16_fraction": sum(size < 16 for size in sizes) / len(sizes),
        "tiny_examples": examples, "sources_with_most_tiny_chunks": tiny_sources.most_common(5),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, default=ROOT / "data/processed/techqa/chunks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase3/fragmentation_audit.json")
    args = parser.parse_args()
    runs = []
    for directory in sorted(args.chunks_dir.iterdir()):
        if not (directory / "report.json").exists() or directory.name.endswith(".partial"):
            continue
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["strategy"] == "structure":
            runs.append(inspect(directory))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps([{k: run[k] for k in ("strategy_version", "chunks", "average_tokens", "under_16_fraction")}
                      for run in runs], indent=2))
    print(f"Audit saved to {args.output}")


if __name__ == "__main__":
    main()
