"""Preview three raw TechQA QA records without adding labels to the corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.data_preparation.prepare_dataset import load_qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-file", type=Path, default=Path("data/raw/techqa/train.json"))
    parser.add_argument("--ids", nargs="+", default=["TRAIN_Q000", "TRAIN_Q001", "TRAIN_Q002"])
    args = parser.parse_args()
    by_id = {str(row.get("id")): row for row in load_qa(args.qa_file)}
    for record_id in args.ids:
        row = by_id.get(record_id)
        if row is None:
            print(f"{record_id}: not found")
            continue
        preview = {"id": row.get("id"), "question": row.get("question"),
                   "answer": row.get("answer"), "is_impossible": row.get("is_impossible"),
                   "contexts": [{"filename": context.get("filename"),
                                 "text_preview": str(context.get("text", ""))[:300]}
                                for context in row.get("contexts", []) if isinstance(context, dict)]}
        print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
