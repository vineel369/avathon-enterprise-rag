"""Build reproducible, source-only chunk manifests; no embedding or QA access."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
from itertools import islice
import json
from pathlib import Path
import re
import time

from scripts.data_preparation.prepare_dataset import KNOWN_HEADINGS, normalize_text

ROOT = Path(__file__).resolve().parents[2]
MODEL = "BAAI/bge-base-en-v1.5"
STRATEGIES = ("fixed", "recursive", "structure")
STRATEGY_VERSIONS = {"fixed": "1", "recursive": "1", "structure": "2"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "fixed"
    target_tokens: int = 384
    overlap_tokens: int = 64
    max_input_tokens: int = 512

    def validate(self, special_tokens: int) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("Require 0 <= overlap < target")
        if self.target_tokens + special_tokens > self.max_input_tokens:
            raise ValueError("Target plus special tokens exceeds model input limit")


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def headings(text: str, conservative: bool = False) -> list[tuple[int, str]]:
    result = []
    offset = 0
    lines = text.splitlines(keepends=True)
    explicit = KNOWN_HEADINGS | {"summary", "question", "answer", "local fix", "apar status",
        "apar information", "problem summary", "recommendation", "comments", "requirements",
        "vulnerability details", "affected products and versions", "remediation/fixes",
        "workarounds and mitigations", "references", "related information", "change history"}
    for index, line in enumerate(lines):
        label = line.strip().rstrip(":").strip()
        upper = label.isupper() and any(c.isalpha() for c in label) and len(label.split()) <= 10
        if conservative:
            # Unknown headings need prose-like words and blank-line context.
            # Codes, status values, flags and single product names are not headings.
            upper = (upper and 2 <= len(label.split()) <= 8
                     and all(c.isalpha() or c.isspace() for c in label)
                     and (index == 0 or not lines[index - 1].strip())
                     and index + 1 < len(lines) and not lines[index + 1].strip())
        known = explicit if conservative else KNOWN_HEADINGS
        if label and len(label) <= 100 and (label.casefold() in known or upper):
            result.append((offset, label))
        offset += len(line)
    return result


def chunk_document(text: str, tokenizer, config: ChunkConfig) -> list[dict]:
    """Return exact slices, packing complete small sections/paragraphs together.

    Structure overlap applies to oversized blocks. Heading-only paragraph breaks
    are suppressed so a heading stays with its first body text where it fits.
    """
    special = len(tokenizer.encode("", add_special_tokens=True).ids)
    config.validate(special)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = [(a, b) for a, b in encoding.offsets if b > a]
    if not offsets:
        return []
    starts = [a for a, _ in offsets]
    sections = headings(text, conservative=config.strategy == "structure")
    section_starts = [a for a, _ in sections]
    bounds = [0, len(offsets)]
    # Separator priority is paragraph, line, sentence, then a token boundary.
    separators = [
        [m.end() for m in re.finditer(r"\n[ \t]*\n+", text)],
        [m.end() for m in re.finditer(r"\n", text)],
        [m.end() for m in re.finditer(r"[.!?](?:[ \t]+|\n)", text)],
    ]
    structure_ends = set()
    if config.strategy == "structure":
        structure_ends = {bisect_left(starts, a) for a in separators[0] + section_starts}
        structure_ends.add(len(offsets))
        for char_start, _ in sections:
            line_end = text.find("\n", char_start)
            if line_end < 0:
                line_end = len(text)
            body = line_end
            while body < len(text) and text[body].isspace():
                body += 1
            heading_token = bisect_left(starts, char_start)
            body_token = bisect_left(starts, body)
            structure_ends.difference_update(cut for cut in list(structure_ends)
                                            if heading_token < cut <= body_token)
        structure_ends = sorted(structure_ends)
    rows = []
    for low, high in zip(bounds, bounds[1:]):
        start = low
        while start < high:
            end = min(start + config.target_tokens, high)
            packed_block = False
            if config.strategy != "fixed" and end < high:
                # Avoid tiny fragments just to honor an early separator.
                minimum = start + max(config.overlap_tokens + 1, config.target_tokens // 2)
                if config.strategy == "structure":
                    pos = bisect_right(structure_ends, end) - 1
                    if pos >= 0 and structure_ends[pos] > start:
                        end = structure_ends[pos]
                        packed_block = True
                for priority, candidates in enumerate(separators):
                    if packed_block:
                        break
                    pos = bisect_right(candidates, offsets[end - 1][1]) - 1
                    if pos >= 0:
                        proposed = bisect_left(starts, candidates[pos])
                        threshold = minimum
                        if threshold <= proposed <= end:
                            end = proposed
                            break
            # Do not split multiple tokens representing one Unicode character.
            while end > start and end < high and offsets[end - 1][1] > offsets[end][0]:
                end -= 1
            while end > start:
                char_start, char_end = offsets[start][0], offsets[end - 1][1]
                piece = text[char_start:char_end]
                complete = tokenizer.encode(piece, add_special_tokens=True)
                total = len(complete.ids)
                count = total - sum(complete.special_tokens_mask)
                if count <= config.target_tokens and total <= config.max_input_tokens:
                    break
                end -= 1
            if end <= start:
                raise ValueError("Cannot fit a complete source character within token budget")
            section = bisect_right(section_starts, char_start) - 1
            row = {"text": piece, "char_start": char_start, "char_end": char_end,
                         "token_start": start, "token_end": end, "token_count": count,
                         "embedding_input_token_count": total,
                         "section_path": [sections[section][1]] if section >= 0 else []}
            if config.strategy == "structure":
                # A list of labels covered by this chunk, not a fabricated hierarchy.
                labels = row["section_path"] + [label for at, label in sections if char_start <= at < char_end]
                row["section_labels"] = list(dict.fromkeys(labels))
            rows.append(row)
            if end == high:
                break
            following = end if packed_block else max(start + 1, end - config.overlap_tokens)
            while following < end and offsets[following][0] < offsets[following - 1][1]:
                following += 1
            start = following
    return rows


def build(manifest: Path, tokenizer_path: Path, output: Path, config: ChunkConfig,
          tokenizer_model: str = MODEL, tokenizer_revision: str = "local-sha256") -> Path:
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer_model == MODEL and config.max_input_tokens > 512:
        raise ValueError("BGE base input limit is 512; use a compatible model for longer experiments")
    config.validate(len(tokenizer.encode("", add_special_tokens=True).ids))
    manifest_bytes = manifest.read_bytes()
    documents = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    if not documents or any("status" not in row or "source_path" not in row for row in documents):
        raise ValueError("Expected a Phase-2 corpus manifest, not a QA manifest")
    ready = sorted((row for row in documents if row["status"] == "ready"), key=lambda r: r["document_id"])
    if not ready or len({r["document_id"] for r in ready}) != len(ready):
        raise ValueError("Corpus must contain unique ready documents")
    versions = {r["corpus_version"] for r in ready}
    if len(versions) != 1:
        raise ValueError("Cannot mix corpus versions")
    version = STRATEGY_VERSIONS[config.strategy]
    settings = {**asdict(config), "strategy_version": version,
                "corpus_version": next(iter(versions)), "manifest_sha256": digest(manifest_bytes),
                "tokenizer_model": tokenizer_model, "tokenizer_revision": tokenizer_revision,
                "tokenizer_sha256": digest(tokenizer_path.read_bytes()),
                "tokenizers_version": importlib.metadata.version("tokenizers"),
                "embedding_text_policy": "exact_source_slice_no_prefix",
                "offset_unit": "unicode_character_in_processed_source"}
    run_id = digest(canonical(settings))[:20]
    destination = output / run_id
    if destination.exists():
        report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
        if (destination / "config.json").read_bytes() != canonical(settings):
            raise ValueError("Existing configuration does not match")
        if digest((destination / "chunks.jsonl").read_bytes()) != report["chunks_sha256"]:
            raise ValueError("Existing chunk artifact failed integrity check")
        print(f"Verified existing artifact: {destination}")
        return destination
    output.mkdir(parents=True, exist_ok=True)
    # A failed run never publishes a completed directory. Retry replaces partial files.
    staging = output / (run_id + ".partial")
    staging.mkdir(exist_ok=True)
    sizes = []
    covered_tokens = indexed_tokens = 0
    started = time.perf_counter()

    def prepare(document):
        path = Path(document["processed_path"])
        path = path if path.is_absolute() else ROOT / path
        text = path.read_text(encoding="utf-8")
        if digest(normalize_text(text).encode("utf-8")) != document["normalized_sha256"]:
            raise ValueError(f"Source content hash mismatch: {document['document_id']}")
        chunks = chunk_document(text, tokenizer, config)
        if not chunks:
            raise ValueError(f"No tokens in ready source: {document['document_id']}")
        return document, chunks

    def prepared(executor):
        documents = iter(ready)
        while batch := list(islice(documents, 32)):
            yield from executor.map(prepare, batch)

    with ThreadPoolExecutor(max_workers=4) as executor, (staging / "chunks.jsonl").open("wb") as stream:
        for number, (document, chunks) in enumerate(prepared(executor), 1):
            previous_end = 0
            for index, chunk in enumerate(chunks):
                # Count duplicated source tokens rather than estimating from requested overlap.
                indexed_tokens += chunk["token_end"] - chunk["token_start"]
                covered_tokens += max(0, chunk["token_end"] - max(previous_end, chunk["token_start"]))
                previous_end = chunk["token_end"]
                content_hash = digest(chunk["text"].encode("utf-8"))
                identity = [run_id, document["document_id"], index, chunk["char_start"], chunk["char_end"], content_hash]
                row = {**chunk, "chunk_id": digest(canonical(identity)), "source_id": document["document_id"],
                       "source_filename": document["source_filename"], "source_path": document["source_path"],
                       "title": document.get("title"), "chunk_index": index, "content_hash": content_hash,
                       "corpus_version": settings["corpus_version"], "strategy": config.strategy,
                       "strategy_version": version, "configuration_id": run_id}
                stream.write(canonical(row))
                sizes.append(chunk["token_count"])
            if number % 2000 == 0:
                print(f"{config.strategy}: {number}/{len(ready)} documents", flush=True)
    ordered = sorted(sizes)
    report = {"configuration_id": run_id, "status": "chunks_ready_retrieval_pending",
              "documents": len(ready), "workers": 4, "document_batch_size": 32, "excluded_status_counts": dict(Counter(r["status"] for r in documents if r["status"] != "ready")),
              "chunks": len(sizes), "average_tokens": sum(sizes) / len(sizes),
              "p95_tokens": ordered[min(len(ordered) - 1, int(len(ordered) * .95))], "max_tokens": max(sizes),
              "median_tokens": ordered[len(ordered) // 2],
              "chunks_under_16_tokens": sum(size < 16 for size in sizes),
              "chunks_under_64_tokens": sum(size < 64 for size in sizes),
              "duplicated_token_ratio": (indexed_tokens - covered_tokens) / indexed_tokens,
              "truncation_count": 0, "elapsed_seconds": time.perf_counter() - started,
              "manifest_bytes": (staging / "chunks.jsonl").stat().st_size,
              "chunks_sha256": digest((staging / "chunks.jsonl").read_bytes()),
              "retrieval_metrics": None, "winner": None}
    (staging / "config.json").write_bytes(canonical(settings))
    (staging / "report.json").write_bytes(canonical(report))
    staging.rename(destination)
    print(json.dumps(report, indent=2), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/techqa/corpus_manifest.jsonl")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Pinned tokenizer.json; checksum recorded")
    parser.add_argument("--tokenizer-model", default=MODEL)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/techqa/chunks")
    parser.add_argument("--strategy", choices=(*STRATEGIES, "all"), default="all")
    parser.add_argument("--target-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    args = parser.parse_args()
    for strategy in STRATEGIES if args.strategy == "all" else (args.strategy,):
        build(args.manifest, args.tokenizer, args.output,
              ChunkConfig(strategy, args.target_tokens, args.overlap_tokens, args.max_input_tokens),
              args.tokenizer_model, args.tokenizer_revision)


if __name__ == "__main__":
    main()
