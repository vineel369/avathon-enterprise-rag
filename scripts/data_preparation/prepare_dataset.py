"""Prepare a pinned, auditable TechQA corpus and isolated evaluation manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import urllib.request
import zipfile


REPO = "nvidia/TechQA-RAG-Eval"
REVISION = "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a"
FILES = ("corpus.zip", "train.json")
MIN_HELD_OUT_ANSWERABLE = 20
KNOWN_HEADINGS = {
    "abstract", "applies to", "cause", "conclusion", "content", "diagnosing the problem",
    "environment", "error description", "introduction", "problem", "problem conclusion",
    "problem determination", "resolving the problem", "solution", "symptom", "workaround",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw_dir / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"Using existing {target}; verify its SHA-256 in inspection_report.json")
            continue
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
        temporary = target.with_suffix(target.suffix + ".partial")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "avathon-techqa-prep/2.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError(f"Unexpected HTML response for {name}")
                shutil.copyfileobj(response, output)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"Downloaded {target}")


def load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        for key in ("data", "train"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = list(data.values())
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("train.json must contain a list of QA records")
    return data


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_question(text: str) -> str:
    return normalize_text(text).casefold()


def question_fingerprint(text: str) -> str:
    return re.sub(r"[^\w]+", " ", normalize_question(text), flags=re.UNICODE).strip()


def extract_title(text: str) -> str | None:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first.casefold().startswith("title:"):
        return first.split(":", 1)[1].strip() or None
    return first[:200] or None


def extract_section_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        value = line.strip().rstrip(":").strip()
        if not value or len(value) > 100:
            continue
        lowered = value.casefold()
        short_upper = len(value.split()) <= 10 and any(c.isalpha() for c in value) and value == value.upper()
        if lowered in KNOWN_HEADINGS or short_upper:
            if value not in headings:
                headings.append(value)
    return headings


def decode_document(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("document is not valid UTF-8 or Windows-1252 text")


def safe_archive_path(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.name or ".." in path.parts or any(":" in part for part in path.parts):
        return None
    return path


def write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite changed versioned artifact: {path}")
        return
    path.write_bytes(data)


def build_corpus(
    archive_path: Path,
    processed_dir: Path,
    raw_extract_dir: Path | None = None,
    corpus_version: str = "fixture",
) -> tuple[list[dict], dict[str, list[str]], dict]:
    version = corpus_version[:12]
    raw_extract_dir = raw_extract_dir or processed_dir / "raw_extracted"
    extract_dir = raw_extract_dir / version
    doc_dir = processed_dir / "documents" / version
    manifest: list[dict] = []
    by_basename: dict[str, list[str]] = defaultdict(list)
    content_owners: dict[str, str] = {}
    statuses: Counter = Counter()

    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            safe_path = safe_archive_path(member.filename)
            row = {
                "source_path": member.filename,
                "source_filename": PurePosixPath(member.filename.replace("\\", "/")).name,
                "corpus_version": corpus_version,
            }
            if safe_path is None:
                row["status"] = "unsafe_path"
            elif safe_path.suffix.casefold() != ".txt":
                row["status"] = "unsupported_format"
            else:
                try:
                    raw = archive.read(member)
                    text, encoding = decode_document(raw)
                    write_immutable(extract_dir.joinpath(*safe_path.parts), raw)
                except (RuntimeError, ValueError, zipfile.BadZipFile) as error:
                    row.update(status="malformed", error=str(error)[:200])
                else:
                    canonical = normalize_text(text)
                    if not canonical:
                        row["status"] = "empty"
                    else:
                        digest = sha256(canonical.encode("utf-8"))
                        output_path = doc_dir / f"{digest}.txt"
                        row.update(
                            document_id=digest,
                            sha256=sha256(raw),
                            normalized_sha256=digest,
                            encoding=encoding,
                            title=extract_title(text),
                            section_headings=extract_section_headings(text),
                            byte_count=len(raw),
                            character_count=len(text),
                            line_count=len(text.splitlines()),
                            raw_extracted_path=extract_dir.joinpath(*safe_path.parts).as_posix(),
                            processed_path=output_path.as_posix(),
                        )
                        if digest in content_owners:
                            row.update(status="duplicate_content", duplicate_of=content_owners[digest])
                        else:
                            row["status"] = "ready"
                            content_owners[digest] = member.filename
                            write_immutable(output_path, text.encode("utf-8"))
                        by_basename[safe_path.name].append(digest)
            statuses[row["status"]] += 1
            manifest.append(row)
    return manifest, dict(by_basename), dict(statuses)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_qa(
    rows: list[dict],
    by_basename: dict[str, list[str]],
    corpus_hashes: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    corpus_hashes = corpus_hashes or {}
    manifest: list[dict] = []
    seen_ids: set[str] = set()
    counts: Counter = Counter()
    for position, source in enumerate(rows):
        record_id = _text(source.get("id"))
        question = _text(source.get("question"))
        answer = _text(source.get("answer"))
        impossible = source.get("is_impossible")
        contexts = source.get("contexts")
        contexts = contexts if isinstance(contexts, list) else []
        context_rows = [item for item in contexts if isinstance(item, dict)]
        filenames = sorted({
            PurePosixPath(_text(item.get("filename")).replace("\\", "/")).name
            for item in context_rows if _text(item.get("filename"))
        })
        matches = {name: sorted(set(by_basename.get(name, []))) for name in filenames}
        missing = [name for name, ids in matches.items() if not ids]
        ambiguous = [name for name, ids in matches.items() if len(ids) > 1]
        context_hashes: dict[str, list[str]] = defaultdict(list)
        answer_in_gold = False
        for item in context_rows:
            name = PurePosixPath(_text(item.get("filename")).replace("\\", "/")).name
            gold_text = _text(item.get("text"))
            if name and gold_text:
                context_hashes[name].append(sha256(normalize_text(gold_text).encode("utf-8")))
                answer_in_gold = answer_in_gold or bool(answer and answer != "-" and normalize_question(answer) in normalize_question(gold_text))
        mismatched = sorted(
            name for name, hashes in context_hashes.items()
            if name in matches and len(matches[name]) == 1
            and matches[name][0] in corpus_hashes and corpus_hashes[matches[name][0]] not in hashes
        )
        issues: list[str] = []
        audits: list[str] = []
        if not record_id or record_id in seen_ids:
            issues.append("missing_or_duplicate_id")
        if not question or not isinstance(impossible, bool):
            issues.append("invalid_question_or_answerability")
        if impossible is False and (not answer or answer == "-" or not filenames):
            issues.append("answerable_without_answer_or_source")
        if missing:
            issues.append("missing_source")
        if ambiguous:
            issues.append("ambiguous_source_filename")
        if impossible is True and filenames:
            audits.append("unanswerable_has_gold_source")
        if mismatched:
            audits.append("gold_context_differs_from_corpus_document")
        seen_ids.add(record_id)
        result = {
            "id": record_id,
            "source_position": position,
            "question": question,
            "answer": answer,
            "is_impossible": impossible,
            "gold_source_filenames": filenames,
            "gold_document_ids": sorted({identifier for ids in matches.values() for identifier in ids}),
            "gold_context_normalized_sha256": {key: sorted(set(value)) for key, value in context_hashes.items()},
            "gold_context_mismatch_filenames": mismatched,
            "answer_text_found_in_gold_context": answer_in_gold if impossible is False else None,
            "missing_source_filenames": missing,
            "ambiguous_source_filenames": ambiguous,
            "normalized_question": normalize_question(question),
            "question_fingerprint": question_fingerprint(question),
            "issues": issues,
            "audit_flags": audits,
        }
        manifest.append(result)
        counts["impossible" if impossible is True else "answerable"] += 1
        for issue in issues:
            counts[f"issue:{issue}"] += 1
        for audit in audits:
            counts[f"audit:{audit}"] += 1
    return manifest, dict(counts)


def choose_groups(rows: list[dict], target: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Keep near-duplicate questions and shared gold documents together."""
    parent = list(range(len(rows)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[root(left)] = root(right)

    owners: dict[str, int] = {}
    for index, row in enumerate(rows):
        fingerprint = row.get("question_fingerprint") or row.get("normalized_question", "")
        keys = [f"q:{fingerprint}"] + [f"d:{value}" for value in row["gold_document_ids"]]
        for key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
    groups: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    groups_list = list(groups.values())
    rng.shuffle(groups_list)
    chosen: list[dict] = []
    remaining: list[dict] = []
    for group in groups_list:
        (chosen if len(chosen) < target else remaining).extend(group)
    return chosen, remaining


def make_splits(
    rows: list[dict],
    seed: int,
    dev_target: int,
    test_target: int,
    impossible_dev_target: int,
    impossible_test_target: int,
) -> dict[str, list[str]]:
    if test_target < MIN_HELD_OUT_ANSWERABLE:
        raise ValueError(f"held-out answerable target must be at least {MIN_HELD_OUT_ANSWERABLE}")
    rng = random.Random(seed)
    usable = [row for row in rows if not row["issues"]]
    answerable = [row for row in usable if row["is_impossible"] is False]
    impossible = [row for row in usable if row["is_impossible"] is True]
    test, rest = choose_groups(answerable, test_target, rng)
    dev, _ = choose_groups(rest, dev_target, rng)
    impossible_test, rest_impossible = choose_groups(impossible, impossible_test_target, rng)
    impossible_dev, _ = choose_groups(rest_impossible, impossible_dev_target, rng)
    return {
        "development_answerable": sorted(row["id"] for row in dev),
        "test_answerable": sorted(row["id"] for row in test),
        "development_unanswerable": sorted(row["id"] for row in impossible_dev),
        "test_unanswerable": sorted(row["id"] for row in impossible_test),
    }


def split_overlap_report(rows: list[dict], splits: dict[str, list[str]]) -> dict:
    by_id = {row["id"]: row for row in rows}
    dev_ids = set(splits["development_answerable"] + splits["development_unanswerable"])
    test_ids = set(splits["test_answerable"] + splits["test_unanswerable"])
    dev_docs = {doc for item in dev_ids for doc in by_id[item]["gold_document_ids"]}
    test_docs = {doc for item in test_ids for doc in by_id[item]["gold_document_ids"]}
    dev_questions = {by_id[item]["question_fingerprint"] for item in dev_ids}
    test_questions = {by_id[item]["question_fingerprint"] for item in test_ids}
    return {
        "record_ids": sorted(dev_ids & test_ids),
        "gold_document_ids": sorted(dev_docs & test_docs),
        "question_fingerprints": sorted(dev_questions & test_questions),
    }


def qa_schema_profile(rows: list[dict]) -> dict:
    fields: Counter = Counter()
    types: dict[str, Counter] = defaultdict(Counter)
    missing: Counter = Counter()
    for row in rows:
        for key, value in row.items():
            fields[key] += 1
            types[key][type(value).__name__] += 1
        for key in ("id", "question", "answer", "is_impossible", "contexts"):
            if key not in row or row[key] is None:
                missing[key] += 1
    return {
        "fields": dict(sorted(fields.items())),
        "types": {key: dict(sorted(value.items())) for key, value in sorted(types.items())},
        "missing_required_fields": dict(sorted(missing.items())),
    }


def write_evaluation_splits(output_dir: Path, rows: list[dict], splits: dict[str, list[str]]) -> None:
    by_id = {row["id"]: row for row in rows}
    evaluation_dir = output_dir / "evaluation"
    for split_name, identifiers in splits.items():
        write_jsonl(evaluation_dir / f"{split_name}.jsonl", [by_id[value] for value in identifiers])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/techqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/techqa"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-answerable", type=int, default=100)
    parser.add_argument("--test-answerable", type=int, default=100)
    parser.add_argument("--development-unanswerable", type=int, default=20)
    parser.add_argument("--test-unanswerable", type=int, default=20)
    args = parser.parse_args()
    if not args.skip_download:
        download(args.raw_dir)
    source_paths = {name: args.raw_dir / name for name in FILES}
    for name, path in source_paths.items():
        if not path.is_file():
            parser.error(f"Missing {path}; download it or omit --skip-download")

    source_bytes = {name: path.read_bytes() for name, path in source_paths.items()}
    source_hashes = {name: sha256(value) for name, value in source_bytes.items()}
    corpus_version = sha256((REVISION + source_hashes["corpus.zip"]).encode())
    corpus, by_basename, corpus_counts = build_corpus(
        source_paths["corpus.zip"],
        args.output_dir,
        args.raw_dir / "extracted",
        corpus_version,
    )
    corpus_hashes = {
        row["document_id"]: row["normalized_sha256"]
        for row in corpus if row["status"] in {"ready", "duplicate_content"}
    }
    source_qa = load_qa(source_paths["train.json"])
    qa, qa_counts = build_qa(source_qa, by_basename, corpus_hashes)
    splits = make_splits(
        qa, args.seed, args.development_answerable, args.test_answerable,
        args.development_unanswerable, args.test_unanswerable,
    )
    overlaps = split_overlap_report(qa, splits)
    if any(overlaps.values()):
        raise ValueError(f"Development/test leakage detected: {overlaps}")

    write_jsonl(args.output_dir / "corpus_manifest.jsonl", corpus)
    write_jsonl(args.output_dir / "qa_manifest.jsonl", qa)
    write_json(args.output_dir / "splits.json", {"seed": args.seed, **splits})
    write_evaluation_splits(args.output_dir, qa, splits)
    basename_collisions = {
        name: sorted(set(ids)) for name, ids in by_basename.items() if len(set(ids)) > 1
    }
    duplicate_questions: dict[str, list[str]] = defaultdict(list)
    for row in qa:
        duplicate_questions[row["question_fingerprint"]].append(row["id"])
    report = {
        "dataset": REPO,
        "revision": REVISION,
        "corpus_version": corpus_version,
        "source_files": {
            name: {"bytes": len(source_bytes[name]), "sha256": source_hashes[name]}
            for name in FILES
        },
        "qa_schema": qa_schema_profile(source_qa),
        "corpus_counts": corpus_counts,
        "qa_counts": qa_counts,
        "qa_total": len(qa),
        "coverage": {
            "qa_without_blocking_issues": sum(not row["issues"] for row in qa),
            "qa_with_all_gold_sources_resolved": sum(not row["missing_source_filenames"] for row in qa),
            "unique_gold_filenames": len({name for row in qa for name in row["gold_source_filenames"]}),
            "resolved_unique_gold_filenames": len({
                name for row in qa for name in row["gold_source_filenames"] if by_basename.get(name)
            }),
        },
        "split_counts": {key: len(value) for key, value in splits.items()},
        "split_overlap": overlaps,
        "unused_qa_records": len(qa) - sum(len(value) for value in splits.values()),
        "duplicate_question_groups": sorted(
            (sorted(ids) for ids in duplicate_questions.values() if len(ids) > 1),
            key=lambda ids: ids[0],
        ),
        "basename_collisions": basename_collisions,
        "gold_filenames_missing": sorted({
            name for row in qa for name in row["missing_source_filenames"]
        }),
        "gold_filenames_ambiguous": sorted({
            name for row in qa for name in row["ambiguous_source_filenames"]
        }),
    }
    write_json(args.output_dir / "inspection_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
