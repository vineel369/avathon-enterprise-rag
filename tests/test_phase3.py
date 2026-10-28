"""Chunk boundary, provenance, input-limit, and artifact regression tests."""
import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers, processors

from scripts.chunking.build_chunks import ChunkConfig, STRATEGIES, STRATEGY_VERSIONS, build, chunk_document, digest, headings
from scripts.chunking.compare_chunking_strategies import classify_runs, experiment_plan
from scripts.data_preparation.prepare_dataset import normalize_text


def fixture_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    return tokenizer


class Phase3Tests(unittest.TestCase):
    def test_all_strategies_cover_source_and_respect_limits(self):
        text = "Title: Example\n\nSYMPTOM\n" + "one two three four five six. " * 80 + "\n\nSOLUTION\n" + "step A step B. " * 30
        tokenizer = fixture_tokenizer()
        for strategy in STRATEGIES:
            chunks = chunk_document(text, tokenizer, ChunkConfig(strategy, 32, 8, 34))
            coverage = set()
            for row in chunks:
                self.assertEqual(row["text"], text[row["char_start"]:row["char_end"]])
                self.assertLessEqual(row["embedding_input_token_count"], 34)
                self.assertLessEqual(row["token_count"], 32)
                coverage.update(range(row["char_start"], row["char_end"]))
            self.assertTrue(all(i in coverage for i, c in enumerate(text) if not c.isspace()), strategy)
            self.assertEqual(chunks, chunk_document(text, tokenizer, ChunkConfig(strategy, 32, 8, 34)))
            if strategy == "structure":
                self.assertTrue(any("SOLUTION" in row["section_labels"] for row in chunks))
                self.assertFalse(any(row["text"].strip() in {"SYMPTOM", "SOLUTION"} for row in chunks))

    def test_short_sections_pack_together_and_keep_all_labels(self):
        text = "SUMMARY\n\nA brief summary.\n\nCAUSE\n\nA small cause.\n\nSOLUTION\n\nApply the fix."
        rows = chunk_document(text, fixture_tokenizer(), ChunkConfig("structure", 32, 8, 34))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["section_labels"], ["SUMMARY", "CAUSE", "SOLUTION"])
        self.assertEqual(rows[0]["text"], text)

    def test_error_codes_and_status_values_are_not_headings(self):
        text = "SUMMARY\n\nAn error occurred.\n\nMQRC_OBJECT_NOT_FOUND.\nIC62341\n* CLOSED AS PROGRAM ERROR.\n\nSOLUTION\n\nInstall the fix."
        labels = [label for _, label in headings(text, conservative=True)]
        self.assertEqual(labels, ["SUMMARY", "SOLUTION"])

    def test_heading_stays_with_first_body_when_near_budget(self):
        text = " ".join(["intro"] * 27) + "\n\nSOLUTION\n\n" + " ".join(["body"] * 40)
        rows = chunk_document(text, fixture_tokenizer(), ChunkConfig("structure", 32, 8, 34))
        self.assertEqual(rows[0]["text"], " ".join(["intro"] * 27))
        solution = next(row for row in rows if "SOLUTION" in row["text"])
        self.assertIn("body", solution["text"])
        self.assertEqual(STRATEGY_VERSIONS, {"fixed": "1", "recursive": "1", "structure": "2"})

    def test_fixed_overlap_and_invalid_budgets(self):
        tokenizer = fixture_tokenizer()
        chunks = chunk_document(" ".join(["word"] * 100), tokenizer, ChunkConfig("fixed", 32, 8, 34))
        self.assertEqual([row["token_start"] for row in chunks], [0, 24, 48, 72])
        for config in [ChunkConfig("bad"), ChunkConfig(overlap_tokens=384),
                       ChunkConfig(target_tokens=512), ChunkConfig(overlap_tokens=-1)]:
            with self.assertRaises(ValueError):
                chunk_document("word", tokenizer, config)

    def test_structure_packs_complete_blocks_before_splitting_oversized_ones(self):
        tokenizer = fixture_tokenizer()
        first = " ".join(["first"] * 10)
        second = " ".join(["second"] * 30)
        text = first + "\n\n" + second
        chunks = chunk_document(text, tokenizer, ChunkConfig("structure", 32, 8, 34))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], first)
        self.assertEqual(chunks[1]["text"], second)

    def test_matrix_contains_size_overlap_and_dimension_controls(self):
        plan = experiment_plan()
        stages = {stage["name"]: stage for stage in plan["stages"]}
        self.assertEqual(stages["size"]["sizes"], [256, 384, 480])
        self.assertEqual(stages["overlap"]["overlaps"], [32, 64, 96])
        self.assertEqual(stages["long_context"]["sizes"], [768, 1024])
        gemini = next(model for model in plan["models"] if model["id"] == "gemini-embedding-2")
        self.assertEqual(gemini["dimensions"], [768, 1536, 3072])

    def test_manifest_is_source_only_versioned_and_integrity_checked(self):
        root = Path(__file__).resolve().parents[1] / "data/.test-tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            folder = Path(folder)
            source = folder / "source.txt"
            text = "Title: Example\n\nSOLUTION\n" + "fix setting restart " * 30
            source.write_text(text, encoding="utf-8")
            tokenizer_path = folder / "tokenizer.json"
            fixture_tokenizer().save(str(tokenizer_path))
            source_hash = digest(normalize_text(text).encode())
            row = {"status": "ready", "source_path": "corpus/source.txt", "source_filename": "source.txt",
                   "document_id": source_hash, "normalized_sha256": source_hash,
                   "corpus_version": "fixture", "processed_path": str(source.resolve()), "title": "Example"}
            manifest = folder / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n" + json.dumps({"status": "duplicate_content", "source_path": "dup.txt"}) + "\n")
            output = build(manifest, tokenizer_path, folder / "chunks", ChunkConfig("fixed", 32, 8, 34))
            original = (output / "chunks.jsonl").read_bytes()
            self.assertEqual(output, build(manifest, tokenizer_path, folder / "chunks", ChunkConfig("fixed", 32, 8, 34)))
            rows = [json.loads(line) for line in original.splitlines()]
            self.assertTrue(all(row["source_id"] == source_hash for row in rows))
            self.assertEqual(len({r["chunk_id"] for r in rows}), len(rows))
            self.assertEqual(json.loads((output / "report.json").read_text())["documents"], 1)
            (output / "chunks.jsonl").write_bytes(original + b"corrupted")
            with self.assertRaises(ValueError):
                build(manifest, tokenizer_path, folder / "chunks", ChunkConfig("fixed", 32, 8, 34))
            qa = folder / "qa.jsonl"
            qa.write_text(json.dumps({"question": "secret", "answer": "gold"}) + "\n")
            with self.assertRaises(ValueError):
                build(qa, tokenizer_path, folder / "chunks", ChunkConfig("fixed", 32, 8, 34))
            source.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                build(manifest, tokenizer_path, folder / "other", ChunkConfig("fixed", 32, 8, 34))

    def test_plan_does_not_claim_unrun_selection(self):
        plan = experiment_plan()
        self.assertFalse(plan["paid_usage_allowed"])
        self.assertIsNone(plan["winner"])
        self.assertEqual(plan["selection_split"], "development_only")
        self.assertEqual(len(plan["models"]), 4)

    def test_superseded_structure_is_excluded_from_active_comparison(self):
        runs = [{"config": {"strategy": strategy, "strategy_version": version}}
                for strategy, version in [("fixed", "1"), ("structure", "1"), ("structure", "2")]]
        active, superseded = classify_runs(runs)
        self.assertEqual(active, [runs[0], runs[2]])
        self.assertEqual(superseded, [runs[1]])


if __name__ == "__main__":
    unittest.main()
