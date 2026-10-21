"""Small fixtures for leakage, provenance, split, and metric correctness."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_dataset import build_corpus, build_qa, make_splits  # noqa: E402
from ragas_evaluation import score_generation  # noqa: E402
from retrieval_metrics import (  # noqa: E402
    IDBasedMeanReciprocalRank,
    precision_recall_at_k,
    reciprocal_rank,
    retrieval_scores_at_k,
)


class Phase2Tests(unittest.TestCase):
    def temporary_directory(self):
        workspace_temp = Path(__file__).resolve().parents[1] / "data" / ".test-tmp"
        workspace_temp.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=workspace_temp)

    def test_corpus_uses_archive_and_flags_duplicates_and_bad_paths(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            archive = root / "corpus.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("a/one.txt", "Title: Example\nSOLUTION\nA useful fact.")
                output.writestr("b/two.txt", "Title: Example\nSOLUTION\nA useful fact.")
                output.writestr("../escape.txt", "not safe")
            manifest, by_name, counts = build_corpus(
                archive, root / "processed", root / "raw-extracted", "abc123"
            )
            self.assertEqual(counts["ready"], 1)
            self.assertEqual(counts["duplicate_content"], 1)
            self.assertEqual(counts["unsafe_path"], 1)
            self.assertEqual(len(list((root / "processed/documents/abc123").glob("*.txt"))), 1)
            self.assertTrue((root / "raw-extracted/abc123/a/one.txt").is_file())
            ready = next(row for row in manifest if row["source_filename"] == "one.txt")
            self.assertEqual(ready["section_headings"], ["SOLUTION"])
            self.assertEqual(ready["corpus_version"], "abc123")
            self.assertEqual(by_name["one.txt"], by_name["two.txt"])
            self.assertEqual(len(manifest), 3)

    def test_qa_labels_are_separate_and_missing_source_is_flagged(self):
        rows = [{"id": "q1", "question": "What is the fix?", "answer": "Use setting X",
                 "is_impossible": False, "contexts": [{"filename": "one.txt", "text": "GOLD ONLY"}]},
                {"id": "q2", "question": "What is unknown?", "answer": "-",
                 "is_impossible": True, "contexts": []}]
        manifest, counts = build_qa(
            rows,
            {"one.txt": ["document-a"]},
            {"document-a": "different-hash"},
        )
        self.assertEqual(counts["answerable"], 1)
        self.assertEqual(manifest[0]["gold_document_ids"], ["document-a"])
        self.assertNotIn("GOLD ONLY", json.dumps(manifest))
        self.assertIn("gold_context_differs_from_corpus_document", manifest[0]["audit_flags"])
        self.assertEqual(manifest[1]["issues"], [])

    def test_shared_evidence_stays_in_one_split(self):
        rows = [{"id": f"q{i}", "is_impossible": False, "issues": [],
                 "normalized_question": f"question {i}",
                 "question_fingerprint": "same question" if i < 2 else f"question {i}",
                 "gold_document_ids": ["shared" if i < 2 else str(i)]}
                for i in range(50)]
        splits = make_splits(rows, 42, 20, 20, 0, 0)
        self.assertFalse(set(splits["development_answerable"]) & set(splits["test_answerable"]))
        self.assertEqual("q0" in splits["test_answerable"], "q1" in splits["test_answerable"])

    def test_source_metrics_deduplicate_rankings(self):
        precision, recall = precision_recall_at_k(["a", "a", "b"], {"b"}, 2)
        self.assertEqual((precision, recall), (0.5, 1.0))
        self.assertEqual(reciprocal_rank(["a", "a", "b"], {"b"}), 0.5)
        scores = retrieval_scores_at_k(["a", "a", "b"], {"b"}, 2)
        self.assertEqual(scores, {"precision@2": 0.5, "recall@2": 1.0, "mrr": 0.5})
        self.assertEqual(IDBasedMeanReciprocalRank().name, "id_based_mean_reciprocal_rank")

    def test_ragas_generation_metrics_are_orchestrated(self):
        class Metric:
            def __init__(self, value):
                self.value = value
                self.calls = []

            async def ascore(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(value=self.value)

        faithfulness = Metric(0.75)
        correctness = Metric(0.5)
        coroutine = score_generation(
            user_input="What is the fix?",
            response="Apply setting X.",
            reference="Use setting X.",
            retrieved_contexts=["Setting X resolves the problem."],
            faithfulness_metric=faithfulness,
            answer_correctness_metric=correctness,
        )
        with self.assertRaises(StopIteration) as completed:
            coroutine.send(None)
        scores = completed.exception.value
        self.assertEqual(scores.to_dict(), {"faithfulness": 0.75, "answer_correctness": 0.5})
        self.assertIn("retrieved_contexts", faithfulness.calls[0])
        self.assertIn("reference", correctness.calls[0])


if __name__ == "__main__":
    unittest.main()
