"""RAGAS document-level retrieval metrics for frozen TechQA evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import typing as t

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from ragas import SingleTurnSample  # noqa: E402
from ragas.metrics._context_precision import IDBasedContextPrecision  # noqa: E402
from ragas.metrics._context_recall import IDBasedContextRecall  # noqa: E402
from ragas.metrics.base import (  # noqa: E402
    MetricOutputType,
    MetricType,
    SingleTurnMetric,
)


@dataclass
class IDBasedMeanReciprocalRank(SingleTurnMetric):
    """RAGAS-compatible reciprocal-rank metric for one evaluation sample."""

    name: str = "id_based_mean_reciprocal_rank"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            }
        }
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS

    def init(self, run_config: object) -> None:
        """The deterministic ID metric has no model or run configuration."""

    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: object,
    ) -> float:
        retrieved = sample.retrieved_context_ids
        relevant = sample.reference_context_ids
        if retrieved is None:
            raise ValueError("retrieved_context_ids is required")
        if not relevant:
            raise ValueError("reference_context_ids must be nonempty")
        relevant_ids = {str(value) for value in relevant}
        for rank, identifier in enumerate(dict.fromkeys(map(str, retrieved)), start=1):
            if identifier in relevant_ids:
                return 1.0 / rank
        return 0.0


def _sample_at_k(retrieved: list[str], relevant: set[str], k: int) -> SingleTurnSample:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("relevant source IDs must be nonempty")
    unique = list(dict.fromkeys(map(str, retrieved)))
    if len(unique) < k:
        raise ValueError(f"retrieval returned {len(unique)} unique documents; expected at least k={k}")
    return SingleTurnSample(
        retrieved_context_ids=unique[:k],
        reference_context_ids=sorted(map(str, relevant)),
    )


def _score_deterministic_metric(metric: SingleTurnMetric, sample: SingleTurnSample) -> float:
    """Run an ID-only RAGAS coroutine that is guaranteed not to suspend."""
    coroutine = metric._single_turn_ascore(sample, callbacks=[])
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return float(completed.value)
    finally:
        coroutine.close()
    raise RuntimeError(f"{metric.name} unexpectedly required asynchronous I/O")


def retrieval_scores_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> dict[str, float]:
    """Score one ranked retrieval result through the RAGAS metric interface."""
    sample = _sample_at_k(retrieved, relevant, k)
    return {
        f"precision@{k}": _score_deterministic_metric(IDBasedContextPrecision(), sample),
        f"recall@{k}": _score_deterministic_metric(IDBasedContextRecall(), sample),
        "mrr": _score_deterministic_metric(IDBasedMeanReciprocalRank(), sample),
    }


def precision_recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> tuple[float, float]:
    """Backward-compatible tuple backed by RAGAS ID metrics."""
    scores = retrieval_scores_at_k(retrieved, relevant, k)
    return scores[f"precision@{k}"], scores[f"recall@{k}"]


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Backward-compatible reciprocal rank backed by a RAGAS metric."""
    return retrieval_scores_at_k(retrieved, relevant, len(set(retrieved)))["mrr"]
