"""RAGAS generation metrics configured for the project's fixed evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Sequence

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

DEFAULT_EVALUATOR_MODEL = "evaluation-judge"
DEFAULT_CORRECTNESS_WEIGHTS = (1.0, 0.0)


@dataclass(frozen=True)
class GenerationScores:
    faithfulness: float
    answer_correctness: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def create_evaluator_llm(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> Any:
    """Create the RAGAS evaluator through the OpenAI-compatible LiteLLM proxy."""
    resolved_url = (base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000")).rstrip("/")
    resolved_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_MASTER_KEY")
    if not resolved_key:
        raise ValueError("LITELLM_API_KEY or LITELLM_MASTER_KEY is required for RAGAS evaluation")
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(api_key=resolved_key, base_url=f"{resolved_url}/v1")
    return llm_factory(
        model or os.getenv("RAGAS_EVALUATOR_MODEL", DEFAULT_EVALUATOR_MODEL),
        provider="openai",
        client=client,
        adapter="instructor",
        temperature=0,
    )


async def score_generation(
    *,
    user_input: str,
    response: str,
    reference: str,
    retrieved_contexts: Sequence[str],
    evaluator_llm: Any | None = None,
    faithfulness_metric: Any | None = None,
    answer_correctness_metric: Any | None = None,
) -> GenerationScores:
    """Score one answer with RAGAS faithfulness and answer correctness."""
    if not user_input or not response or not reference or not retrieved_contexts:
        raise ValueError("question, response, reference, and retrieved contexts are required")
    llm = evaluator_llm
    if faithfulness_metric is None or answer_correctness_metric is None:
        llm = llm or create_evaluator_llm()
    if faithfulness_metric is None:
        from ragas.metrics.collections import Faithfulness

        faithfulness_metric = Faithfulness(llm=llm)
    if answer_correctness_metric is None:
        from ragas.metrics.collections import AnswerCorrectness

        answer_correctness_metric = AnswerCorrectness(
            llm=llm,
            weights=list(DEFAULT_CORRECTNESS_WEIGHTS),
        )
    faithfulness = await faithfulness_metric.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=list(retrieved_contexts),
    )
    correctness = await answer_correctness_metric.ascore(
        user_input=user_input,
        response=response,
        reference=reference,
    )
    return GenerationScores(
        faithfulness=float(faithfulness.value),
        answer_correctness=float(correctness.value),
    )
