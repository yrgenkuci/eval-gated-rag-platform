"""Evaluation runner for RAG system."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.evaluation.gold_set import GoldSet, GoldSetItem
from src.evaluation.metrics import EvaluationMetric, MetricResult
from src.logging_config import get_logger

logger = get_logger(__name__)


class EvaluationResult(BaseModel):
    """Result from evaluating a single item.

    Attributes:
        item_id: ID of the evaluated item.
        question: The input question.
        expected: Expected answer.
        predicted: Predicted answer.
        metrics: Metric results for this item.
        passed: Whether all metrics passed thresholds.
    """

    item_id: str = Field(description="Item ID")
    question: str = Field(description="Input question")
    expected: str = Field(description="Expected answer")
    predicted: str = Field(description="Predicted answer")
    metrics: list[MetricResult] = Field(description="Metric results")
    passed: bool = Field(description="Whether thresholds passed")


class EvaluationSummary(BaseModel):
    """Summary of evaluation run.

    Attributes:
        gold_set_name: Name of the gold set.
        total_items: Total items evaluated.
        passed_items: Items that passed thresholds.
        average_scores: Average score per metric.
        threshold: Minimum score threshold used.
        pass_rate: Percentage of items passed.
        timestamp: When evaluation was run.
    """

    gold_set_name: str = Field(description="Gold set name")
    total_items: int = Field(description="Total items")
    passed_items: int = Field(description="Items passed")
    average_scores: dict[str, float] = Field(description="Average per metric")
    threshold: float = Field(description="Score threshold")
    pass_rate: float = Field(description="Pass percentage")
    timestamp: str = Field(description="Evaluation timestamp")


class EvaluationRunner:
    """Runs evaluation against a gold set.

    Usage:
        runner = EvaluationRunner(metrics=[ROUGEMetric(), BLEUMetric()])
        results, summary = await runner.evaluate(
            gold_set=gold_set,
            predict_fn=my_rag_pipeline.query_simple,
            threshold=0.7,
        )
    """

    def __init__(self, metrics: list[EvaluationMetric]) -> None:
        """Initialize the evaluation runner.

        Args:
            metrics: List of metrics to compute.
        """
        self._metrics = metrics

    async def evaluate(
        self,
        gold_set: GoldSet,
        predict_fn: Callable[[str], Any],
        threshold: float = 0.7,
        limit: int | None = None,
    ) -> tuple[list[EvaluationResult], EvaluationSummary]:
        """Run evaluation on a gold set.

        Args:
            gold_set: Gold set to evaluate against.
            predict_fn: Async function that takes a question and returns answer.
            threshold: Minimum average score to pass.
            limit: Optional limit on items to evaluate.

        Returns:
            Tuple of (results, summary).
        """
        items = list(gold_set.items)
        if limit:
            items = items[:limit]

        logger.info(
            f"Starting evaluation: {gold_set.name}",
            extra={"items": len(items), "threshold": threshold},
        )

        results: list[EvaluationResult] = []
        metric_totals: dict[str, float] = {m.name: 0.0 for m in self._metrics}

        for i, item in enumerate(items):
            result = await self._evaluate_item(item, predict_fn, threshold)
            results.append(result)

            # Accumulate scores
            for metric_result in result.metrics:
                metric_totals[metric_result.name] += metric_result.score

            if (i + 1) % 10 == 0:
                logger.debug(f"Evaluated {i + 1}/{len(items)} items")

        # Compute averages
        num_items = len(items)
        average_scores = {
            name: round(total / num_items, 4) if num_items > 0 else 0.0
            for name, total in metric_totals.items()
        }

        passed_items = sum(1 for r in results if r.passed)
        pass_rate = passed_items / num_items if num_items > 0 else 0.0

        summary = EvaluationSummary(
            gold_set_name=gold_set.name,
            total_items=num_items,
            passed_items=passed_items,
            average_scores=average_scores,
            threshold=threshold,
            pass_rate=round(pass_rate, 4),
            timestamp=datetime.now(UTC).isoformat(),
        )

        logger.info(
            f"Evaluation complete: {gold_set.name}",
            extra={
                "pass_rate": summary.pass_rate,
                "averages": average_scores,
            },
        )

        return results, summary

    async def _evaluate_item(
        self,
        item: GoldSetItem,
        predict_fn: Callable[[str], Any],
        threshold: float,
    ) -> EvaluationResult:
        """Evaluate a single item.

        Args:
            item: Gold set item.
            predict_fn: Prediction function.
            threshold: Pass threshold.

        Returns:
            EvaluationResult for this item.
        """
        # Get prediction
        prediction = await predict_fn(item.question)
        if not isinstance(prediction, str):
            prediction = str(prediction)

        # Compute all metrics
        metric_results = [
            metric.compute(prediction, item.expected_answer)
            for metric in self._metrics
        ]

        # Check if average score meets threshold
        avg_score = sum(m.score for m in metric_results) / len(metric_results)
        passed = avg_score >= threshold

        return EvaluationResult(
            item_id=item.id,
            question=item.question,
            expected=item.expected_answer,
            predicted=prediction,
            metrics=metric_results,
            passed=passed,
        )

    def evaluate_single(
        self,
        prediction: str,
        reference: str,
    ) -> list[MetricResult]:
        """Evaluate a single prediction synchronously.

        Args:
            prediction: Predicted text.
            reference: Reference text.

        Returns:
            List of metric results.
        """
        return [metric.compute(prediction, reference) for metric in self._metrics]

