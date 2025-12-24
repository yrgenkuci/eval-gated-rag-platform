"""Evaluation metrics for RAG system."""

import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from src.logging_config import get_logger

logger = get_logger(__name__)


class MetricResult(BaseModel):
    """Result from a metric computation.

    Attributes:
        name: Metric name.
        score: Computed score (0-1 range).
        details: Additional metric-specific details.
    """

    name: str = Field(description="Metric name")
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details",
    )


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name."""
        ...

    @abstractmethod
    def compute(self, prediction: str, reference: str) -> MetricResult:
        """Compute the metric.

        Args:
            prediction: Generated/predicted text.
            reference: Reference/expected text.

        Returns:
            MetricResult with score and details.
        """
        ...


class ROUGEMetric(EvaluationMetric):
    """ROUGE-L metric for evaluating text overlap.

    Computes longest common subsequence based F1 score.
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "rouge_l"

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return re.findall(r"\w+", text.lower())

    def _lcs_length(self, x: list[str], y: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0

        # Use space-optimized DP
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, prev

        return prev[n]

    def compute(self, prediction: str, reference: str) -> MetricResult:
        """Compute ROUGE-L score.

        Args:
            prediction: Generated text.
            reference: Reference text.

        Returns:
            MetricResult with F1 score.
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            )

        lcs_len = self._lcs_length(pred_tokens, ref_tokens)

        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )

        return MetricResult(
            name=self.name,
            score=f1,
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "lcs_length": lcs_len,
            },
        )


class BLEUMetric(EvaluationMetric):
    """BLEU metric for evaluating n-gram overlap.

    Simplified BLEU implementation focusing on precision.
    """

    def __init__(self, max_n: int = 4) -> None:
        """Initialize BLEU metric.

        Args:
            max_n: Maximum n-gram size (default 4).
        """
        self._max_n = max_n

    @property
    def name(self) -> str:
        """Return metric name."""
        return "bleu"

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return re.findall(r"\w+", text.lower())

    def _get_ngrams(self, tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
        """Extract n-grams from tokens."""
        if n > len(tokens):
            return Counter()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def compute(self, prediction: str, reference: str) -> MetricResult:
        """Compute BLEU score.

        Args:
            prediction: Generated text.
            reference: Reference text.

        Returns:
            MetricResult with BLEU score.
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"n_gram_precisions": {}},
            )

        # Compute n-gram precisions
        precisions: dict[str, float] = {}
        total_precision = 1.0

        for n in range(1, self._max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                precisions[f"{n}-gram"] = 0.0
                total_precision = 0.0
                continue

            # Count matches (clipped by reference count)
            matches = 0
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))

            precision = matches / sum(pred_ngrams.values())
            precisions[f"{n}-gram"] = round(precision, 4)

            # Geometric mean
            if precision > 0:
                total_precision *= precision ** (1 / self._max_n)
            else:
                total_precision = 0.0

        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(ref_tokens):
            bp = min(1.0, (len(pred_tokens) + 1) / (len(ref_tokens) + 1))

        score = bp * total_precision

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={
                "n_gram_precisions": precisions,
                "brevity_penalty": round(bp, 4),
            },
        )


class SemanticSimilarityMetric(EvaluationMetric):
    """Semantic similarity using simple word overlap.

    For production, this would use embedding cosine similarity.
    This simplified version uses Jaccard similarity.
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "semantic_similarity"

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into unique words."""
        return set(re.findall(r"\w+", text.lower()))

    def compute(self, prediction: str, reference: str) -> MetricResult:
        """Compute semantic similarity score.

        Uses Jaccard similarity as a simple baseline.
        For production, use embedding-based cosine similarity.

        Args:
            prediction: Generated text.
            reference: Reference text.

        Returns:
            MetricResult with similarity score.
        """
        pred_words = self._tokenize(prediction)
        ref_words = self._tokenize(reference)

        if not pred_words and not ref_words:
            return MetricResult(
                name=self.name,
                score=1.0,  # Both empty = identical
                details={"method": "jaccard", "intersection": 0, "union": 0},
            )

        if not pred_words or not ref_words:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"method": "jaccard", "intersection": 0, "union": 0},
            )

        intersection = len(pred_words & ref_words)
        union = len(pred_words | ref_words)

        score = intersection / union if union > 0 else 0.0

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={
                "method": "jaccard",
                "intersection": intersection,
                "union": union,
                "pred_words": len(pred_words),
                "ref_words": len(ref_words),
            },
        )

