"""Tests for evaluation harness."""

import json
import tempfile

import pytest

from src.evaluation.gold_set import GoldSet, GoldSetItem, GoldSetLoader
from src.evaluation.metrics import (
    BLEUMetric,
    ROUGEMetric,
    SemanticSimilarityMetric,
)
from src.evaluation.runner import EvaluationRunner
from src.exceptions import EvaluationError


class TestGoldSetItem:
    """Tests for GoldSetItem model."""

    def test_create_item(self) -> None:
        """Item can be created with required fields."""
        item = GoldSetItem(
            id="q1",
            question="What is AI?",
            expected_answer="AI is artificial intelligence.",
        )
        assert item.id == "q1"
        assert item.context == []
        assert item.metadata == {}

    def test_item_with_context(self) -> None:
        """Item can have context chunks."""
        item = GoldSetItem(
            id="q1",
            question="What is AI?",
            expected_answer="AI is artificial intelligence.",
            context=["Chunk 1", "Chunk 2"],
        )
        assert len(item.context) == 2


class TestGoldSet:
    """Tests for GoldSet model."""

    def test_create_gold_set(self) -> None:
        """Gold set can be created."""
        gs = GoldSet(
            name="test_set",
            items=[
                GoldSetItem(id="q1", question="Q1?", expected_answer="A1"),
                GoldSetItem(id="q2", question="Q2?", expected_answer="A2"),
            ],
        )
        assert len(gs) == 2
        assert gs.version == "1.0.0"

    def test_iteration(self) -> None:
        """Gold set can be iterated."""
        gs = GoldSet(
            name="test",
            items=[
                GoldSetItem(id="q1", question="Q1?", expected_answer="A1"),
            ],
        )
        items = list(gs)
        assert len(items) == 1

    def test_filter_by_metadata(self) -> None:
        """Gold set can be filtered by metadata."""
        gs = GoldSet(
            name="test",
            items=[
                GoldSetItem(
                    id="q1",
                    question="Q1?",
                    expected_answer="A1",
                    metadata={"category": "science"},
                ),
                GoldSetItem(
                    id="q2",
                    question="Q2?",
                    expected_answer="A2",
                    metadata={"category": "history"},
                ),
            ],
        )
        filtered = gs.filter_by_metadata("category", "science")
        assert len(filtered) == 1
        assert filtered.items[0].id == "q1"


class TestGoldSetLoader:
    """Tests for GoldSetLoader."""

    def test_load_from_file(self) -> None:
        """Loader can load from JSON file."""
        data = {
            "name": "test_gold",
            "version": "1.0.0",
            "items": [
                {"id": "q1", "question": "What?", "expected_answer": "Answer"}
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()

            gs = GoldSetLoader.load_from_file(f.name)

        assert gs.name == "test_gold"
        assert len(gs) == 1

    def test_load_file_not_found(self) -> None:
        """Loader raises error for missing file."""
        with pytest.raises(EvaluationError) as exc_info:
            GoldSetLoader.load_from_file("/nonexistent/path.json")

        assert "not found" in str(exc_info.value.message)

    def test_load_invalid_json(self) -> None:
        """Loader raises error for invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {")
            f.flush()

            with pytest.raises(EvaluationError) as exc_info:
                GoldSetLoader.load_from_file(f.name)

        assert "Invalid JSON" in str(exc_info.value.message)

    def test_load_from_dict(self) -> None:
        """Loader can load from dictionary."""
        data = {
            "name": "test",
            "items": [{"id": "q1", "question": "Q?", "expected_answer": "A"}],
        }
        gs = GoldSetLoader.load_from_dict(data)
        assert gs.name == "test"


class TestROUGEMetric:
    """Tests for ROUGE-L metric."""

    def test_identical_text(self) -> None:
        """ROUGE-L score is 1.0 for identical text."""
        metric = ROUGEMetric()
        result = metric.compute("hello world", "hello world")
        assert result.score == 1.0
        assert result.name == "rouge_l"

    def test_no_overlap(self) -> None:
        """ROUGE-L score is 0.0 for no overlap."""
        metric = ROUGEMetric()
        result = metric.compute("hello world", "foo bar")
        assert result.score == 0.0

    def test_partial_overlap(self) -> None:
        """ROUGE-L computes correct score for partial overlap."""
        metric = ROUGEMetric()
        result = metric.compute(
            "the cat sat on the mat",
            "the cat sat on the floor",
        )
        assert 0.5 < result.score < 1.0
        assert "precision" in result.details
        assert "recall" in result.details

    def test_empty_input(self) -> None:
        """ROUGE-L handles empty input."""
        metric = ROUGEMetric()
        result = metric.compute("", "hello")
        assert result.score == 0.0


class TestBLEUMetric:
    """Tests for BLEU metric."""

    def test_identical_text(self) -> None:
        """BLEU score is high for identical text."""
        metric = BLEUMetric()
        result = metric.compute(
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog",
        )
        assert result.score > 0.9
        assert result.name == "bleu"

    def test_no_overlap(self) -> None:
        """BLEU score is 0.0 for no overlap."""
        metric = BLEUMetric()
        result = metric.compute("hello world", "foo bar baz")
        assert result.score == 0.0

    def test_partial_overlap(self) -> None:
        """BLEU computes reasonable score for partial match."""
        metric = BLEUMetric()
        result = metric.compute(
            "the cat sat on the mat",
            "the cat sat on the floor",
        )
        assert 0.0 < result.score < 1.0
        assert "n_gram_precisions" in result.details

    def test_empty_input(self) -> None:
        """BLEU handles empty input."""
        metric = BLEUMetric()
        result = metric.compute("", "hello")
        assert result.score == 0.0


class TestSemanticSimilarityMetric:
    """Tests for semantic similarity metric."""

    def test_identical_text(self) -> None:
        """Similarity is 1.0 for identical text."""
        metric = SemanticSimilarityMetric()
        result = metric.compute("hello world", "hello world")
        assert result.score == 1.0
        assert result.name == "semantic_similarity"

    def test_no_overlap(self) -> None:
        """Similarity is 0.0 for no word overlap."""
        metric = SemanticSimilarityMetric()
        result = metric.compute("hello world", "foo bar")
        assert result.score == 0.0

    def test_partial_overlap(self) -> None:
        """Similarity reflects word overlap."""
        metric = SemanticSimilarityMetric()
        result = metric.compute("hello world test", "hello world foo")
        # 2 common words (hello, world) out of 4 unique
        assert 0.4 < result.score < 0.6

    def test_both_empty(self) -> None:
        """Both empty returns 1.0 (identical)."""
        metric = SemanticSimilarityMetric()
        result = metric.compute("", "")
        assert result.score == 1.0


class TestEvaluationRunner:
    """Tests for EvaluationRunner."""

    def _create_gold_set(self) -> GoldSet:
        """Create a test gold set."""
        return GoldSet(
            name="test_gold",
            items=[
                GoldSetItem(
                    id="q1",
                    question="What is AI?",
                    expected_answer="AI is artificial intelligence.",
                ),
                GoldSetItem(
                    id="q2",
                    question="What is ML?",
                    expected_answer="ML is machine learning.",
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_evaluate_perfect_predictions(self) -> None:
        """Runner evaluates perfect predictions."""
        gold_set = self._create_gold_set()

        # Mock prediction function that returns exact answers
        async def perfect_predict(question: str) -> str:
            if "AI" in question:
                return "AI is artificial intelligence."
            return "ML is machine learning."

        runner = EvaluationRunner(metrics=[ROUGEMetric()])
        results, summary = await runner.evaluate(
            gold_set=gold_set,
            predict_fn=perfect_predict,
            threshold=0.8,
        )

        assert len(results) == 2
        assert summary.pass_rate == 1.0
        assert summary.passed_items == 2

    @pytest.mark.asyncio
    async def test_evaluate_with_limit(self) -> None:
        """Runner respects item limit."""
        gold_set = self._create_gold_set()

        async def mock_predict(_: str) -> str:
            return "Some answer"

        runner = EvaluationRunner(metrics=[ROUGEMetric()])
        results, summary = await runner.evaluate(
            gold_set=gold_set,
            predict_fn=mock_predict,
            limit=1,
        )

        assert len(results) == 1
        assert summary.total_items == 1

    @pytest.mark.asyncio
    async def test_evaluate_multiple_metrics(self) -> None:
        """Runner computes multiple metrics."""
        gold_set = self._create_gold_set()

        async def mock_predict(_: str) -> str:
            return "AI is artificial intelligence."

        runner = EvaluationRunner(
            metrics=[ROUGEMetric(), BLEUMetric(), SemanticSimilarityMetric()]
        )
        results, summary = await runner.evaluate(
            gold_set=gold_set,
            predict_fn=mock_predict,
        )

        assert len(results[0].metrics) == 3
        assert "rouge_l" in summary.average_scores
        assert "bleu" in summary.average_scores
        assert "semantic_similarity" in summary.average_scores

    def test_evaluate_single(self) -> None:
        """Runner can evaluate single prediction."""
        runner = EvaluationRunner(metrics=[ROUGEMetric(), BLEUMetric()])
        # Use longer text so BLEU n-grams can be computed
        text = "the quick brown fox jumps over the lazy dog"
        results = runner.evaluate_single(
            prediction=text,
            reference=text,
        )

        assert len(results) == 2
        # ROUGE-L should be 1.0 for identical text
        assert results[0].score == 1.0
        # BLEU should be high for identical text
        assert results[1].score > 0.9

