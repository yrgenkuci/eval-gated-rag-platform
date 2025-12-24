#!/usr/bin/env python
"""Run evaluation against a gold set.

Usage:
    python -m scripts.run_eval --gold-set data/gold/test.json --threshold 0.85

This script is designed to be run in CI/CD pipelines to gate deployments
based on evaluation metrics.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.evaluation.gold_set import GoldSetLoader
from src.evaluation.metrics import BLEUMetric, ROUGEMetric, SemanticSimilarityMetric
from src.evaluation.runner import EvaluationRunner
from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


async def mock_predict(question: str) -> str:
    """Mock prediction function for testing without live services.

    In production, this would call the actual RAG pipeline.
    For CI testing, we return a mock response based on the question.
    """
    # Simple mock: return a response that partially matches expected patterns
    if "AI" in question or "artificial intelligence" in question.lower():
        return "AI is artificial intelligence, a field of computer science."
    if "ML" in question or "machine learning" in question.lower():
        return "ML is machine learning, a subset of AI."
    return f"I understand you are asking about: {question}"


async def run_evaluation(
    gold_set_path: Path,
    threshold: float,
    output_path: Path | None = None,
    limit: int | None = None,
) -> bool:
    """Run evaluation and return whether it passed.

    Args:
        gold_set_path: Path to gold set JSON file.
        threshold: Minimum pass rate required.
        output_path: Optional path to save results JSON.
        limit: Optional limit on items to evaluate.

    Returns:
        True if evaluation passed threshold, False otherwise.
    """
    setup_logging(level="INFO")

    logger.info(f"Loading gold set from {gold_set_path}")
    gold_set = GoldSetLoader.load_from_file(gold_set_path)

    logger.info(f"Creating evaluation runner with threshold {threshold}")
    runner = EvaluationRunner(
        metrics=[
            ROUGEMetric(),
            BLEUMetric(),
            SemanticSimilarityMetric(),
        ]
    )

    logger.info(f"Running evaluation on {len(gold_set)} items")
    results, summary = await runner.evaluate(
        gold_set=gold_set,
        predict_fn=mock_predict,
        threshold=threshold,
        limit=limit,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Gold Set: {summary.gold_set_name}")
    print(f"Total Items: {summary.total_items}")
    print(f"Passed Items: {summary.passed_items}")
    print(f"Pass Rate: {summary.pass_rate:.2%}")
    print(f"Threshold: {summary.threshold:.2%}")
    print("\nAverage Scores:")
    for metric, score in summary.average_scores.items():
        print(f"  {metric}: {score:.4f}")
    print("=" * 60)

    # Save results if output path provided
    if output_path:
        output_data = {
            "summary": {
                "gold_set_name": summary.gold_set_name,
                "total_items": summary.total_items,
                "passed_items": summary.passed_items,
                "pass_rate": summary.pass_rate,
                "threshold": summary.threshold,
                "average_scores": summary.average_scores,
                "timestamp": summary.timestamp,
            },
            "results": [
                {
                    "item_id": r.item_id,
                    "question": r.question,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "passed": r.passed,
                    "metrics": [
                        {"name": m.name, "score": m.score, "details": m.details}
                        for m in r.metrics
                    ],
                }
                for r in results
            ],
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        logger.info(f"Results saved to {output_path}")

    # Determine pass/fail
    passed = summary.pass_rate >= threshold
    if passed:
        print(f"\nRESULT: PASSED (pass rate {summary.pass_rate:.2%} >= {threshold:.2%})")
    else:
        print(f"\nRESULT: FAILED (pass rate {summary.pass_rate:.2%} < {threshold:.2%})")

    return passed


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run evaluation against a gold set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gold-set",
        type=Path,
        required=True,
        help="Path to gold set JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum pass rate threshold (0-1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to evaluate",
    )

    args = parser.parse_args()

    passed = asyncio.run(
        run_evaluation(
            gold_set_path=args.gold_set,
            threshold=args.threshold,
            output_path=args.output,
            limit=args.limit,
        )
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

