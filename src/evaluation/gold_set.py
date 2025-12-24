"""Gold set schema and loader for evaluation."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.exceptions import ErrorCode, EvaluationError
from src.logging_config import get_logger

logger = get_logger(__name__)


class GoldSetItem(BaseModel):
    """A single evaluation example in the gold set.

    Attributes:
        id: Unique identifier for this example.
        question: The input question.
        expected_answer: The expected/reference answer.
        context: Optional context that should be retrieved.
        metadata: Additional metadata for filtering/analysis.
    """

    id: str = Field(description="Unique identifier")
    question: str = Field(description="Input question")
    expected_answer: str = Field(description="Expected reference answer")
    context: list[str] = Field(
        default_factory=list,
        description="Expected context chunks",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class GoldSet(BaseModel):
    """Collection of gold set items for evaluation.

    Attributes:
        name: Name of the gold set.
        version: Version string for tracking.
        items: List of evaluation items.
        description: Optional description.
    """

    name: str = Field(description="Gold set name")
    version: str = Field(default="1.0.0", description="Version string")
    items: list[GoldSetItem] = Field(
        default_factory=list,
        description="Evaluation items",
    )
    description: str = Field(default="", description="Optional description")

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.items)

    def __iter__(self) -> Any:
        """Iterate over items."""
        return iter(self.items)

    def filter_by_metadata(self, key: str, value: Any) -> "GoldSet":
        """Filter items by metadata key-value pair.

        Args:
            key: Metadata key to filter on.
            value: Value to match.

        Returns:
            New GoldSet with filtered items.
        """
        filtered = [
            item for item in self.items if item.metadata.get(key) == value
        ]
        return GoldSet(
            name=f"{self.name}_filtered",
            version=self.version,
            items=filtered,
            description=f"Filtered by {key}={value}",
        )


class GoldSetLoader:
    """Loader for gold set files.

    Supports JSON format with the following structure:
    {
        "name": "my_gold_set",
        "version": "1.0.0",
        "items": [
            {
                "id": "q1",
                "question": "What is AI?",
                "expected_answer": "AI is artificial intelligence."
            }
        ]
    }
    """

    @staticmethod
    def load_from_file(path: Path | str) -> GoldSet:
        """Load a gold set from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Loaded GoldSet.

        Raises:
            EvaluationError: If file cannot be loaded or parsed.
        """
        path = Path(path)

        if not path.exists():
            raise EvaluationError(
                message=f"Gold set file not found: {path}",
                code=ErrorCode.GOLD_SET_ERROR,
                details={"path": str(path)},
            )

        if not path.suffix.lower() == ".json":
            raise EvaluationError(
                message=f"Unsupported file format: {path.suffix}",
                code=ErrorCode.GOLD_SET_ERROR,
                details={"path": str(path), "expected": ".json"},
            )

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise EvaluationError(
                message=f"Invalid JSON in gold set file: {e}",
                code=ErrorCode.GOLD_SET_ERROR,
                details={"path": str(path)},
            ) from e
        except OSError as e:
            raise EvaluationError(
                message=f"Failed to read gold set file: {e}",
                code=ErrorCode.GOLD_SET_ERROR,
                details={"path": str(path)},
            ) from e

        try:
            gold_set = GoldSet.model_validate(data)
        except ValueError as e:
            raise EvaluationError(
                message=f"Invalid gold set schema: {e}",
                code=ErrorCode.GOLD_SET_ERROR,
                details={"path": str(path)},
            ) from e

        logger.info(
            f"Loaded gold set: {gold_set.name}",
            extra={"items": len(gold_set), "version": gold_set.version},
        )

        return gold_set

    @staticmethod
    def load_from_dict(data: dict[str, Any]) -> GoldSet:
        """Load a gold set from a dictionary.

        Args:
            data: Dictionary with gold set data.

        Returns:
            Loaded GoldSet.

        Raises:
            EvaluationError: If schema is invalid.
        """
        try:
            return GoldSet.model_validate(data)
        except ValueError as e:
            raise EvaluationError(
                message=f"Invalid gold set schema: {e}",
                code=ErrorCode.GOLD_SET_ERROR,
            ) from e

