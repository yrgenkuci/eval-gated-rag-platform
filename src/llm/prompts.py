"""Prompt templates for RAG."""

from abc import ABC, abstractmethod
from typing import Any


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables.

        Args:
            **kwargs: Template variables.

        Returns:
            Formatted prompt string.
        """
        ...


class RAGPromptTemplate(PromptTemplate):
    """Prompt template for RAG queries.

    Formats context and question into a structured prompt.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Answer ONLY based on the provided context
- If the context does not contain enough information, say so
- Be concise and direct
- Cite the source when possible"""

    DEFAULT_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        system_prompt: str | None = None,
        user_template: str | None = None,
    ) -> None:
        """Initialize the RAG prompt template.

        Args:
            system_prompt: Custom system prompt.
            user_template: Custom user message template.
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_template = user_template or self.DEFAULT_USER_TEMPLATE

    def format(self, **kwargs: Any) -> str:
        """Format the user template.

        Args:
            **kwargs: Must include 'context' and 'question'.

        Returns:
            Formatted user prompt.
        """
        return self.user_template.format(**kwargs)

    def format_context(self, chunks: list[str], separator: str = "\n\n---\n\n") -> str:
        """Format multiple chunks into a single context string.

        Args:
            chunks: List of text chunks.
            separator: Separator between chunks.

        Returns:
            Combined context string.
        """
        return separator.join(chunks)

    def build_prompt(
        self,
        question: str,
        chunks: list[str],
    ) -> tuple[str, str]:
        """Build complete prompt from question and chunks.

        Args:
            question: User question.
            chunks: Retrieved context chunks.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        context = self.format_context(chunks)
        user_prompt = self.format(context=context, question=question)
        return self.system_prompt, user_prompt

