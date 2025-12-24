"""RAG pipeline orchestrator."""

from src.llm.client import LLMClient
from src.llm.prompts import RAGPromptTemplate
from src.logging_config import get_logger
from src.rag.models import RAGQuery, RAGResponse, SourceAttribution
from src.retrieval.retriever import Retriever

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrates the RAG pipeline.

    Combines retrieval and generation into a single query interface.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        prompt_template: RAGPromptTemplate | None = None,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            retriever: Document retriever.
            llm_client: LLM client for generation.
            prompt_template: Prompt template for RAG.
        """
        self._retriever = retriever
        self._llm_client = llm_client
        self._prompt_template = prompt_template or RAGPromptTemplate()

    async def query(self, request: RAGQuery) -> RAGResponse:
        """Execute a RAG query.

        Args:
            request: The RAG query request.

        Returns:
            RAGResponse with answer and sources.
        """
        logger.info(
            "Processing RAG query",
            extra={"question_length": len(request.question), "top_k": request.top_k},
        )

        # Step 1: Retrieve relevant documents
        retrieval_results = await self._retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        # Filter by score threshold
        filtered_results = [
            r for r in retrieval_results if r.score >= request.score_threshold
        ]

        logger.debug(
            f"Retrieved {len(filtered_results)} documents",
            extra={
                "total_retrieved": len(retrieval_results),
                "after_threshold": len(filtered_results),
            },
        )

        # Handle no results
        if not filtered_results:
            return RAGResponse(
                answer="I could not find relevant information to answer your question.",
                sources=[],
                model=self._llm_client.model_name,
                tokens_used=0,
            )

        # Step 2: Build prompt with context
        chunks = [r.content for r in filtered_results]
        system_prompt, user_prompt = self._prompt_template.build_prompt(
            question=request.question,
            chunks=chunks,
        )

        # Step 3: Generate answer
        generation_result = await self._llm_client.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        # Step 4: Build source attributions
        sources = [
            SourceAttribution(
                source=r.source,
                content=r.content[:200] + "..." if len(r.content) > 200 else r.content,
                score=r.score,
            )
            for r in filtered_results
        ]

        logger.info(
            "RAG query completed",
            extra={
                "sources_count": len(sources),
                "tokens_used": generation_result.total_tokens,
            },
        )

        return RAGResponse(
            answer=generation_result.content,
            sources=sources,
            model=generation_result.model,
            tokens_used=generation_result.total_tokens,
        )

    async def query_simple(self, question: str, top_k: int = 5) -> str:
        """Simple query interface returning just the answer.

        Args:
            question: The question to answer.
            top_k: Number of documents to retrieve.

        Returns:
            Generated answer string.
        """
        response = await self.query(RAGQuery(question=question, top_k=top_k))
        return response.answer

