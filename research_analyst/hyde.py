"""
HyDE — Hypothetical Document Embeddings
========================================
Instead of embedding the raw query and searching for similar documents,
HyDE generates a hypothetical answer first, embeds THAT, and searches
for documents similar to the hypothetical answer.

Why this works:
    Queries and documents live in different semantic spaces.
    "What is the relationship between X and Y?" embeds very differently
    from an article that actually explains the relationship.
    A hypothetical answer ("X and Y are connected because...") embeds
    much closer to real documents that contain similar content.

Paper: Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)
       https://arxiv.org/abs/2212.10496

Measured improvement: 10-20% better retrieval precision on open-domain QA.

Integration:
    Called from web_search.search_with_expansion() before embedding the query.
    The hypothetical document replaces or augments the query embedding.

    In orchestrator._execute_retrieval():
        # Generate hypothetical answer for better retrieval embedding
        if self.hyde.is_enabled():
            hyde_query = self.hyde.generate_hypothetical_document(
                query_text=state.normalized_query.normalized_text,
                intent=state.normalized_query.intent,
                domain=state.normalized_query.domain,
            )
            # Use hyde_query for embedding instead of original query
            state.ranked_documents = self.ranker.rank_documents(
                processed_docs,
                hyde_query,  # embed this, not the original query
            )
        else:
            state.ranked_documents = self.ranker.rank_documents(
                processed_docs,
                state.normalized_query.normalized_text,
            )
"""

import json
from typing import Optional

from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.llm_client import get_llm_client
from research_analyst.core.models import QueryIntent


logger = get_logger()


# ---------------------------------------------------------------------------
# HyDE prompt templates per intent type
# ---------------------------------------------------------------------------

_HYDE_PROMPTS = {
    QueryIntent.SEMANTIC: """Write a concise, factual paragraph that directly answers this question.
Write as if you are an expert explaining the concept. Include key terminology and relationships.
Keep it under 150 words. Write only the paragraph, nothing else.

Question: {query}

Answer:""",

    QueryIntent.ENTITY: """Write a concise factual paragraph about the entity mentioned in this question.
Include: who/what they are, key facts, why they are notable, relevant connections.
Keep it under 150 words. Write only the paragraph, nothing else.

Question: {query}

Answer:""",

    QueryIntent.RELATIONAL: """Write a concise factual paragraph explaining the relationships and connections
asked about in this question. Describe how the entities are connected, why the relationship
exists, and what its significance is.
Keep it under 150 words. Write only the paragraph, nothing else.

Question: {query}

Answer:""",

    QueryIntent.TEMPORAL: """Write a concise factual paragraph describing the historical development or
timeline asked about in this question. Include key events, dates, and how things changed.
Keep it under 150 words. Write only the paragraph, nothing else.

Question: {query}

Answer:""",

    QueryIntent.HYBRID: """Write a concise factual paragraph that covers both the conceptual aspects
and the specific entity relationships asked about in this question.
Keep it under 150 words. Write only the paragraph, nothing else.

Question: {query}

Answer:""",
}

_DEFAULT_HYDE_PROMPT = """Write a concise factual paragraph that directly answers this question.
Keep it under 150 words. Write only the paragraph, nothing else.

Question: {query}

Answer:"""


class HyDE:
    """
    Hypothetical Document Embeddings for improved retrieval.

    Generates a short hypothetical answer to the query, then uses
    that hypothetical text for embedding-based document ranking
    instead of the raw query string.

    This bridges the semantic gap between question-style text
    and answer-style document text.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.llm_client = get_llm_client()

        self.enabled: bool = getattr(self.settings, "hyde_enabled", True)
        # Only use HyDE on RESEARCH path — FAST path is already quick enough
        self.research_only: bool = getattr(self.settings, "hyde_research_only", True)

        self.logger.debug(
            "HyDE initialised",
            enabled=self.enabled,
            research_only=self.research_only,
        )

    def generate_hypothetical_document(
        self,
        query_text: str,
        intent: Optional[QueryIntent] = None,
        domain: Optional[str] = None,
    ) -> str:
        """
        Generate a hypothetical answer document for the query.

        Args:
            query_text: The original query.
            intent:     Query intent for selecting the right prompt template.
            domain:     Optional domain for context enrichment.

        Returns:
            Hypothetical answer text to use for embedding.
            Falls back to original query_text on any failure.
        """
        if not self.enabled:
            return query_text

        try:
            # Select prompt template based on intent
            prompt_template = _HYDE_PROMPTS.get(intent, _DEFAULT_HYDE_PROMPT)
            prompt = prompt_template.format(query=query_text)

            # Add domain context if available
            system_prompt = (
                f"You are an expert in {domain}. "
                if domain
                else "You are a knowledgeable research assistant. "
            )
            system_prompt += (
                "Write a brief, factual paragraph that answers the question directly. "
                "Do not include preamble, headers, or JSON. Just write the paragraph."
            )

            # Sanitize to avoid Groq JSON mode issues with special chars
            prompt = prompt.encode("ascii", errors="ignore").decode("ascii")

            hypothetical = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=200,
                temperature=0.3,   # Slightly higher than 0 for diversity
                json_mode=False,   # Plain text output
            )

            hypothetical = hypothetical.strip()

            if not hypothetical or len(hypothetical) < 20:
                self.logger.warning(
                    "HyDE generated empty/short document, using original query"
                )
                return query_text

            self.logger.info(
                "HyDE hypothetical document generated",
                original_query=query_text[:80],
                hypothetical_preview=hypothetical[:100],
                length=len(hypothetical),
            )

            return hypothetical

        except Exception as e:
            self.logger.warning(
                "HyDE generation failed, falling back to original query",
                error=str(e),
            )
            return query_text

    def augment_query(
        self,
        query_text: str,
        intent: Optional[QueryIntent] = None,
        domain: Optional[str] = None,
    ) -> str:
        """
        Return an augmented query string that combines the original query
        with the hypothetical document. This gives the embedding model
        context from both the question and the expected answer space.

        Args:
            query_text: Original query.
            intent:     Query intent.
            domain:     Optional domain.

        Returns:
            Augmented query string for embedding.
        """
        if not self.enabled:
            return query_text

        hypothetical = self.generate_hypothetical_document(
            query_text, intent, domain
        )

        if hypothetical == query_text:
            # Generation failed or disabled — return original
            return query_text

        # Combine: original query + hypothetical answer
        # The embedding of this combined text sits closer to
        # real documents than the query alone.
        augmented = f"{query_text}\n\n{hypothetical}"
        return augmented

    def is_enabled(self) -> bool:
        return bool(self.enabled)