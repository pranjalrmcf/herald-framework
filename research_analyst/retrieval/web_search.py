"""
Web search module for the research analyst system.
Supports Serper (Google Search API), DuckDuckGo, and Mock search (DEV only).

Changes vs original:
    - search_with_expansion() accepts skip_expansion=False param
      so orchestrator can skip the LLM query expansion call on FAST path
      saving 3-5 seconds per FAST path query
"""

from typing import List, Optional
from datetime import datetime
import requests

from duckduckgo_search import DDGS

from research_analyst.core.models import Document, SourceType, NormalizedQuery
from research_analyst.core.exceptions import SearchEngineError, NoResultsError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id, extract_domain, deduplicate_by_key

logger = get_logger()


class WebSearch:
    """Web search with fallback strategy (Serper -> DuckDuckGo -> Mock)."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        time_range: Optional[str] = None,
    ) -> List[Document]:

        if max_results is None:
            max_results = self.settings.max_search_results

        # DEV MODE MOCK
        if self.settings.mock_search and self.settings.dev_mode:
            self.logger.warning("Using MOCK search backend (DEV MODE)")
            return self._mock_results(query)

        documents: List[Document] = []

        # SERPER (PRIMARY)
        if self.settings.search_engine == "serper":
            try:
                documents = self._search_serper(query, max_results)
            except Exception as e:
                self.logger.error(
                    "Serper failed, falling back to DuckDuckGo",
                    error=str(e),
                )

        # DUCKDUCKGO (SECONDARY)
        if not documents:
            try:
                documents = self._search_duckduckgo(
                    query=query,
                    max_results=max_results,
                    time_range=time_range,
                )
            except Exception as e:
                self.logger.error("DuckDuckGo failed", error=str(e))

        # FINAL FALLBACK
        if not documents:
            if self.settings.dev_mode:
                self.logger.warning("All engines failed, using MOCK results")
                return self._mock_results(query)
            raise SearchEngineError(
                "All search engines failed",
                details={"query": query},
            )

        # DEDUP
        unique = deduplicate_by_key(
            [doc.__dict__ for doc in documents],
            key="url",
        )
        documents = [Document(**d) for d in unique]

        if len(documents) < 3:
            self.logger.warning(
                "Low document count, supplementing with DuckDuckGo",
                count=len(documents),
            )
            try:
                supplement = self._search_duckduckgo(query, 3, time_range)
                documents.extend(supplement)
            except Exception:
                pass

        return documents[:max_results]

    # =========================================================================
    # SERPER SEARCH
    # =========================================================================

    def _search_serper(self, query: str, max_results: int) -> List[Document]:
        if not self.settings.serper_api_key:
            raise SearchEngineError("SERPER_API_KEY missing")

        headers = {
            "X-API-KEY": self.settings.serper_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": max_results}
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=self.settings.search_timeout,
        )
        response.raise_for_status()
        data = response.json()
        documents: List[Document] = []

        for item in data.get("organic", []):
            documents.append(Document(
                doc_id=generate_id("doc"),
                url=item["link"],
                title=item.get("title", "Untitled"),
                content=item.get("snippet", ""),
                snippet=item.get("snippet", ""),
                source_type=self._classify_source_type(item["link"]),
                retrieved_at=datetime.utcnow(),
                credibility_score=self._estimate_credibility(item["link"]),
                metadata={"engine": "serper"},
            ))

        if not documents:
            raise NoResultsError("No results from Serper")
        return documents

    # =========================================================================
    # DUCKDUCKGO SEARCH
    # =========================================================================

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        time_range: Optional[str],
    ) -> List[Document]:
        try:
            ddgs = DDGS()
            raw_results = ddgs.text(
                keywords=query,
                max_results=max_results,
                timelimit=time_range,
            )
            documents: List[Document] = []
            for r in raw_results:
                url = r.get("href")
                if not url:
                    continue
                documents.append(Document(
                    doc_id=generate_id("doc"),
                    url=url,
                    title=r.get("title", "Untitled"),
                    content=r.get("body", ""),
                    snippet=r.get("body", ""),
                    source_type=self._classify_source_type(url),
                    retrieved_at=datetime.utcnow(),
                    credibility_score=self._estimate_credibility(url),
                    metadata={"engine": "duckduckgo"},
                ))
            if not documents:
                raise NoResultsError("No DuckDuckGo results")
            return documents
        except Exception as e:
            raise SearchEngineError(f"DuckDuckGo failed: {e}")

    # =========================================================================
    # MOCK SEARCH (DEV ONLY)
    # =========================================================================

    def _mock_results(self, query: str) -> List[Document]:
        return [Document(
            doc_id="mock_graph_rag",
            url="https://example.com/graph-rag",
            title="Graph RAG Explained",
            content=(
                "Graph RAG integrates knowledge graphs with retrieval-augmented "
                "generation to enable multi-hop reasoning and structured context."
            ),
            snippet="Graph RAG integrates knowledge graphs with RAG pipelines.",
            source_type=SourceType.WEB,
            retrieved_at=datetime.utcnow(),
            credibility_score=0.9,
            metadata={"engine": "mock"},
        )]

    # =========================================================================
    # QUERY EXPANSION — supports skip_expansion for FAST path
    # =========================================================================

    def search_with_expansion(
        self,
        normalized_query: NormalizedQuery,
        max_total_results: Optional[int] = None,
        skip_expansion: bool = False,          # NEW: set True on FAST path
    ) -> List[Document]:
        """
        Search with optional query expansion.

        Args:
            normalized_query:  Normalized query object.
            max_total_results: Max docs to retrieve total.
            skip_expansion:    If True, skip LLM expansion and search directly.
                               Use on FAST path to save 3-5 seconds.

        Returns:
            List of retrieved Document objects.
        """
        if max_total_results is None:
            max_total_results = self.settings.max_search_results

        # ── FAST PATH: skip LLM expansion entirely ─────────────────────
        if skip_expansion:
            self.logger.info(
                "Skipping query expansion (fast path)",
                query=normalized_query.normalized_text[:80],
            )
            return self.search(
                normalized_query.normalized_text,
                max_results=max_total_results,
            )

        # ── RESEARCH PATH: full LLM-based expansion ─────────────────────
        from research_analyst.query_processing.normalizer import QueryNormalizer
        from research_analyst.core.models import QueryComplexity

        normalizer = QueryNormalizer()
        expanded = normalizer.expand_query(normalized_query)

        if normalized_query.complexity == QueryComplexity.SIMPLE:
            expanded = expanded[:2]
        elif normalized_query.complexity == QueryComplexity.MEDIUM:
            expanded = expanded[:3]
        else:
            expanded = expanded[:4]

        self.logger.info(
            "Searching with expansion",
            original_query=normalized_query.normalized_text,
            num_expansions=len(expanded),
        )

        return self.multi_query_search(
            expanded,
            max_results_per_query=max(1, max_total_results // len(expanded)),
        )

    # =========================================================================
    # MULTI QUERY SEARCH
    # =========================================================================

    def multi_query_search(
        self,
        queries: List[str],
        max_results_per_query: Optional[int] = None,
    ) -> List[Document]:

        if max_results_per_query is None:
            max_results_per_query = max(
                3,
                self.settings.max_search_results // len(queries),
            )

        self.logger.info(
            "Executing multi-query search",
            num_queries=len(queries),
            max_per_query=max_results_per_query,
        )

        all_documents: List[Document] = []
        for q in queries:
            try:
                all_documents.extend(self.search(q, max_results_per_query))
            except Exception as e:
                self.logger.warning(
                    "Search failed for expanded query",
                    query=q,
                    error=str(e),
                )

        if not all_documents:
            raise NoResultsError("No results from any expanded query")

        unique = deduplicate_by_key(
            [doc.__dict__ for doc in all_documents],
            "url",
        )
        return [Document(**d) for d in unique]

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _classify_source_type(self, url: str) -> SourceType:
        domain = extract_domain(url).lower()
        if any(x in domain for x in ["arxiv", "edu", "nature", "science"]):
            return SourceType.ACADEMIC
        if any(x in domain for x in ["bbc", "reuters", "nytimes"]):
            return SourceType.NEWS
        if any(x in domain for x in ["gov", "mil"]):
            return SourceType.GOVERNMENT
        if any(x in domain for x in ["twitter", "reddit"]):
            return SourceType.SOCIAL_MEDIA
        return SourceType.WEB

    def _estimate_credibility(self, url: str) -> float:
        domain = extract_domain(url).lower()
        if any(x in domain for x in ["gov", "edu", "arxiv", "nature"]):
            return 0.9
        if any(x in domain for x in ["bbc", "reuters", "nytimes"]):
            return 0.8
        if "wikipedia" in domain:
            return 0.7
        return 0.5