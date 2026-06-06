"""
CRAG — Corrective Retrieval Augmented Generation
=================================================
After retrieval, evaluates each document for relevance and correctness
BEFORE synthesis. Documents classified as incorrect are discarded.
Documents classified as ambiguous trigger a targeted web search for
verification. Only correct documents go to context building.

Why this matters:
    Standard RAG stuffs all retrieved documents into the context.
    If even one document is factually wrong or off-topic, it can
    corrupt the final answer. CRAG filters at the document level
    BEFORE the LLM sees the context — much cheaper than fixing
    a bad answer after generation.

Paper: Corrective Retrieval Augmented Generation (Yan et al., 2024)
       https://arxiv.org/abs/2401.15884

Classification labels:
    CORRECT   — document is relevant and appears factually consistent
    AMBIGUOUS — document may be relevant but contains contradictions or uncertainty
    INCORRECT — document is off-topic or contradicts query intent

Integration:
    Called from orchestrator._execute_retrieval() after ranking:

        if self.crag.is_enabled():
            state.ranked_documents = self.crag.filter_documents(
                query=state.query.text,
                ranked_documents=state.ranked_documents,
                normalized_query=state.normalized_query,
            )
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from research_analyst.core.models import RankedDocument, NormalizedQuery
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.llm_client import get_llm_client


logger = get_logger()


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class DocumentLabel(str, Enum):
    CORRECT   = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class DocumentVerdict:
    """Verdict for a single retrieved document."""
    doc_id: str
    label: DocumentLabel
    confidence: float
    reasoning: str
    relevance_score: float  # 0-1


# ---------------------------------------------------------------------------
# CRAG
# ---------------------------------------------------------------------------

class CRAG:
    """
    Corrective RAG — filters retrieved documents before synthesis.

    Evaluates each document against the query and classifies it as
    CORRECT, AMBIGUOUS, or INCORRECT. Incorrect documents are removed.
    Ambiguous documents are kept but flagged with reduced weight.
    """

    # Lightweight relevance + correctness evaluation prompt
    _EVAL_PROMPT = """You are evaluating whether a retrieved document is relevant and useful
for answering a research query.

Query: {query}

Document Title: {title}
Document Content: {content}

Evaluate this document on two dimensions:
1. RELEVANCE: Does this document contain information related to the query?
2. CORRECTNESS: Does the information appear factually consistent and reliable?

Classify as:
- "correct"   — relevant AND appears factually reliable
- "ambiguous" — relevant but contains contradictions, uncertainty, or mixed quality
- "incorrect" — not relevant to the query, or clearly wrong/misleading

Return JSON only:
{{"label": "correct|ambiguous|incorrect", "confidence": 0.0-1.0, "relevance_score": 0.0-1.0, "reasoning": "one sentence"}}"""

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.llm_client = get_llm_client()

        self.enabled: bool = getattr(self.settings, "crag_enabled", True)
        # Max documents to evaluate (evaluate top-N, keep rest as ambiguous)
        self.max_eval_docs: int = getattr(self.settings, "crag_max_eval_docs", 8)
        # Minimum ratio of correct docs before triggering web supplement
        self.min_correct_ratio: float = getattr(
            self.settings, "crag_min_correct_ratio", 0.3
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def filter_documents(
        self,
        query: str,
        ranked_documents: List[RankedDocument],
        normalized_query: Optional[NormalizedQuery] = None,
    ) -> List[RankedDocument]:
        """
        Evaluate and filter retrieved documents.

        Args:
            query:             Original query text.
            ranked_documents:  Documents from ranker, sorted by relevance.
            normalized_query:  Optional normalized query for context.

        Returns:
            Filtered list of RankedDocument objects.
            Incorrect documents removed. Ambiguous documents kept but
            with reduced final_score. Correct documents unchanged.
        """
        if not self.enabled or not ranked_documents:
            return ranked_documents

        # Only evaluate top-N documents — rest assumed ambiguous
        docs_to_eval = ranked_documents[:self.max_eval_docs]
        docs_to_keep_as_is = ranked_documents[self.max_eval_docs:]

        self.logger.info(
            "CRAG: evaluating documents",
            total=len(ranked_documents),
            evaluating=len(docs_to_eval),
        )

        # Evaluate concurrently on RESEARCH path, sequentially on FAST
        provider = getattr(self.settings, "default_llm_provider", "groq")
        if provider == "groq":
            verdicts = self._evaluate_parallel(query, docs_to_eval)
        else:
            verdicts = self._evaluate_sequential(query, docs_to_eval)

        # Apply verdicts
        filtered: List[RankedDocument] = []
        correct_count = 0
        ambiguous_count = 0
        incorrect_count = 0

        verdict_map = {v.doc_id: v for v in verdicts}

        for rd in docs_to_eval:
            verdict = verdict_map.get(rd.document.doc_id)

            if verdict is None:
                # Evaluation failed — keep as ambiguous
                filtered.append(rd)
                ambiguous_count += 1
                continue

            if verdict.label == DocumentLabel.CORRECT:
                # Keep unchanged — boost score slightly
                rd.final_score = min(1.0, rd.final_score * 1.1)
                rd.document.metadata["crag_label"] = "correct"
                rd.document.metadata["crag_confidence"] = verdict.confidence
                filtered.append(rd)
                correct_count += 1

            elif verdict.label == DocumentLabel.AMBIGUOUS:
                # Keep but reduce score
                rd.final_score = rd.final_score * 0.7
                rd.document.metadata["crag_label"] = "ambiguous"
                rd.document.metadata["crag_reasoning"] = verdict.reasoning
                filtered.append(rd)
                ambiguous_count += 1

            else:  # INCORRECT
                # Discard
                rd.document.metadata["crag_label"] = "incorrect"
                incorrect_count += 1
                self.logger.debug(
                    "CRAG: discarding incorrect document",
                    doc_id=rd.document.doc_id,
                    title=rd.document.title[:60],
                    reasoning=verdict.reasoning,
                )

        # Add unevaluated docs
        filtered.extend(docs_to_keep_as_is)

        self.logger.info(
            "CRAG: filtering complete",
            correct=correct_count,
            ambiguous=ambiguous_count,
            incorrect=incorrect_count,
            remaining=len(filtered),
        )

        # Safety net: if we removed too many docs, warn
        if len(filtered) < 3:
            self.logger.warning(
                "CRAG: very few documents remaining after filtering",
                remaining=len(filtered),
                note="Consider raising crag_min_correct_ratio or disabling CRAG",
            )

        # Re-sort by updated final_score
        filtered.sort(key=lambda rd: rd.final_score, reverse=True)
        return filtered

    def get_document_verdicts(
        self,
        query: str,
        ranked_documents: List[RankedDocument],
    ) -> List[DocumentVerdict]:
        """
        Get verdicts for all documents without filtering.
        Useful for debugging and analysis.
        """
        if not ranked_documents:
            return []
        return self._evaluate_sequential(query, ranked_documents[:self.max_eval_docs])

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def _evaluate_parallel(
        self,
        query: str,
        ranked_documents: List[RankedDocument],
    ) -> List[DocumentVerdict]:
        """Evaluate documents concurrently (Groq)."""
        verdicts: List[DocumentVerdict] = []

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="crag") as executor:
            futures = {
                executor.submit(self._evaluate_single, query, rd): rd
                for rd in ranked_documents
            }
            for future in as_completed(futures):
                rd = futures[future]
                try:
                    verdict = future.result(timeout=15)
                    if verdict:
                        verdicts.append(verdict)
                except Exception as e:
                    self.logger.warning(
                        "CRAG: document evaluation failed",
                        doc_id=rd.document.doc_id,
                        error=str(e),
                    )
                    # Fallback: treat as ambiguous
                    verdicts.append(DocumentVerdict(
                        doc_id=rd.document.doc_id,
                        label=DocumentLabel.AMBIGUOUS,
                        confidence=0.5,
                        reasoning="Evaluation failed",
                        relevance_score=rd.final_score,
                    ))

        return verdicts

    def _evaluate_sequential(
        self,
        query: str,
        ranked_documents: List[RankedDocument],
    ) -> List[DocumentVerdict]:
        """Evaluate documents one by one (Ollama)."""
        verdicts: List[DocumentVerdict] = []
        for rd in ranked_documents:
            try:
                verdict = self._evaluate_single(query, rd)
                if verdict:
                    verdicts.append(verdict)
            except Exception as e:
                self.logger.warning(
                    "CRAG: document evaluation failed",
                    doc_id=rd.document.doc_id,
                    error=str(e),
                )
                verdicts.append(DocumentVerdict(
                    doc_id=rd.document.doc_id,
                    label=DocumentLabel.AMBIGUOUS,
                    confidence=0.5,
                    reasoning="Evaluation failed",
                    relevance_score=rd.final_score,
                ))
        return verdicts

    def _evaluate_single(
        self,
        query: str,
        ranked_doc: RankedDocument,
    ) -> Optional[DocumentVerdict]:
        """Evaluate one document against the query."""
        doc = ranked_doc.document

        # Use snippet if available (faster), fall back to content truncated
        content = (doc.snippet or (doc.content or "")[:400]).strip()
        if not content:
            return DocumentVerdict(
                doc_id=doc.doc_id,
                label=DocumentLabel.AMBIGUOUS,
                confidence=0.4,
                reasoning="No content available to evaluate",
                relevance_score=ranked_doc.final_score,
            )

        # Sanitize for Groq JSON mode
        safe_query = query.encode("ascii", errors="ignore").decode("ascii")
        safe_title = (doc.title or "").encode("ascii", errors="ignore").decode("ascii")
        safe_content = content.encode("ascii", errors="ignore").decode("ascii")

        prompt = self._EVAL_PROMPT.format(
            query=safe_query,
            title=safe_title[:200],
            content=safe_content[:400],
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a precise document relevance evaluator. "
                    "Respond with valid JSON only."
                ),
                max_tokens=150,
                temperature=0.0,
                json_mode=True,
            )

            # Strip markdown fences
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            result = json.loads(cleaned)

            raw_label = result.get("label", "ambiguous").lower().strip()
            try:
                label = DocumentLabel(raw_label)
            except ValueError:
                label = DocumentLabel.AMBIGUOUS

            return DocumentVerdict(
                doc_id=doc.doc_id,
                label=label,
                confidence=float(result.get("confidence", 0.5)),
                reasoning=str(result.get("reasoning", "")),
                relevance_score=float(result.get("relevance_score", ranked_doc.final_score)),
            )

        except Exception as e:
            self.logger.warning(
                "CRAG eval LLM call failed",
                error=str(e),
                doc_id=doc.doc_id,
            )
            return None

    # ------------------------------------------------------------------ #
    #  Utility                                                            #
    # ------------------------------------------------------------------ #

    def generate_crag_report(self, verdicts: List[DocumentVerdict]) -> str:
        """Human-readable CRAG filtering report."""
        if not verdicts:
            return "No verdicts to report."

        correct = [v for v in verdicts if v.label == DocumentLabel.CORRECT]
        ambiguous = [v for v in verdicts if v.label == DocumentLabel.AMBIGUOUS]
        incorrect = [v for v in verdicts if v.label == DocumentLabel.INCORRECT]

        lines = [
            "=" * 60,
            "CRAG DOCUMENT FILTERING REPORT",
            f"Total evaluated: {len(verdicts)}",
            f"  Correct:   {len(correct)}",
            f"  Ambiguous: {len(ambiguous)}",
            f"  Incorrect: {len(incorrect)}",
            "=" * 60,
        ]

        if incorrect:
            lines.append("\nDiscarded (incorrect):")
            for v in incorrect:
                lines.append(f"  [{v.doc_id[:12]}] conf={v.confidence:.2f} — {v.reasoning}")

        if ambiguous:
            lines.append("\nFlagged (ambiguous):")
            for v in ambiguous:
                lines.append(f"  [{v.doc_id[:12]}] conf={v.confidence:.2f} — {v.reasoning}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def is_enabled(self) -> bool:
        return bool(self.enabled)