"""
Answer generator for the research analyst system.
Synthesizes final answers from structured evidence using LLM.

Updates vs original:
    - ASCII sanitization before every LLM call (fixes Groq JSON mode crashes on
      non-ASCII characters like Cyrillic glyphs that appear in retrieved content)
    - try/except with json_mode=False fallback on every generation call
    - G-Eval chain-of-thought scoring integrated into quality evaluation
    - Markdown fence stripping for models that ignore json_mode
"""

import json
import re
from typing import List, Tuple
from datetime import datetime

from research_analyst.core.models import (
    Query,
    NormalizedQuery,
    Evidence,
    Answer,
    Citation,
    ExecutionPath
)
from research_analyst.core.exceptions import AnswerGenerationError, LLMError
from research_analyst.config import get_settings, prompts
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id
from research_analyst.utils.llm_client import get_llm_client


logger = get_logger()


def _sanitize(text: str) -> str:
    """Remove non-ASCII characters that break Groq JSON mode."""
    return text.encode("ascii", errors="ignore").decode("ascii")


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        # Take the part after the first fence
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
    return cleaned


def _safe_json_generate(llm_client, prompt: str, system_prompt: str,
                         max_tokens: int, temperature: float,
                         logger_ref) -> dict:
    """
    Call LLM with json_mode=True, fall back to json_mode=False if Groq
    rejects due to special characters or formatting issues.
    Returns parsed dict or raises on total failure.
    """
    # Attempt 1 — strict JSON mode
    try:
        response = llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
        return json.loads(_strip_fences(response))
    except Exception as e1:
        logger_ref.warning(
            "JSON mode generation failed, retrying without json_mode",
            error=str(e1)[:200],
        )

    # Attempt 2 — plain text mode, parse manually
    try:
        response = llm_client.generate(
            prompt=prompt,
            system_prompt=(
                system_prompt
                + " Respond with a JSON object only. "
                  "No markdown, no preamble, no special characters outside ASCII."
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=False,
        )
        return json.loads(_strip_fences(response))
    except Exception as e2:
        logger_ref.warning(
            "Both JSON generation attempts failed",
            error=str(e2)[:200],
        )
        raise


class AnswerGenerator:
    """Generate answers from evidence using LLM."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.llm_client = get_llm_client()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def generate_answer(
        self,
        query: Query,
        normalized_query: NormalizedQuery,
        evidence: Evidence,
        execution_path: ExecutionPath
    ) -> Answer:
        """
        Generate answer from evidence.

        Args:
            query:            Original query
            normalized_query: Normalized query
            evidence:         Structured evidence
            execution_path:   Execution path taken

        Returns:
            Answer object
        """
        self.logger.info(
            "Generating answer",
            query=query.text[:100],
            num_claims=len(evidence.claims),
            num_documents=len(evidence.supporting_documents)
        )

        try:
            if evidence.relationship_chains:
                answer_text, confidence, sources_used = self._generate_with_graph(
                    query.text, evidence
                )
            else:
                answer_text, confidence, sources_used = self._generate_standard(
                    query.text, evidence
                )

            citations = self._extract_citations(answer_text, evidence)

            answer = Answer(
                answer_id=generate_id("answer"),
                query=query.text,
                text=answer_text,
                citations=citations,
                confidence=confidence,
                generated_at=datetime.utcnow(),
                execution_path=execution_path,
                metadata={
                    "num_claims": len(evidence.claims),
                    "num_sources": len(evidence.supporting_documents),
                    "sources_used": sources_used,
                    "has_graph_data": bool(evidence.relationship_chains),
                }
            )

            self.logger.log_answer_generation(
                query_id=query.metadata.get("query_id", "unknown"),
                answer_length=len(answer_text),
                num_citations=len(citations),
                confidence=confidence,
                generation_time_ms=0,
            )

            return answer

        except Exception as e:
            self.logger.error(
                "Answer generation failed",
                query=query.text[:50],
                error=str(e)
            )
            raise AnswerGenerationError(
                f"Failed to generate answer: {str(e)}",
                details={"query": query.text}
            )

    # ------------------------------------------------------------------ #
    #  Standard generation (no graph)                                    #
    # ------------------------------------------------------------------ #

    def _generate_standard(
        self,
        query: str,
        evidence: Evidence,
    ) -> Tuple[str, float, List[int]]:
        from research_analyst.synthesis import ContextBuilder
        context_builder = ContextBuilder()
        formatted_evidence = context_builder.format_for_synthesis(evidence)

        prompt = _sanitize(prompts.format_prompt(
            prompts.ANSWER_GENERATION,
            query=query,
            evidence=formatted_evidence,
        ))

        system_prompt = (
            "You must produce a long-form analytical report. "
            "Minimum length: 700-1000 words. "
            "Structure the answer with: Overview, Section per major company, "
            "Partnership analysis, Competition analysis, Dependency chains, "
            "Future implications. "
            "Each section must be multi-paragraph. "
            "Add inline citations [1], [2], etc. after EVERY factual claim."
        )

        try:
            result = _safe_json_generate(
                self.llm_client, prompt, system_prompt,
                max_tokens=10000,
                temperature=self.settings.temperature,
                logger_ref=self.logger,
            )
            answer_text = result.get("answer", "")
            confidence = float(result.get("confidence", 0.7))
            sources_used = result.get("sources_used", [])
            return answer_text, confidence, sources_used

        except Exception:
            # Last resort: generate without JSON
            self.logger.warning("All JSON attempts failed, generating plain text answer")
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a research analyst. Write a comprehensive, well-structured answer "
                    "using only the evidence provided. "
                    "Structure: Overview (1 paragraph) → Key findings per entity (2-3 paragraphs each) "
                    "→ Synthesis (1 paragraph). "
                    "Each section must add new information — never repeat a point from a previous section. "
                    "Add inline citations [1], [2], etc. after every factual claim. "
                    "Do not pad or repeat. Write only what the evidence supports."
                ),
                max_tokens=3000,
                temperature=self.settings.temperature,
                json_mode=False,
            )
            return response.strip(), 0.6, []

    # ------------------------------------------------------------------ #
    #  Graph-aware generation                                             #
    # ------------------------------------------------------------------ #

    def _generate_with_graph(
        self,
        query: str,
        evidence: Evidence,
    ) -> Tuple[str, float, List[int]]:
        from research_analyst.synthesis import ContextBuilder
        context_builder = ContextBuilder()
        formatted_evidence = context_builder.format_for_synthesis(evidence)

        graph_relationships = ""
        if evidence.relationship_chains:
            chains = []
            for chain in evidence.relationship_chains[:5]:
                chain_str = " → ".join([
                    f"{r.subject} --[{r.predicate}]--> {r.object}"
                    for r in chain
                ])
                chains.append(chain_str)
            graph_relationships = "\n".join(chains)

        prompt = _sanitize(prompts.format_prompt(
            prompts.ANSWER_GENERATION_WITH_GRAPH,
            query=query,
            textual_evidence=formatted_evidence,
            graph_relationships=graph_relationships,
        ))

        system_prompt = (
            "You are a research analyst providing comprehensive analysis. "
            "CRITICAL REQUIREMENTS: "
            "Minimum length: 800-1200 words. "
            "Structure with clear sections and subsections. "
            "Multiple detailed paragraphs per section. "
            "Explain relationships and dependencies thoroughly. "
            "Include specific examples from evidence. "
            "Add inline citations [1][2][3] for EVERY claim. "
            "Write a DETAILED, COMPREHENSIVE analysis - not a brief summary."
        )

        try:
            result = _safe_json_generate(
                self.llm_client, prompt, system_prompt,
                max_tokens=12000,
                temperature=self.settings.temperature,
                logger_ref=self.logger,
            )
            answer_text = result.get("answer", "")
            confidence = float(result.get("confidence", 0.75))
            sources_used = result.get("sources_used", [])
            return answer_text, confidence, sources_used

        except Exception:
            self.logger.warning("Graph generation JSON failed, using plain text")
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a research analyst. Write a comprehensive multi-hop analysis "
                    "using the evidence and graph relationships provided. "
                    "Structure: Executive Summary → Entity-by-entity analysis → "
                    "Relationship chains → Implications. "
                    "Each section must cover new ground — no repetition across sections. "
                    "Use graph relationships to explain multi-hop connections. "
                    "Add inline citations [1], [2], etc. after every factual claim."
                ),
                max_tokens=3000,
                temperature=self.settings.temperature,
                json_mode=False,
            )
            return response.strip(), 0.65, []

    # ------------------------------------------------------------------ #
    #  G-Eval chain-of-thought scoring                                   #
    # ------------------------------------------------------------------ #

    def geval_score(
        self,
        query: str,
        answer_text: str,
        evidence_summary: str,
    ) -> dict:
        """
        G-Eval: chain-of-thought based quality scoring.

        Unlike standard LLM Judge which gives scores directly,
        G-Eval first generates evaluation steps (chain-of-thought),
        then uses those steps to produce more calibrated scores.

        Returns dict with keys:
            coherence, consistency, fluency, relevance  (each 1-5)
            composite  (average, normalised 0-1)
            reasoning  (the CoT steps used)
        """
        # Step 1: Generate evaluation criteria steps
        cot_prompt = _sanitize(f"""You are evaluating the quality of a research answer.

Query: {query[:500]}

Answer: {answer_text[:1500]}

Evidence Summary: {evidence_summary[:800]}

First, write out the steps you will follow to evaluate this answer.
Think about: factual accuracy, completeness, citation quality, coherence.
Then provide scores.

Return JSON:
{{
    "evaluation_steps": ["step 1", "step 2", "step 3"],
    "coherence": 1-5,
    "consistency": 1-5,
    "fluency": 1-5,
    "relevance": 1-5,
    "reasoning": "one paragraph explaining your scores"
}}""")

        try:
            result = _safe_json_generate(
                self.llm_client,
                cot_prompt,
                "You are a precise answer quality evaluator. Use chain-of-thought reasoning before scoring.",
                max_tokens=600,
                temperature=0.0,
                logger_ref=self.logger,
            )

            scores = {
                "coherence":    min(5, max(1, int(result.get("coherence", 3)))),
                "consistency":  min(5, max(1, int(result.get("consistency", 3)))),
                "fluency":      min(5, max(1, int(result.get("fluency", 3)))),
                "relevance":    min(5, max(1, int(result.get("relevance", 3)))),
                "reasoning":    result.get("reasoning", ""),
                "steps":        result.get("evaluation_steps", []),
            }
            # Normalise composite to 0-1
            scores["composite"] = round(
                (scores["coherence"] + scores["consistency"]
                 + scores["fluency"] + scores["relevance"]) / 20.0,
                4,
            )

            self.logger.info(
                "G-Eval scoring complete",
                composite=scores["composite"],
                coherence=scores["coherence"],
                consistency=scores["consistency"],
            )
            return scores

        except Exception as e:
            self.logger.warning("G-Eval scoring failed", error=str(e))
            return {
                "coherence": 3, "consistency": 3,
                "fluency": 3, "relevance": 3,
                "composite": 0.6, "reasoning": "G-Eval failed",
                "steps": [],
            }

    # ------------------------------------------------------------------ #
    #  Citation extraction                                                #
    # ------------------------------------------------------------------ #

    def _extract_citations(
        self,
        answer_text: str,
        evidence: Evidence,
    ) -> List[Citation]:
        citation_markers = re.findall(r'\[(\d+)\]', answer_text)
        citation_indices = sorted(set(int(m) for m in citation_markers))

        citations = []
        for idx in citation_indices:
            doc_idx = idx - 1
            if 0 <= doc_idx < len(evidence.supporting_documents):
                doc = evidence.supporting_documents[doc_idx]
                excerpt = doc.snippet or (doc.content[:200] if doc.content else None)
                citations.append(Citation(
                    source_url=doc.url,
                    source_title=doc.title,
                    excerpt=excerpt,
                    relevance=0.8,
                ))
        return citations

    # ------------------------------------------------------------------ #
    #  Streaming / refine (unchanged)                                     #
    # ------------------------------------------------------------------ #

    def generate_streaming_answer(
        self,
        query: Query,
        normalized_query: NormalizedQuery,
        evidence: Evidence,
        execution_path: ExecutionPath,
    ):
        answer = self.generate_answer(query, normalized_query, evidence, execution_path)
        chunk_size = 50
        for i in range(0, len(answer.text), chunk_size):
            yield answer.text[i:i + chunk_size]

    def refine_answer(
        self,
        original_answer: Answer,
        additional_evidence: Evidence,
    ) -> Answer:
        self.logger.info("Refining answer", answer_id=original_answer.answer_id)

        combined_prompt = _sanitize(f"""
ORIGINAL ANSWER (preserve ALL sections, ALL content, ALL structure):
{original_answer.text}

TASK: Add inline citations [1], [2], [3], etc. after every factual claim that lacks one.
Do NOT remove, summarise, or shorten any section.
Do NOT change the section structure or headers.
Only insert [N] citation markers where they are missing.
Every factual sentence must end with at least one citation.

Available sources (use these index numbers):
{additional_evidence.summary[:800]}

Respond in JSON format: {{"answer": "<full answer with citations added>", "confidence": 0.0-1.0, "sources_used": [1,2,3]}}
""")

        try:
            result = _safe_json_generate(
                self.llm_client,
                combined_prompt,
                (
                    "You are a research analyst adding citations to an existing answer. "
                    "CRITICAL: Preserve the COMPLETE original answer text and all section structure. "
                    "Only add [N] citation markers — do not remove or shorten any content. "
                    "Always respond with valid JSON."
                ),
                max_tokens=4000,
                temperature=0.0,
                logger_ref=self.logger,
            )
            refined_text = result.get("answer", original_answer.text)
            # Safety check: if refined is much shorter than original, keep original
            if len(refined_text) < len(original_answer.text) * 0.6:
                self.logger.warning(
                    "Refined answer significantly shorter than original — keeping original",
                    original_len=len(original_answer.text),
                    refined_len=len(refined_text),
                )
                refined_text = original_answer.text
            return Answer(
                answer_id=generate_id("answer"),
                query=original_answer.query,
                text=refined_text,
                citations=original_answer.citations,
                confidence=float(result.get("confidence", original_answer.confidence)),
                generated_at=datetime.utcnow(),
                execution_path=original_answer.execution_path,
                metadata={
                    **original_answer.metadata,
                    "refined": True,
                    "original_answer_id": original_answer.answer_id,
                }
            )
        except Exception as e:
            self.logger.warning("Answer refinement failed, returning original", error=str(e))
            return original_answer

    def format_answer_with_sources(self, answer: Answer) -> str:
        lines = [answer.text]
        if answer.citations:
            lines.append("\n\n**Sources:**")
            for i, citation in enumerate(answer.citations, 1):
                lines.append(f"\n[{i}] {citation.source_title}")
                lines.append(f"    {citation.source_url}")
                if citation.excerpt:
                    lines.append(f"    \"{citation.excerpt[:100]}...\"")
        return "\n".join(lines)