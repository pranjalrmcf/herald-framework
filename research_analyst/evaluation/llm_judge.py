"""
LLM Judge v2 — parallelized evaluation calls for Groq, sequential for Ollama.

Groq: all 4 dimensions run in parallel via ThreadPoolExecutor (~70% faster)
Ollama: all 4 dimensions run sequentially (Ollama handles one request at a time)

Also adds:
  - Answer hash-based score caching (skip re-evaluation if same answer)
  - Structured error isolation (one dimension failing doesn't kill the rest)
  - Safe issues deduplication (handles both string and dict issues from Ollama)
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from typing import Dict, Any, Optional

from research_analyst.core.models import Answer, Evidence, LLMJudgeScore
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.llm_client import get_llm_client
from research_analyst.utils.helpers import generate_hash


logger = get_logger()

# Timeout per individual eval dimension (seconds)
_EVAL_DIMENSION_TIMEOUT = 60


class LLMJudge:
    """
    LLM-as-Judge evaluator.
    Uses parallel execution for Groq, sequential for Ollama.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.llm_client = get_llm_client()

        # Thread pool for parallel mode (Groq only)
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="llm_judge",
        )

        # Score cache: answer_hash -> LLMJudgeScore
        self._score_cache: Dict[str, LLMJudgeScore] = {}
        self._cache_enabled: bool = getattr(
            self.settings, "cache_llm_judge_scores", True
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def evaluate_answer(
        self,
        query: str,
        answer: Answer,
        evidence: Evidence,
    ) -> LLMJudgeScore:
        """
        Evaluate answer quality across grounding, factuality, relevance, completeness.
        Uses parallel execution for Groq, sequential for Ollama.
        """
        if not self.settings.enable_llm_judge:
            self.logger.info("LLM Judge disabled — returning default scores")
            return self._default_score()

        # Cache check
        answer_hash = generate_hash(answer.text)
        if self._cache_enabled and answer_hash in self._score_cache:
            self.logger.debug("LLM Judge: cache hit", answer_id=answer.answer_id)
            return self._score_cache[answer_hash]

        provider = getattr(self.settings, "default_llm_provider", "groq")
        mode = "sequential" if provider == "ollama" else "parallel"

        self.logger.info(
            "LLM Judge evaluation started",
            answer_id=answer.answer_id,
            mode=mode,
        )

        dimension_fns = {
            "grounding":    lambda: self._evaluate_grounding(query, answer, evidence),
            "factuality":   lambda: self._evaluate_factuality(query, answer, evidence),
            "relevance":    lambda: self._evaluate_relevance(query, answer),
            "completeness": lambda: self._evaluate_completeness(query, answer, evidence),
        }

        if provider == "ollama":
            results = self._run_sequential(dimension_fns)
        else:
            results = self._run_parallel(dimension_fns)

        # Aggregate results
        all_issues = []
        reasoning_parts = []
        for dim in ("grounding", "factuality", "relevance", "completeness"):
            r = results.get(dim, self._fallback_dimension())
            # Safe issue collection — Ollama sometimes returns dicts instead of strings
            for issue in r.get("issues", []):
                all_issues.append(str(issue) if isinstance(issue, dict) else issue)
            reasoning_parts.append(f"{dim.capitalize()}: {r.get('reasoning', '')}")

        judge_score = LLMJudgeScore(
            grounding_score=float(results.get("grounding", {}).get("score", 0.5)),
            factuality_score=float(results.get("factuality", {}).get("score", 0.5)),
            relevance_score=float(results.get("relevance", {}).get("score", 0.5)),
            completeness_score=float(results.get("completeness", {}).get("score", 0.5)),
            reasoning=" | ".join(reasoning_parts),
            issues_found=list({i for i in all_issues if i}),
        )

        self.logger.info(
            "LLM Judge evaluation complete",
            answer_id=answer.answer_id,
            grounding=judge_score.grounding_score,
            factuality=judge_score.factuality_score,
            relevance=judge_score.relevance_score,
            completeness=judge_score.completeness_score,
        )

        if self._cache_enabled:
            self._score_cache[answer_hash] = judge_score

        return judge_score

    # ------------------------------------------------------------------ #
    #  Execution modes                                                    #
    # ------------------------------------------------------------------ #

    def _run_sequential(
        self,
        dimension_fns: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Run all 4 dimensions one by one. Used for Ollama."""
        results = {}
        for dim, fn in dimension_fns.items():
            try:
                self.logger.debug(f"LLM Judge evaluating: {dim}")
                results[dim] = fn()
            except Exception as e:
                self.logger.warning(
                    f"LLM Judge {dim} failed",
                    error=str(e),
                )
                results[dim] = self._fallback_dimension()
        return results

    def _run_parallel(
        self,
        dimension_fns: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Run all 4 dimensions simultaneously. Used for Groq."""
        # Pre-fill with fallbacks so partial results still work
        results = {dim: self._fallback_dimension() for dim in dimension_fns}

        futures = {
            self._executor.submit(fn): dim
            for dim, fn in dimension_fns.items()
        }

        try:
            for future in as_completed(futures, timeout=_EVAL_DIMENSION_TIMEOUT + 5):
                dim = futures[future]
                try:
                    results[dim] = future.result(timeout=_EVAL_DIMENSION_TIMEOUT)
                except FutureTimeoutError:
                    self.logger.warning(
                        "LLM Judge dimension timed out",
                        dimension=dim,
                        timeout=_EVAL_DIMENSION_TIMEOUT,
                    )
                    results[dim] = self._fallback_dimension()
                except Exception as e:
                    self.logger.warning(
                        "LLM Judge dimension failed",
                        dimension=dim,
                        error=str(e),
                    )
                    results[dim] = self._fallback_dimension()

        except TimeoutError:
            # as_completed itself timed out — use whatever results we have
            self.logger.warning(
                "LLM Judge parallel timeout — using partial results with fallbacks",
                completed=sum(
                    1 for d in results.values()
                    if d.get("reasoning") != "Evaluation failed"
                ),
                total=len(futures),
            )
            for future in futures:
                future.cancel()

        return results

    # ------------------------------------------------------------------ #
    #  Dimension evaluations                                              #
    # ------------------------------------------------------------------ #

    def _evaluate_grounding(
        self,
        query: str,
        answer: Answer,
        evidence: Evidence,
    ) -> Dict[str, Any]:
        evidence_text = self._format_evidence(evidence)
        prompt = f"""You are an expert evaluator assessing answer quality.

QUERY: {query}

ANSWER: {answer.text}

EVIDENCE (from retrieved sources):
{evidence_text}

TASK: Evaluate how well the ANSWER is grounded in the EVIDENCE.

SCORING CRITERIA:
- 1.0: Every claim in the answer is directly supported by evidence
- 0.8: Most claims supported, minor unsupported details
- 0.6: Some claims supported, some speculation
- 0.4: Weak support, significant speculation
- 0.2: Minimal grounding, mostly speculation
- 0.0: No grounding in evidence

Return JSON only:
{{"score": 0.0, "reasoning": "brief explanation", "issues": ["issue1"]}}"""

        return self._call_judge(prompt)

    def _evaluate_factuality(
        self,
        query: str,
        answer: Answer,
        evidence: Evidence,
    ) -> Dict[str, Any]:
        evidence_text = self._format_evidence(evidence)
        prompt = f"""You are an expert fact-checker.

QUERY: {query}

ANSWER: {answer.text}

EVIDENCE:
{evidence_text}

TASK: Check the FACTUAL ACCURACY of claims in the answer.

SCORING CRITERIA:
- 1.0: All facts verifiable and accurate
- 0.8: Mostly accurate, minor errors
- 0.6: Some inaccuracies or unverifiable claims
- 0.4: Multiple factual errors
- 0.2: Significant misinformation
- 0.0: Mostly false information

Return JSON only:
{{"score": 0.0, "reasoning": "brief explanation", "issues": ["issue1"]}}"""

        return self._call_judge(prompt)

    def _evaluate_relevance(
        self,
        query: str,
        answer: Answer,
    ) -> Dict[str, Any]:
        prompt = f"""You are an expert evaluator.

QUERY: {query}

ANSWER: {answer.text}

TASK: Evaluate how RELEVANT the answer is to the query.

SCORING CRITERIA:
- 1.0: Directly addresses all aspects of the query
- 0.8: Addresses main query, minor aspects missed
- 0.6: Partially relevant, some drift
- 0.4: Somewhat off-topic
- 0.2: Mostly irrelevant
- 0.0: Completely off-topic

Return JSON only:
{{"score": 0.0, "reasoning": "brief explanation", "issues": ["issue1"]}}"""

        return self._call_judge(prompt)

    def _evaluate_completeness(
        self,
        query: str,
        answer: Answer,
        evidence: Evidence,
    ) -> Dict[str, Any]:
        prompt = f"""You are an expert evaluator.

QUERY: {query}

ANSWER: {answer.text}

AVAILABLE EVIDENCE: {len(evidence.supporting_documents)} sources

TASK: Evaluate how COMPLETE the answer is.

SCORING CRITERIA:
- 1.0: Comprehensive, covers all important aspects
- 0.8: Good coverage, minor gaps
- 0.6: Adequate, some aspects missing
- 0.4: Incomplete, significant gaps
- 0.2: Very incomplete
- 0.0: Barely addresses query

Return JSON only:
{{"score": 0.0, "reasoning": "brief explanation", "issues": ["issue1"]}}"""

        return self._call_judge(prompt)

    # ------------------------------------------------------------------ #
    #  Shared LLM call                                                    #
    # ------------------------------------------------------------------ #

    def _call_judge(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a precise evaluator. "
                    "Always respond with valid JSON only. "
                    "No markdown, no explanation outside JSON."
                ),
                max_tokens=400,
                temperature=0.0,
                json_mode=True,
            )

            # Strip markdown fences if Ollama ignored json_mode
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            result = json.loads(cleaned)
            return {
                "score": float(result.get("score", 0.5)),
                "reasoning": str(result.get("reasoning", "")),
                "issues": result.get("issues", []),
            }
        except json.JSONDecodeError as e:
            self.logger.warning(
                "Judge JSON parse failed",
                error=str(e),
                response_preview=response[:200] if 'response' in dir() else "no response",
            )
            return self._fallback_dimension()
        except Exception as e:
            self.logger.warning("Judge LLM call failed", error=str(e))
            return self._fallback_dimension()

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fallback_dimension() -> Dict[str, Any]:
        return {"score": 0.5, "reasoning": "Evaluation failed", "issues": []}

    def _format_evidence(self, evidence: Evidence) -> str:
        lines = ["CLAIMS:"]
        for i, claim in enumerate(evidence.claims[:10], 1):
            lines.append(f"{i}. {claim.text} (confidence: {claim.confidence:.2f})")
        lines.append("\nSOURCES:")
        for i, doc in enumerate(evidence.supporting_documents[:5], 1):
            snippet = (doc.snippet or doc.content or "")[:200]
            lines.append(f"[{i}] {doc.title}: {snippet}...")
        return "\n".join(lines)

    def _default_score(self) -> LLMJudgeScore:
        return LLMJudgeScore(
            grounding_score=0.5,
            factuality_score=0.5,
            relevance_score=0.5,
            completeness_score=0.5,
            reasoning="LLM Judge evaluation not performed",
            issues_found=[],
        )

    def __del__(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)