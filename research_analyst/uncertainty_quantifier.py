"""
Aleatoric-Epistemic Credibility Banding for the HERALD research analyst.

Terminology alignment (paper → code):
    "Uncertainty Quantifier"          → UncertaintyQuantifier
    "Aleatoric-Epistemic Banding"     → the five-signal credible interval
    "Iterative Posterior Refinement"  → two-pass quantify() calls in orchestrator

Before this module:
    Claim.confidence = 0.73   (single scalar, no context)

After first pass (pre-LLM-judge):
    Claim.uncertainty_band  = UncertaintyBand(lower=0.58, upper=0.88, level="low", ...)
    Claim.uncertainty_level = "low"

After second pass (post-LLM-judge, factuality blended in):
    Claim.uncertainty_band  = UncertaintyBand(lower=0.51, upper=0.84, level="medium", ...)
    Claim.uncertainty_level = "medium"
    Answer text: "OpenAI raised $13 billion [uncertain — limited sources] [1]"

Five uncertainty signals:
    1. Supporting source count      — more sources → narrower band
    2. Source domain diversity      — diverse domains → narrower band
    3. Claim controversy flag       — controversial → wider band
    4. LLM judge factuality score   — low factuality → lower blended confidence
    5. Source credibility scores    — low credibility docs → wider band

The quantifier rewrites flagged citation markers inline.
Markers are inserted before [N] refs; sentences are never rewritten
to avoid introducing factual errors.
"""

import re
from typing import Dict, List, Optional, Tuple

from research_analyst.core.models import (
    Answer,
    Claim,
    Evidence,
    LLMJudgeScore,
    UncertaintyBand,
)
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


logger = get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence thresholds that define uncertainty level
_LEVEL_THRESHOLDS = {
    "high":   (0.00, 0.45),
    "medium": (0.45, 0.70),
    "low":    (0.70, 1.01),
}

# Inline markers injected into answer text when level is high or medium
_INLINE_MARKERS = {
    "high":   "[uncertain — very limited sources]",
    "medium": "[uncertain — limited sources]",
}
# "low" uncertainty → no inline marker (clean output)


# ---------------------------------------------------------------------------
# UncertaintyQuantifier
# ---------------------------------------------------------------------------

class UncertaintyQuantifier:
    """
    Computes aleatoric-epistemic credibility bands for claims
    and flags uncertain assertions in the answer text.

    Designed for two-pass integration (Iterative Posterior Refinement):
        Pass 1  — called after answer generation, before LLM judge
                  (llm_judge_scores=None)
        Pass 2  — called after LLM judge evaluation
                  (llm_judge_scores=<scores>)
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.enabled: bool = getattr(self.settings, "uncertainty_enabled", True)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def quantify(
        self,
        answer: Answer,
        evidence: Evidence,
        llm_judge_scores: Optional[LLMJudgeScore] = None,
    ) -> Tuple[Answer, Evidence]:
        """
        Main entry point. Computes uncertainty bands for all claims,
        then injects inline markers into the answer text.

        Args:
            answer:           Generated Answer object.
            evidence:         Evidence containing claims to score.
            llm_judge_scores: Optional. If provided, factuality score is
                              blended into claim confidence (second pass).

        Returns:
            (updated Answer, updated Evidence)
        """
        if not self.enabled or not evidence.claims:
            return answer, evidence

        source_map = {
            doc.doc_id: doc for doc in evidence.supporting_documents
        }
        flagged_claims: List[Claim] = []

        for claim in evidence.claims:
            band = self._compute_band(claim, evidence, source_map, llm_judge_scores)
            # Write proper fields — no monkeypatching
            claim.uncertainty_band  = band
            claim.uncertainty_level = band.level

            if band.level in _INLINE_MARKERS:
                flagged_claims.append(claim)

        if flagged_claims:
            answer = self._inject_markers(answer, flagged_claims, evidence)

        high_count   = sum(1 for c in evidence.claims if c.uncertainty_level == "high")
        medium_count = sum(1 for c in evidence.claims if c.uncertainty_level == "medium")

        self.logger.info(
            "Credibility banding complete",
            total_claims  = len(evidence.claims),
            high_uncertainty   = high_count,
            medium_uncertainty = medium_count,
            pass_type = "second" if llm_judge_scores else "first",
        )

        return answer, evidence

    # ------------------------------------------------------------------ #
    #  Band computation                                                   #
    # ------------------------------------------------------------------ #

    def _compute_band(
        self,
        claim: Claim,
        evidence: Evidence,
        source_map: Dict,
        llm_judge_scores: Optional[LLMJudgeScore],
    ) -> UncertaintyBand:
        """
        Compute a credible interval for a single claim.

        Band half-width = source_penalty + diversity_penalty + controversy_penalty
        Lower bound     = blended_confidence − half_width
        Upper bound     = blended_confidence + half_width * 0.5
        """
        # --- Source count penalty ---
        n = len(claim.supporting_sources)
        if n == 0:
            source_penalty = 0.20
        elif n == 1:
            source_penalty = 0.12
        elif n == 2:
            source_penalty = 0.08
        else:
            source_penalty = max(0.04, 0.12 / n)

        # --- Source diversity signal ---
        diversity = self._source_diversity(claim.supporting_sources, source_map)
        diversity_penalty = 0.10 * (1.0 - diversity)

        # --- Controversy penalty ---
        controversy_penalty = 0.10 if claim.is_controversial else 0.0

        # --- Source credibility signal ---
        credibility = self._mean_credibility(claim.supporting_sources, source_map)
        credibility_adjustment = (credibility - 0.5) * 0.05  # ±0.05 at extremes

        # --- LLM factuality blend (second pass) ---
        if llm_judge_scores is not None:
            blended = claim.confidence * 0.80 + llm_judge_scores.factuality_score * 0.20
        else:
            blended = claim.confidence

        # Apply credibility adjustment
        blended = max(0.0, min(1.0, blended + credibility_adjustment))

        # --- Compute interval ---
        half_width = source_penalty + diversity_penalty + controversy_penalty
        lower = max(0.0, round(blended - half_width,       3))
        upper = min(1.0, round(blended + half_width * 0.5, 3))
        level = self._confidence_to_level(blended)

        return UncertaintyBand(
            lower            = lower,
            upper            = upper,
            level            = level,
            num_sources      = n,
            source_diversity = round(diversity, 3),
            is_controversial = claim.is_controversial,
        )

    def _source_diversity(
        self,
        source_ids: List[str],
        source_map: Dict,
    ) -> float:
        """Fraction of unique domains among supporting documents."""
        if not source_ids:
            return 0.0
        domains = set()
        for sid in source_ids:
            doc = source_map.get(sid)
            if doc:
                domain = doc.metadata.get("domain", "")
                if domain:
                    domains.add(domain)
        if not domains:
            return 0.5  # unknown → neutral
        return min(1.0, len(domains) / len(source_ids))

    def _mean_credibility(
        self,
        source_ids: List[str],
        source_map: Dict,
    ) -> float:
        """Average credibility score across supporting documents."""
        if not source_ids:
            return 0.5
        scores = [
            source_map[sid].credibility_score
            for sid in source_ids
            if sid in source_map
        ]
        return sum(scores) / len(scores) if scores else 0.5

    @staticmethod
    def _confidence_to_level(confidence: float) -> str:
        for level, (lo, hi) in _LEVEL_THRESHOLDS.items():
            if lo <= confidence < hi:
                return level
        return "low"

    # ------------------------------------------------------------------ #
    #  Answer text rewriting                                              #
    # ------------------------------------------------------------------ #

    def _inject_markers(
        self,
        answer: Answer,
        flagged_claims: List[Claim],
        evidence: Evidence,
    ) -> Answer:
        """
        Insert inline uncertainty markers before citation references [N]
        in the answer text.

        Strategy: for each flagged claim, find its supporting document(s)
        in the evidence list, map to 1-based citation index, and prepend
        the appropriate marker before that [N] reference.

        Sentences are never rewritten — only citation markers are annotated.
        """
        uncertain_indices: Dict[int, str] = {}

        for claim in flagged_claims:
            level = claim.uncertainty_level or "low"
            if level not in _INLINE_MARKERS:
                continue
            for doc_id in claim.supporting_sources:
                for idx, doc in enumerate(evidence.supporting_documents, 1):
                    if doc.doc_id == doc_id:
                        existing = uncertain_indices.get(idx, "low")
                        uncertain_indices[idx] = (
                            "high" if "high" in (level, existing) else "medium"
                        )

        if not uncertain_indices:
            return answer

        def _replace(match: re.Match) -> str:
            idx   = int(match.group(1))
            level = uncertain_indices.get(idx)
            if level and level in _INLINE_MARKERS:
                return f"{_INLINE_MARKERS[level]} [{idx}]"
            return match.group(0)

        new_text = re.sub(r"\[(\d+)\]", _replace, answer.text)
        if new_text != answer.text:
            answer.text = new_text
            self.logger.debug(
                "Uncertainty markers injected",
                answer_id          = answer.answer_id,
                uncertain_indices  = uncertain_indices,
            )

        return answer

    # ------------------------------------------------------------------ #
    #  Reporting                                                          #
    # ------------------------------------------------------------------ #

    def generate_report(self, evidence: Evidence) -> str:
        """Human-readable credibility banding report."""
        if not evidence.claims:
            return "No claims to report."

        counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "unscored": 0}
        for c in evidence.claims:
            counts[c.uncertainty_level or "unscored"] += 1

        lines = [
            "=" * 60,
            "ALEATORIC-EPISTEMIC CREDIBILITY BANDING REPORT",
            f"Total claims: {len(evidence.claims)}",
            "=" * 60,
            f"  Low uncertainty:    {counts['low']}",
            f"  Medium uncertainty: {counts['medium']}",
            f"  High uncertainty:   {counts['high']}",
            f"  Unscored:           {counts['unscored']}",
            "",
            "High-uncertainty claims (up to 5):",
        ]
        for c in evidence.claims:
            if c.uncertainty_level == "high" and c.uncertainty_band:
                b = c.uncertainty_band
                lines.append(
                    f"  [{b.lower:.2f}, {b.upper:.2f}] "
                    f"n_src={b.num_sources}  "
                    f"diversity={b.source_diversity:.2f}  "
                    f"| {c.text[:80]}..."
                )
        lines.append("=" * 60)
        return "\n".join(lines)