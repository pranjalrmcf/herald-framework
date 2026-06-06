"""
Unit tests for research_analyst.core.models

Run with: pytest tests/test_models.py -v
"""

import pytest
from datetime import datetime

from research_analyst.core.models import (
    Answer,
    Claim,
    Evidence,
    ExecutionPath,
    GEvalScores,
    LLMJudgeScore,
    QualityMetrics,
    UncertaintyBand,
)


# ---------------------------------------------------------------------------
# UncertaintyBand
# ---------------------------------------------------------------------------

class TestUncertaintyBand:
    def test_to_dict_rounds_values(self):
        band = UncertaintyBand(
            lower=0.51234, upper=0.88765,
            level="medium",
            num_sources=2,
            source_diversity=0.6667,
            is_controversial=False,
        )
        d = band.to_dict()
        assert d["lower"] == 0.512
        assert d["upper"] == 0.888
        assert d["level"] == "medium"

    def test_levels(self):
        for level in ("high", "medium", "low"):
            b = UncertaintyBand(
                lower=0.0, upper=1.0, level=level,
                num_sources=1, source_diversity=0.5, is_controversial=False,
            )
            assert b.level == level


# ---------------------------------------------------------------------------
# GEvalScores
# ---------------------------------------------------------------------------

class TestGEvalScores:
    def test_from_dict_clamps(self):
        scores = GEvalScores.from_dict({
            "coherence":   10,   # should clamp to 5
            "consistency": 0,    # should clamp to 1
            "fluency":     3,
            "relevance":   4,
        })
        assert scores.coherence   == 5
        assert scores.consistency == 1
        assert scores.fluency     == 3

    def test_composite_calculation(self):
        scores = GEvalScores.from_dict({
            "coherence": 4, "consistency": 4, "fluency": 4, "relevance": 4
        })
        # (4+4+4+4) / 20 = 0.8
        assert abs(scores.composite - 0.8) < 0.001

    def test_from_dict_all_threes(self):
        scores = GEvalScores.from_dict({})
        assert scores.coherence   == 3
        assert scores.consistency == 3
        # (3*4)/20 = 0.6
        assert abs(scores.composite - 0.6) < 0.001


# ---------------------------------------------------------------------------
# Claim (uncertainty fields)
# ---------------------------------------------------------------------------

class TestClaim:
    def test_claim_without_uncertainty(self):
        c = Claim(claim_id="c1", text="test", confidence=0.8)
        assert c.uncertainty_band  is None
        assert c.uncertainty_level is None

    def test_claim_with_uncertainty_band(self):
        band = UncertaintyBand(
            lower=0.5, upper=0.9, level="low",
            num_sources=3, source_diversity=0.7, is_controversial=False,
        )
        c = Claim(
            claim_id="c1", text="test", confidence=0.8,
            uncertainty_band=band, uncertainty_level="low",
        )
        assert c.uncertainty_band.level == "low"
        assert c.uncertainty_level == "low"


# ---------------------------------------------------------------------------
# QualityMetrics.calculate_composite_score
# ---------------------------------------------------------------------------

class TestQualityMetrics:
    def _base_metrics(self, **kwargs):
        defaults = dict(
            citation_coverage  = 0.8,
            grounding_score    = 0.8,
            coherence_score    = 0.8,
            answer_completeness= 0.8,
            source_diversity   = 0.8,
            passes_threshold   = True,
        )
        defaults.update(kwargs)
        return QualityMetrics(**defaults)

    def test_heuristic_only(self):
        m = self._base_metrics()
        score = m.calculate_composite_score()
        # 0.8 * (0.15+0.25+0.15+0.20+0.10) = 0.8 * 0.85 = 0.68
        assert abs(score - 0.68) < 0.001

    def test_with_llm_judge(self):
        judge = LLMJudgeScore(
            grounding_score=1.0, factuality_score=1.0,
            relevance_score=1.0, completeness_score=1.0,
            reasoning="", issues_found=[],
        )
        m = self._base_metrics(llm_judge_scores=judge)
        score = m.calculate_composite_score()
        # heuristic=0.68, llm=1.0 → 0.6*0.68 + 0.4*1.0 = 0.808
        assert abs(score - 0.808) < 0.001

    def test_with_geval(self):
        geval = GEvalScores(
            coherence=5, consistency=5, fluency=5, relevance=5, composite=1.0
        )
        m = self._base_metrics(geval_scores=geval)
        score = m.calculate_composite_score()
        # base=0.68, geval=1.0 → 0.8*0.68 + 0.2*1.0 = 0.744
        assert abs(score - 0.744) < 0.001

    def test_geval_composite_alias(self):
        """geval_composite field takes precedence when set directly."""
        m = self._base_metrics(geval_composite=0.9)
        score = m.calculate_composite_score()
        # base=0.68, geval=0.9 → 0.8*0.68 + 0.2*0.9 = 0.724
        assert abs(score - 0.724) < 0.001

    def test_composite_clamped_to_one(self):
        judge = LLMJudgeScore(
            grounding_score=1.0, factuality_score=1.0,
            relevance_score=1.0, completeness_score=1.0,
            reasoning="", issues_found=[],
        )
        m = self._base_metrics(
            citation_coverage=1.0, grounding_score=1.0, coherence_score=1.0,
            answer_completeness=1.0, source_diversity=1.0,
            llm_judge_scores=judge, geval_composite=1.0,
        )
        score = m.calculate_composite_score()
        assert score <= 1.0