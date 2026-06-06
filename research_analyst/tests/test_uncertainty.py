"""
Unit tests for research_analyst.uncertainty_quantifier

Run with: pytest tests/test_uncertainty.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from research_analyst.core.models import (
    Answer,
    Citation,
    Claim,
    Document,
    Evidence,
    ExecutionPath,
    LLMJudgeScore,
    SourceType,
    UncertaintyBand,
)
from research_analyst.uncertainty_quantifier import UncertaintyQuantifier
from research_analyst.utils.helpers import generate_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, domain: str = "news", credibility: float = 0.7) -> Document:
    return Document(
        doc_id=doc_id, url=f"https://example.com/{doc_id}",
        title=f"Document {doc_id}", content="Some content.",
        source_type=SourceType.NEWS,
        credibility_score=credibility,
        metadata={"domain": domain},
    )


def _make_claim(
    claim_id: str,
    confidence: float,
    sources: list,
    controversial: bool = False,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=f"Test claim {claim_id}",
        supporting_sources=sources,
        confidence=confidence,
        is_controversial=controversial,
    )


def _make_evidence(claims: list, docs: list) -> Evidence:
    return Evidence(
        evidence_id=generate_id("ev"),
        claims=claims,
        supporting_documents=docs,
        summary="Test summary",
    )


def _make_answer(text: str = "Answer text with [1] and [2].") -> Answer:
    return Answer(
        answer_id=generate_id("ans"),
        query="test query",
        text=text,
        citations=[
            Citation(source_url="https://a.com", source_title="Source A", relevance=0.9),
            Citation(source_url="https://b.com", source_title="Source B", relevance=0.8),
        ],
        confidence=0.75,
        execution_path=ExecutionPath.RESEARCH,
    )


# ---------------------------------------------------------------------------
# Band computation
# ---------------------------------------------------------------------------

class TestBandComputation:
    def setup_method(self):
        self.uq = UncertaintyQuantifier()

    def test_no_sources_gives_wide_band(self):
        doc  = _make_doc("d1")
        claim = _make_claim("c1", confidence=0.8, sources=[])
        ev   = _make_evidence([claim], [doc])
        ans  = _make_answer()
        _, updated_ev = self.uq.quantify(ans, ev)
        c = updated_ev.claims[0]
        assert c.uncertainty_band is not None
        width = c.uncertainty_band.upper - c.uncertainty_band.lower
        assert width > 0.15  # no-source band should be wide

    def test_many_sources_gives_narrow_band(self):
        docs   = [_make_doc(f"d{i}") for i in range(5)]
        claim  = _make_claim("c1", confidence=0.85, sources=[d.doc_id for d in docs])
        ev     = _make_evidence([claim], docs)
        ans    = _make_answer()
        _, updated_ev = self.uq.quantify(ans, ev)
        c = updated_ev.claims[0]
        width = c.uncertainty_band.upper - c.uncertainty_band.lower
        assert width < 0.15  # many sources → narrow

    def test_controversial_flag_widens_band(self):
        doc  = _make_doc("d1")
        c_normal = _make_claim("c1", confidence=0.7, sources=["d1"], controversial=False)
        c_contr  = _make_claim("c2", confidence=0.7, sources=["d1"], controversial=True)
        ev1 = _make_evidence([c_normal], [doc])
        ev2 = _make_evidence([c_contr],  [doc])
        ans = _make_answer()
        _, ev1u = self.uq.quantify(_make_answer(), ev1)
        _, ev2u = self.uq.quantify(_make_answer(), ev2)
        w1 = ev1u.claims[0].uncertainty_band.upper - ev1u.claims[0].uncertainty_band.lower
        w2 = ev2u.claims[0].uncertainty_band.upper - ev2u.claims[0].uncertainty_band.lower
        assert w2 > w1

    def test_high_confidence_gives_low_uncertainty_level(self):
        doc   = _make_doc("d1")
        claim = _make_claim("c1", confidence=0.95, sources=["d1"])
        ev    = _make_evidence([claim], [doc])
        ans   = _make_answer()
        _, updated_ev = self.uq.quantify(ans, ev)
        assert updated_ev.claims[0].uncertainty_level == "low"

    def test_low_confidence_gives_high_uncertainty_level(self):
        doc   = _make_doc("d1")
        claim = _make_claim("c1", confidence=0.30, sources=[])
        ev    = _make_evidence([claim], [doc])
        ans   = _make_answer()
        _, updated_ev = self.uq.quantify(ans, ev)
        assert updated_ev.claims[0].uncertainty_level == "high"


# ---------------------------------------------------------------------------
# LLM judge blending (second pass)
# ---------------------------------------------------------------------------

class TestSecondPassBlending:
    def setup_method(self):
        self.uq = UncertaintyQuantifier()

    def test_low_factuality_lowers_blended_confidence(self):
        doc   = _make_doc("d1")
        claim = _make_claim("c1", confidence=0.80, sources=["d1"])
        ev    = _make_evidence([claim], [doc])

        judge = LLMJudgeScore(
            grounding_score=0.5, factuality_score=0.10,  # very low factuality
            relevance_score=0.5, completeness_score=0.5,
            reasoning="", issues_found=[],
        )
        ans = _make_answer()
        _, updated_ev = self.uq.quantify(ans, ev, llm_judge_scores=judge)
        c = updated_ev.claims[0]
        # Blended = 0.80*0.8 + 0.10*0.2 = 0.66 → should not be "low"
        assert c.uncertainty_level in ("medium", "high")


# ---------------------------------------------------------------------------
# Marker injection
# ---------------------------------------------------------------------------

class TestMarkerInjection:
    def setup_method(self):
        self.uq = UncertaintyQuantifier()

    def test_high_uncertainty_marker_injected(self):
        doc   = _make_doc("d1")
        claim = _make_claim("c1", confidence=0.20, sources=["d1"])  # high uncertainty
        ev    = _make_evidence([claim], [doc])
        ans   = _make_answer(text="Some claim [1] and more text.")
        updated_ans, _ = self.uq.quantify(ans, ev)
        assert "[uncertain" in updated_ans.text

    def test_low_uncertainty_no_marker(self):
        docs  = [_make_doc(f"d{i}") for i in range(4)]
        claim = _make_claim("c1", confidence=0.95, sources=[d.doc_id for d in docs])
        ev    = _make_evidence([claim], docs)
        ans   = _make_answer(text="Some claim [1] and more text.")
        updated_ans, _ = self.uq.quantify(ans, ev)
        assert "[uncertain" not in updated_ans.text

    def test_disabled_returns_unchanged(self):
        self.uq.enabled = False
        doc   = _make_doc("d1")
        claim = _make_claim("c1", confidence=0.10, sources=[])
        ev    = _make_evidence([claim], [doc])
        ans   = _make_answer()
        original_text = ans.text
        updated_ans, _ = self.uq.quantify(ans, ev)
        assert updated_ans.text == original_text


# ---------------------------------------------------------------------------
# Proper field assignment (no monkeypatching)
# ---------------------------------------------------------------------------

class TestNoMonkeypatching:
    def test_uncertainty_band_is_pydantic_field(self):
        """Verify that uncertainty_band is a proper Pydantic field, not __dict__ mutation."""
        doc   = _make_doc("d1")
        claim = _make_claim("c1", confidence=0.4, sources=["d1"])
        ev    = _make_evidence([claim], [doc])
        ans   = _make_answer()
        uq    = UncertaintyQuantifier()
        _, updated_ev = uq.quantify(ans, ev)
        c = updated_ev.claims[0]
        # Should be a proper UncertaintyBand instance, not a raw dict
        assert isinstance(c.uncertainty_band, UncertaintyBand)
        # Should not be in __dict__ as a workaround (would mean it's unset as field)
        assert "uncertainty_band" not in c.__dict__ or isinstance(
            c.__dict__.get("uncertainty_band"), UncertaintyBand
        )