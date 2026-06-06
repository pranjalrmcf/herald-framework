"""
Unit tests for research_analyst.evaluation.calibration

Run with: pytest tests/test_calibration.py -v
"""

import pytest

from research_analyst.core.models import Claim, UncertaintyBand
from research_analyst.evaluation.calibration import CalibrationEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim(confidence: float, level: str, lower: float = None, upper: float = None) -> Claim:
    band = UncertaintyBand(
        lower            = lower if lower is not None else max(0.0, confidence - 0.1),
        upper            = upper if upper is not None else min(1.0, confidence + 0.1),
        level            = level,
        num_sources      = 1,
        source_diversity = 0.5,
        is_controversial = False,
    )
    return Claim(
        claim_id          = f"c_{confidence}",
        text              = "test",
        confidence        = confidence,
        uncertainty_band  = band,
        uncertainty_level = level,
    )


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

class TestECE:
    def setup_method(self):
        self.ev = CalibrationEvaluator(n_bins=5)

    def test_perfect_calibration_gives_zero_ece(self):
        # Confidence == correctness → ECE should be ~0
        claims      = [_claim(0.9, "low")] * 5 + [_claim(0.5, "medium")] * 5
        correctness = [0.9] * 5 + [0.5] * 5
        result = self.ev.evaluate(claims, correctness)
        assert result.ece < 0.05

    def test_overconfident_gives_positive_ece(self):
        # Model thinks it is right (conf=0.9) but is wrong (acc=0.0)
        claims      = [_claim(0.9, "low")]  * 10
        correctness = [0.0] * 10
        result = self.ev.evaluate(claims, correctness)
        assert result.ece > 0.5

    def test_empty_claims(self):
        result = self.ev.evaluate([], [])
        assert result.ece == 0.0
        assert result.n_claims == 0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            self.ev.evaluate([_claim(0.5, "medium")], [1.0, 0.0])

    def test_well_calibrated_threshold(self):
        claims      = [_claim(0.8, "low")] * 10
        correctness = [0.8] * 10
        result = self.ev.evaluate(claims, correctness)
        assert result.is_well_calibrated(ece_threshold=0.10)


# ---------------------------------------------------------------------------
# Per-level stats
# ---------------------------------------------------------------------------

class TestPerLevelStats:
    def test_high_uncertainty_has_lower_accuracy(self):
        ev = CalibrationEvaluator()
        # High-uncertainty claims should be wrong more often
        high_claims = [_claim(0.3, "high")] * 5
        low_claims  = [_claim(0.9, "low")]  * 5
        all_claims  = high_claims + low_claims
        correctness = [0.0] * 5 + [1.0] * 5  # high=wrong, low=right
        result = ev.evaluate(all_claims, correctness)
        assert result.level_stats["high"]["mean_acc"]  < 0.1
        assert result.level_stats["low"]["mean_acc"]   > 0.9


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

class TestReliabilityDiagramData:
    def test_returns_lists(self):
        ev      = CalibrationEvaluator(n_bins=5)
        claims  = [_claim(0.8, "low")]  * 5
        flags   = [1.0] * 5
        data    = ev.reliability_diagram_data(claims, flags)
        assert "bin_confidences" in data
        assert "bin_accuracies"  in data
        assert "bin_counts"      in data
        assert isinstance(data["bin_confidences"], list)

    def test_non_empty_bins(self):
        ev      = CalibrationEvaluator(n_bins=5)
        claims  = [_claim(0.9, "low")] * 10
        flags   = [1.0] * 10
        data    = ev.reliability_diagram_data(claims, flags)
        assert len(data["bin_confidences"]) > 0


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_brier(self):
        ev          = CalibrationEvaluator()
        claims      = [_claim(1.0, "low")] * 5
        correctness = [1.0] * 5
        result      = ev.evaluate(claims, correctness)
        assert result.brier_score < 0.01

    def test_worst_brier(self):
        ev          = CalibrationEvaluator()
        claims      = [_claim(1.0, "low")] * 5
        correctness = [0.0] * 5   # perfectly wrong
        result      = ev.evaluate(claims, correctness)
        assert result.brier_score > 0.9