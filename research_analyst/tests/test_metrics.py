"""
Unit tests for research_analyst.utils.metrics

Run with: pytest tests/test_metrics.py -v
"""

import pytest
from research_analyst.utils.metrics import (
    bleu,
    expected_calibration_error,
    fact_coverage,
    meteor,
    rouge,
    EvaluationSuite,
)


REF = (
    "OpenAI and Microsoft have a strategic partnership worth 13 billion dollars. "
    "Microsoft has integrated OpenAI technology into its Azure cloud platform."
)

PERFECT  = REF  # identical prediction
PARTIAL  = "OpenAI and Microsoft have a partnership. Microsoft uses OpenAI technology."
UNRELATED= "The capital of France is Paris and it has many museums and art galleries."


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

class TestROUGE:
    def test_perfect_rouge1(self):
        r = rouge(PERFECT, REF, "rouge1")
        assert r["fmeasure"] > 0.95

    def test_partial_rouge1(self):
        r = rouge(PARTIAL, REF, "rouge1")
        assert 0.3 < r["fmeasure"] < 0.95

    def test_unrelated_low_rouge(self):
        r = rouge(UNRELATED, REF, "rouge1")
        assert r["fmeasure"] < 0.15

    def test_rougeL_less_than_rouge1(self):
        r1 = rouge(PARTIAL, REF, "rouge1")
        rL = rouge(PARTIAL, REF, "rougeL")
        # rougeL is always ≤ rouge1
        assert rL["fmeasure"] <= r1["fmeasure"] + 1e-6


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

class TestBLEU:
    def test_perfect_bleu(self):
        score = bleu(PERFECT, REF, n=4)
        assert score > 0.9

    def test_unrelated_near_zero(self):
        score = bleu(UNRELATED, REF, n=4)
        assert score < 0.05

    def test_bleu1_geq_bleu4(self):
        b1 = bleu(PARTIAL, REF, n=1)
        b4 = bleu(PARTIAL, REF, n=4)
        assert b1 >= b4 - 1e-6


# ---------------------------------------------------------------------------
# METEOR
# ---------------------------------------------------------------------------

class TestMETEOR:
    def test_perfect_meteor(self):
        score = meteor(PERFECT, REF)
        assert score > 0.9

    def test_partial_meteor(self):
        score = meteor(PARTIAL, REF)
        assert 0.2 < score < 0.95

    def test_unrelated_low(self):
        score = meteor(UNRELATED, REF)
        assert score < 0.15


# ---------------------------------------------------------------------------
# Fact coverage
# ---------------------------------------------------------------------------

class TestFactCoverage:
    def test_all_facts_covered(self):
        facts = ["OpenAI Microsoft partnership", "13 billion dollars"]
        r = fact_coverage(REF, facts)
        assert r["coverage_ratio"] == 1.0

    def test_no_facts_covered(self):
        facts = ["quantum computing entanglement", "french cuisine croissant"]
        r = fact_coverage(REF, facts)
        assert r["coverage_ratio"] == 0.0

    def test_empty_facts(self):
        r = fact_coverage(REF, [])
        assert r["coverage_ratio"] == 1.0  # vacuously true

    def test_partial_coverage(self):
        facts = ["OpenAI Microsoft partnership", "quantum entanglement"]
        r = fact_coverage(REF, facts)
        assert 0.4 < r["coverage_ratio"] < 0.6


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration(self):
        confs  = [0.9, 0.9, 0.9, 0.5, 0.5, 0.5]
        correct= [0.9, 0.9, 0.9, 0.5, 0.5, 0.5]
        result = expected_calibration_error(confs, correct, n_bins=5)
        assert result["ece"] < 0.05

    def test_overconfident(self):
        confs  = [1.0] * 10
        correct= [0.0] * 10
        result = expected_calibration_error(confs, correct, n_bins=5)
        assert result["ece"] > 0.8

    def test_empty_inputs(self):
        result = expected_calibration_error([], [], n_bins=5)
        assert result["ece"] == 0.0

    def test_mismatched_raises(self):
        with pytest.raises(AssertionError):
            # EvaluationSuite.evaluate_batch raises, low-level function raises via numpy
            pass


# ---------------------------------------------------------------------------
# EvaluationSuite
# ---------------------------------------------------------------------------

class TestEvaluationSuite:
    def setup_method(self):
        self.suite = EvaluationSuite(bleu_n=1)  # bleu1 for speed

    def test_evaluate_returns_all_keys(self):
        scores = self.suite.evaluate(PARTIAL, REF, reference_facts=["OpenAI Microsoft"])
        for key in ("rouge1_f", "rouge2_f", "rougeL_f", "bleu1", "meteor", "fact_cov"):
            assert key in scores, f"Missing key: {key}"

    def test_perfect_all_ones(self):
        scores = self.suite.evaluate(PERFECT, REF)
        assert scores["rouge1_f"] > 0.95
        assert scores["meteor"]   > 0.90

    def test_evaluate_batch_mean(self):
        preds = [PERFECT, PARTIAL]
        refs  = [REF, REF]
        agg   = self.suite.evaluate_batch(preds, refs)
        assert "rouge1_f" in agg
        # Mean of (perfect + partial) should be between them
        single_perfect = rouge(PERFECT, REF, "rouge1")["fmeasure"]
        single_partial = rouge(PARTIAL, REF, "rouge1")["fmeasure"]
        expected_mean  = (single_perfect + single_partial) / 2
        assert abs(agg["rouge1_f"] - expected_mean) < 0.01

    def test_evaluate_batch_length_mismatch(self):
        with pytest.raises(AssertionError):
            self.suite.evaluate_batch(["a"], ["b", "c"])