"""
Uncertainty Calibration Evaluator for HERALD.

Measures how well the Aleatoric-Epistemic Credibility Banding module
is calibrated — i.e. whether claims labelled "medium uncertainty"
are actually wrong more often than claims labelled "low uncertainty".

Key metric: Expected Calibration Error (ECE)
    ECE = Σ_b (|b| / N) * |avg_confidence(b) - avg_accuracy(b)|

Secondary metrics:
    MCE   — Maximum Calibration Error across bins
    ACE   — Average Calibration Error (equal-weight bins, no frequency weighting)
    Brier — Brier Score (MSE of confidence vs correctness)

Usage:
    evaluator = CalibrationEvaluator()
    result    = evaluator.evaluate(claims, ground_truth_flags)
    print(evaluator.format_report(result))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research_analyst.core.models import Claim
from research_analyst.utils.logger import get_logger


logger = get_logger()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Full calibration evaluation result.

    Attributes:
        ece          Expected Calibration Error (lower is better).
        mce          Maximum Calibration Error.
        ace          Average Calibration Error (unweighted bins).
        brier_score  Mean squared error: confidence vs correctness.
        n_bins       Number of calibration bins used.
        n_claims     Total number of claims evaluated.
        bin_stats    Per-bin breakdown.
        level_stats  Per uncertainty-level breakdown.
    """
    ece:         float
    mce:         float
    ace:         float
    brier_score: float
    n_bins:      int
    n_claims:    int
    bin_stats:   List[Dict[str, Any]] = field(default_factory=list)
    level_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def is_well_calibrated(self, ece_threshold: float = 0.10) -> bool:
        """Returns True if ECE is below threshold."""
        return self.ece < ece_threshold


# ---------------------------------------------------------------------------
# CalibrationEvaluator
# ---------------------------------------------------------------------------

class CalibrationEvaluator:
    """
    Evaluates calibration of HERALD's uncertainty bands.

    Expected input:
        claims             — List[Claim] with uncertainty_band populated.
        ground_truth_flags — List[float] where 1.0 = claim is factually
                             correct and 0.0 = incorrect. Float values
                             (e.g. from a human judge on a 0-1 scale) are
                             also accepted.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins  = n_bins
        self.logger  = get_logger()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        claims:              List[Claim],
        ground_truth_flags:  List[float],
    ) -> CalibrationResult:
        """
        Compute all calibration metrics.

        Args:
            claims:             Claims with uncertainty_band set.
            ground_truth_flags: 0.0 (wrong) / 1.0 (correct) per claim.

        Returns:
            CalibrationResult
        """
        if len(claims) != len(ground_truth_flags):
            raise ValueError(
                f"claims ({len(claims)}) and ground_truth_flags "
                f"({len(ground_truth_flags)}) must have the same length"
            )
        if not claims:
            return CalibrationResult(
                ece=0.0, mce=0.0, ace=0.0, brier_score=0.0,
                n_bins=self.n_bins, n_claims=0,
            )

        confidences = self._extract_confidences(claims)
        correctness = list(ground_truth_flags)

        ece, mce, ace, bin_stats = self._compute_calibration_bins(
            confidences, correctness
        )
        brier = self._brier_score(confidences, correctness)
        level_stats = self._per_level_stats(claims, correctness)

        result = CalibrationResult(
            ece         = round(ece,   4),
            mce         = round(mce,   4),
            ace         = round(ace,   4),
            brier_score = round(brier, 4),
            n_bins      = self.n_bins,
            n_claims    = len(claims),
            bin_stats   = bin_stats,
            level_stats = level_stats,
        )

        self.logger.info(
            "Calibration evaluation complete",
            ece    = result.ece,
            mce    = result.mce,
            brier  = result.brier_score,
            claims = result.n_claims,
        )
        return result

    def reliability_diagram_data(
        self,
        claims:             List[Claim],
        ground_truth_flags: List[float],
    ) -> Dict[str, List[float]]:
        """
        Return data needed to plot a reliability diagram.

        Returns:
            dict with keys:
                bin_confidences  — x-axis: mean confidence per bin
                bin_accuracies   — y-axis: mean accuracy per bin
                bin_counts       — bar heights (counts)
        """
        confidences = self._extract_confidences(claims)
        _, _, _, bin_stats = self._compute_calibration_bins(
            confidences, list(ground_truth_flags)
        )
        non_empty = [b for b in bin_stats if b["count"] > 0]
        return {
            "bin_confidences": [b["avg_conf"] for b in non_empty],
            "bin_accuracies":  [b["avg_acc"]  for b in non_empty],
            "bin_counts":      [b["count"]    for b in non_empty],
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_confidences(claims: List[Claim]) -> List[float]:
        """Use mid-point of uncertainty_band if available, else claim.confidence."""
        result = []
        for c in claims:
            if c.uncertainty_band is not None:
                mid = (c.uncertainty_band.lower + c.uncertainty_band.upper) / 2.0
            else:
                mid = c.confidence
            result.append(float(mid))
        return result

    def _compute_calibration_bins(
        self,
        confidences: List[float],
        correctness: List[float],
    ) -> Tuple[float, float, float, List[Dict[str, Any]]]:
        """
        Partition predictions into n_bins equal-width bins and compute
        ECE, MCE, ACE.
        """
        edges     = np.linspace(0.0, 1.0, self.n_bins + 1)
        n         = len(confidences)
        ece       = 0.0
        mce       = 0.0
        ace_sum   = 0.0
        ace_bins  = 0
        bin_stats = []

        for i in range(self.n_bins):
            lo, hi = edges[i], edges[i + 1]
            idx = [
                j for j, c in enumerate(confidences)
                if lo <= c < hi or (hi == 1.0 and c == 1.0)
            ]
            if not idx:
                bin_stats.append({
                    "bin":      f"[{lo:.2f},{hi:.2f})",
                    "avg_conf": 0.0,
                    "avg_acc":  0.0,
                    "count":    0,
                    "gap":      0.0,
                })
                continue

            avg_conf = float(np.mean([confidences[j] for j in idx]))
            avg_acc  = float(np.mean([correctness[j]  for j in idx]))
            gap      = abs(avg_conf - avg_acc)
            weight   = len(idx) / n

            ece     += weight * gap
            mce      = max(mce, gap)
            ace_sum += gap
            ace_bins += 1

            bin_stats.append({
                "bin":      f"[{lo:.2f},{hi:.2f})",
                "avg_conf": round(avg_conf, 4),
                "avg_acc":  round(avg_acc,  4),
                "count":    len(idx),
                "gap":      round(gap, 4),
            })

        ace = ace_sum / ace_bins if ace_bins > 0 else 0.0
        return ece, mce, ace, bin_stats

    @staticmethod
    def _brier_score(
        confidences: List[float],
        correctness: List[float],
    ) -> float:
        """Mean squared error between confidence and correctness."""
        if not confidences:
            return 0.0
        mse = float(np.mean(
            [(c - a) ** 2 for c, a in zip(confidences, correctness)]
        ))
        return mse

    @staticmethod
    def _per_level_stats(
        claims:      List[Claim],
        correctness: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute accuracy per uncertainty level (high / medium / low).
        Useful to verify that high-uncertainty claims are actually wrong more often.
        """
        level_data: Dict[str, List[float]] = {"high": [], "medium": [], "low": [], "unscored": []}

        for claim, acc in zip(claims, correctness):
            level = claim.uncertainty_level or "unscored"
            level_data.setdefault(level, []).append(acc)

        stats: Dict[str, Dict[str, float]] = {}
        for level, accs in level_data.items():
            if not accs:
                continue
            stats[level] = {
                "count":    len(accs),
                "mean_acc": round(float(np.mean(accs)), 4),
                "std_acc":  round(float(np.std(accs)),  4),
            }
        return stats

    # ------------------------------------------------------------------ #
    #  Report                                                             #
    # ------------------------------------------------------------------ #

    def format_report(self, result: CalibrationResult) -> str:
        """Generate a human-readable calibration report."""
        calibrated = "✓ WELL-CALIBRATED" if result.is_well_calibrated() else "✗ NEEDS CALIBRATION"
        lines = [
            "=" * 60,
            f"UNCERTAINTY CALIBRATION REPORT  {calibrated}",
            f"Claims evaluated: {result.n_claims} | Bins: {result.n_bins}",
            "=" * 60,
            f"  ECE   (Expected Calibration Error): {result.ece:.4f}",
            f"  MCE   (Maximum Calibration Error):  {result.mce:.4f}",
            f"  ACE   (Average Calibration Error):  {result.ace:.4f}",
            f"  Brier Score (MSE):                  {result.brier_score:.4f}",
            "",
            "Per-level accuracy (high uncertainty should be least accurate):",
        ]
        for level in ["high", "medium", "low"]:
            stats = result.level_stats.get(level, {})
            if stats:
                lines.append(
                    f"  {level:8s}  n={stats['count']:4d}  "
                    f"acc={stats['mean_acc']:.3f} ± {stats['std_acc']:.3f}"
                )
        lines += [
            "",
            "Calibration bins (conf → accuracy):",
        ]
        for b in result.bin_stats:
            if b["count"] == 0:
                continue
            bar = "█" * min(20, int(b["avg_acc"] * 20))
            lines.append(
                f"  {b['bin']:14s}  n={b['count']:4d}  "
                f"conf={b['avg_conf']:.3f}  acc={b['avg_acc']:.3f}  gap={b['gap']:.3f}  {bar}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)