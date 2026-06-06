"""
Evaluation metrics for HERALD.

Metrics implemented:
    - ROUGE-1/2/L        (lexical overlap, recall-focused)
    - BLEU-1/4           (precision-focused n-gram overlap)
    - METEOR             (synonym-aware unigram alignment)
    - Semantic Similarity (sentence-transformers cosine)
    - Fact Coverage      (claim-level evidence coverage)
    - Expected Calibration Error (ECE)
                         (how well uncertainty_band.lower/upper correlates
                          with actual error rate)

All functions are pure; no external state. The EvaluationSuite class
bundles them for convenient batch evaluation.

Dependencies:
    rouge-score   pip install rouge-score
    nltk          pip install nltk
    sentence-transformers (already in requirements for retrieval)

Usage:
    from research_analyst.utils.metrics import EvaluationSuite
    suite = EvaluationSuite()
    scores = suite.evaluate(prediction="...", reference="...", evidence=evidence)
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False
    warnings.warn("rouge-score not installed; ROUGE metrics will return 0.0")

try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("punkt",   quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    from nltk.translate.bleu_score  import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    from nltk.tokenize import word_tokenize
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False
    warnings.warn("nltk not installed; BLEU and METEOR will return 0.0")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    warnings.warn("sentence-transformers not installed; semantic similarity will return 0.0")


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def rouge(
    prediction: str,
    reference:  str,
    variant:    str = "rougeL",
) -> Dict[str, float]:
    """
    Compute ROUGE scores for a single prediction/reference pair.

    Args:
        prediction: Generated answer text.
        reference:  Ground-truth answer text.
        variant:    One of "rouge1", "rouge2", "rougeL".
                    If "all", returns all three.

    Returns:
        dict with keys precision, recall, fmeasure.
    """
    if not _HAS_ROUGE:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    variants = ["rouge1", "rouge2", "rougeL"] if variant == "all" else [variant]
    scorer   = _rouge_scorer.RougeScorer(variants, use_stemmer=True)
    scores   = scorer.score(reference, prediction)
    if variant == "all":
        return {
            v: {
                "precision": scores[v].precision,
                "recall":    scores[v].recall,
                "fmeasure":  scores[v].fmeasure,
            }
            for v in variants
        }
    s = scores[variant]
    return {"precision": s.precision, "recall": s.recall, "fmeasure": s.fmeasure}


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def bleu(
    prediction: str,
    reference:  str,
    n:          int = 4,
) -> float:
    """
    Compute sentence-level BLEU-n score.

    Args:
        prediction: Generated answer text.
        reference:  Ground-truth answer text.
        n:          Maximum n-gram order (1 or 4 typical).

    Returns:
        BLEU score in [0, 1].
    """
    if not _HAS_NLTK:
        return 0.0
    weights = tuple(1.0 / n for _ in range(n))
    hyp = word_tokenize(prediction.lower())
    ref = word_tokenize(reference.lower())
    sf  = SmoothingFunction().method4
    try:
        return float(sentence_bleu([ref], hyp, weights=weights, smoothing_function=sf))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# METEOR
# ---------------------------------------------------------------------------

def meteor(prediction: str, reference: str) -> float:
    """
    Compute METEOR score (synonym-aware unigram alignment).

    Args:
        prediction: Generated answer text.
        reference:  Ground-truth answer text.

    Returns:
        METEOR score in [0, 1].
    """
    if not _HAS_NLTK:
        return 0.0
    try:
        hyp = word_tokenize(prediction.lower())
        ref = word_tokenize(reference.lower())
        return float(nltk_meteor([ref], hyp))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Semantic Similarity
# ---------------------------------------------------------------------------

_st_model: Optional[Any] = None

def _get_st_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[Any]:
    global _st_model
    if _st_model is None and _HAS_ST:
        _st_model = SentenceTransformer(model_name)
    return _st_model


def semantic_similarity(
    prediction: str,
    reference:  str,
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """
    Cosine similarity between sentence embeddings of prediction and reference.

    Args:
        prediction: Generated answer text.
        reference:  Ground-truth answer text.
        model_name: sentence-transformers model name.

    Returns:
        Cosine similarity in [−1, 1], typically [0, 1] for short texts.
    """
    model = _get_st_model(model_name)
    if model is None:
        return 0.0
    try:
        embs = model.encode([prediction, reference], convert_to_numpy=True)
        a, b = embs[0], embs[1]
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Fact Coverage
# ---------------------------------------------------------------------------

def fact_coverage(
    prediction: str,
    reference_facts: List[str],
    threshold:  float = 0.3,
) -> Dict[str, float]:
    """
    Estimate what fraction of reference facts appear in the prediction.

    Coverage is determined by Jaccard word-overlap between each fact
    and the prediction text (no external models required).

    Args:
        prediction:      Generated answer text.
        reference_facts: List of atomic fact strings from ground truth.
        threshold:       Minimum Jaccard overlap to count a fact as covered.

    Returns:
        dict with keys:
            covered_count   — int
            total_count     — int
            coverage_ratio  — float [0, 1]
    """
    if not reference_facts:
        return {"covered_count": 0, "total_count": 0, "coverage_ratio": 1.0}

    pred_words = set(re.findall(r"\w+", prediction.lower()))
    covered    = 0

    for fact in reference_facts:
        fact_words = set(re.findall(r"\w+", fact.lower()))
        if not fact_words:
            continue
        intersection = pred_words & fact_words
        union        = pred_words | fact_words
        jaccard      = len(intersection) / len(union) if union else 0.0
        if jaccard >= threshold:
            covered += 1

    return {
        "covered_count":  covered,
        "total_count":    len(reference_facts),
        "coverage_ratio": round(covered / len(reference_facts), 4),
    }


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: List[float],
    correctness: List[float],
    n_bins:      int = 10,
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error for the uncertainty module.

    A well-calibrated system should have:
        P(correct | confidence ≈ c) ≈ c

    Args:
        confidences: List of claim confidence scores ∈ [0, 1].
        correctness: List of binary correctness labels ∈ {0, 1}
                     (or soft scores from a judge).
        n_bins:      Number of equal-width calibration bins.

    Returns:
        dict with keys:
            ece          — Expected Calibration Error (lower is better)
            mce          — Maximum Calibration Error
            bin_stats    — per-bin dict list with avg_conf, avg_acc, count
    """
    if len(confidences) != len(correctness) or not confidences:
        return {"ece": 0.0, "mce": 0.0, "bin_stats": []}

    bins      = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats = []
    ece       = 0.0
    mce       = 0.0
    n         = len(confidences)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx    = [
            j for j, c in enumerate(confidences)
            if lo <= c < hi or (hi == 1.0 and c == 1.0)
        ]
        if not idx:
            bin_stats.append({
                "bin":      f"[{lo:.2f},{hi:.2f})",
                "avg_conf": 0.0,
                "avg_acc":  0.0,
                "count":    0,
            })
            continue

        avg_conf = float(np.mean([confidences[j] for j in idx]))
        avg_acc  = float(np.mean([correctness[j]  for j in idx]))
        gap      = abs(avg_conf - avg_acc)
        weight   = len(idx) / n

        ece += weight * gap
        mce  = max(mce, gap)

        bin_stats.append({
            "bin":      f"[{lo:.2f},{hi:.2f})",
            "avg_conf": round(avg_conf, 4),
            "avg_acc":  round(avg_acc,  4),
            "count":    len(idx),
        })

    return {
        "ece":       round(ece, 4),
        "mce":       round(mce, 4),
        "bin_stats": bin_stats,
    }


def calibration_from_claims(
    claims,  # List[Claim] with uncertainty_band populated
    correctness_scores: Optional[List[float]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute ECE directly from a list of Claim objects.

    If correctness_scores is None, uses claim.confidence as a proxy
    (self-calibration check — useful when ground-truth is unavailable).

    Args:
        claims:              List of Claim objects (uncertainty_band must be set).
        correctness_scores:  Optional external correctness labels.
        n_bins:              Calibration bins.

    Returns:
        ECE result dict (same as expected_calibration_error).
    """
    confidences = []
    for c in claims:
        if c.uncertainty_band is not None:
            # Use mid-point of the band as the effective confidence
            mid = (c.uncertainty_band.lower + c.uncertainty_band.upper) / 2.0
        else:
            mid = c.confidence
        confidences.append(mid)

    if correctness_scores is None:
        correctness_scores = [c.confidence for c in claims]

    return expected_calibration_error(confidences, correctness_scores, n_bins)


# ---------------------------------------------------------------------------
# EvaluationSuite — bundles all metrics for benchmark evaluation
# ---------------------------------------------------------------------------

class EvaluationSuite:
    """
    Single-call evaluation bundle for HERALD answer quality.

    Computes all metrics in one pass and returns a flat dict
    suitable for logging or CSV export.
    """

    def __init__(
        self,
        semantic_model: str = "all-MiniLM-L6-v2",
        rouge_variants: List[str] = None,
        bleu_n:         int = 4,
        ece_bins:       int = 10,
    ):
        self.semantic_model = semantic_model
        self.rouge_variants = rouge_variants or ["rouge1", "rouge2", "rougeL"]
        self.bleu_n         = bleu_n
        self.ece_bins       = ece_bins

    def evaluate(
        self,
        prediction:      str,
        reference:       str,
        reference_facts: Optional[List[str]] = None,
        claims          = None,   # Optional[List[Claim]]
        correctness     = None,   # Optional[List[float]]
    ) -> Dict[str, float]:
        """
        Evaluate a prediction against a reference answer.

        Returns a flat dict of all metric values, e.g.:
            {
                "rouge1_f": 0.612,
                "rouge2_f": 0.341,
                "rougeL_f": 0.588,
                "bleu4":    0.229,
                "meteor":   0.443,
                "sem_sim":  0.871,
                "fact_cov": 0.750,
                "ece":      0.043,
            }
        """
        results: Dict[str, float] = {}

        # ROUGE
        for v in self.rouge_variants:
            r = rouge(prediction, reference, variant=v)
            key = v.replace("rouge", "rouge").lower()
            results[f"{key}_f"] = round(r["fmeasure"], 4)
            results[f"{key}_p"] = round(r["precision"], 4)
            results[f"{key}_r"] = round(r["recall"],    4)

        # BLEU
        results[f"bleu{self.bleu_n}"] = round(
            bleu(prediction, reference, n=self.bleu_n), 4
        )

        # METEOR
        results["meteor"] = round(meteor(prediction, reference), 4)

        # Semantic Similarity
        results["sem_sim"] = round(
            semantic_similarity(prediction, reference, self.semantic_model), 4
        )

        # Fact Coverage
        if reference_facts:
            fc = fact_coverage(prediction, reference_facts)
            results["fact_cov"] = fc["coverage_ratio"]

        # ECE
        if claims:
            ece_result = calibration_from_claims(claims, correctness, self.ece_bins)
            results["ece"] = ece_result["ece"]
            results["mce"] = ece_result["mce"]

        return results

    def evaluate_batch(
        self,
        predictions:      List[str],
        references:       List[str],
        reference_facts:  Optional[List[List[str]]] = None,
        claims_per_query  = None,
    ) -> Dict[str, float]:
        """
        Evaluate a batch and return mean scores across all examples.

        Args:
            predictions:     List of generated answer texts.
            references:      List of ground-truth answer texts.
            reference_facts: Optional per-query list of atomic facts.
            claims_per_query:Optional per-query list of Claim objects.

        Returns:
            dict of mean metric values.
        """
        assert len(predictions) == len(references), (
            "predictions and references must have the same length"
        )
        all_scores: List[Dict[str, float]] = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            facts  = reference_facts[i]  if reference_facts  else None
            claims = claims_per_query[i] if claims_per_query else None
            all_scores.append(self.evaluate(pred, ref, facts, claims))

        if not all_scores:
            return {}

        keys   = all_scores[0].keys()
        return {
            k: round(float(np.mean([s[k] for s in all_scores if k in s])), 4)
            for k in keys
        }