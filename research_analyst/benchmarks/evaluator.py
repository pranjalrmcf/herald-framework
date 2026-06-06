"""
Benchmark Evaluation Harness for HERALD.

Supports three standard multi-hop QA benchmarks:
    - HotpotQA        (Yang et al., 2018)
    - MuSiQue         (Trivedi et al., 2022)
    - 2WikiMultiHopQA (Ho et al., 2020)

Usage:
    python -m benchmarks.evaluator \\
        --dataset hotpotqa \\
        --split   validation \\
        --n_samples 200 \\
        --output_dir results/

The harness:
    1. Loads dataset samples (expects HuggingFace datasets format).
    2. Runs HERALD pipeline for each question.
    3. Computes all metrics from utils.metrics.EvaluationSuite.
    4. Saves per-sample results to JSON and aggregate to CSV.

Dependencies:
    pip install datasets   (HuggingFace datasets)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from research_analyst.orchestration.orchestrator import ResearchAnalyst
from research_analyst.utils.metrics import EvaluationSuite
from research_analyst.utils.logger import get_logger


logger = get_logger()

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_hotpotqa(split: str, n_samples: int) -> List[Dict[str, Any]]:
    """
    Load HotpotQA samples.

    Returns list of dicts with keys: id, question, answer, supporting_facts.
    """
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "fullwiki", split=split, trust_remote_code=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        # Flatten supporting facts to a list of strings
        facts: List[str] = []
        titles  = item.get("supporting_facts", {}).get("title", [])
        sent_ids= item.get("supporting_facts", {}).get("sent_id", [])
        context_titles   = item.get("context", {}).get("title", [])
        context_sentences= item.get("context", {}).get("sentences", [])
        title_to_sents   = dict(zip(context_titles, context_sentences))
        for title, sid in zip(titles, sent_ids):
            sents = title_to_sents.get(title, [])
            if sid < len(sents):
                facts.append(sents[sid])
        samples.append({
            "id":       item["id"],
            "question": item["question"],
            "answer":   item["answer"],
            "facts":    facts,
            "dataset":  "hotpotqa",
        })
    return samples


def load_musique(split: str, n_samples: int) -> List[Dict[str, Any]]:
    """
    Load MuSiQue samples.

    Returns list of dicts with keys: id, question, answer, facts.
    """
    from datasets import load_dataset
    ds = load_dataset("drt/musique", split=split, trust_remote_code=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        # Extract decomposition steps as facts
        facts = [
            para["paragraph_text"]
            for para in item.get("paragraphs", [])
            if para.get("is_supporting", False)
        ]
        samples.append({
            "id":       item["id"],
            "question": item["question"],
            "answer":   item["answer"],
            "facts":    facts,
            "dataset":  "musique",
        })
    return samples


def load_2wikimultihopqa(split: str, n_samples: int) -> List[Dict[str, Any]]:
    """
    Load 2WikiMultiHopQA samples.

    Returns list of dicts with keys: id, question, answer, facts.
    """
    from datasets import load_dataset
    ds = load_dataset("williamf/2wikimultihopqa", split=split, trust_remote_code=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        facts = [
            " ".join(f[1])
            for f in item.get("supporting_facts", [])
        ]
        samples.append({
            "id":       item["_id"],
            "question": item["question"],
            "answer":   item["answer"],
            "facts":    facts,
            "dataset":  "2wikimultihopqa",
        })
    return samples


DATASET_LOADERS = {
    "hotpotqa":        load_hotpotqa,
    "musique":         load_musique,
    "2wikimultihopqa": load_2wikimultihopqa,
}


# ---------------------------------------------------------------------------
# Per-sample result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    sample_id:      str
    dataset:        str
    question:       str
    reference:      str
    prediction:     str
    execution_path: str
    execution_time_ms:float
    quality_composite:float
    metrics:        Dict[str, float] = field(default_factory=dict)
    error:          Optional[str]    = None


# ---------------------------------------------------------------------------
# BenchmarkEvaluator
# ---------------------------------------------------------------------------

class BenchmarkEvaluator:
    """
    Runs HERALD against a benchmark dataset and collects evaluation metrics.
    """

    def __init__(
        self,
        output_dir:     str  = "results",
        semantic_model: str  = "all-MiniLM-L6-v2",
    ):
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyst        = ResearchAnalyst()
        self.eval_suite     = EvaluationSuite(semantic_model=semantic_model)
        self.logger         = get_logger()

    def run(
        self,
        dataset:    str,
        split:      str  = "validation",
        n_samples:  int  = 100,
        resume:     bool = True,
    ) -> Dict[str, float]:
        """
        Run full benchmark evaluation.

        Args:
            dataset:   One of "hotpotqa", "musique", "2wikimultihopqa".
            split:     Dataset split name.
            n_samples: Number of samples to evaluate.
            resume:    If True, skip samples already in output file.

        Returns:
            Aggregate metric dict.
        """
        if dataset not in DATASET_LOADERS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Choose from: {list(DATASET_LOADERS.keys())}"
            )

        self.logger.info(
            "Benchmark evaluation starting",
            dataset   = dataset,
            split     = split,
            n_samples = n_samples,
        )

        samples = DATASET_LOADERS[dataset](split, n_samples)

        # Resume: load already-completed results
        out_path = self.output_dir / f"{dataset}_{split}_{n_samples}.jsonl"
        completed_ids: set = set()
        results: List[SampleResult] = []

        if resume and out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        completed_ids.add(r["sample_id"])
                        results.append(SampleResult(**r))
                    except Exception:
                        pass
            self.logger.info(f"Resuming: {len(completed_ids)} samples already done")

        # Process remaining samples
        with open(out_path, "a") as out_f:
            for idx, sample in enumerate(samples):
                if sample["id"] in completed_ids:
                    continue

                self.logger.info(
                    f"Sample {idx + 1}/{len(samples)}",
                    sample_id = sample["id"],
                )

                result = self._evaluate_sample(sample)
                results.append(result)
                out_f.write(json.dumps(asdict(result)) + "\n")
                out_f.flush()

        # Aggregate
        aggregate = self._aggregate(results)

        # Save aggregate
        agg_path = self.output_dir / f"{dataset}_{split}_{n_samples}_aggregate.json"
        with open(agg_path, "w") as f:
            json.dump({
                "dataset":    dataset,
                "split":      split,
                "n_samples":  len(results),
                "evaluated_at": datetime.utcnow().isoformat(),
                "metrics":    aggregate,
            }, f, indent=2)

        self.logger.info(
            "Benchmark complete",
            dataset = dataset,
            n       = len(results),
            rouge1  = aggregate.get("rouge1_f", 0.0),
            rougeL  = aggregate.get("rougeL_f", 0.0),
            sem_sim = aggregate.get("sem_sim",   0.0),
        )
        return aggregate

    def _evaluate_sample(self, sample: Dict[str, Any]) -> SampleResult:
        """Run HERALD on one sample and compute all metrics."""
        t0 = time.time()
        try:
            response = self.analyst.query(query_text=sample["question"])
            elapsed  = (time.time() - t0) * 1000

            if response.success and response.answer:
                prediction = response.answer.text
                exc_path   = str(response.execution_path or "unknown")
                composite  = (
                    response.quality_metrics.composite_score
                    if response.quality_metrics else 0.0
                ) or 0.0
            else:
                prediction = ""
                exc_path   = "error"
                composite  = 0.0

            metrics = self.eval_suite.evaluate(
                prediction      = prediction,
                reference       = sample["answer"],
                reference_facts = sample.get("facts"),
            )

            return SampleResult(
                sample_id         = sample["id"],
                dataset           = sample["dataset"],
                question          = sample["question"],
                reference         = sample["answer"],
                prediction        = prediction,
                execution_path    = exc_path,
                execution_time_ms = round(elapsed, 1),
                quality_composite = round(composite, 4),
                metrics           = metrics,
            )

        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            self.logger.error(
                "Sample evaluation failed",
                sample_id = sample["id"],
                error     = str(e),
            )
            return SampleResult(
                sample_id         = sample["id"],
                dataset           = sample["dataset"],
                question          = sample["question"],
                reference         = sample["answer"],
                prediction        = "",
                execution_path    = "error",
                execution_time_ms = round(elapsed, 1),
                quality_composite = 0.0,
                error             = str(e),
            )

    @staticmethod
    def _aggregate(results: List[SampleResult]) -> Dict[str, float]:
        """Compute mean metrics across all samples."""
        import numpy as np

        successful = [r for r in results if not r.error]
        if not successful:
            return {}

        all_keys = set()
        for r in successful:
            all_keys.update(r.metrics.keys())

        agg: Dict[str, float] = {}
        for k in sorted(all_keys):
            vals = [r.metrics[k] for r in successful if k in r.metrics]
            agg[k] = round(float(np.mean(vals)), 4)

        agg["mean_execution_time_ms"] = round(
            float(np.mean([r.execution_time_ms for r in successful])), 1
        )
        agg["mean_quality_composite"] = round(
            float(np.mean([r.quality_composite for r in successful])), 4
        )
        agg["success_rate"] = round(len(successful) / len(results), 4)

        return agg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HERALD benchmark evaluator"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_LOADERS.keys()),
        required=True,
        help="Benchmark dataset name",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory for result files (default: results/)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start fresh (do not resume from existing results)",
    )
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(output_dir=args.output_dir)
    aggregate = evaluator.run(
        dataset   = args.dataset,
        split     = args.split,
        n_samples = args.n_samples,
        resume    = not args.no_resume,
    )

    print("\n=== AGGREGATE RESULTS ===")
    for k, v in sorted(aggregate.items()):
        print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()