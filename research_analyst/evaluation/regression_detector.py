"""
Regression detection system for monitoring answer quality degradation.
Uses statistical analysis to detect performance drops compared to baseline.
"""

import statistics
from typing import List, Optional, Dict, Tuple
from datetime import datetime

from research_analyst.core.models import (
    QualityMetrics,
    PerformanceBaseline,
    RegressionAlert,
    EvaluationRecord
)
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id


logger = get_logger()


class RegressionDetector:
    """Detect performance regressions using statistical analysis."""
    
    def __init__(self):
        """Initialize regression detector."""
        self.settings = get_settings()
        self.logger = get_logger()
    
    def detect_regression(
        self,
        current_metrics: QualityMetrics,
        baselines: Dict[str, PerformanceBaseline],
        query_id: str
    ) -> List[RegressionAlert]:
        """
        Detect regressions in current metrics vs baselines.
        
        Args:
            current_metrics: Current quality metrics
            baselines: Dict of metric_name -> baseline
            query_id: Current query ID
            
        Returns:
            List of regression alerts
        """
        if not self.settings.enable_regression_detection:
            return []
        
        alerts = []
        
        # Calculate composite score if not set
        if current_metrics.composite_score is None:
            current_metrics.composite_score = current_metrics.calculate_composite_score()
        
        # Check each metric
        metrics_to_check = {
            'citation_coverage': current_metrics.citation_coverage,
            'grounding_score': current_metrics.grounding_score,
            'coherence_score': current_metrics.coherence_score,
            'answer_completeness': current_metrics.answer_completeness,
            'source_diversity': current_metrics.source_diversity,
            'composite_score': current_metrics.composite_score
        }
        
        # Add LLM judge scores if available
        if current_metrics.llm_judge_scores:
            metrics_to_check.update({
                'llm_grounding': current_metrics.llm_judge_scores.grounding_score,
                'llm_factuality': current_metrics.llm_judge_scores.factuality_score,
                'llm_relevance': current_metrics.llm_judge_scores.relevance_score,
                'llm_completeness': current_metrics.llm_judge_scores.completeness_score
            })
        
        for metric_name, current_value in metrics_to_check.items():
            baseline = baselines.get(metric_name)
            
            if baseline:
                alert = self._check_metric_regression(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline=baseline,
                    query_id=query_id
                )
                
                if alert:
                    alerts.append(alert)
        
        if alerts:
            self.logger.warning(
                f"Detected {len(alerts)} regression(s)",
                query_id=query_id,
                metrics=[a.metric_name for a in alerts]
            )
        
        return alerts
    
    def _check_metric_regression(
        self,
        metric_name: str,
        current_value: float,
        baseline: PerformanceBaseline,
        query_id: str
    ) -> Optional[RegressionAlert]:
        """
        Check if a single metric shows regression.
        
        Args:
            metric_name: Name of metric
            current_value: Current metric value
            baseline: Historical baseline
            query_id: Current query ID
            
        Returns:
            RegressionAlert if regression detected, else None
        """
        # Calculate z-score (how many std devs from mean)
        if baseline.std_dev == 0:
            # No variation in baseline, can't detect regression
            return None
        
        z_score = (baseline.mean_value - current_value) / baseline.std_dev
        
        # Check if regression (current value significantly below mean)
        if z_score >= self.settings.regression_threshold:
            # Determine severity
            severity = self._determine_severity(z_score)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                metric_name=metric_name,
                current_value=current_value,
                baseline_mean=baseline.mean_value,
                severity=severity
            )
            
            alert = RegressionAlert(
                alert_id=generate_id("alert"),
                metric_name=metric_name,
                current_value=current_value,
                baseline_mean=baseline.mean_value,
                baseline_std_dev=baseline.std_dev,
                z_score=z_score,
                severity=severity,
                query_id=query_id,
                recommendation=recommendation
            )
            
            return alert
        
        return None
    
    def _determine_severity(self, z_score: float) -> str:
        """
        Determine severity based on z-score.
        
        Args:
            z_score: Standard deviations from mean
            
        Returns:
            Severity level
        """
        if z_score >= 4.0:
            return "critical"
        elif z_score >= 3.0:
            return "high"
        elif z_score >= 2.0:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendation(
        self,
        metric_name: str,
        current_value: float,
        baseline_mean: float,
        severity: str
    ) -> str:
        """
        Generate actionable recommendation for regression.
        
        Args:
            metric_name: Metric that regressed
            current_value: Current value
            baseline_mean: Expected value
            severity: Severity level
            
        Returns:
            Recommendation string
        """
        recommendations = {
            'citation_coverage': "Trigger re-synthesis with emphasis on citations. Check if retrieval quality degraded.",
            'grounding_score': "Re-retrieve with broader search. Verify evidence quality. May need query expansion.",
            'coherence_score': "Review LLM temperature settings. Check for prompt issues. Consider synthesis refinement.",
            'answer_completeness': "Expand query. Retrieve more sources. Check if complex query routed to FAST path.",
            'source_diversity': "Broaden search parameters. Check if search engine limiting results.",
            'composite_score': "Overall quality drop. Trigger self-healing. Review recent system changes.",
            'llm_grounding': "LLM judge detected weak grounding. Re-retrieve with higher quality sources.",
            'llm_factuality': "Factual accuracy concerns. Verify source credibility. Check for hallucination.",
            'llm_relevance': "Answer drift detected. Review query normalization. Check intent classification.",
            'llm_completeness': "Incomplete answer per LLM judge. Expand retrieval or refine synthesis."
        }
        
        base_rec = recommendations.get(metric_name, "Investigate metric degradation.")
        
        if severity in ["critical", "high"]:
            return f"URGENT: {base_rec} (Degradation: {baseline_mean:.2f} → {current_value:.2f})"
        else:
            return f"{base_rec} (Degradation: {baseline_mean:.2f} → {current_value:.2f})"
    
    def calculate_baseline(
        self,
        evaluations: List[EvaluationRecord],
        metric_name: str
    ) -> Optional[PerformanceBaseline]:
        """
        Calculate performance baseline from historical evaluations.
        
        Args:
            evaluations: List of historical evaluations
            metric_name: Metric to calculate baseline for
            
        Returns:
            PerformanceBaseline or None if insufficient data
        """
        if len(evaluations) < self.settings.min_baseline_samples:
            self.logger.warning(
                f"Insufficient samples for baseline ({len(evaluations)} < {self.settings.min_baseline_samples})"
            )
            return None
        
        # Extract metric values
        values = []
        for eval_rec in evaluations:
            value = self._extract_metric_value(eval_rec.quality_metrics, metric_name)
            if value is not None:
                values.append(value)
        
        if not values:
            return None
        
        # Calculate statistics
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        min_value = min(values)
        max_value = max(values)
        
        # Get time window
        timestamps = [e.timestamp for e in evaluations]
        window_start = min(timestamps)
        window_end = max(timestamps)
        
        baseline = PerformanceBaseline(
            baseline_id=generate_id("baseline"),
            metric_name=metric_name,
            mean_value=mean_value,
            std_dev=std_dev,
            min_value=min_value,
            max_value=max_value,
            sample_size=len(values),
            window_start=window_start,
            window_end=window_end
        )
        
        self.logger.info(
            f"Calculated baseline for {metric_name}",
            mean=mean_value,
            std_dev=std_dev,
            samples=len(values)
        )
        
        return baseline
    
    def _extract_metric_value(
        self,
        metrics: QualityMetrics,
        metric_name: str
    ) -> Optional[float]:
        """Extract metric value from QualityMetrics object."""
        # Calculate composite if needed
        if metric_name == 'composite_score':
            if metrics.composite_score is None:
                return metrics.calculate_composite_score()
            return metrics.composite_score
        
        # Heuristic metrics
        heuristic_metrics = {
            'citation_coverage': metrics.citation_coverage,
            'grounding_score': metrics.grounding_score,
            'coherence_score': metrics.coherence_score,
            'answer_completeness': metrics.answer_completeness,
            'source_diversity': metrics.source_diversity
        }
        
        if metric_name in heuristic_metrics:
            return heuristic_metrics[metric_name]
        
        # LLM judge metrics
        if metrics.llm_judge_scores:
            llm_metrics = {
                'llm_grounding': metrics.llm_judge_scores.grounding_score,
                'llm_factuality': metrics.llm_judge_scores.factuality_score,
                'llm_relevance': metrics.llm_judge_scores.relevance_score,
                'llm_completeness': metrics.llm_judge_scores.completeness_score
            }
            
            if metric_name in llm_metrics:
                return llm_metrics[metric_name]
        
        return None
    
    def calculate_all_baselines(
        self,
        evaluations: List[EvaluationRecord]
    ) -> Dict[str, PerformanceBaseline]:
        """
        Calculate baselines for all metrics.
        
        Args:
            evaluations: Historical evaluations
            
        Returns:
            Dict of metric_name -> baseline
        """
        metric_names = [
            'citation_coverage',
            'grounding_score',
            'coherence_score',
            'answer_completeness',
            'source_diversity',
            'composite_score'
        ]
        
        # Add LLM judge metrics if any evaluation has them
        if any(e.quality_metrics.llm_judge_scores for e in evaluations):
            metric_names.extend([
                'llm_grounding',
                'llm_factuality',
                'llm_relevance',
                'llm_completeness'
            ])
        
        baselines = {}
        for metric_name in metric_names:
            baseline = self.calculate_baseline(evaluations, metric_name)
            if baseline:
                baselines[metric_name] = baseline
        
        self.logger.info(f"Calculated {len(baselines)} baselines")
        return baselines
    
    def should_trigger_healing_from_regression(
        self,
        alerts: List[RegressionAlert]
    ) -> Tuple[bool, str]:
        """
        Determine if regression alerts should trigger self-healing.
        
        Args:
            alerts: List of regression alerts
            
        Returns:
            Tuple of (should_trigger, reason)
        """
        if not alerts:
            return False, "No regressions detected"
        
        # Check severity
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        high_alerts = [a for a in alerts if a.severity == "high"]
        
        if critical_alerts:
            return True, f"Critical regression in {critical_alerts[0].metric_name}"
        
        if len(high_alerts) >= 2:
            return True, f"Multiple high-severity regressions: {', '.join(a.metric_name for a in high_alerts)}"
        
        if len(alerts) >= 3:
            return True, f"Multiple regressions detected ({len(alerts)} metrics affected)"
        
        return False, "Regressions present but not severe enough to trigger healing"
    
    def generate_regression_report(
        self,
        alerts: List[RegressionAlert]
    ) -> str:
        """
        Generate human-readable regression report.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Report string
        """
        if not alerts:
            return "No regressions detected."
        
        lines = [
            "=" * 60,
            f"REGRESSION DETECTION REPORT - {len(alerts)} ISSUE(S) FOUND",
            "=" * 60,
            ""
        ]
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_alerts = sorted(alerts, key=lambda a: severity_order[a.severity])
        
        for alert in sorted_alerts:
            lines.extend([
                f"[{alert.severity.upper()}] {alert.metric_name}",
                f"  Current: {alert.current_value:.3f}",
                f"  Expected: {alert.baseline_mean:.3f} ± {alert.baseline_std_dev:.3f}",
                f"  Z-Score: {alert.z_score:.2f} σ below mean",
                f"  Recommendation: {alert.recommendation}",
                ""
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)