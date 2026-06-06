# """
# Self-healing module for the research analyst system.
# Implements corrective RAG to improve answer quality.
# """

# from typing import List, Optional, Tuple, Dict
# from datetime import datetime

# from research_analyst.core.models import (
#     Query,
#     NormalizedQuery,
#     Answer,
#     Evidence,
#     QualityMetrics,
#     SelfHealingAction,
#     ExecutionPath
# )
# from research_analyst.core.exceptions import SelfHealingError, MaxRetriesExceeded
# from research_analyst.config import get_settings
# from research_analyst.utils.logger import get_logger
# from research_analyst.utils.helpers import generate_id


# logger = get_logger()


# class SelfHealer:
#     """Implement self-healing and corrective RAG."""
    
#     def __init__(self):
#         """Initialize self-healer."""
#         self.settings = get_settings()
#         self.logger = get_logger()
    
#     def should_trigger_healing(
#         self,
#         quality_metrics: QualityMetrics,
#         answer: Answer
#     ) -> Tuple[bool, str]:
#         """
#         Determine if self-healing should be triggered.
        
#         Args:
#             quality_metrics: Quality metrics
#             answer: Generated answer
            
#         Returns:
#             Tuple of (should_heal, reason)
#         """
#         # Check if already passes threshold
#         if quality_metrics.passes_threshold:
#             return False, "Quality metrics pass threshold"
        
#         # Identify specific issue
#         if quality_metrics.citation_coverage < self.settings.min_citation_coverage:
#             return True, "Insufficient citation coverage"
        
#         if quality_metrics.grounding_score < 0.6:
#             return True, "Weak grounding in evidence"
        
#         if answer.confidence < self.settings.min_confidence_threshold:
#             return True, "Low confidence score"
        
#         if quality_metrics.answer_completeness < 0.5:
#             return True, "Incomplete answer"
        
#         # Check for specific issues
#         if "Insufficient citations" in quality_metrics.issues:
#             return True, "Insufficient citations detected"
        
#         return False, "No critical issues detected"
    
#     def heal(
#         self,
#         query: Query,
#         normalized_query: NormalizedQuery,
#         answer: Answer,
#         evidence: Evidence,
#         quality_metrics: QualityMetrics,
#         current_attempt: int = 0
#     ) -> Tuple[Optional[Answer], List[SelfHealingAction]]:
#         """
#         Attempt to heal/improve answer quality.
        
#         Args:
#             query: Original query
#             normalized_query: Normalized query
#             answer: Current answer
#             evidence: Current evidence
#             quality_metrics: Quality metrics
#             current_attempt: Current healing attempt number
            
#         Returns:
#             Tuple of (improved_answer or None, list of actions taken)
#         """
#         self.logger.info(
#             "Attempting self-healing",
#             answer_id=answer.answer_id,
#             attempt=current_attempt + 1,
#             max_attempts=self.settings.max_self_healing_attempts
#         )
        
#         # Check max attempts
#         if current_attempt >= self.settings.max_self_healing_attempts:
#             raise MaxRetriesExceeded(
#                 attempts=current_attempt,
#                 details={
#                     "answer_id": answer.answer_id,
#                     "final_metrics": quality_metrics.__dict__
#                 }
#             )
        
#         actions = []
        
#         try:
#             # Determine healing strategy based on issues
#             strategy = self._determine_strategy(quality_metrics, evidence)
            
#             self.logger.log_self_healing(
#                 query_id=query.metadata.get("query_id", "unknown"),
#                 action_type=strategy,
#                 attempt=current_attempt + 1,
#                 reasoning=f"Quality issues: {', '.join(quality_metrics.issues)}"
#             )
            
#             # Execute strategy
#             if strategy == "re-retrieve":
#                 improved_answer = self._re_retrieve(
#                     query,
#                     normalized_query,
#                     answer,
#                     evidence,
#                     actions
#                 )
#             elif strategy == "expand-query":
#                 improved_answer = self._expand_and_retrieve(
#                     query,
#                     normalized_query,
#                     answer,
#                     evidence,
#                     actions
#                 )
#             elif strategy == "switch-path":
#                 improved_answer = self._switch_execution_path(
#                     query,
#                     normalized_query,
#                     answer,
#                     evidence,
#                     actions
#                 )
#             elif strategy == "refine-synthesis":
#                 improved_answer = self._refine_synthesis(
#                     query,
#                     answer,
#                     evidence,
#                     actions
#                 )
#             else:
#                 # No viable strategy
#                 self.logger.warning(
#                     "No viable healing strategy",
#                     answer_id=answer.answer_id
#                 )
#                 return None, actions
            
#             return improved_answer, actions
            
#         except Exception as e:
#             self.logger.error(
#                 "Self-healing failed",
#                 answer_id=answer.answer_id,
#                 attempt=current_attempt + 1,
#                 error=str(e)
#             )
#             raise SelfHealingError(
#                 f"Self-healing failed: {str(e)}",
#                 attempts=current_attempt + 1,
#                 details={"answer_id": answer.answer_id}
#             )
    
#     def _determine_strategy(
#         self,
#         quality_metrics: QualityMetrics,
#         evidence: Evidence
#     ) -> str:
#         """
#         Determine best healing strategy based on issues.
        
#         Args:
#             quality_metrics: Quality metrics
#             evidence: Current evidence
            
#         Returns:
#             Strategy name
#         """
#         # If not enough documents retrieved
#         if len(evidence.supporting_documents) < 3:
#             return "expand-query"
        
#         # If weak grounding
#         if quality_metrics.grounding_score < 0.5:
#             return "re-retrieve"
        
#         # If low citation coverage
#         if quality_metrics.citation_coverage < 0.3:
#             return "refine-synthesis"
        
#         # If low completeness
#         if quality_metrics.answer_completeness < 0.5:
#             return "expand-query"
        
#         # Default: try re-retrieval
#         return "re-retrieve"
    
#     def _re_retrieve(
#         self,
#         query: Query,
#         normalized_query: NormalizedQuery,
#         answer: Answer,
#         evidence: Evidence,
#         actions: List[SelfHealingAction]
#     ) -> Optional[Answer]:
#         """
#         Re-retrieve with different parameters.
        
#         Args:
#             query: Original query
#             normalized_query: Normalized query
#             answer: Current answer
#             evidence: Current evidence
#             actions: List to append actions to
            
#         Returns:
#             Improved answer or None
#         """
#         action = SelfHealingAction(
#             action_type="re-retrieve",
#             reasoning="Re-retrieving with broader search to get more relevant sources",
#             parameters={
#                 "increase_results": True,
#                 "relax_filters": True
#             }
#         )
#         actions.append(action)
        
#         # This would trigger actual re-retrieval
#         # For now, return None to indicate need for pipeline re-execution
#         return None
    
#     def _expand_and_retrieve(
#         self,
#         query: Query,
#         normalized_query: NormalizedQuery,
#         answer: Answer,
#         evidence: Evidence,
#         actions: List[SelfHealingAction]
#     ) -> Optional[Answer]:
#         """
#         Expand query and retrieve additional documents.
        
#         Args:
#             query: Original query
#             normalized_query: Normalized query
#             answer: Current answer
#             evidence: Current evidence
#             actions: List to append actions to
            
#         Returns:
#             Improved answer or None
#         """
#         # Generate expanded query terms
#         expanded_terms = self._generate_expanded_terms(normalized_query)
        
#         action = SelfHealingAction(
#             action_type="expand-query",
#             reasoning="Expanding query with related terms to find additional sources",
#             parameters={
#                 "expanded_terms": expanded_terms,
#                 "original_query": query.text
#             }
#         )
#         actions.append(action)
        
#         # Return None to trigger pipeline re-execution with expanded query
#         return None
    
#     def _switch_execution_path(
#         self,
#         query: Query,
#         normalized_query: NormalizedQuery,
#         answer: Answer,
#         evidence: Evidence,
#         actions: List[SelfHealingAction]
#     ) -> Optional[Answer]:
#         """
#         Switch execution path (e.g., from fast to research).
        
#         Args:
#             query: Original query
#             normalized_query: Normalized query
#             answer: Current answer
#             evidence: Current evidence
#             actions: List to append actions to
            
#         Returns:
#             Improved answer or None
#         """
#         # Determine new path
#         current_path = answer.execution_path
#         new_path = (
#             ExecutionPath.RESEARCH if current_path == ExecutionPath.FAST
#             else ExecutionPath.FAST
#         )
        
#         action = SelfHealingAction(
#             action_type="switch-path",
#             reasoning=f"Switching from {current_path.value} to {new_path.value} path for better results",
#             parameters={
#                 "from_path": current_path.value,
#                 "to_path": new_path.value
#             }
#         )
#         actions.append(action)
        
#         # Return None to trigger pipeline with new path
#         return None
    
#     def _refine_synthesis(
#         self,
#         query: Query,
#         answer: Answer,
#         evidence: Evidence,
#         actions: List[SelfHealingAction]
#     ) -> Optional[Answer]:
#         """
#         Refine answer synthesis without new retrieval.
        
#         Args:
#             query: Original query
#             answer: Current answer
#             evidence: Current evidence
#             actions: List to append actions to
            
#         Returns:
#             Improved answer or None
#         """
#         action = SelfHealingAction(
#             action_type="refine-synthesis",
#             reasoning="Regenerating answer with emphasis on citations and grounding",
#             parameters={
#                 "focus": "citations_and_grounding",
#                 "use_existing_evidence": True
#             }
#         )
#         actions.append(action)
        
#         # Use answer generator to refine
#         try:
#             from research_analyst.synthesis import AnswerGenerator
#             generator = AnswerGenerator()
            
#             refined_answer = generator.refine_answer(answer, evidence)
            
#             return refined_answer
            
#         except Exception as e:
#             self.logger.warning(
#                 "Synthesis refinement failed",
#                 error=str(e)
#             )
#             return None
    
#     def _generate_expanded_terms(self, normalized_query: NormalizedQuery) -> List[str]:
#         """
#         Generate expanded query terms.
        
#         Args:
#             normalized_query: Normalized query
            
#         Returns:
#             List of expanded terms
#         """
#         expanded = []
        
#         # Add entities
#         expanded.extend(normalized_query.entities_mentioned)
        
#         # Add domain-specific terms
#         if normalized_query.domain:
#             expanded.append(normalized_query.domain)
        
#         # Add synonyms (simplified - would use WordNet or similar)
#         keywords = normalized_query.normalized_text.split()[:5]
#         expanded.extend(keywords)
        
#         return list(set(expanded))
    
#     def analyze_healing_effectiveness(
#         self,
#         original_metrics: QualityMetrics,
#         improved_metrics: QualityMetrics
#     ) -> Dict:
#         """
#         Analyze effectiveness of healing.
        
#         Args:
#             original_metrics: Original quality metrics
#             improved_metrics: Improved quality metrics
            
#         Returns:
#             Analysis dictionary
#         """
#         improvements = {
#             'citation_coverage': improved_metrics.citation_coverage - original_metrics.citation_coverage,
#             'grounding_score': improved_metrics.grounding_score - original_metrics.grounding_score,
#             'coherence_score': improved_metrics.coherence_score - original_metrics.coherence_score,
#             'completeness': improved_metrics.answer_completeness - original_metrics.answer_completeness,
#             'diversity': improved_metrics.source_diversity - original_metrics.source_diversity,
#         }
        
#         # Calculate overall improvement
#         overall_improvement = sum(improvements.values()) / len(improvements)
        
#         # Determine if healing was successful
#         was_successful = (
#             improved_metrics.passes_threshold and
#             not original_metrics.passes_threshold
#         )
        
#         analysis = {
#             'improvements': improvements,
#             'overall_improvement': overall_improvement,
#             'was_successful': was_successful,
#             'original_passed': original_metrics.passes_threshold,
#             'improved_passed': improved_metrics.passes_threshold
#         }
        
#         return analysis
    
#     def generate_healing_report(
#         self,
#         actions: List[SelfHealingAction],
#         original_metrics: QualityMetrics,
#         improved_metrics: Optional[QualityMetrics]
#     ) -> str:
#         """
#         Generate report of healing attempts.
        
#         Args:
#             actions: Actions taken
#             original_metrics: Original metrics
#             improved_metrics: Improved metrics (if available)
            
#         Returns:
#             Report string
#         """
#         lines = [
#             "=" * 60,
#             "SELF-HEALING REPORT",
#             "=" * 60,
#             "",
#             f"Healing Attempts: {len(actions)}",
#             ""
#         ]
        
#         # List actions
#         lines.append("ACTIONS TAKEN:")
#         for i, action in enumerate(actions, 1):
#             lines.append(f"{i}. {action.action_type}")
#             lines.append(f"   Reasoning: {action.reasoning}")
#             lines.append("")
        
#         # Show metrics comparison
#         if improved_metrics:
#             lines.extend([
#                 "METRICS COMPARISON:",
#                 f"  Citation Coverage:  {original_metrics.citation_coverage:.2%} → {improved_metrics.citation_coverage:.2%}",
#                 f"  Grounding Score:    {original_metrics.grounding_score:.2%} → {improved_metrics.grounding_score:.2%}",
#                 f"  Coherence:          {original_metrics.coherence_score:.2%} → {improved_metrics.coherence_score:.2%}",
#                 f"  Completeness:       {original_metrics.answer_completeness:.2%} → {improved_metrics.answer_completeness:.2%}",
#                 "",
#                 f"RESULT: {'✓ IMPROVED' if improved_metrics.passes_threshold else '✗ STILL INSUFFICIENT'}",
#                 ""
#             ])
        
#         lines.append("=" * 60)
        
#         return "\n".join(lines)


"""
Self-healing module for the research analyst system.
Implements corrective RAG to improve answer quality.
"""

from typing import List, Optional, Tuple, Dict
from datetime import datetime

from research_analyst.core.models import (
    Query,
    NormalizedQuery,
    Answer,
    Evidence,
    QualityMetrics,
    SelfHealingAction,
    ExecutionPath,
    RegressionAlert
)
from research_analyst.core.exceptions import SelfHealingError, MaxRetriesExceeded
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id


logger = get_logger()


class SelfHealer:
    """Implement self-healing and corrective RAG."""
    
    def __init__(self):
        """Initialize self-healer."""
        self.settings = get_settings()
        self.logger = get_logger()
    
    def should_trigger_healing(
        self,
        quality_metrics: QualityMetrics,
        answer: Answer,
        regression_alerts: Optional[List[RegressionAlert]] = None
    ) -> Tuple[bool, str]:
        """
        Determine if self-healing should be triggered.
        
        Args:
            quality_metrics: Quality metrics
            answer: Generated answer
            regression_alerts: Optional regression alerts
            
        Returns:
            Tuple of (should_heal, reason)
        """
        # Check if already passes threshold and no regressions
        if quality_metrics.passes_threshold and not regression_alerts:
            return False, "Quality metrics pass threshold and no regressions"
        
        # Check for regression-based triggers
        if regression_alerts:
            should_heal, reason = self._check_regression_triggers(regression_alerts)
            if should_heal:
                return True, f"Regression detected: {reason}"
        
        # Identify specific quality issues
        if quality_metrics.citation_coverage < self.settings.min_citation_coverage:
            return True, "Insufficient citation coverage"
        
        if quality_metrics.grounding_score < 0.6:
            return True, "Weak grounding in evidence"
        
        if answer.confidence < self.settings.min_confidence_threshold:
            return True, "Low confidence score"
        
        if quality_metrics.answer_completeness < 0.5:
            return True, "Incomplete answer"
        
        # Check LLM Judge scores if available
        if quality_metrics.llm_judge_scores:
            llm_scores = quality_metrics.llm_judge_scores
            if llm_scores.grounding_score < 0.6:
                return True, "LLM Judge: Weak grounding"
            if llm_scores.factuality_score < 0.6:
                return True, "LLM Judge: Factual accuracy concerns"
        
        # Check for specific issues
        if "Insufficient citations" in quality_metrics.issues:
            return True, "Insufficient citations detected"
        
        return False, "No critical issues detected"
    
    def _check_regression_triggers(
        self,
        alerts: List[RegressionAlert]
    ) -> Tuple[bool, str]:
        """
        Check if regressions should trigger healing.
        
        Args:
            alerts: Regression alerts
            
        Returns:
            Tuple of (should_trigger, reason)
        """
        if not alerts:
            return False, "No regressions"
        
        # Critical or high severity triggers immediately
        critical = [a for a in alerts if a.severity in ["critical", "high"]]
        if critical:
            return True, f"{critical[0].severity} regression in {critical[0].metric_name}"
        
        # Multiple medium regressions trigger
        if len(alerts) >= 3:
            return True, f"Multiple regressions ({len(alerts)} metrics)"
        
        return False, "Regressions not severe enough"
    
    def heal(
        self,
        query: Query,
        normalized_query: NormalizedQuery,
        answer: Answer,
        evidence: Evidence,
        quality_metrics: QualityMetrics,
        current_attempt: int = 0
    ) -> Tuple[Optional[Answer], List[SelfHealingAction]]:
        """
        Attempt to heal/improve answer quality.
        
        Args:
            query: Original query
            normalized_query: Normalized query
            answer: Current answer
            evidence: Current evidence
            quality_metrics: Quality metrics
            current_attempt: Current healing attempt number
            
        Returns:
            Tuple of (improved_answer or None, list of actions taken)
        """
        self.logger.info(
            "Attempting self-healing",
            answer_id=answer.answer_id,
            attempt=current_attempt + 1,
            max_attempts=self.settings.max_self_healing_attempts
        )
        
        # Check max attempts
        if current_attempt >= self.settings.max_self_healing_attempts:
            raise MaxRetriesExceeded(
                attempts=current_attempt,
                details={
                    "answer_id": answer.answer_id,
                    "final_metrics": quality_metrics.__dict__
                }
            )
        
        actions = []
        
        try:
            # Determine healing strategy based on issues
            strategy = self._determine_strategy(quality_metrics, evidence)
            
            self.logger.log_self_healing(
                query_id=query.metadata.get("query_id", "unknown"),
                action_type=strategy,
                attempt=current_attempt + 1,
                reasoning=f"Quality issues: {', '.join(quality_metrics.issues)}"
            )
            
            # Execute strategy
            if strategy == "re-retrieve":
                improved_answer = self._re_retrieve(
                    query,
                    normalized_query,
                    answer,
                    evidence,
                    actions
                )
            elif strategy == "expand-query":
                improved_answer = self._expand_and_retrieve(
                    query,
                    normalized_query,
                    answer,
                    evidence,
                    actions
                )
            elif strategy == "switch-path":
                improved_answer = self._switch_execution_path(
                    query,
                    normalized_query,
                    answer,
                    evidence,
                    actions
                )
            elif strategy == "refine-synthesis":
                improved_answer = self._refine_synthesis(
                    query,
                    answer,
                    evidence,
                    actions
                )
            else:
                # No viable strategy
                self.logger.warning(
                    "No viable healing strategy",
                    answer_id=answer.answer_id
                )
                return None, actions
            
            return improved_answer, actions
            
        except Exception as e:
            self.logger.error(
                "Self-healing failed",
                answer_id=answer.answer_id,
                attempt=current_attempt + 1,
                error=str(e)
            )
            raise SelfHealingError(
                f"Self-healing failed: {str(e)}",
                attempts=current_attempt + 1,
                details={"answer_id": answer.answer_id}
            )
    
    def _determine_strategy(
        self,
        quality_metrics: QualityMetrics,
        evidence: Evidence
    ) -> str:
        """
        Determine best healing strategy based on issues.
        
        Args:
            quality_metrics: Quality metrics
            evidence: Current evidence
            
        Returns:
            Strategy name
        """
        # If not enough documents retrieved
        if len(evidence.supporting_documents) < 3:
            return "expand-query"
        
        # If weak grounding
        if quality_metrics.grounding_score < 0.5:
            return "re-retrieve"
        
        # If low citation coverage
        if quality_metrics.citation_coverage < 0.3:
            return "refine-synthesis"
        
        # If low completeness
        if quality_metrics.answer_completeness < 0.5:
            return "expand-query"
        
        # Default: try re-retrieval
        return "re-retrieve"
    
    def _re_retrieve(
        self,
        query: Query,
        normalized_query: NormalizedQuery,
        answer: Answer,
        evidence: Evidence,
        actions: List[SelfHealingAction]
    ) -> Optional[Answer]:
        """
        Re-retrieve with different parameters.
        
        Args:
            query: Original query
            normalized_query: Normalized query
            answer: Current answer
            evidence: Current evidence
            actions: List to append actions to
            
        Returns:
            Improved answer or None
        """
        action = SelfHealingAction(
            action_type="re-retrieve",
            reasoning="Re-retrieving with broader search to get more relevant sources",
            parameters={
                "increase_results": True,
                "relax_filters": True
            }
        )
        actions.append(action)
        
        # This would trigger actual re-retrieval
        # For now, return None to indicate need for pipeline re-execution
        return None
    
    def _expand_and_retrieve(
        self,
        query: Query,
        normalized_query: NormalizedQuery,
        answer: Answer,
        evidence: Evidence,
        actions: List[SelfHealingAction]
    ) -> Optional[Answer]:
        """
        Expand query and retrieve additional documents.
        
        Args:
            query: Original query
            normalized_query: Normalized query
            answer: Current answer
            evidence: Current evidence
            actions: List to append actions to
            
        Returns:
            Improved answer or None
        """
        # Generate expanded query terms
        expanded_terms = self._generate_expanded_terms(normalized_query)
        
        action = SelfHealingAction(
            action_type="expand-query",
            reasoning="Expanding query with related terms to find additional sources",
            parameters={
                "expanded_terms": expanded_terms,
                "original_query": query.text
            }
        )
        actions.append(action)
        
        # Return None to trigger pipeline re-execution with expanded query
        return None
    
    def _switch_execution_path(
        self,
        query: Query,
        normalized_query: NormalizedQuery,
        answer: Answer,
        evidence: Evidence,
        actions: List[SelfHealingAction]
    ) -> Optional[Answer]:
        """
        Switch execution path (e.g., from fast to research).
        
        Args:
            query: Original query
            normalized_query: Normalized query
            answer: Current answer
            evidence: Current evidence
            actions: List to append actions to
            
        Returns:
            Improved answer or None
        """
        # Determine new path
        current_path = answer.execution_path
        new_path = (
            ExecutionPath.RESEARCH if current_path == ExecutionPath.FAST
            else ExecutionPath.FAST
        )
        
        action = SelfHealingAction(
            action_type="switch-path",
            reasoning=f"Switching from {current_path.value} to {new_path.value} path for better results",
            parameters={
                "from_path": current_path.value,
                "to_path": new_path.value
            }
        )
        actions.append(action)
        
        # Return None to trigger pipeline with new path
        return None
    
    def _refine_synthesis(
        self,
        query: Query,
        answer: Answer,
        evidence: Evidence,
        actions: List[SelfHealingAction]
    ) -> Optional[Answer]:
        """
        Refine answer synthesis without new retrieval.
        
        Args:
            query: Original query
            answer: Current answer
            evidence: Current evidence
            actions: List to append actions to
            
        Returns:
            Improved answer or None
        """
        action = SelfHealingAction(
            action_type="refine-synthesis",
            reasoning="Regenerating answer with emphasis on citations and grounding",
            parameters={
                "focus": "citations_and_grounding",
                "use_existing_evidence": True
            }
        )
        actions.append(action)
        
        # Use answer generator to refine
        try:
            from research_analyst.synthesis import AnswerGenerator
            generator = AnswerGenerator()
            
            refined_answer = generator.refine_answer(answer, evidence)
            
            return refined_answer
            
        except Exception as e:
            self.logger.warning(
                "Synthesis refinement failed",
                error=str(e)
            )
            return None
    
    def _generate_expanded_terms(self, normalized_query: NormalizedQuery) -> List[str]:
        """
        Generate expanded query terms.
        
        Args:
            normalized_query: Normalized query
            
        Returns:
            List of expanded terms
        """
        expanded = []
        
        # Add entities
        expanded.extend(normalized_query.entities_mentioned)
        
        # Add domain-specific terms
        if normalized_query.domain:
            expanded.append(normalized_query.domain)
        
        # Add synonyms (simplified - would use WordNet or similar)
        keywords = normalized_query.normalized_text.split()[:5]
        expanded.extend(keywords)
        
        return list(set(expanded))
    
    def analyze_healing_effectiveness(
        self,
        original_metrics: QualityMetrics,
        improved_metrics: QualityMetrics
    ) -> Dict:
        """
        Analyze effectiveness of healing.
        
        Args:
            original_metrics: Original quality metrics
            improved_metrics: Improved quality metrics
            
        Returns:
            Analysis dictionary
        """
        improvements = {
            'citation_coverage': improved_metrics.citation_coverage - original_metrics.citation_coverage,
            'grounding_score': improved_metrics.grounding_score - original_metrics.grounding_score,
            'coherence_score': improved_metrics.coherence_score - original_metrics.coherence_score,
            'completeness': improved_metrics.answer_completeness - original_metrics.answer_completeness,
            'diversity': improved_metrics.source_diversity - original_metrics.source_diversity,
        }
        
        # Calculate overall improvement
        overall_improvement = sum(improvements.values()) / len(improvements)
        
        # Determine if healing was successful
        was_successful = (
            improved_metrics.passes_threshold and
            not original_metrics.passes_threshold
        )
        
        analysis = {
            'improvements': improvements,
            'overall_improvement': overall_improvement,
            'was_successful': was_successful,
            'original_passed': original_metrics.passes_threshold,
            'improved_passed': improved_metrics.passes_threshold
        }
        
        return analysis
    
    def generate_healing_report(
        self,
        actions: List[SelfHealingAction],
        original_metrics: QualityMetrics,
        improved_metrics: Optional[QualityMetrics]
    ) -> str:
        """
        Generate report of healing attempts.
        
        Args:
            actions: Actions taken
            original_metrics: Original metrics
            improved_metrics: Improved metrics (if available)
            
        Returns:
            Report string
        """
        lines = [
            "=" * 60,
            "SELF-HEALING REPORT",
            "=" * 60,
            "",
            f"Healing Attempts: {len(actions)}",
            ""
        ]
        
        # List actions
        lines.append("ACTIONS TAKEN:")
        for i, action in enumerate(actions, 1):
            lines.append(f"{i}. {action.action_type}")
            lines.append(f"   Reasoning: {action.reasoning}")
            lines.append("")
        
        # Show metrics comparison
        if improved_metrics:
            lines.extend([
                "METRICS COMPARISON:",
                f"  Citation Coverage:  {original_metrics.citation_coverage:.2%} → {improved_metrics.citation_coverage:.2%}",
                f"  Grounding Score:    {original_metrics.grounding_score:.2%} → {improved_metrics.grounding_score:.2%}",
                f"  Coherence:          {original_metrics.coherence_score:.2%} → {improved_metrics.coherence_score:.2%}",
                f"  Completeness:       {original_metrics.answer_completeness:.2%} → {improved_metrics.answer_completeness:.2%}",
                "",
                f"RESULT: {'✓ IMPROVED' if improved_metrics.passes_threshold else '✗ STILL INSUFFICIENT'}",
                ""
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)