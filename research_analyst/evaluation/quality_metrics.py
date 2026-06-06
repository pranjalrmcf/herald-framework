# """
# Quality metrics for the research analyst system.
# Evaluates answer quality using multiple metrics.
# """

# import re
# from typing import Dict, List, Tuple
# import json

# from research_analyst.core.models import Answer, Evidence, QualityMetrics
# from research_analyst.core.exceptions import EvaluationError, LLMError
# from research_analyst.config import get_settings, prompts
# from research_analyst.utils.logger import get_logger
# from research_analyst.utils.helpers import calculate_similarity
# from research_analyst.utils.llm_client import get_llm_client


# logger = get_logger()


# class QualityEvaluator:
#     """Evaluate answer quality."""
    
#     def __init__(self):
#         """Initialize quality evaluator."""
#         self.settings = get_settings()
#         self.logger = get_logger()
        
#         # Use unified LLM client (Groq)
#         self.llm_client = get_llm_client()
    
#     def evaluate(self, answer: Answer, evidence: Evidence) -> QualityMetrics:
#         """
#         Evaluate answer quality.
        
#         Args:
#             answer: Generated answer
#             evidence: Evidence used for generation
            
#         Returns:
#             QualityMetrics object
#         """
#         self.logger.info(
#             "Evaluating answer quality",
#             answer_id=answer.answer_id
#         )
        
#         try:
#             # Calculate individual metrics
#             citation_coverage = self._calculate_citation_coverage(answer, evidence)
#             grounding_score = self._calculate_grounding_score(answer, evidence)
#             coherence_score = self._calculate_coherence_score(answer)
#             answer_completeness = self._calculate_completeness(answer, evidence)
#             source_diversity = self._calculate_source_diversity(answer, evidence)
            
#             # Determine if passes thresholds
#             passes_threshold = (
#                 citation_coverage >= self.settings.min_citation_coverage and
#                 answer.confidence >= self.settings.min_confidence_threshold and
#                 grounding_score >= 0.6  # Additional grounding threshold
#             )
            
#             # Identify issues
#             issues = self._identify_issues(
#                 citation_coverage,
#                 grounding_score,
#                 coherence_score,
#                 answer_completeness,
#                 answer.confidence
#             )
            
#             # Create metrics object
#             metrics = QualityMetrics(
#                 citation_coverage=citation_coverage,
#                 grounding_score=grounding_score,
#                 coherence_score=coherence_score,
#                 answer_completeness=answer_completeness,
#                 source_diversity=source_diversity,
#                 passes_threshold=passes_threshold,
#                 issues=issues
#             )
            
#             self.logger.log_quality_metrics(
#                 query_id=answer.answer_id,
#                 metrics={
#                     'citation_coverage': citation_coverage,
#                     'grounding_score': grounding_score,
#                     'coherence_score': coherence_score,
#                     'completeness': answer_completeness,
#                     'diversity': source_diversity
#                 },
#                 passes_threshold=passes_threshold
#             )
            
#             return metrics
            
#         except Exception as e:
#             self.logger.error(
#                 "Quality evaluation failed",
#                 answer_id=answer.answer_id,
#                 error=str(e)
#             )
#             raise EvaluationError(
#                 f"Failed to evaluate quality: {str(e)}",
#                 details={"answer_id": answer.answer_id}
#             )
    
#     def _calculate_citation_coverage(
#         self,
#         answer: Answer,
#         evidence: Evidence
#     ) -> float:
#         """
#         Calculate citation coverage score.
        
#         Args:
#             answer: Answer object
#             evidence: Evidence object
            
#         Returns:
#             Coverage score 0.0-1.0
#         """
#         # Count citation markers in answer
#         citation_markers = re.findall(r'\[(\d+)\]', answer.text)
#         num_citations = len(set(citation_markers))
        
#         # Count sentences in answer
#         sentences = [s for s in answer.text.split('.') if len(s.strip()) > 10]
#         num_sentences = len(sentences)
        
#         if num_sentences == 0:
#             return 0.0
        
#         # Calculate coverage (citations per sentence)
#         coverage = min(1.0, num_citations / num_sentences)
        
#         # Penalize if no citations at all
#         if num_citations == 0:
#             return 0.0
        
#         return coverage
    
#     def _calculate_grounding_score(
#         self,
#         answer: Answer,
#         evidence: Evidence
#     ) -> float:
#         """
#         Calculate how well answer is grounded in evidence.
        
#         Args:
#             answer: Answer object
#             evidence: Evidence object
            
#         Returns:
#             Grounding score 0.0-1.0
#         """
#         if self.settings.mock_llm_calls:
#             # Simple heuristic for testing
#             return self._heuristic_grounding_score(answer, evidence)
        
#         # Use LLM to evaluate grounding
#         try:
#             # Format evidence summary
#             evidence_text = "\n".join([
#                 f"- {claim.text}" for claim in evidence.claims[:10]
#             ])
            
#             # Format prompt
#             prompt = prompts.format_prompt(
#                 prompts.GROUNDING_EVALUATION,
#                 answer=answer.text,
#                 evidence=evidence_text
#             )
            
#             # Use unified LLM client
#             response_text = self.llm_client.generate(
#                 prompt=prompt,
#                 system_prompt="You are an evaluation expert. Always respond with valid JSON.",
#                 max_tokens=500,
#                 temperature=0.1,
#                 json_mode=True
#             )
            
#             # Parse result
#             result = json.loads(response_text)
#             grounding_score = result.get('grounding_score', 0.5)
            
#             return grounding_score
            
#         except Exception as e:
#             self.logger.warning(
#                 "LLM grounding evaluation failed, using heuristic",
#                 error=str(e)
#             )
#             return self._heuristic_grounding_score(answer, evidence)
    
#     def _heuristic_grounding_score(
#         self,
#         answer: Answer,
#         evidence: Evidence
#     ) -> float:
#         """
#         Calculate grounding score using heuristics.
        
#         Args:
#             answer: Answer object
#             evidence: Evidence object
            
#         Returns:
#             Grounding score 0.0-1.0
#         """
#         # Calculate overlap between answer and claims
#         answer_words = set(answer.text.lower().split())
        
#         overlaps = []
#         for claim in evidence.claims[:10]:  # Top 10 claims
#             claim_words = set(claim.text.lower().split())
#             overlap = len(answer_words & claim_words) / len(claim_words) if claim_words else 0
#             overlaps.append(overlap)
        
#         # Average overlap
#         avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        
#         return min(1.0, avg_overlap * 2)  # Scale up
    
#     def _calculate_coherence_score(self, answer: Answer) -> float:
#         """
#         Calculate answer coherence score.
        
#         Args:
#             answer: Answer object
            
#         Returns:
#             Coherence score 0.0-1.0
#         """
#         score = 1.0
        
#         # Check for very short answers
#         if len(answer.text) < 50:
#             score -= 0.3
        
#         # Check for incomplete sentences
#         if answer.text.count('.') == 0:
#             score -= 0.2
        
#         # Check for excessive hedging
#         hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could be']
#         hedge_count = sum(answer.text.lower().count(word) for word in hedging_words)
#         if hedge_count > 3:
#             score -= 0.2
        
#         # Check for repetition (simple check)
#         words = answer.text.lower().split()
#         if len(words) > 10:
#             unique_ratio = len(set(words)) / len(words)
#             if unique_ratio < 0.5:  # Too much repetition
#                 score -= 0.3
        
#         return max(0.0, score)
    
#     def _calculate_completeness(
#         self,
#         answer: Answer,
#         evidence: Evidence
#     ) -> float:
#         """
#         Calculate answer completeness.
        
#         Args:
#             answer: Answer object
#             evidence: Evidence object
            
#         Returns:
#             Completeness score 0.0-1.0
#         """
#         # Check if answer addresses multiple aspects
        
#         # Count distinct claims referenced
#         citation_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', answer.text))
#         coverage_ratio = len(citation_indices) / min(len(evidence.supporting_documents), 5)
        
#         # Check length adequacy
#         word_count = len(answer.text.split())
#         if word_count < 50:
#             length_score = 0.3
#         elif word_count < 100:
#             length_score = 0.6
#         elif word_count < 200:
#             length_score = 0.8
#         else:
#             length_score = 1.0
        
#         # Combine scores
#         completeness = (coverage_ratio * 0.6 + length_score * 0.4)
        
#         return min(1.0, completeness)
    
#     def _calculate_source_diversity(
#         self,
#         answer: Answer,
#         evidence: Evidence
#     ) -> float:
#         """
#         Calculate source diversity score.
        
#         Args:
#             answer: Answer object
#             evidence: Evidence object
            
#         Returns:
#             Diversity score 0.0-1.0
#         """
#         if not evidence.supporting_documents:
#             return 0.0
        
#         # Count unique sources cited
#         citation_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', answer.text))
        
#         if not citation_indices:
#             return 0.0
        
#         # Get source types and domains
#         cited_docs = [
#             evidence.supporting_documents[i-1]
#             for i in citation_indices
#             if 0 <= i-1 < len(evidence.supporting_documents)
#         ]
        
#         if not cited_docs:
#             return 0.0
        
#         # Count unique source types
#         source_types = set(doc.source_type for doc in cited_docs)
#         type_diversity = len(source_types) / 5  # Normalize by max possible types
        
#         # Count unique domains
#         domains = set(doc.metadata.get('domain', '') for doc in cited_docs)
#         domain_diversity = min(1.0, len(domains) / 3)  # Expect at least 3 different domains for full score
        
#         # Combine
#         diversity = (type_diversity * 0.5 + domain_diversity * 0.5)
        
#         return min(1.0, diversity)
    
#     def _identify_issues(
#         self,
#         citation_coverage: float,
#         grounding_score: float,
#         coherence_score: float,
#         completeness: float,
#         confidence: float
#     ) -> List[str]:
#         """
#         Identify specific quality issues.
        
#         Args:
#             citation_coverage: Citation coverage score
#             grounding_score: Grounding score
#             coherence_score: Coherence score
#             completeness: Completeness score
#             confidence: Answer confidence
            
#         Returns:
#             List of issue descriptions
#         """
#         issues = []
        
#         if citation_coverage < 0.3:
#             issues.append("Insufficient citations - answer lacks source attribution")
        
#         if grounding_score < 0.5:
#             issues.append("Weak grounding - answer not well-supported by evidence")
        
#         if coherence_score < 0.7:
#             issues.append("Low coherence - answer may be unclear or repetitive")
        
#         if completeness < 0.5:
#             issues.append("Incomplete answer - may not fully address the query")
        
#         if confidence < 0.5:
#             issues.append("Low confidence - answer may be uncertain or speculative")
        
#         return issues
    
#     def evaluate_citation_accuracy(
#         self,
#         answer: Answer,
#         evidence: Evidence
#     ) -> Tuple[int, int]:
#         """
#         Evaluate citation accuracy.
        
#         Args:
#             answer: Answer object
#             evidence: Evidence object
            
#         Returns:
#             Tuple of (correct_citations, total_citations)
#         """
#         citation_markers = re.findall(r'\[(\d+)\]', answer.text)
#         total_citations = len(citation_markers)
        
#         # Check if each citation index is valid
#         correct_citations = 0
#         for marker in citation_markers:
#             idx = int(marker) - 1
#             if 0 <= idx < len(evidence.supporting_documents):
#                 correct_citations += 1
        
#         return correct_citations, total_citations
    
#     def generate_quality_report(
#         self,
#         answer: Answer,
#         evidence: Evidence,
#         metrics: QualityMetrics
#     ) -> str:
#         """
#         Generate human-readable quality report.
        
#         Args:
#             answer: Answer object
#             evidence: Evidence object
#             metrics: Quality metrics
            
#         Returns:
#             Quality report string
#         """
#         report_lines = [
#             "=" * 60,
#             "ANSWER QUALITY REPORT",
#             "=" * 60,
#             "",
#             f"Answer ID: {answer.answer_id}",
#             f"Query: {answer.query[:100]}...",
#             "",
#             "METRICS:",
#             f"  Citation Coverage:   {metrics.citation_coverage:.2%}",
#             f"  Grounding Score:     {metrics.grounding_score:.2%}",
#             f"  Coherence Score:     {metrics.coherence_score:.2%}",
#             f"  Completeness:        {metrics.answer_completeness:.2%}",
#             f"  Source Diversity:    {metrics.source_diversity:.2%}",
#             f"  Answer Confidence:   {answer.confidence:.2%}",
#             "",
#             f"OVERALL: {'✓ PASSED' if metrics.passes_threshold else '✗ FAILED'}",
#             ""
#         ]
        
#         if metrics.issues:
#             report_lines.extend([
#                 "ISSUES IDENTIFIED:",
#                 *[f"  • {issue}" for issue in metrics.issues],
#                 ""
#             ])
        
#         # Citation analysis
#         correct, total = self.evaluate_citation_accuracy(answer, evidence)
#         if total > 0:
#             report_lines.extend([
#                 "CITATIONS:",
#                 f"  Total citations: {total}",
#                 f"  Valid citations: {correct}",
#                 f"  Accuracy: {correct/total:.2%}",
#                 ""
#             ])
        
#         report_lines.append("=" * 60)
        
#         return "\n".join(report_lines)



"""
Quality metrics for the research analyst system.
Evaluates answer quality using multiple metrics.
"""

import re
from typing import Dict, List, Tuple
import json

from research_analyst.core.models import Answer, Evidence, QualityMetrics
from research_analyst.core.exceptions import EvaluationError, LLMError
from research_analyst.config import get_settings, prompts
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import calculate_similarity
from research_analyst.utils.llm_client import get_llm_client


logger = get_logger()


class QualityEvaluator:
    """Evaluate answer quality."""
    
    def __init__(self):
        """Initialize quality evaluator."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Use unified LLM client (Groq)
        self.llm_client = get_llm_client()
        
        # Initialize LLM Judge if enabled
        if self.settings.enable_llm_judge:
            from research_analyst.evaluation.llm_judge import LLMJudge
            self.llm_judge = LLMJudge()
        else:
            self.llm_judge = None
    
    def evaluate(self, answer: Answer, evidence: Evidence) -> QualityMetrics:
        """
        Evaluate answer quality.
        
        Args:
            answer: Generated answer
            evidence: Evidence used for generation
            
        Returns:
            QualityMetrics object
        """
        self.logger.info(
            "Evaluating answer quality",
            answer_id=answer.answer_id
        )
        
        try:
            # Calculate heuristic metrics
            citation_coverage = self._calculate_citation_coverage(answer, evidence)
            grounding_score = self._calculate_grounding_score(answer, evidence)
            coherence_score = self._calculate_coherence_score(answer)
            answer_completeness = self._calculate_completeness(answer, evidence)
            source_diversity = self._calculate_source_diversity(answer, evidence)
            
            # Get LLM Judge scores if enabled
            llm_judge_scores = None
            if self.llm_judge:
                try:
                    llm_judge_scores = self.llm_judge.evaluate_answer(
                        query=answer.query,
                        answer=answer,
                        evidence=evidence
                    )
                except Exception as e:
                    self.logger.warning(f"LLM Judge evaluation failed: {e}")
            
            # Create metrics object
            metrics = QualityMetrics(
                citation_coverage=citation_coverage,
                grounding_score=grounding_score,
                coherence_score=coherence_score,
                answer_completeness=answer_completeness,
                source_diversity=source_diversity,
                llm_judge_scores=llm_judge_scores,
                passes_threshold=False,  # Will be set below
                issues=[]
            )
            
            # Calculate composite score
            metrics.composite_score = metrics.calculate_composite_score()
            
            # Determine if passes thresholds
            passes_threshold = (
                citation_coverage >= self.settings.min_citation_coverage and
                answer.confidence >= self.settings.min_confidence_threshold and
                grounding_score >= 0.6  # Additional grounding threshold
            )
            
            # If LLM judge enabled, also check judge scores
            if llm_judge_scores:
                llm_passes = (
                    llm_judge_scores.grounding_score >= 0.6 and
                    llm_judge_scores.factuality_score >= 0.6
                )
                passes_threshold = passes_threshold and llm_passes
            
            metrics.passes_threshold = passes_threshold
            
            # Identify issues
            metrics.issues = self._identify_issues(
                citation_coverage,
                grounding_score,
                coherence_score,
                answer_completeness,
                answer.confidence,
                llm_judge_scores
            )
            
            self.logger.log_quality_metrics(
                query_id=answer.answer_id,
                metrics={
                    'citation_coverage': citation_coverage,
                    'grounding_score': grounding_score,
                    'coherence_score': coherence_score,
                    'completeness': answer_completeness,
                    'diversity': source_diversity,
                    'composite_score': metrics.composite_score,
                    'llm_grounding': llm_judge_scores.grounding_score if llm_judge_scores else None,
                    'llm_factuality': llm_judge_scores.factuality_score if llm_judge_scores else None
                },
                passes_threshold=passes_threshold
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(
                "Quality evaluation failed",
                answer_id=answer.answer_id,
                error=str(e)
            )
            raise EvaluationError(
                f"Failed to evaluate quality: {str(e)}",
                details={"answer_id": answer.answer_id}
            )
    
    def _calculate_citation_coverage(
        self,
        answer: Answer,
        evidence: Evidence
    ) -> float:
        """
        Calculate citation coverage score.
        
        Args:
            answer: Answer object
            evidence: Evidence object
            
        Returns:
            Coverage score 0.0-1.0
        """
        # Count sentences that contain at least one citation marker
        sentences = [s for s in answer.text.split('.') if len(s.strip()) > 10]
        num_sentences = len(sentences)

        if num_sentences == 0:
            return 0.0

        # Count sentences that contain a citation marker [N]
        cited_sentences = sum(
            1 for s in sentences
            if re.search(r'\[\d+\]', s)
        )

        # Penalize if no citations at all
        if cited_sentences == 0:
            return 0.0

        coverage = cited_sentences / num_sentences
        return min(1.0, coverage)
    
    def _calculate_grounding_score(
        self,
        answer: Answer,
        evidence: Evidence
    ) -> float:
        """
        Calculate how well answer is grounded in evidence.
        
        Args:
            answer: Answer object
            evidence: Evidence object
            
        Returns:
            Grounding score 0.0-1.0
        """
        if self.settings.mock_llm_calls:
            # Simple heuristic for testing
            return self._heuristic_grounding_score(answer, evidence)
        
        # Use LLM to evaluate grounding
        try:
            # Format evidence summary
            evidence_text = "\n".join([
                f"- {claim.text}" for claim in evidence.claims[:10]
            ])
            
            # Format prompt
            prompt = prompts.format_prompt(
                prompts.GROUNDING_EVALUATION,
                answer=answer.text,
                evidence=evidence_text
            )
            
            # Use unified LLM client
            response_text = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are an evaluation expert. Always respond with valid JSON.",
                max_tokens=500,
                temperature=0.1,
                json_mode=True
            )
            
            # Parse result
            result = json.loads(response_text)
            grounding_score = result.get('grounding_score', 0.5)
            
            return grounding_score
            
        except Exception as e:
            self.logger.warning(
                "LLM grounding evaluation failed, using heuristic",
                error=str(e)
            )
            return self._heuristic_grounding_score(answer, evidence)
    
    def _heuristic_grounding_score(
        self,
        answer: Answer,
        evidence: Evidence
    ) -> float:
        """
        Calculate grounding score using heuristics.
        
        Args:
            answer: Answer object
            evidence: Evidence object
            
        Returns:
            Grounding score 0.0-1.0
        """
        # Calculate overlap between answer and claims
        answer_words = set(answer.text.lower().split())
        
        overlaps = []
        for claim in evidence.claims[:10]:  # Top 10 claims
            claim_words = set(claim.text.lower().split())
            overlap = len(answer_words & claim_words) / len(claim_words) if claim_words else 0
            overlaps.append(overlap)
        
        # Average overlap
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        
        return min(1.0, avg_overlap * 2)  # Scale up
    
    def _calculate_coherence_score(self, answer: Answer) -> float:
        """
        Calculate answer coherence score.
        
        Args:
            answer: Answer object
            
        Returns:
            Coherence score 0.0-1.0
        """
        score = 1.0
        
        # Check for very short answers
        if len(answer.text) < 50:
            score -= 0.3
        
        # Check for incomplete sentences
        if answer.text.count('.') == 0:
            score -= 0.2
        
        # Check for excessive hedging
        hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could be']
        hedge_count = sum(answer.text.lower().count(word) for word in hedging_words)
        if hedge_count > 3:
            score -= 0.2
        
        # Check for repetition (simple check)
        words = answer.text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:  # Too much repetition
                score -= 0.3
        
        return max(0.0, score)
    
    def _calculate_completeness(
        self,
        answer: Answer,
        evidence: Evidence
    ) -> float:
        """
        Calculate answer completeness.
        
        Args:
            answer: Answer object
            evidence: Evidence object
            
        Returns:
            Completeness score 0.0-1.0
        """
        # Check if answer addresses multiple aspects
        
        # Count distinct claims referenced
        citation_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', answer.text))
        coverage_ratio = len(citation_indices) / min(len(evidence.supporting_documents), 5)
        
        # Check length adequacy
        word_count = len(answer.text.split())
        if word_count < 50:
            length_score = 0.3
        elif word_count < 100:
            length_score = 0.6
        elif word_count < 200:
            length_score = 0.8
        else:
            length_score = 1.0
        
        # Combine scores
        completeness = (coverage_ratio * 0.6 + length_score * 0.4)
        
        return min(1.0, completeness)
    
    def _calculate_source_diversity(
        self,
        answer: Answer,
        evidence: Evidence
    ) -> float:
        """
        Calculate source diversity score.
        
        Args:
            answer: Answer object
            evidence: Evidence object
            
        Returns:
            Diversity score 0.0-1.0
        """
        if not evidence.supporting_documents:
            return 0.0
        
        # Count unique sources cited
        citation_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', answer.text))
        
        if not citation_indices:
            return 0.0
        
        # Get source types and domains
        cited_docs = [
            evidence.supporting_documents[i-1]
            for i in citation_indices
            if 0 <= i-1 < len(evidence.supporting_documents)
        ]
        
        if not cited_docs:
            return 0.0
        
        # Count unique source types
        source_types = set(doc.source_type for doc in cited_docs)
        type_diversity = len(source_types) / 5  # Normalize by max possible types
        
        # Count unique domains
        domains = set(doc.metadata.get('domain', '') for doc in cited_docs)
        domain_diversity = min(1.0, len(domains) / 3)  # Expect at least 3 different domains for full score
        
        # Combine
        diversity = (type_diversity * 0.5 + domain_diversity * 0.5)
        
        return min(1.0, diversity)
    
    def _identify_issues(
        self,
        citation_coverage: float,
        grounding_score: float,
        coherence_score: float,
        completeness: float,
        confidence: float,
        llm_judge_scores=None
    ) -> List[str]:
        """
        Identify specific quality issues.
        
        Args:
            citation_coverage: Citation coverage score
            grounding_score: Grounding score
            coherence_score: Coherence score
            completeness: Completeness score
            confidence: Answer confidence
            llm_judge_scores: Optional LLM judge scores
            
        Returns:
            List of issue descriptions
        """
        issues = []
        
        # Heuristic issues
        if citation_coverage < 0.3:
            issues.append("Insufficient citations - answer lacks source attribution")
        
        if grounding_score < 0.5:
            issues.append("Weak grounding - answer not well-supported by evidence")
        
        if coherence_score < 0.7:
            issues.append("Low coherence - answer may be unclear or repetitive")
        
        if completeness < 0.5:
            issues.append("Incomplete answer - may not fully address the query")
        
        if confidence < 0.5:
            issues.append("Low confidence - answer may be uncertain or speculative")
        
        # LLM Judge issues
        if llm_judge_scores:
            if llm_judge_scores.grounding_score < 0.6:
                issues.append("LLM Judge: Weak grounding in evidence")
            
            if llm_judge_scores.factuality_score < 0.6:
                issues.append("LLM Judge: Factual accuracy concerns")
            
            if llm_judge_scores.relevance_score < 0.6:
                issues.append("LLM Judge: Answer relevance to query questionable")
            
            if llm_judge_scores.completeness_score < 0.6:
                issues.append("LLM Judge: Answer incomplete")
            
            # Add specific issues found by judge
            issues.extend(llm_judge_scores.issues_found)
        
        return issues
    
    def evaluate_citation_accuracy(
        self,
        answer: Answer,
        evidence: Evidence
    ) -> Tuple[int, int]:
        """
        Evaluate citation accuracy.
        
        Args:
            answer: Answer object
            evidence: Evidence object
            
        Returns:
            Tuple of (correct_citations, total_citations)
        """
        citation_markers = re.findall(r'\[(\d+)\]', answer.text)
        total_citations = len(citation_markers)
        
        # Check if each citation index is valid
        correct_citations = 0
        for marker in citation_markers:
            idx = int(marker) - 1
            if 0 <= idx < len(evidence.supporting_documents):
                correct_citations += 1
        
        return correct_citations, total_citations
    
    def generate_quality_report(
        self,
        answer: Answer,
        evidence: Evidence,
        metrics: QualityMetrics
    ) -> str:
        """
        Generate human-readable quality report.
        
        Args:
            answer: Answer object
            evidence: Evidence object
            metrics: Quality metrics
            
        Returns:
            Quality report string
        """
        report_lines = [
            "=" * 60,
            "ANSWER QUALITY REPORT",
            "=" * 60,
            "",
            f"Answer ID: {answer.answer_id}",
            f"Query: {answer.query[:100]}...",
            "",
            "METRICS:",
            f"  Citation Coverage:   {metrics.citation_coverage:.2%}",
            f"  Grounding Score:     {metrics.grounding_score:.2%}",
            f"  Coherence Score:     {metrics.coherence_score:.2%}",
            f"  Completeness:        {metrics.answer_completeness:.2%}",
            f"  Source Diversity:    {metrics.source_diversity:.2%}",
            f"  Answer Confidence:   {answer.confidence:.2%}",
            "",
            f"OVERALL: {'✓ PASSED' if metrics.passes_threshold else '✗ FAILED'}",
            ""
        ]
        
        if metrics.issues:
            report_lines.extend([
                "ISSUES IDENTIFIED:",
                *[f"  • {issue}" for issue in metrics.issues],
                ""
            ])
        
        # Citation analysis
        correct, total = self.evaluate_citation_accuracy(answer, evidence)
        if total > 0:
            report_lines.extend([
                "CITATIONS:",
                f"  Total citations: {total}",
                f"  Valid citations: {correct}",
                f"  Accuracy: {correct/total:.2%}",
                ""
            ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)