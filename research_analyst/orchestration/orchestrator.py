"""
Orchestrator for the research analyst system.
Main pipeline coordinator that ties all components together.

Pipeline phases:
    1.  Input guardrails
    2.  Query processing (normalize + classify)
    3.  Routing
    4.  Retrieval (+ HyDE augmented ranking + CRAG document filtering)
    5.  Graph construction + temporal decay + graph scoring (RESEARCH path)
    6.  Context building (simple on FAST, full LLM on RESEARCH)
    7.  Memory context injection
    8.  Answer generation (Speculative RAG on RESEARCH path)
    9.  Uncertainty quantification
    10. Quality evaluation (parallelised LLM Judge + G-Eval)
    11. Self-healing (RESEARCH path only)
    12. Output guardrails
    13. Regression detection + evaluation storage
    14. Memory session save
"""

import time
from typing import Optional, Tuple, List
from datetime import datetime

from research_analyst.core.models import (
    Query, NormalizedQuery, Answer, Evidence,
    PipelineResponse, PipelineState, ExecutionPath,
    QualityMetrics, EvaluationRecord, RegressionAlert,
)
from research_analyst.core.exceptions import ResearchAnalystException, MaxRetriesExceeded
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id

from research_analyst.guardrails import InputGuardrails, OutputGuardrails
from research_analyst.query_processing import QueryNormalizer, IntentClassifier
from research_analyst.routing import Router
from research_analyst.retrieval import WebSearch, DocumentProcessor, VectorStore, Ranker
from research_analyst.graph import (
    EntityExtractor, RelationshipExtractor,
    GraphBuilder, GraphStore, GraphQuerier,
)
from research_analyst.synthesis import ContextBuilder, AnswerGenerator
from research_analyst.evaluation import (
    QualityEvaluator, SelfHealer, EvaluationStore, RegressionDetector,
)
from research_analyst.caching import CacheManager
from research_analyst.orchestration import AsyncExecutor

from research_analyst.speculative_rag import SpeculativeRAG
from research_analyst.memory_manager import MemoryManager
from research_analyst.temporal_decay import TemporalDecayScorer
from research_analyst.graph_scorer import GraphScorer
from research_analyst.uncertainty_quantifier import UncertaintyQuantifier
from research_analyst.evaluation.llm_judge import LLMJudge
from research_analyst.hyde import HyDE
from research_analyst.crag import CRAG

logger = get_logger()


class ResearchAnalyst:
    """Main orchestrator for the research analyst system."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self._init_components()
        self.logger.info("Research Analyst initialised")

    def _init_components(self):
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.query_normalizer = QueryNormalizer()
        self.intent_classifier = IntentClassifier()
        self.router = Router()
        self.web_search = WebSearch()
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.ranker = Ranker()
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.graph_builder = GraphBuilder()
        self.graph_store = GraphStore()
        self.graph_querier = GraphQuerier(self.graph_store)
        self.context_builder = ContextBuilder()
        self.answer_generator = AnswerGenerator()
        self.quality_evaluator = QualityEvaluator()
        self.self_healer = SelfHealer()
        self.llm_judge = LLMJudge()

        if self.settings.enable_regression_detection:
            self.evaluation_store = EvaluationStore()
            self.regression_detector = RegressionDetector()
            self._update_baselines()
        else:
            self.evaluation_store = None
            self.regression_detector = None

        self.cache_manager = CacheManager()
        self.async_executor = AsyncExecutor()
        self.spec_rag = SpeculativeRAG()
        self.memory_manager = MemoryManager()
        self.temporal_decay_scorer = TemporalDecayScorer()
        self.graph_scorer = GraphScorer()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.hyde = HyDE()
        self.crag = CRAG()

        if getattr(self.settings, "cache_warm_on_startup", False) and self.evaluation_store:
            self._warm_cache_on_startup()

    def query(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PipelineResponse:
        start_time = time.time()
        query_id = generate_id("query")

        query = Query(
            text=query_text,
            user_id=user_id,
            session_id=session_id,
            metadata={"query_id": query_id, "use_cache": use_cache},
        )

        self.logger.log_query(query_text, query_id, user_id)
        state = PipelineState(query=query)

        try:
            if use_cache:
                cached_response = self.cache_manager.get_pipeline_response(query)
                if cached_response:
                    self.logger.info("Returning cached response", query_id=query_id)
                    return cached_response

            answer, execution_path, quality_metrics, cost = self._execute_pipeline(state)
            total_time = (time.time() - start_time) * 1000

            response = PipelineResponse(
                success=True,
                answer=answer,
                execution_path=execution_path,
                quality_metrics=quality_metrics,
                execution_time_ms=total_time,
                cost_estimate=cost,
                metadata={
                    "query_id": query_id,
                    "self_healing_attempts": len(state.self_healing_actions),
                    "winning_strategy": state.metadata.get("winning_strategy") if state.metadata else None,
                    "speculation_scores": state.metadata.get("speculation_scores") if state.metadata else None,
                    "memory_context_used": bool(state.metadata and state.metadata.get("memory_context")),
                    "hyde_used": bool(state.metadata and state.metadata.get("hyde_used")),
                    "crag_filtered": state.metadata.get("crag_filtered", 0) if state.metadata else 0,
                },
            )

            if use_cache and self.router.should_cache(state.normalized_query, state.routing_decision):
                ttl = self.router.get_cache_ttl(state.normalized_query)
                self.cache_manager.cache_pipeline_response(query, response, ttl)

            self.logger.log_pipeline_completion(
                query_id=query_id, execution_path=execution_path,
                total_time_ms=total_time, cost_estimate=cost, success=True,
            )
            return response

        except MaxRetriesExceeded as e:
            self.logger.error("Max retries exceeded", query_id=query_id, error=str(e))
            return self._error_response(query, e, time.time() - start_time)
        except ResearchAnalystException as e:
            self.logger.error("Pipeline failed", query_id=query_id, error=str(e))
            return self._error_response(query, e, time.time() - start_time)

    def _execute_pipeline(self, state: PipelineState) -> Tuple[Answer, ExecutionPath, QualityMetrics, float]:
        start = time.time()

        # Phase 1: Input guardrails
        state.query = self.input_guardrails.validate_and_sanitize(state.query)

        # Phase 2: Query processing
        state.normalized_query = self.query_normalizer.normalize(state.query)
        state.normalized_query = self.intent_classifier.classify(state.normalized_query)
        state.query.metadata["intent"] = (
            state.normalized_query.intent.value
            if hasattr(state.normalized_query.intent, "value")
            else str(state.normalized_query.intent)
        )

        # Phase 3: Routing
        state.routing_decision = self.router.route(state.normalized_query)
        execution_path = state.routing_decision.execution_path

        # Phase 4: Retrieval + HyDE + CRAG
        state = self._execute_retrieval(
            state,
            skip_expansion=(execution_path == ExecutionPath.FAST),
        )

        # Phase 5: Graph + temporal decay + graph scoring (RESEARCH only)
        if execution_path == ExecutionPath.RESEARCH:
            state = self._execute_graph_processing(state)

            if state.relevant_subgraph and self.temporal_decay_scorer.enabled:
                doc_map = {doc.doc_id: doc for doc in state.retrieved_documents}
                state.relevant_subgraph = self.temporal_decay_scorer.apply_decay(
                    subgraph=state.relevant_subgraph, document_map=doc_map,
                )

            if state.relevant_subgraph:
                state.relevant_subgraph = self.graph_scorer.score_and_rank(
                    subgraph=state.relevant_subgraph,
                    normalized_query=state.normalized_query,
                )
                state.relevant_subgraph = self.graph_scorer.score_relationships_by_centrality(
                    state.relevant_subgraph
                )

        # Phase 6: Context building
        if execution_path == ExecutionPath.FAST:
            state.evidence = self.context_builder.build_simple_context(state.ranked_documents)
        else:
            state.evidence = self.context_builder.build_context(
                state.ranked_documents, state.relevant_subgraph,
            )

        # Phase 7: Memory context injection
        if self.memory_manager.enabled:
            memory_ctx = self.memory_manager.retrieve_context(
                query_text=state.query.text,
                entities=state.normalized_query.entities_mentioned,
                session_id=state.query.session_id,
            )
            if not memory_ctx.is_empty:
                state.evidence.summary = memory_ctx.inject_into_evidence_summary(state.evidence.summary)
                if state.metadata is None:
                    state.metadata = {}
                state.metadata["memory_context"] = memory_ctx.to_dict()

        # Phase 8: Answer generation
        if execution_path == ExecutionPath.RESEARCH and self.spec_rag.is_enabled():
            winner, all_candidates = self.spec_rag.run(
                query=state.query,
                normalized_query=state.normalized_query,
                base_evidence=state.evidence,
                ranked_docs=state.ranked_documents,
                subgraph=state.relevant_subgraph,
                answer_generator=self.answer_generator,
            )
            state.answer = winner.answer
            state.evidence = winner.evidence
            if state.metadata is None:
                state.metadata = {}
            state.metadata["speculation_candidates"] = [c.to_dict() for c in all_candidates]
            state.metadata["winning_strategy"] = winner.strategy.value
            state.metadata["speculation_scores"] = {
                c.strategy.value: c.composite_judge_score for c in all_candidates
            }
        else:
            state.answer = self.answer_generator.generate_answer(
                state.query, state.normalized_query, state.evidence, execution_path,
            )

        # Phase 9: Uncertainty quantification (first pass)
        if state.answer and state.evidence:
            state.answer, state.evidence = self.uncertainty_quantifier.quantify(
                answer=state.answer, evidence=state.evidence, llm_judge_scores=None,
            )

        # Phase 10: Quality evaluation
        state.quality_metrics = self.quality_evaluator.evaluate(state.answer, state.evidence)

        if self.settings.enable_llm_judge:
            llm_scores = self.llm_judge.evaluate_answer(
                query=state.query.text, answer=state.answer, evidence=state.evidence,
            )
            state.quality_metrics.llm_judge_scores = llm_scores

            if state.answer and state.evidence:
                state.answer, state.evidence = self.uncertainty_quantifier.quantify(
                    answer=state.answer, evidence=state.evidence, llm_judge_scores=llm_scores,
                )

            # G-Eval chain-of-thought scoring
            try:
                geval_scores = self.answer_generator.geval_score(
                    query=state.query.text,
                    answer_text=state.answer.text,
                    evidence_summary=state.evidence.summary or "",
                )
                if state.metadata is None:
                    state.metadata = {}
                state.metadata["geval"] = geval_scores
                state.quality_metrics.geval_composite = geval_scores.get("composite", 0.6)
                self.logger.info(
                    "G-Eval scoring complete",
                    composite=geval_scores.get("composite"),
                    coherence=geval_scores.get("coherence"),
                    consistency=geval_scores.get("consistency"),
                )
            except Exception as e:
                self.logger.warning("G-Eval failed, skipping", error=str(e))

        # Composite score — blend G-Eval at 20% weight if available
        state.quality_metrics.composite_score = state.quality_metrics.calculate_composite_score()
        geval_composite = getattr(state.quality_metrics, "geval_composite", None)
        if geval_composite is not None:
            state.quality_metrics.composite_score = round(
                state.quality_metrics.composite_score * 0.8 + geval_composite * 0.2, 4,
            )

        # Phase 10.5: Regression detection
        regression_alerts: List[RegressionAlert] = []
        if self.regression_detector:
            regression_alerts = self._detect_regressions(
                state.quality_metrics, state.query.metadata.get("query_id"),
            )

        # Phase 11: Self-healing (RESEARCH path only)
        should_heal, reason = self.self_healer.should_trigger_healing(
            state.quality_metrics, state.answer,
        )
        if should_heal and execution_path == ExecutionPath.RESEARCH:
            state = self._execute_self_healing(state, regression_alerts)

        # Phase 12: Output guardrails
        validated_answer, formatted_text = self.output_guardrails.validate_and_format(
            state.answer, state.quality_metrics,
        )
        state.answer = validated_answer

        # Phase 13: Store evaluation
        if self.evaluation_store:
            self._store_evaluation(state, regression_alerts)

        # Phase 14: Save session to memory
        if self.memory_manager.enabled:
            self.memory_manager.save_session(state)

        elapsed_ms = (time.time() - start) * 1000
        state.execution_time_ms = elapsed_ms
        cost_estimate = state.routing_decision.estimated_cost

        return (state.answer, state.routing_decision.execution_path, state.quality_metrics, cost_estimate)

    def _execute_retrieval(
        self,
        state: PipelineState,
        max_results_override: Optional[int] = None,
        skip_expansion: bool = False,
    ) -> PipelineState:
        max_results = max_results_override or self.settings.max_search_results

        documents = self.web_search.search_with_expansion(
            state.normalized_query,
            max_total_results=max_results,
            skip_expansion=skip_expansion,
        )
        state.retrieved_documents = documents

        processed_docs, chunks = self.document_processor.process_documents(
            documents, fetch_content=True, create_chunks=True,
        )
        state.retrieved_documents = processed_docs

        if chunks:
            chunks_with_embeddings = self.vector_store.embed_chunks(chunks)
            self.vector_store.build_index(chunks_with_embeddings)

        # HyDE: augment ranking query with hypothetical document (RESEARCH only)
        ranking_query = state.normalized_query.normalized_text
        if self.hyde.is_enabled() and not skip_expansion:
            try:
                ranking_query = self.hyde.augment_query(
                    query_text=state.normalized_query.normalized_text,
                    intent=state.normalized_query.intent,
                    domain=state.normalized_query.domain,
                )
                if state.metadata is None:
                    state.metadata = {}
                state.metadata["hyde_used"] = (ranking_query != state.normalized_query.normalized_text)
            except Exception as e:
                self.logger.warning("HyDE augmentation failed", error=str(e))

        state.ranked_documents = self.ranker.rank_documents(processed_docs, ranking_query)

        # CRAG: filter documents before context building (RESEARCH only)
        if self.crag.is_enabled() and not skip_expansion:
            try:
                before_count = len(state.ranked_documents)
                state.ranked_documents = self.crag.filter_documents(
                    query=state.query.text,
                    ranked_documents=state.ranked_documents,
                    normalized_query=state.normalized_query,
                )
                if state.metadata is None:
                    state.metadata = {}
                state.metadata["crag_filtered"] = before_count - len(state.ranked_documents)
            except Exception as e:
                self.logger.warning("CRAG filtering failed, using all documents", error=str(e))

        return state

    def _execute_graph_processing(self, state: PipelineState) -> PipelineState:
        top_docs = [rd.document for rd in state.ranked_documents[:8]]
        entities = self.entity_extractor.extract_from_multiple_documents(top_docs, use_llm=False)
        relationships = self.relationship_extractor.extract_from_multiple_documents(
            top_docs, entities, use_llm=not self.settings.mock_llm_calls,
        )
        knowledge_graph = self.graph_builder.build_graph(entities, relationships)
        state.knowledge_graph = knowledge_graph
        self.graph_store.store_graph(knowledge_graph)
        state.relevant_subgraph = self.graph_querier.query_for_normalized_query(state.normalized_query)
        return state

    def _execute_self_healing(
        self,
        state: PipelineState,
        regression_alerts: Optional[List[RegressionAlert]] = None,
    ) -> PipelineState:
        max_attempts = self.settings.max_self_healing_attempts

        for attempt in range(max_attempts):
            try:
                improved_answer, actions = self.self_healer.heal(
                    state.query, state.normalized_query, state.answer,
                    state.evidence, state.quality_metrics, current_attempt=attempt,
                )
                state.self_healing_actions.extend(actions)

                if improved_answer:
                    new_metrics = self.quality_evaluator.evaluate(improved_answer, state.evidence)
                    state.answer = improved_answer
                    state.quality_metrics = new_metrics
                    if new_metrics.passes_threshold:
                        self.logger.info("Self-healing successful", attempt=attempt + 1)
                        break
                    else:
                        self.logger.info("Self-healing improved but still below threshold", attempt=attempt + 1)
                else:
                    self.logger.info("Self-healing: re-executing retrieval with expanded results", attempt=attempt + 1)
                    expanded_max = min(self.settings.max_search_results + 5, 20)
                    state = self._execute_retrieval(state, max_results_override=expanded_max, skip_expansion=False)
                    state.evidence = self.context_builder.build_context(state.ranked_documents, state.relevant_subgraph)
                    state.answer = self.answer_generator.generate_answer(
                        state.query, state.normalized_query, state.evidence,
                        state.routing_decision.execution_path,
                    )
                    new_metrics = self.quality_evaluator.evaluate(state.answer, state.evidence)
                    state.quality_metrics = new_metrics

                    if new_metrics.passes_threshold:
                        self.logger.info("Self-healing with re-execution successful", attempt=attempt + 1)
                        break
                    else:
                        self.logger.info("Self-healing with re-execution still insufficient", attempt=attempt + 1)

            except MaxRetriesExceeded:
                raise
            except Exception as e:
                self.logger.warning("Self-healing attempt failed", attempt=attempt + 1, error=str(e))
                break

        return state

    def _detect_regressions(self, quality_metrics: QualityMetrics, query_id: str) -> List[RegressionAlert]:
        if not self.regression_detector or not hasattr(self, "baselines"):
            return []
        try:
            alerts = self.regression_detector.detect_regression(
                current_metrics=quality_metrics, baselines=self.baselines, query_id=query_id,
            )
            if alerts:
                self.logger.warning(f"Regression detected: {len(alerts)} metrics below baseline", query_id=query_id)
                report = self.regression_detector.generate_regression_report(alerts)
                self.logger.info(f"Regression Report:\n{report}")
            return alerts
        except Exception as e:
            self.logger.error(f"Regression detection failed: {e}")
            return []

    def _store_evaluation(self, state: PipelineState, regression_alerts: List[RegressionAlert]) -> None:
        if not self.evaluation_store:
            return
        try:
            record = EvaluationRecord(
                evaluation_id=generate_id("eval"),
                query_id=state.query.metadata.get("query_id", "unknown"),
                query_text=state.query.text,
                answer_id=state.answer.answer_id,
                quality_metrics=state.quality_metrics,
                execution_path=state.routing_decision.execution_path,
                execution_time_ms=state.execution_time_ms or 0.0,
                cost_estimate=state.cost_estimate or 0.0,
                self_healing_triggered=len(state.self_healing_actions) > 0,
                self_healing_attempts=len(state.self_healing_actions),
            )
            self.evaluation_store.save_evaluation(record)
            if self.regression_detector:
                import random
                if random.random() < 0.1:
                    self._update_baselines()
        except Exception as e:
            self.logger.error(f"Failed to store evaluation: {e}")

    def _update_baselines(self) -> None:
        if not self.evaluation_store or not self.regression_detector:
            return
        try:
            recent = self.evaluation_store.get_recent_evaluations(limit=self.settings.baseline_window_size)
            if len(recent) < self.settings.min_baseline_samples:
                self.logger.info(f"Insufficient samples for baseline ({len(recent)} < {self.settings.min_baseline_samples})")
                self.baselines = {}
                return
            self.baselines = self.regression_detector.calculate_all_baselines(recent)
            for metric_name, baseline in self.baselines.items():
                self.evaluation_store.save_baseline(baseline)
            self.logger.info(f"Updated {len(self.baselines)} baselines from {len(recent)} evaluations")
        except Exception as e:
            self.logger.error(f"Baseline update failed: {e}")
            self.baselines = {}

    def _warm_cache_on_startup(self) -> None:
        try:
            from collections import Counter
            top_n = self.settings.cache_warm_top_n
            recent = self.evaluation_store.get_recent_evaluations(limit=200)
            query_counts = Counter(r.query_text for r in recent)
            top_queries = [q for q, _ in query_counts.most_common(top_n)]
            if top_queries:
                self.cache_manager.warm_cache(top_queries)
                self.logger.info("Cache pre-warmed on startup", num_queries=len(top_queries))
        except Exception as e:
            self.logger.warning("Cache warm on startup failed", error=str(e))

    def _error_response(self, query: Query, exception: Exception, elapsed_seconds: float) -> PipelineResponse:
        from research_analyst.core.exceptions import get_error_details
        error_details = get_error_details(exception)
        return PipelineResponse(
            success=False, answer=None, error=error_details["message"],
            execution_time_ms=elapsed_seconds * 1000, cost_estimate=0.0,
            metadata={"query_id": query.metadata.get("query_id"), "error_details": error_details},
        )

    def clear_caches(self):
        self.cache_manager.clear()
        self.graph_store.clear()
        self.logger.info("All caches cleared")

    def get_stats(self) -> dict:
        stats = {
            "cache": self.cache_manager.get_cache_stats(),
            "graph": self.graph_store.get_graph_stats(),
            "memory": self.memory_manager.get_stats(),
        }
        if self.evaluation_store:
            stats["evaluation"] = self.evaluation_store.get_stats()
            stats["baselines"] = (
                {name: {"mean": b.mean_value, "std_dev": b.std_dev, "samples": b.sample_size}
                 for name, b in self.baselines.items()}
                if hasattr(self, "baselines") else {}
            )
        return stats