"""
Persistent storage for evaluation history.
Supports SQLite and JSON backends for regression detection.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from pathlib import Path

from research_analyst.core.models import (
    EvaluationRecord,
    PerformanceBaseline,
    QualityMetrics,
    ExecutionPath
)
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id


logger = get_logger()


class EvaluationStore:
    """Persistent storage for evaluation history."""
    
    def __init__(self):
        """Initialize evaluation store."""
        self.settings = get_settings()
        self.logger = get_logger()
        self.backend = self.settings.evaluation_storage_backend
        
        if self.backend == "sqlite":
            self._init_sqlite()
        elif self.backend == "json":
            self._init_json()
        
        self.logger.info(f"Initialized evaluation store ({self.backend})")
    
    def _init_sqlite(self):
        """Initialize SQLite database."""
        db_path = Path(self.settings.evaluation_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                answer_id TEXT NOT NULL,
                
                -- Quality metrics
                citation_coverage REAL,
                grounding_score REAL,
                coherence_score REAL,
                answer_completeness REAL,
                source_diversity REAL,
                composite_score REAL,
                
                -- LLM Judge scores (optional)
                llm_grounding_score REAL,
                llm_factuality_score REAL,
                llm_relevance_score REAL,
                llm_completeness_score REAL,
                
                -- Metadata
                execution_path TEXT,
                execution_time_ms REAL,
                cost_estimate REAL,
                self_healing_triggered INTEGER,
                self_healing_attempts INTEGER,
                timestamp TEXT NOT NULL,
                
                -- Full JSON for complex data
                quality_metrics_json TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                baseline_id TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                mean_value REAL NOT NULL,
                std_dev REAL NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                calculated_at TEXT NOT NULL
            )
        """)
        
        # Create indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluations(timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_query_id ON evaluations(query_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metric_name ON baselines(metric_name)"
        )
        
        self.conn.commit()
    
    def _init_json(self):
        """Initialize JSON file storage."""
        json_path = Path(self.settings.evaluation_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.json_path = json_path
        
        # Initialize file if doesn't exist
        if not json_path.exists():
            with open(json_path, 'w') as f:
                json.dump({
                    'evaluations': [],
                    'baselines': []
                }, f)
    
    def save_evaluation(self, record: EvaluationRecord) -> None:
        """
        Save evaluation record.
        
        Args:
            record: Evaluation record to save
        """
        if self.backend == "sqlite":
            self._save_evaluation_sqlite(record)
        else:
            self._save_evaluation_json(record)
        
        self.logger.info(
            "Evaluation saved",
            evaluation_id=record.evaluation_id,
            composite_score=record.quality_metrics.composite_score
        )
    
    def _save_evaluation_sqlite(self, record: EvaluationRecord) -> None:
        """Save to SQLite."""
        llm_scores = record.quality_metrics.llm_judge_scores
        
        self.conn.execute("""
            INSERT INTO evaluations VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?
            )
        """, (
            record.evaluation_id,
            record.query_id,
            record.query_text,
            record.answer_id,
            # Heuristic metrics
            record.quality_metrics.citation_coverage,
            record.quality_metrics.grounding_score,
            record.quality_metrics.coherence_score,
            record.quality_metrics.answer_completeness,
            record.quality_metrics.source_diversity,
            record.quality_metrics.composite_score,
            # LLM Judge scores
            llm_scores.grounding_score if llm_scores else None,
            llm_scores.factuality_score if llm_scores else None,
            llm_scores.relevance_score if llm_scores else None,
            llm_scores.completeness_score if llm_scores else None,
            # Metadata
            record.execution_path,
            record.execution_time_ms,
            record.cost_estimate,
            int(record.self_healing_triggered),
            record.self_healing_attempts,
            record.timestamp.isoformat(),
            # Full JSON
            record.quality_metrics.model_dump_json()
        ))
        
        self.conn.commit()
    
    def _save_evaluation_json(self, record: EvaluationRecord) -> None:
        """Save to JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        data['evaluations'].append(record.model_dump(mode='json'))
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_recent_evaluations(
        self,
        limit: int = 100,
        metric_name: Optional[str] = None
    ) -> List[EvaluationRecord]:
        """
        Get recent evaluation records.
        
        Args:
            limit: Number of records to retrieve
            metric_name: Optional filter by specific metric
            
        Returns:
            List of evaluation records
        """
        if self.backend == "sqlite":
            return self._get_recent_sqlite(limit, metric_name)
        else:
            return self._get_recent_json(limit, metric_name)
    
    def _get_recent_sqlite(
        self,
        limit: int,
        metric_name: Optional[str]
    ) -> List[EvaluationRecord]:
        """Get recent from SQLite."""
        cursor = self.conn.execute("""
            SELECT * FROM evaluations
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        records = []
        for row in cursor.fetchall():
            # Reconstruct QualityMetrics
            quality_metrics_dict = json.loads(row['quality_metrics_json'])
            quality_metrics = QualityMetrics(**quality_metrics_dict)
            
            record = EvaluationRecord(
                evaluation_id=row['evaluation_id'],
                query_id=row['query_id'],
                query_text=row['query_text'],
                answer_id=row['answer_id'],
                quality_metrics=quality_metrics,
                execution_path=ExecutionPath(row['execution_path']),
                execution_time_ms=row['execution_time_ms'],
                cost_estimate=row['cost_estimate'],
                self_healing_triggered=bool(row['self_healing_triggered']),
                self_healing_attempts=row['self_healing_attempts'],
                timestamp=datetime.fromisoformat(row['timestamp'])
            )
            records.append(record)
        
        return records
    
    def _get_recent_json(
        self,
        limit: int,
        metric_name: Optional[str]
    ) -> List[EvaluationRecord]:
        """Get recent from JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        evaluations = data.get('evaluations', [])
        # Sort by timestamp descending
        evaluations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        records = []
        for eval_dict in evaluations[:limit]:
            records.append(EvaluationRecord(**eval_dict))
        
        return records
    
    def save_baseline(self, baseline: PerformanceBaseline) -> None:
        """
        Save performance baseline.
        
        Args:
            baseline: Baseline to save
        """
        if self.backend == "sqlite":
            self._save_baseline_sqlite(baseline)
        else:
            self._save_baseline_json(baseline)
        
        self.logger.info(
            "Baseline saved",
            metric=baseline.metric_name,
            mean=baseline.mean_value
        )
    
    def _save_baseline_sqlite(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to SQLite."""
        # Delete old baseline for this metric
        self.conn.execute(
            "DELETE FROM baselines WHERE metric_name = ?",
            (baseline.metric_name,)
        )
        
        # Insert new baseline
        self.conn.execute("""
            INSERT INTO baselines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            baseline.baseline_id,
            baseline.metric_name,
            baseline.mean_value,
            baseline.std_dev,
            baseline.min_value,
            baseline.max_value,
            baseline.sample_size,
            baseline.window_start.isoformat(),
            baseline.window_end.isoformat(),
            baseline.calculated_at.isoformat()
        ))
        
        self.conn.commit()
    
    def _save_baseline_json(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Remove old baseline for this metric
        data['baselines'] = [
            b for b in data.get('baselines', [])
            if b.get('metric_name') != baseline.metric_name
        ]
        
        # Add new baseline
        data['baselines'].append(baseline.model_dump(mode='json'))
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_baseline(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """
        Get baseline for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Baseline or None
        """
        if self.backend == "sqlite":
            return self._get_baseline_sqlite(metric_name)
        else:
            return self._get_baseline_json(metric_name)
    
    def _get_baseline_sqlite(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """Get baseline from SQLite."""
        cursor = self.conn.execute(
            "SELECT * FROM baselines WHERE metric_name = ?",
            (metric_name,)
        )
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return PerformanceBaseline(
            baseline_id=row['baseline_id'],
            metric_name=row['metric_name'],
            mean_value=row['mean_value'],
            std_dev=row['std_dev'],
            min_value=row['min_value'],
            max_value=row['max_value'],
            sample_size=row['sample_size'],
            window_start=datetime.fromisoformat(row['window_start']),
            window_end=datetime.fromisoformat(row['window_end']),
            calculated_at=datetime.fromisoformat(row['calculated_at'])
        )
    
    def _get_baseline_json(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """Get baseline from JSON."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        baselines = data.get('baselines', [])
        for baseline_dict in baselines:
            if baseline_dict.get('metric_name') == metric_name:
                return PerformanceBaseline(**baseline_dict)
        
        return None
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        if self.backend == "sqlite":
            cursor = self.conn.execute("SELECT COUNT(*) FROM evaluations")
            eval_count = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM baselines")
            baseline_count = cursor.fetchone()[0]
        else:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            eval_count = len(data.get('evaluations', []))
            baseline_count = len(data.get('baselines', []))
        
        return {
            'backend': self.backend,
            'total_evaluations': eval_count,
            'total_baselines': baseline_count
        }
    
    def cleanup_old_evaluations(self, days: int = 90) -> int:
        """
        Delete evaluations older than specified days.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of records deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        if self.backend == "sqlite":
            cursor = self.conn.execute(
                "DELETE FROM evaluations WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            deleted = cursor.rowcount
            self.conn.commit()
        else:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            original_count = len(data.get('evaluations', []))
            data['evaluations'] = [
                e for e in data.get('evaluations', [])
                if datetime.fromisoformat(e['timestamp']) >= cutoff
            ]
            deleted = original_count - len(data['evaluations'])
            
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Deleted {deleted} old evaluations (>{days} days)")
        return deleted