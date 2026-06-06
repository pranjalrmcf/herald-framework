"""
Temporal Decay module for the research analyst system.

Scores knowledge graph relationships by freshness and downweights stale
edges before synthesis. Prevents outdated facts from corrupting answers
about the current state of affairs.

Decay model:
    freshness_score = exp(-lambda * age_days)
    where lambda = ln(2) / halflife_days   (exponential half-life decay)

    Default halflife = 90 days → a 90-day-old relationship scores 0.5
                                  a 180-day-old relationship scores 0.25
                                  a 30-day-old relationship scores 0.79

Temporal signals used (in priority order):
    1. relationship.temporal_info['parsed_date']  — explicit date from LLM extraction
    2. relationship.temporal_info['year']          — year-level precision
    3. source document published_date              — date of the source article
    4. source document retrieved_at                — fallback: when we fetched it

Integration:
    graph_querier._rank_subgraph_elements():
        from research_analyst.temporal_decay import TemporalDecayScorer
        scorer = TemporalDecayScorer()
        subgraph = scorer.apply_decay(subgraph, document_map)

    The scorer mutates relationship.metadata['freshness_score'] and
    relationship.confidence by blending in the freshness score.
    It also attaches a 'staleness_warning' flag for relationships that
    are too old to be trusted.
"""

import math
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from research_analyst.core.models import Relationship, Subgraph, Document
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


logger = get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HALFLIFE_DAYS = 90
_STALE_THRESHOLD_DAYS = 365        # Relationships older than this get a warning
_FRESHNESS_CONFIDENCE_BLEND = 0.3  # How much freshness shifts original confidence
                                   # 0.0 = no effect, 1.0 = full replacement


# ---------------------------------------------------------------------------
# TemporalDecayScorer
# ---------------------------------------------------------------------------

class TemporalDecayScorer:
    """
    Applies exponential temporal decay to knowledge graph relationships.

    After calling apply_decay():
      - Each Relationship gets metadata['freshness_score'] ∈ [0, 1]
      - Each Relationship gets metadata['age_days'] (float)
      - Relationships older than _STALE_THRESHOLD_DAYS get
        metadata['staleness_warning'] = True
      - relationship.confidence is blended with freshness_score
        (weighted by _FRESHNESS_CONFIDENCE_BLEND)
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()

        self.enabled: bool = getattr(
            self.settings, "temporal_decay_enabled", True
        )
        halflife: int = getattr(
            self.settings, "temporal_decay_halflife_days", _DEFAULT_HALFLIFE_DAYS
        )
        # Decay constant lambda so that score = 0.5 at halflife
        self._lambda: float = math.log(2) / max(halflife, 1)
        self._halflife_days = halflife

        self.logger.debug(
            "TemporalDecayScorer initialised",
            enabled=self.enabled,
            halflife_days=self._halflife_days,
            lambda_=round(self._lambda, 6),
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def apply_decay(
        self,
        subgraph: Subgraph,
        document_map: Optional[Dict[str, Document]] = None,
    ) -> Subgraph:
        """
        Apply temporal decay to all relationships in a subgraph.

        Args:
            subgraph:     Subgraph whose relationships will be scored.
            document_map: Optional dict of doc_id -> Document for date lookup.
                          If None, only temporal_info on the relationship is used.

        Returns:
            The same Subgraph object with updated relationship scores.
        """
        if not self.enabled:
            return subgraph

        if not subgraph.relationships:
            return subgraph

        now = datetime.utcnow()
        scored_count = 0
        stale_count = 0
        doc_map = document_map or {}

        for rel in subgraph.relationships:
            date = self._extract_date(rel, doc_map)
            freshness, age_days = self._compute_freshness(date, now)

            rel.metadata["freshness_score"] = round(freshness, 4)
            rel.metadata["age_days"] = round(age_days, 1) if age_days is not None else None
            rel.metadata["date_used"] = date.isoformat() if date else None

            if age_days is not None and age_days > _STALE_THRESHOLD_DAYS:
                rel.metadata["staleness_warning"] = True
                stale_count += 1
            else:
                rel.metadata["staleness_warning"] = False

            # Blend freshness into confidence
            # new_conf = original_conf * (1 - blend) + freshness * blend
            original_conf = rel.confidence
            blended_conf = (
                original_conf * (1.0 - _FRESHNESS_CONFIDENCE_BLEND)
                + freshness * _FRESHNESS_CONFIDENCE_BLEND
            )
            rel.confidence = round(max(0.0, min(1.0, blended_conf)), 4)
            rel.metadata["original_confidence"] = original_conf

            scored_count += 1

        self.logger.info(
            "Temporal decay applied",
            total_relationships=len(subgraph.relationships),
            scored=scored_count,
            stale=stale_count,
            halflife_days=self._halflife_days,
        )

        return subgraph

    def score_relationship(
        self,
        rel: Relationship,
        document_map: Optional[Dict[str, Document]] = None,
    ) -> float:
        """
        Score a single relationship's freshness.

        Args:
            rel:          Relationship to score.
            document_map: Optional doc_id -> Document map.

        Returns:
            Freshness score ∈ [0, 1].
        """
        doc_map = document_map or {}
        date = self._extract_date(rel, doc_map)
        freshness, _ = self._compute_freshness(date, datetime.utcnow())
        return freshness

    def get_stale_relationships(
        self,
        subgraph: Subgraph,
        threshold_days: int = _STALE_THRESHOLD_DAYS,
    ) -> List[Relationship]:
        """
        Return relationships flagged as stale.

        Args:
            subgraph:       Subgraph (apply_decay must have been called first).
            threshold_days: Age threshold in days.

        Returns:
            List of stale Relationship objects.
        """
        return [
            r for r in subgraph.relationships
            if r.metadata.get("staleness_warning", False)
            or (
                r.metadata.get("age_days") is not None
                and r.metadata["age_days"] > threshold_days
            )
        ]

    # ------------------------------------------------------------------ #
    #  Date extraction helpers                                            #
    # ------------------------------------------------------------------ #

    def _extract_date(
        self,
        rel: Relationship,
        doc_map: Dict[str, Document],
    ) -> Optional[datetime]:
        """
        Extract the best available date for a relationship.
        Priority: temporal_info parsed_date > year > doc published_date > doc retrieved_at
        """
        # 1. Check relationship.temporal_info
        ti = rel.temporal_info or {}

        if ti.get("parsed_date"):
            parsed = self._parse_iso(ti["parsed_date"])
            if parsed:
                return parsed

        if ti.get("year"):
            try:
                year = int(str(ti["year"])[:4])
                if 1900 <= year <= datetime.utcnow().year:
                    return datetime(year, 7, 1)  # Mid-year as proxy
            except (ValueError, TypeError):
                pass

        if ti.get("keyword"):
            parsed = self._parse_keyword(ti["keyword"])
            if parsed:
                return parsed

        # 2. Fall back to source document dates
        doc = doc_map.get(rel.source_doc_id)
        if doc:
            if doc.published_date:
                return doc.published_date
            if doc.retrieved_at:
                return doc.retrieved_at

        # 3. No date available
        return None

    def _compute_freshness(
        self,
        date: Optional[datetime],
        now: datetime,
    ) -> tuple:
        """
        Compute freshness score from a date.

        Returns:
            Tuple of (freshness_score, age_days_or_None).
        """
        if date is None:
            # Unknown date — assign a neutral score (not penalised, not rewarded)
            return 0.7, None

        # Ensure timezone-naive comparison
        if date.tzinfo is not None:
            date = date.replace(tzinfo=None)

        age_days = max(0.0, (now - date).total_seconds() / 86400.0)
        freshness = math.exp(-self._lambda * age_days)
        freshness = max(0.0, min(1.0, freshness))

        return freshness, age_days

    @staticmethod
    def _parse_iso(date_str: str) -> Optional[datetime]:
        """Try parsing an ISO datetime string."""
        if not date_str:
            return None
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d",
            "%Y-%m",
        ):
            try:
                return datetime.strptime(date_str[:len(fmt)], fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_keyword(keyword: str) -> Optional[datetime]:
        """Parse simple temporal keywords to approximate datetime."""
        now = datetime.utcnow()
        kw = keyword.lower().strip()
        mapping = {
            "today": now,
            "yesterday": now - timedelta(days=1),
            "last week": now - timedelta(weeks=1),
            "last month": now - timedelta(days=30),
            "last year": now - timedelta(days=365),
            "this year": datetime(now.year, 1, 1),
            "this month": datetime(now.year, now.month, 1),
            "recent": now - timedelta(days=30),
            "recently": now - timedelta(days=30),
            "current": now - timedelta(days=7),
            "latest": now - timedelta(days=7),
        }
        return mapping.get(kw)

    # ------------------------------------------------------------------ #
    #  Reporting                                                          #
    # ------------------------------------------------------------------ #

    def generate_decay_report(self, subgraph: Subgraph) -> str:
        """
        Generate a human-readable freshness report for a subgraph.
        Useful for debugging and monitoring.
        """
        if not subgraph.relationships:
            return "No relationships in subgraph."

        lines = [
            "=" * 60,
            "TEMPORAL DECAY REPORT",
            f"Halflife: {self._halflife_days} days | "
            f"Relationships: {len(subgraph.relationships)}",
            "=" * 60,
        ]

        scored = [
            r for r in subgraph.relationships
            if "freshness_score" in r.metadata
        ]
        unscored = len(subgraph.relationships) - len(scored)

        if scored:
            avg_freshness = sum(
                r.metadata["freshness_score"] for r in scored
            ) / len(scored)
            stale = sum(1 for r in scored if r.metadata.get("staleness_warning"))

            lines += [
                f"Scored:          {len(scored)}",
                f"Unscored:        {unscored} (no date available)",
                f"Stale (>{_STALE_THRESHOLD_DAYS}d): {stale}",
                f"Avg freshness:   {avg_freshness:.3f}",
                "",
                "Top 5 freshest relationships:",
            ]

            top5 = sorted(
                scored,
                key=lambda r: r.metadata.get("freshness_score", 0),
                reverse=True,
            )[:5]
            for r in top5:
                age = r.metadata.get("age_days")
                age_str = f"{age:.0f}d" if age is not None else "unknown age"
                lines.append(
                    f"  {r.subject} --[{r.predicate}]--> {r.object} | "
                    f"freshness={r.metadata['freshness_score']:.3f} | {age_str}"
                )

            stale_rels = [r for r in scored if r.metadata.get("staleness_warning")]
            if stale_rels:
                lines.append(f"\nStale relationships ({len(stale_rels)}):")
                for r in stale_rels[:5]:
                    age = r.metadata.get("age_days")
                    lines.append(
                        f"  [STALE] {r.subject} --[{r.predicate}]--> {r.object} | "
                        f"age={age:.0f}d"
                    )

        lines.append("=" * 60)
        return "\n".join(lines)