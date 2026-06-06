"""
Intent Classifier for the HERALD research analyst system.

Classifies NormalizedQuery.intent into QueryIntent, determines
requires_graph, and enriches the entities_mentioned list.

Backend: unified LLMClient (groq or ollama).
         Mock path available via settings.mock_llm_calls=True.

Ollama robustness:
  - Strips "entity|semantic" or "entity/semantic" compound responses
    to a single valid intent value.
  - Falls back to "semantic" on any parse failure.
"""

import json
from typing import Any, Dict, Tuple

from research_analyst.core.models import NormalizedQuery, QueryIntent
from research_analyst.core.exceptions import IntentClassificationError
from research_analyst.config import get_settings, prompts
from research_analyst.utils.llm_client import get_llm_client
from research_analyst.utils.logger import get_logger


logger = get_logger()

_VALID_INTENTS = {i.value for i in QueryIntent}


def _parse_intent(raw: str) -> str:
    """
    Normalise a raw intent string from the LLM.

    Handles compound responses such as "entity|semantic" or "entity/semantic"
    by taking the first valid token.
    """
    token = raw.strip().lower().replace("/", "|").split("|")[0].strip()
    return token if token in _VALID_INTENTS else "semantic"


class IntentClassifier:
    """
    Classify query intent and requirements via LLM or mock heuristic.
    """

    def __init__(self):
        self.settings   = get_settings()
        self.logger     = get_logger()
        self.llm_client = get_llm_client()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def classify(self, normalized_query: NormalizedQuery) -> NormalizedQuery:
        """
        Classify query intent and update the NormalizedQuery in-place.

        Args:
            normalized_query: Normalised query with placeholder intent.

        Returns:
            Updated NormalizedQuery with intent, requires_graph, domain,
            and entities_mentioned populated.

        Raises:
            IntentClassificationError: on LLM or parse failure.
        """
        self.logger.info(
            "Classifying query intent",
            query = normalized_query.normalized_text[:100],
        )
        try:
            if self.settings.mock_llm_calls:
                result = self._mock_classification(normalized_query)
            else:
                result = self._llm_classification(normalized_query)

            normalized_query.intent        = QueryIntent(_parse_intent(result["intent"]))
            normalized_query.requires_graph= bool(result.get("requires_graph", False))

            if result.get("domain"):
                normalized_query.domain = result["domain"]

            if result.get("entities_mentioned"):
                existing = set(normalized_query.entities_mentioned)
                new_ents = set(result["entities_mentioned"])
                normalized_query.entities_mentioned = list(existing | new_ents)

            self.logger.info(
                "Query classified",
                intent        = normalized_query.intent.value,
                requires_graph= normalized_query.requires_graph,
                domain        = normalized_query.domain,
            )
            return normalized_query

        except IntentClassificationError:
            raise
        except Exception as e:
            raise IntentClassificationError(
                f"Intent classification failed: {e}",
                details={"query": normalized_query.normalized_text},
            )

    def classify_with_reasoning(
        self,
        normalized_query: NormalizedQuery,
    ) -> Tuple[NormalizedQuery, str]:
        """
        Classify and return the LLM's reasoning string.

        Returns:
            (updated NormalizedQuery, reasoning_string)
        """
        if self.settings.mock_llm_calls:
            result = self._mock_classification(normalized_query)
        else:
            result = self._llm_classification(normalized_query)

        normalized_query.intent         = QueryIntent(_parse_intent(result["intent"]))
        normalized_query.requires_graph = bool(result.get("requires_graph", False))
        if result.get("domain"):
            normalized_query.domain = result["domain"]

        reasoning = result.get("reasoning", "No reasoning provided")
        return normalized_query, reasoning

    # ------------------------------------------------------------------ #
    #  LLM classification                                                 #
    # ------------------------------------------------------------------ #

    def _llm_classification(
        self,
        normalized_query: NormalizedQuery,
    ) -> Dict[str, Any]:
        """Call the LLM and parse the JSON classification response."""
        prompt = prompts.format_prompt(
            prompts.INTENT_CLASSIFICATION,
            query=normalized_query.normalized_text,
        )
        response = self.llm_client.generate(
            prompt        = prompt,
            system_prompt = "You are a query intent classifier. Always respond with valid JSON.",
            max_tokens    = 500,
            temperature   = self.settings.temperature,
            json_mode     = True,
        )
        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            raise IntentClassificationError(
                "LLM returned invalid JSON",
                details={"response_preview": response[:200]},
            )
        for required in ("intent", "requires_graph"):
            if required not in result:
                raise IntentClassificationError(
                    f"Missing required field '{required}' in classification response"
                )
        return result

    # ------------------------------------------------------------------ #
    #  Mock classification (for testing / mock_llm_calls=True)           #
    # ------------------------------------------------------------------ #

    def _mock_classification(
        self,
        normalized_query: NormalizedQuery,
    ) -> Dict[str, Any]:
        """Keyword-heuristic classification — no LLM call."""
        text = normalized_query.normalized_text.lower()

        if any(w in text for w in ["relationship", "connection", "between", "versus", "vs"]):
            intent, requires_graph = "relational", True
        elif any(w in text for w in ["who is", "who are", "tell me about"]):
            intent, requires_graph = "entity", True
        elif any(w in text for w in ["how has", "evolved", "changed", "history of", "over time"]):
            intent, requires_graph = "temporal", True
        elif any(w in text for w in ["compare", "difference", "similar"]):
            intent, requires_graph = "hybrid", True
        else:
            intent, requires_graph = "semantic", False

        return {
            "intent":            intent,
            "reasoning":         "Mock classification based on keywords",
            "confidence":        0.80,
            "domain":            normalized_query.domain,
            "requires_graph":    requires_graph,
            "entities_mentioned":normalized_query.entities_mentioned,
        }