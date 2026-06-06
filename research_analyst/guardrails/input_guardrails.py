"""
Input guardrails for the research analyst system.
Validates and sanitizes incoming queries before processing.
"""

import re
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from research_analyst.core.models import Query
from research_analyst.core.exceptions import (
    GuardrailViolation,
    PromptInjectionDetected,
    UnsafeContentDetected,
    OutOfScopeQuery,
    RateLimitError,
)
from research_analyst.config import get_settings, prompts
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import clean_text, generate_hash


logger = get_logger()


class InputGuardrails:
    """Input validation and safety checks."""
    
    def __init__(self):
        """Initialize input guardrails."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Rate limiting tracking (in-memory, simple implementation)
        self._rate_limit_cache: Dict[str, list] = {}
        
        # Patterns for prompt injection detection
        self._injection_patterns = [
            r"ignore\s+(previous|all|above)\s+instructions",
            r"disregard\s+(previous|all|above)",
            r"forget\s+(previous|all|above)",
            r"new\s+instructions?:",
            r"system\s*:\s*",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"you\s+are\s+now",
            r"pretend\s+(to\s+be|you\s+are)",
            r"roleplay\s+as",
            r"act\s+as\s+(if|a)",
        ]
        
        # Patterns for unsafe content
        self._unsafe_patterns = [
            r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)",
            r"(kill|harm|hurt)\s+(yourself|myself|someone)",
            r"(illegal|unlawful)\s+(drugs|activities|weapons)",
        ]
        
        # Out of scope patterns
        self._out_of_scope_patterns = [
            r"schedule\s+(a\s+)?(meeting|appointment|reminder)",
            r"send\s+(an?\s+)?(email|message|text)",
            r"buy\s+|purchase\s+|order\s+",
            r"book\s+(a\s+)?(flight|hotel|restaurant)",
            r"what\s+(should|can)\s+i\s+(do|eat|wear)",
        ]

        self._malicious_intent_patterns = [
            r"manipulate\s+(search|ranking|algorithms?)",
            r"spread\s+(misinformation|disinformation|propaganda)",
            r"game\s+(search|seo|ranking)",
            r"exploit\s+(algorithms?|systems?)",
            r"deceive\s+(users|people|audience)",
            r"at\s+scale",
        ]
    
    def check(self, query: Query) -> Tuple[bool, Optional[str]]:
        """
        Run all input guardrail checks on a query.
        
        Args:
            query: Query object to validate
            
        Returns:
            Tuple of (is_safe, error_message)
            
        Raises:
            GuardrailViolation: If any check fails
        """
        query_id = query.metadata.get("query_id", "unknown")
        
        self.logger.info(
            "Running input guardrails",
            query_id=query_id,
            query_length=len(query.text)
        )
        
        try:
            # 1. Basic validation
            self._check_basic_validation(query)
            
            # 2. Rate limiting (if user_id provided)
            if query.user_id:
                self._check_rate_limit(query.user_id)
            
            # 3. Prompt injection detection
            self._check_prompt_injection(query.text, query_id)

            self._check_malicious_intent(query.text, query_id)
            
            # 4. Unsafe content detection
            self._check_unsafe_content(query.text, query_id)
            
            # 5. Scope validation
            self._check_scope(query.text, query_id)
            
            self.logger.info(
                "Input guardrails passed",
                query_id=query_id
            )
            
            return True, None
            
        except GuardrailViolation as e:
            self.logger.log_guardrail_violation(
                query_id=query_id,
                violation_type=e.violation_type,
                details=str(e.message)
            )
            raise
    
    def _check_basic_validation(self, query: Query):
        """Check basic query validation rules."""
        # Check minimum length
        if len(query.text.strip()) < 3:
            raise GuardrailViolation(
                "Query too short (minimum 3 characters)",
                violation_type="validation",
                details={"query_length": len(query.text)}
            )
        
        # Check maximum length
        if len(query.text) > 2000:
            raise GuardrailViolation(
                "Query too long (maximum 2000 characters)",
                violation_type="validation",
                details={"query_length": len(query.text)}
            )
        
        # Check for empty or whitespace-only
        if not query.text.strip():
            raise GuardrailViolation(
                "Query is empty or whitespace only",
                violation_type="validation"
            )
    
    def _check_rate_limit(self, user_id: str):
        """
        Check rate limiting for user.
        
        Args:
            user_id: User identifier
            
        Raises:
            RateLimitError: If rate limit exceeded
        """
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        # Get or initialize user's request history
        if user_id not in self._rate_limit_cache:
            self._rate_limit_cache[user_id] = []
        
        # Remove old requests outside the time window
        self._rate_limit_cache[user_id] = [
            ts for ts in self._rate_limit_cache[user_id]
            if ts > window_start
        ]
        
        # Check limit
        if len(self._rate_limit_cache[user_id]) >= self.settings.rate_limit_per_minute:
            raise RateLimitError(
                f"Rate limit exceeded: {self.settings.rate_limit_per_minute} requests per minute",
                retry_after=60,
                details={"user_id": user_id, "current_count": len(self._rate_limit_cache[user_id])}
            )
        
        # Add current request
        self._rate_limit_cache[user_id].append(now)
    
    def _check_prompt_injection(self, text: str, query_id: str):
        """
        Check for prompt injection attempts.
        
        Args:
            text: Query text
            query_id: Query identifier
            
        Raises:
            PromptInjectionDetected: If injection detected
        """
        text_lower = text.lower()
        
        # Check against known patterns
        detected_patterns = []
        for pattern in self._injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        if detected_patterns:
            raise PromptInjectionDetected(
                "Potential prompt injection detected",
                details={
                    "query_id": query_id,
                    "patterns": detected_patterns[:3],  # Limit to first 3
                    "severity": "high" if len(detected_patterns) > 2 else "medium"
                }
            )
        
        # Check for suspicious character sequences
        if any(seq in text for seq in ["<|", "|>", "###", "```system"]):
            raise PromptInjectionDetected(
                "Suspicious character sequence detected",
                details={"query_id": query_id, "severity": "medium"}
            )
    
    def _check_unsafe_content(self, text: str, query_id: str):
        """
        Check for unsafe or inappropriate content.
        
        Args:
            text: Query text
            query_id: Query identifier
            
        Raises:
            UnsafeContentDetected: If unsafe content detected
        """
        text_lower = text.lower()
        
        # Check against unsafe patterns
        detected_patterns = []
        for pattern in self._unsafe_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        if detected_patterns:
            raise UnsafeContentDetected(
                "Unsafe content detected in query",
                details={
                    "query_id": query_id,
                    "patterns": detected_patterns[:3],
                    "severity": "high"
                }
            )
        
        # Check for PII patterns (basic check)
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email (if not in context)
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                self.logger.warning(
                    "Potential PII detected in query",
                    query_id=query_id,
                    pattern=pattern
                )


    def _check_malicious_intent(self, text: str, query_id: str):
        """
        Detect malicious or harmful intent even if phrased as research.
        """
        text_lower = text.lower()
        detected = []

        for pattern in self._malicious_intent_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append(pattern)

        if detected:
            raise UnsafeContentDetected(
                "Malicious intent detected (misinformation / manipulation)",
                details={
                    "query_id": query_id,
                    "patterns": detected[:3],
                    "severity": "high"
                }
            )

    
    def _check_scope(self, text: str, query_id: str):
        """
        Check if query is within system scope.
        
        Args:
            text: Query text
            query_id: Query identifier
            
        Raises:
            OutOfScopeQuery: If query is out of scope
        """
        text_lower = text.lower()
        
        # Check against out-of-scope patterns
        for pattern in self._out_of_scope_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise OutOfScopeQuery(
                    "Query is outside the research analyst's capabilities",
                    details={
                        "query_id": query_id,
                        "suggestion": "This system is designed for research and information retrieval, not task execution or personal assistance.",
                        "pattern": pattern
                    }
                )
        
        # Check for real-time requirements
        realtime_indicators = ["right now", "currently", "at this moment", "live"]
        if any(indicator in text_lower for indicator in realtime_indicators):
            # This is a warning, not a hard block
            self.logger.warning(
                "Query may require real-time data",
                query_id=query_id,
                note="System will attempt to retrieve recent information"
            )
    
    def sanitize_query(self, query: Query) -> Query:
        """
        Sanitize and clean the query.
        
        Args:
            query: Original query
            
        Returns:
            Sanitized query
        """
        # Clean the text
        sanitized_text = clean_text(query.text)
        
        # Remove any potentially harmful characters
        sanitized_text = re.sub(r'[<>{}]', '', sanitized_text)
        
        # Truncate if needed
        if len(sanitized_text) > 2000:
            sanitized_text = sanitized_text[:2000]
        
        # Create new query with sanitized text
        sanitized_query = Query(
            text=sanitized_text,
            user_id=query.user_id,
            session_id=query.session_id,
            timestamp=query.timestamp,
            metadata=query.metadata
        )
        
        return sanitized_query
    
    def validate_and_sanitize(self, query: Query) -> Query:
        """
        Run full validation and sanitization pipeline.
        
        Args:
            query: Input query
            
        Returns:
            Validated and sanitized query
            
        Raises:
            GuardrailViolation: If validation fails
        """
        # First check if guardrails are enabled
        if not self.settings.enable_input_guardrails:
            self.logger.warning("Input guardrails disabled - skipping validation")
            return query
        
        # Run all checks
        self.check(query)
        
        # Sanitize
        sanitized_query = self.sanitize_query(query)
        
        return sanitized_query