"""
LLM client for the research analyst system.
Supports Groq (cloud) and Ollama (local) backends.
Controlled by RA_DEFAULT_LLM_PROVIDER in .env
"""

import json
import requests
from typing import Optional

from research_analyst.core.exceptions import LLMError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


logger = get_logger()


class LLMClient:
    """
    Unified LLM client supporting Groq and Ollama.
    Switch via RA_DEFAULT_LLM_PROVIDER=groq|ollama in .env
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.model = self.settings.default_model
        self.provider = self.settings.default_llm_provider

        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "groq":
            self._init_groq()
        else:
            raise LLMError(
                f"Unsupported provider: {self.provider}. Use 'groq' or 'ollama'.",
                provider=self.provider,
            )

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def _init_ollama(self):
        """Initialise Ollama client."""
        self.ollama_base_url = getattr(
            self.settings, "ollama_base_url", "http://localhost:11434"
        )
        # Quick connectivity check
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            self.logger.info(
                "Ollama initialised",
                model=self.model,
                base_url=self.ollama_base_url,
                available_models=available,
            )
            if self.model not in available and not any(
                self.model in m for m in available
            ):
                self.logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}",
                    available=available,
                )
        except requests.exceptions.ConnectionError:
            raise LLMError(
                f"Cannot connect to Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running: ollama serve",
                provider="ollama",
            )
        except Exception as e:
            raise LLMError(
                f"Ollama initialisation failed: {str(e)}",
                provider="ollama",
            )

    def _init_groq(self):
        """Initialise Groq client."""
        try:
            from groq import Groq

            api_key_field = self.settings.groq_api_key
            if api_key_field is None:
                raise LLMError("Groq API key not configured", provider="groq")
            api_key = (
                api_key_field
                if isinstance(api_key_field, str)
                else api_key_field.get_secret_value()
            )
            if not api_key:
                raise LLMError("Groq API key is empty", provider="groq")

            self.groq_client = Groq(api_key=api_key)
            self.logger.info("Groq initialised", model=self.model)

        except ImportError:
            raise LLMError(
                "Groq library not installed. Run: pip install groq",
                provider="groq",
            )

    # ------------------------------------------------------------------ #
    #  Public generate method                                             #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a completion.

        Args:
            prompt:        User message.
            system_prompt: Optional system message.
            max_tokens:    Max tokens to generate.
            temperature:   Sampling temperature.
            json_mode:     Request JSON output (adds instruction if needed).

        Returns:
            Generated text string.
        """
        if max_tokens is None:
            max_tokens = self.settings.max_tokens
        if temperature is None:
            temperature = self.settings.temperature

        # When json_mode is requested, append instruction to system prompt
        # for Ollama (which doesn't have a native json_object response format)
        effective_system = system_prompt or ""
        if json_mode and self.provider == "ollama":
            effective_system = (
                (effective_system + "\n" if effective_system else "")
                + "You must respond with valid JSON only. "
                "No markdown, no explanation, no text outside the JSON object."
            )

        if self.provider == "ollama":
            return self._generate_ollama(
                prompt, effective_system, max_tokens, temperature, json_mode
            )
        else:
            return self._generate_groq(
                prompt, effective_system, max_tokens, temperature, json_mode
            )

    # ------------------------------------------------------------------ #
    #  Ollama backend                                                     #
    # ------------------------------------------------------------------ #

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Ollama native JSON mode
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=getattr(self.settings, "request_timeout", 120),
            )
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]

            self.logger.log_llm_call(
                provider="ollama",
                model=self.model,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                cost_estimate=0.0,   # local model, no cost
                latency_ms=data.get("total_duration", 0) / 1_000_000,
            )

            return content

        except requests.exceptions.Timeout:
            raise LLMError(
                f"Ollama request timed out after {self.settings.request_timeout}s. "
                "Try a smaller model or increase RA_REQUEST_TIMEOUT.",
                provider="ollama",
            )
        except requests.exceptions.ConnectionError:
            raise LLMError(
                f"Lost connection to Ollama at {self.ollama_base_url}.",
                provider="ollama",
            )
        except KeyError:
            raise LLMError(
                f"Unexpected Ollama response format: {data}",
                provider="ollama",
            )
        except Exception as e:
            raise LLMError(
                f"Ollama generation failed: {str(e)}",
                provider="ollama",
            )

    # ------------------------------------------------------------------ #
    #  Groq backend                                                       #
    # ------------------------------------------------------------------ #

    def _generate_groq(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        import time as _time

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        max_retries = 4
        backoff = 10  # seconds — start at 10s, double each attempt

        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(**kwargs)

                self.logger.log_llm_call(
                    provider="groq",
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost_estimate=self._estimate_groq_cost(response.usage),
                    latency_ms=0,
                )
                return response.choices[0].message.content

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()

                if is_rate_limit and attempt < max_retries - 1:
                    self.logger.warning(
                        "Groq rate limit hit, retrying",
                        attempt=attempt + 1,
                        wait_s=backoff,
                    )
                    _time.sleep(backoff)
                    backoff = min(backoff * 2, 60)  # cap at 60s
                    continue

                raise LLMError(
                    f"Groq generation failed: {err_str}",
                    provider="groq",
                    details={"model": self.model},
                )

    def _estimate_groq_cost(self, usage) -> float:
        prompt_cost = usage.prompt_tokens * 0.00000027
        completion_cost = usage.completion_tokens * 0.00000027
        return prompt_cost + completion_cost


# ----------------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------------

_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def reset_llm_client():
    """Force re-initialisation (useful when switching providers)."""
    global _llm_client
    _llm_client = None