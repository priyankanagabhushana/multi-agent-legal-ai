"""LLM interface supporting both LiteLLM (DeepSeek API, OpenAI, etc.) and Ollama.

Primary mode: LiteLLM + DeepSeek V4 Flash (same pattern as RegCheck_v2).
Fallback mode: Ollama with local models for sovereign inference.

Structured output uses instructor (Pydantic-native) with automatic retry/self-correction.
"""

import json
import os
import re
import logging
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = """You are a precise legal reasoning assistant specialized in Swiss Tenancy Law (Mietrecht).
You operate within a multi-agent architecture. Your responses must be:
1. Structured — return valid JSON matching the requested schema exactly.
2. Evidence-based — cite specific statutes (Art. X OR) and cases (BGE X) where relevant.
3. Transparent — explain your reasoning briefly.
4. Honest about uncertainty — use confidence scores (0-100) and flag missing evidence.

Always return ONLY the JSON object, no preamble, no markdown fences, no trailing text.
If you cannot determine something, set confidence appropriately and note the limitation."""


def extract_json(text: str) -> str:
    """Extract JSON from LLM output, handling markdown fences and stray text."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    brace_start = text.find("{")
    if brace_start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[brace_start:], start=brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]
    return text


def json_to_model(text: str, model_class: Type[T]) -> T:
    """Parse JSON string into a Pydantic model, with fallback for missing fields."""
    try:
        return model_class.model_validate_json(text, strict=False)
    except Exception:
        try:
            data = json.loads(text)
            return model_class.model_validate(data, strict=False)
        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM output into {model_class.__name__}: {e}\n"
                f"Raw text (first 500 chars): {text[:500]}"
            )


class LLMConfig:
    """Configuration for LLM backends."""

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.1,
    ):
        self.provider = provider
        self.max_retries = max_retries
        self.temperature = temperature

        if provider == "deepseek":
            self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-chat")
            self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            self.base_url = base_url
            if not self.api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is required for DeepSeek provider")
            os.environ["DEEPSEEK_API_KEY"] = self.api_key
        elif provider == "ollama":
            self.model = model or "llama3.1:8b"
            self.api_key = None
            self.base_url = base_url or "http://localhost:11434"
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.base_url = base_url
            if self.api_key:
                os.environ["OPENAI_API_KEY"] = self.api_key
        else:
            self.model = model or provider
            self.api_key = api_key
            self.base_url = base_url


class LLM:
    """Unified LLM interface supporting LiteLLM (DeepSeek/OpenAI) and Ollama.

    Usage:
        # DeepSeek API (no GPU needed, same pattern as RegCheck_v2)
        llm = LLM(LLMConfig(provider="deepseek", model="deepseek/deepseek-chat"))

        # Local Ollama (sovereign, no API key needed)
        llm = LLM(LLMConfig(provider="ollama", model="llama3.1:8b"))
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = self._resolve_default_config()
        self.config = config
        self._instructor_client = None
        self._ollama_client = None

    @staticmethod
    def _resolve_default_config() -> LLMConfig:
        """Resolve default config: prefer DeepSeek API, fall back to Ollama."""
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            model = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-chat")
            return LLMConfig(provider="deepseek", model=model, api_key=api_key)

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            return LLMConfig(provider="openai", api_key=openai_key)

        logger.info("No API key found, falling back to local Ollama")
        return LLMConfig(provider="ollama")

    @property
    def instructor(self):
        """Lazy-load instructor client for structured output."""
        if self._instructor_client is None and self.config.provider != "ollama":
            try:
                import instructor
                from litellm import completion
                self._instructor_client = instructor.from_litellm(completion)
            except ImportError:
                logger.warning("instructor not installed, falling back to manual JSON parsing")
                self._instructor_client = False
        return self._instructor_client if self._instructor_client is not False else None

    def _call_litellm(self, messages: list[dict], temperature: float) -> str:
        """Call LiteLLM completion API."""
        import litellm  # lazy import

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.config.api_key and self.config.provider == "deepseek":
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content.strip()

    def _call_litellm_json(self, messages: list[dict], temperature: float) -> str:
        """Call LiteLLM with JSON mode enabled."""
        import litellm

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if self.config.api_key and self.config.provider == "deepseek":
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content.strip()

    def _call_ollama(self, prompt: str, system: str, temperature: float) -> str:
        """Call Ollama generate API."""
        import ollama

        if self._ollama_client is None:
            self._ollama_client = ollama.Client(host=self.config.base_url)

        response = self._ollama_client.generate(
            model=self.config.model,
            prompt=prompt,
            system=system,
            options={"temperature": temperature},
        )
        return response["response"].strip()

    def generate(
        self,
        prompt: str,
        system: str = SYSTEM_PROMPT,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a raw text response."""
        temp = temperature if temperature is not None else self.config.temperature

        if self.config.provider == "ollama":
            return self._call_ollama(prompt, system, temp)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self._call_litellm(messages, temp)

    def _build_field_description(self, model_class: Type[T]) -> str:
        """Build a plain-language field description from a Pydantic model."""
        schema = model_class.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])

        lines = ["Respond with a JSON object containing these fields:"]
        for field_name, field_info in props.items():
            req_marker = " (REQUIRED)" if field_name in required else ""
            field_type = field_info.get("type", "any")
            desc = field_info.get("description", "")
            lines.append(f'  - "{field_name}": {field_type}{req_marker} — {desc}')

        return "\n".join(lines)

    def generate_structured(
        self,
        prompt: str,
        output_model: Type[T],
        system: str = SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        schema_hint: bool = True,
    ) -> T:
        """Generate a structured response parsed into a Pydantic model.

        Uses instructor for API-based providers (DeepSeek/OpenAI) with automatic
        retry and self-correction. Falls back to prompt engineering for Ollama.
        """
        temp = temperature if temperature is not None else self.config.temperature

        if self.config.provider == "ollama":
            return self._generate_structured_ollama(prompt, output_model, system, temp, schema_hint)

        client = self.instructor
        if client is not False and client is not None:
            return self._generate_structured_instructor(prompt, output_model, system, temp)

        return self._generate_structured_litellm(prompt, output_model, system, temp, schema_hint)

    def _generate_structured_instructor(
        self,
        prompt: str,
        output_model: Type[T],
        system: str,
        temperature: float,
    ) -> T:
        """Use instructor for Pydantic-native structured output with retry."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self.config.max_retries):
            try:
                response = self.instructor.chat.completions.create(
                    model=self.config.model,
                    response_model=output_model,
                    messages=messages,
                    max_retries=1,
                    temperature=temperature,
                )
                return response
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                messages.append({
                    "role": "user",
                    "content": f"Your previous response had a validation error: {e}\nPlease fix it and return a valid JSON object matching the schema exactly."
                })

        raise RuntimeError("Failed to generate structured output after all retries")

    def _generate_structured_litellm(
        self,
        prompt: str,
        output_model: Type[T],
        system: str,
        temperature: float,
        schema_hint: bool,
    ) -> T:
        """Use LiteLLM JSON mode + manual Pydantic validation with self-correction."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        if schema_hint:
            field_desc = self._build_field_description(output_model)
            messages[1]["content"] += f"\n\n{field_desc}\n\nReturn ONLY the JSON object, no other text."

        for attempt in range(self.config.max_retries):
            try:
                raw = self._call_litellm_json(messages, temperature)
                json_str = extract_json(raw)
                data = json.loads(json_str)
                return output_model.model_validate(data, strict=False)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == self.config.max_retries - 1:
                    raise
                messages.append({
                    "role": "user",
                    "content": f"Your previous response had an error: {e}\nPlease fix it and return a valid JSON object."
                })

        raise RuntimeError("Failed after all retries")

    def _generate_structured_ollama(
        self,
        prompt: str,
        output_model: Type[T],
        system: str,
        temperature: float,
        schema_hint: bool,
    ) -> T:
        """Use Ollama with prompt-based schema hints."""
        if schema_hint:
            field_desc = self._build_field_description(output_model)
            prompt = f"{prompt}\n\n{field_desc}\n\nReturn ONLY the JSON object, no other text."

        raw = self._call_ollama(prompt, system, temperature)
        json_str = extract_json(raw)
        return json_to_model(json_str, output_model)

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
    ) -> str:
        """Multi-turn chat."""
        temp = temperature if temperature is not None else self.config.temperature

        if self.config.provider == "ollama":
            import ollama
            response = ollama.Client(host=self.config.base_url).chat(
                model=self.config.model,
                messages=messages,
                options={"temperature": temp},
            )
            return response["message"]["content"].strip()

        return self._call_litellm(messages, temp)


def create_llm(
    provider: str = "deepseek",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLM:
    """Factory function to create an LLM instance.

    Args:
        provider: "deepseek", "ollama", or "openai"
        model: Model name (defaults to deepseek-chat, llama3.1:8b, or gpt-4o-mini)
        api_key: API key (reads from env var if not provided)
    """
    config = LLMConfig(provider=provider, model=model, api_key=api_key)
    return LLM(config)
