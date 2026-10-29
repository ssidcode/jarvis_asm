"""LLM backends. Anthropic by default; Ollama / any OpenAI-compatible server
for fully-local operation. All of them stream, because in a voice assistant
time-to-first-word is the only latency the user actually feels."""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator

log = logging.getLogger("jarvis.llm")


#: Groq's OpenAI-compatible API root (the /openai/v1 path is part of it).
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMError(RuntimeError):
    pass


class BaseLLM:
    def complete(self, messages: list[dict], system: str) -> str:
        return "".join(self.stream(messages, system))

    def stream(self, messages: list[dict], system: str) -> Iterator[str]:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 400,
        temperature: float = 0.7,
        **_: object,
    ):
        import anthropic

        key = os.environ.get(api_key_env)
        if not key:
            raise LLMError(
                f"{api_key_env} is not set. Export your key, or switch "
                "llm.backend to 'ollama' in config.yaml for a local model."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def stream(self, messages: list[dict], system: str) -> Iterator[str]:
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=messages,
            ) as stream:
                yield from stream.text_stream
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc


class OpenAICompatibleLLM(BaseLLM):
    """Works with Groq, Ollama (``/v1``), llama.cpp, vLLM, LM Studio, OpenAI."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str | None = "http://localhost:11434/v1",
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 400,
        temperature: float = 0.7,
        token_param: str = "max_tokens",
        require_key: bool = False,
        **_: object,
    ):
        self.model = model
        self.base_url = (base_url or "http://localhost:11434/v1").rstrip("/")
        # Strip it: a trailing newline or space from a copy-paste or an
        # env file otherwise produces a baffling 401.
        self.api_key = os.environ.get(api_key_env, "").strip()
        if require_key and not self.api_key:
            raise LLMError(
                f"{api_key_env} is not set. Get a key at "
                "https://console.groq.com/keys and export it:\n"
                f"  export {api_key_env}=gsk_..."
            )
        self.api_key = self.api_key or "not-needed"
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Groq deprecated max_tokens in favour of max_completion_tokens;
        # local servers mostly only understand max_tokens.
        self.token_param = token_param

    def _http_error(self, status: int, body: str) -> LLMError:
        """Turn an HTTP failure into something that names the actual fix.

        A retired model ID is the most likely cause of a 404 here: Groq
        deprecates models on a few months' notice, so a config that worked
        last quarter can start failing with nothing having changed locally.
        """
        detail = body[:300]
        if status == 404 and "model" in body.lower():
            return LLMError(
                f"model {self.model!r} was rejected as unavailable (HTTP 404).\n"
                "  It has most likely been retired. Check what your key can use:\n"
                "      python -m scripts.groq_models\n"
                "  then set llm.model in config.yaml. Current Groq production\n"
                "  models are openai/gpt-oss-120b and openai/gpt-oss-20b;\n"
                "  llama-3.3-70b-versatile and llama-3.1-8b-instant were shut\n"
                "  down on 2026-08-16 (console.groq.com/docs/deprecations).\n"
                f"  server said: {detail}"
            )
        if status in (401, 403):
            return LLMError(
                f"authentication failed (HTTP {status}). Check your API key is "
                f"current and has no stray whitespace.\n  server said: {detail}"
            )
        if status == 429:
            return LLMError(
                f"rate limited (HTTP 429). Wait, or use a smaller model such as "
                f"openai/gpt-oss-20b.\n  server said: {detail}"
            )
        return LLMError(f"{self.base_url} returned HTTP {status}: {detail}")

    def stream(self, messages: list[dict], system: str) -> Iterator[str]:
        import json

        import httpx

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            self.token_param: self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise self._http_error(response.status_code, response.text)
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data in ("", "[DONE]"):
                            continue
                        try:
                            delta = json.loads(data)["choices"][0].get("delta", {})
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        chunk = delta.get("content")
                        if chunk:
                            yield chunk
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.base_url} request failed: {exc}") from exc


def build_llm(cfg_section: dict) -> BaseLLM:
    backend = str(cfg_section.get("backend", "anthropic")).lower()
    kwargs = {
        k: cfg_section.get(k)
        for k in ("model", "api_key_env", "base_url", "max_tokens", "temperature")
        if cfg_section.get(k) is not None
    }
    if backend == "anthropic":
        kwargs.pop("base_url", None)
        return AnthropicLLM(**kwargs)
    if backend == "groq":
        kwargs.setdefault("base_url", GROQ_BASE_URL)
        kwargs.setdefault("api_key_env", "GROQ_API_KEY")
        kwargs.setdefault("model", "openai/gpt-oss-120b")
        kwargs["token_param"] = "max_completion_tokens"
        kwargs["require_key"] = True
        return OpenAICompatibleLLM(**kwargs)
    if backend in ("ollama", "openai-compatible", "openai", "local"):
        if backend == "ollama" and not kwargs.get("base_url"):
            kwargs["base_url"] = "http://localhost:11434/v1"
        return OpenAICompatibleLLM(**kwargs)
    raise LLMError(f"unknown llm backend {backend!r}")


class Conversation:
    """Rolling chat history, trimmed to the last N user/assistant pairs."""

    def __init__(self, system_prompt: str, max_turns: int = 8):
        self.system_prompt = system_prompt.strip()
        self.max_turns = max_turns
        self._messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        if text.strip():
            self._messages.append({"role": "assistant", "content": text})
        elif self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()  # never leave a dangling user turn
        self._trim()

    def _trim(self) -> None:
        limit = self.max_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]
        # The API requires the first message to be from the user.
        while self._messages and self._messages[0]["role"] != "user":
            self._messages.pop(0)

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()