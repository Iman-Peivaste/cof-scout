"""OpenAI-compatible client used by public campaign reruns."""

from __future__ import annotations

import os
from typing import Any

import httpx

_FIXED_SAMPLING_MODELS = frozenset({"gpt-5.6-luna"})


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return "" if content is None else str(content)


class OpenAICompatClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ["OPENAI_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.timeout = timeout_s

    def chat_structured(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.4,
        max_tokens: int = 4096,
        json_object: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
        }
        if self.model not in _FIXED_SAMPLING_MODELS:
            payload["temperature"] = temperature
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        message = data["choices"][0]["message"]
        text = _extract_text(message.get("content") or message.get("reasoning_content")).strip()
        if not text:
            raise RuntimeError("Endpoint returned empty assistant content")
        usage = data.get("usage") or {}
        return text, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "usd": 0.0,
            "estimated": not bool(usage),
        }


def make_client(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 600.0,
) -> OpenAICompatClient:
    if provider != "openai_compat":
        raise ValueError("Public release supports only an OpenAI-compatible endpoint")
    return OpenAICompatClient(model, base_url, api_key, timeout_s)
