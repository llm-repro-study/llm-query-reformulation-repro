"""Unified LLM client for OpenAI API and OpenRouter-hosted models."""

from __future__ import annotations

import os
import time
from typing import List, Optional, Dict, Any

from openai import OpenAI


# ── Model presets ────────────────────────────────────────────────────────────

LLM_CONFIGS: Dict[str, Dict[str, str]] = {
    "gpt-4.1":      {"provider": "openai",     "model_id": "gpt-4.1"},
    "gpt-4.1-nano": {"provider": "openai",     "model_id": "gpt-4.1-nano"},
    "qwen-72b":     {"provider": "openrouter", "model_id": "qwen/qwen-2.5-72b-instruct"},
    "qwen-7b":      {"provider": "openrouter", "model_id": "qwen/qwen-2.5-7b-instruct"},
}


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API.

    Works with both OpenAI models and OpenRouter-hosted models by switching
    the ``base_url`` depending on the provider.

    Parameters
    ----------
    model_name : str
        Key in ``LLM_CONFIGS`` (e.g. ``"gpt-4.1"``, ``"qwen-72b"``),
        or an arbitrary model id (treated as OpenAI by default).
    max_tokens : int
        Default max output tokens for every request.
    temperature : float
        Default sampling temperature.
    max_retries : int
        Number of retries on transient errors.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        cfg = LLM_CONFIGS.get(model_name, {"provider": "openai", "model_id": model_name})
        self.model_id = cfg["model_id"]
        self.provider = cfg["provider"]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        if self.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            base_url = "https://openrouter.ai/api/v1"
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = None  # default OpenAI endpoint

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    # ── public helpers ───────────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
    ) -> List[str]:
        """Send a chat-completion request and return *n* response strings.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style ``[{"role": ..., "content": ...}]`` message list.
        temperature : float, optional
            Override default temperature for this call.
        max_tokens : int, optional
            Override default max_tokens for this call.
        n : int
            Number of completions to return (uses the API ``n`` parameter).

        Returns
        -------
        list[str]
            One string per completion choice.
        """
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temp,
                    max_tokens=mtok,
                    n=n,
                )
                return [c.message.content.strip() for c in resp.choices]
            except Exception as exc:
                if attempt == self.max_retries:
                    raise
                wait = 2 ** attempt
                print(f"  [LLM] attempt {attempt} failed ({exc}), retrying in {wait}s …")
                time.sleep(wait)

    def generate_one(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Convenience wrapper that returns a single completion string."""
        return self.generate(messages, n=1, **kwargs)[0]

