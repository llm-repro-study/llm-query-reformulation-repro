"""Prompt bank: loads and renders prompt templates from a JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class PromptBank:
    """Load prompt templates and render them with variable substitution.

    The prompts file is a JSON object keyed by prompt id.  Each entry
    contains a ``messages`` list (OpenAI-style roles) with ``{variable}``
    placeholders and optional metadata.

    Parameters
    ----------
    path : str | Path
        Path to the prompts JSON file.
    """

    def __init__(self, path: str | Path):
        with open(path) as f:
            self._bank: Dict[str, Any] = json.load(f)

    def render(self, prompt_id: str, **variables) -> List[Dict[str, str]]:
        """Return an OpenAI-style message list with variables filled in.

        Parameters
        ----------
        prompt_id : str
            Key in the prompts JSON (e.g. ``"genqr"``).
        **variables
            Values to substitute into ``{placeholder}`` slots.

        Returns
        -------
        list[dict]
            ``[{"role": ..., "content": ...}, ...]``
        """
        entry = self._bank.get(prompt_id)
        if entry is None:
            raise KeyError(f"Prompt '{prompt_id}' not found in bank.")

        rendered: List[Dict[str, str]] = []
        for msg in entry["messages"]:
            rendered.append({
                "role": msg["role"],
                "content": msg["content"].format(**variables),
            })
        return rendered

    def list_prompts(self) -> List[str]:
        """Return all available prompt ids."""
        return list(self._bank.keys())

