"""Base class and data structures for query reformulation methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from tqdm import tqdm

from ..llm_client import LLMClient
from ..prompts import PromptBank


@dataclass
class Query:
    """A single query with its id and text."""
    qid: str
    text: str


@dataclass
class ReformulatedQuery:
    """Output of a reformulation method."""
    qid: str
    original: str
    reformulated: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMethod:
    """Abstract base for all reformulation methods.

    Subclasses must implement :meth:`reformulate_query`.

    Parameters
    ----------
    llm : LLMClient
        Shared LLM client instance.
    prompts : PromptBank
        Shared prompt bank.
    params : dict
        Method-specific hyper-parameters (from config YAML).
    """

    NAME: str = "base"
    REQUIRES_CONTEXTS: bool = False

    def __init__(self, llm: LLMClient, prompts: PromptBank, params: Dict[str, Any] | None = None):
        self.llm = llm
        self.prompts = prompts
        self.params = params or {}

    # ── interface ─────────────────────────────────────────────────────────

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        """Reformulate a single query. Must be overridden by subclasses."""
        raise NotImplementedError

    def reformulate_batch(
        self,
        queries: List[Query],
        ctx_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[ReformulatedQuery]:
        """Reformulate a list of queries, showing a progress bar."""
        results: List[ReformulatedQuery] = []
        for q in tqdm(queries, desc=self.NAME, unit="query"):
            ctxs = (ctx_map or {}).get(q.qid)
            results.append(self.reformulate_query(q, ctxs))
        return results

    # ── query construction helpers ────────────────────────────────────────

    @staticmethod
    def concat_repeat(query: str, generated: str, repeats: int = 5) -> str:
        """Return ``(query * repeats) + generated``, whitespace-joined."""
        parts = [query] * repeats + [generated]
        return _clean(" ".join(parts))

    @staticmethod
    def concat_adaptive(query: str, generated: str, ratio: int = 5) -> str:
        """Repeat query proportionally to length of generated text."""
        reps = max(1, (len(generated) // max(1, len(query))) // ratio)
        return _clean((query + " ") * reps + generated)

    @staticmethod
    def concat_interleave(query: str, passages: List[str]) -> str:
        """Interleave query between each passage: q p1 q p2 …"""
        parts = []
        for p in passages:
            parts.extend([query, p])
        return _clean(" ".join(parts))


def _clean(text: str) -> str:
    """Normalise whitespace and strip stray quotes.

    Tabs are replaced with spaces so the reformulated text is safe for
    TSV output (qid\\treformulated_query).
    """
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip().strip('"').strip("'")

