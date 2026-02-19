"""Query2Doc (Q2D) — pseudo-document generation in three prompting variants."""

from __future__ import annotations

from typing import List, Optional

from .base import BaseMethod, Query, ReformulatedQuery, _clean


class _Q2DBase(BaseMethod):
    """Shared logic for all Query2Doc variants.

    The LLM generates an answer-style pseudo-document that is concatenated
    with the original query repeated ``query_repeats`` times (default 5).

    Expansion pattern: ``(q × 5) + P``

    Reference: Wang et al., *Query2Doc*, EMNLP 2023.
    """

    PROMPT_ID: str = ""  # overridden by subclasses

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        query_repeats = int(self.params.get("query_repeats", 5))

        msgs = self._build_messages(query)
        passage = self.llm.generate_one(msgs)
        reformulated = self.concat_repeat(query.text, passage, query_repeats)

        return ReformulatedQuery(
            qid=query.qid,
            original=query.text,
            reformulated=reformulated,
            metadata={"pseudo_document": passage},
        )

    def _build_messages(self, query: Query) -> List[dict]:
        return self.prompts.render(self.PROMPT_ID, query=query.text)


class Query2DocZS(_Q2DBase):
    """Zero-shot Query2Doc."""

    NAME = "q2d_zs"
    PROMPT_ID = "q2d_zs"


class Query2DocFS(_Q2DBase):
    """Few-shot Query2Doc with in-context examples."""

    NAME = "q2d_fs"
    PROMPT_ID = "q2d_fs"

    def _build_messages(self, query: Query) -> List[dict]:
        examples = self.params.get("examples", "")
        return self.prompts.render(self.PROMPT_ID, query=query.text, examples=examples)


class Query2DocCoT(_Q2DBase):
    """Chain-of-thought Query2Doc."""

    NAME = "q2d_cot"
    PROMPT_ID = "q2d_cot"

