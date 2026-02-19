"""GenQR — keyword-level expansion via N independent LLM calls."""

from __future__ import annotations

from typing import List, Optional

from .base import BaseMethod, Query, ReformulatedQuery, _clean


class GenQR(BaseMethod):
    """Generate expansion keywords through multiple independent LLM calls.

    For each query the LLM is prompted ``num_calls`` times (default 5) to
    produce comma-separated keywords.  All keyword sets are concatenated
    after the original query (no query repetition).

    Expansion pattern: ``q + K₁ + K₂ + … + Kₙ``

    Reference: Bonifacio et al., *InPars-v2* / GenQR, 2022.
    """

    NAME = "genqr"

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        num_calls = int(self.params.get("num_calls", 5))

        all_keywords: List[str] = []
        for _ in range(num_calls):
            msgs = self.prompts.render("genqr", query=query.text)
            resp = self.llm.generate_one(msgs)
            all_keywords.append(resp)

        keyword_text = " ".join(all_keywords)
        reformulated = _clean(f"{query.text} {keyword_text}")

        return ReformulatedQuery(
            qid=query.qid,
            original=query.text,
            reformulated=reformulated,
            metadata={"keywords": all_keywords},
        )

