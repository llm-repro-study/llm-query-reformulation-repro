"""Query2Keyword (Q2K) — single-call keyword expansion."""

from __future__ import annotations

from typing import List, Optional

from .base import BaseMethod, Query, ReformulatedQuery


class Query2Keyword(BaseMethod):
    """Map a query to semantically related expansion terms in a single LLM
    call, broadening lexical coverage without pseudo-document synthesis.

    Reference: Wang et al., *Exploring the Impact of Query Reformulation*, 2024.
    """

    NAME = "q2k"

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        query_repeats = int(self.params.get("query_repeats", 5))

        msgs = self.prompts.render("q2k", query=query.text)
        keywords = self.llm.generate_one(msgs)

        reformulated = self.concat_repeat(query.text, keywords, query_repeats)

        return ReformulatedQuery(
            qid=query.qid,
            original=query.text,
            reformulated=reformulated,
            metadata={"keywords": keywords},
        )

