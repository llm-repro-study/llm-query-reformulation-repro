"""MUGI — multi-generation integration with adaptive query weighting."""

from __future__ import annotations

from typing import List, Optional

from .base import BaseMethod, Query, ReformulatedQuery


class MUGI(BaseMethod):
    """Generate multiple independent pseudo-documents and consolidate them
    into a single expanded representation.  The original query is repeated
    proportionally to the length of the generated content (adaptive
    weighting).

    Reference: Zhang et al., *MUGI*, SIGIR 2024.
    """

    NAME = "mugi"

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        num_docs = int(self.params.get("num_docs", 5))
        adaptive_ratio = int(self.params.get("adaptive_ratio", 5))

        pseudo_docs: List[str] = []
        for _ in range(num_docs):
            msgs = self.prompts.render("mugi", query=query.text)
            doc = self.llm.generate_one(msgs)
            pseudo_docs.append(doc)

        generated = " ".join(pseudo_docs)
        reformulated = self.concat_adaptive(
            query.text, generated, ratio=adaptive_ratio
        )

        return ReformulatedQuery(
            qid=query.qid,
            original=query.text,
            reformulated=reformulated,
            metadata={"pseudo_docs": pseudo_docs},
        )

