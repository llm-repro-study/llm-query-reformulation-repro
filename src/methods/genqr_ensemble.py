"""GenQREnsemble — keyword expansion using diverse instruction paraphrases."""

from __future__ import annotations

from typing import List, Optional

from .base import BaseMethod, Query, ReformulatedQuery


class GenQREnsemble(BaseMethod):
    """Issue the same query to multiple paraphrased instructions and merge
    the resulting keyword sets into one expanded query.

    The prompt bank contains 10 instruction variants under the ids
    ``genqr_ens_1`` … ``genqr_ens_10``.  Keywords from all variants are
    concatenated and prepended by the original query repeated
    ``query_repeats`` times.

    Reference: Chuklin et al., *GenQREnsemble*, 2023.
    """

    NAME = "genqr_ensemble"
    NUM_INSTRUCTIONS = 10

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        query_repeats = int(self.params.get("query_repeats", 5))

        all_keywords: List[str] = []
        for i in range(1, self.NUM_INSTRUCTIONS + 1):
            prompt_id = f"genqr_ens_{i}"
            msgs = self.prompts.render(prompt_id, query=query.text)
            resp = self.llm.generate_one(msgs)
            all_keywords.append(resp)

        keyword_text = " ".join(all_keywords)
        reformulated = self.concat_repeat(query.text, keyword_text, query_repeats)

        return ReformulatedQuery(
            qid=query.qid,
            original=query.text,
            reformulated=reformulated,
            metadata={"keyword_sets": all_keywords},
        )

