"""LameR — corpus-grounded rewriting conditioned on retrieved evidence."""

from __future__ import annotations

from typing import List, Optional

from .base import BaseMethod, Query, ReformulatedQuery


class LameR(BaseMethod):
    """Retrieve the top-k documents for a query, then condition the LLM on
    that evidence to produce multiple rewrites.  The expanded query
    interleaves the original query between each generated passage.

    Reference: Wang et al., *LameR*, EMNLP 2023.
    """

    NAME = "lamer"
    REQUIRES_CONTEXTS = True

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        num_gen = int(self.params.get("num_passages", 5))
        ctx_k = int(self.params.get("context_k", 10))
        dataset_tag = self.params.get("dataset", "msmarco")

        ctxs = (contexts or [])[:ctx_k]
        ctx_blob = "\n".join(f"{i+1}. {p}" for i, p in enumerate(ctxs))

        # Choose the dataset-specific prompt (falls back to generic)
        prompt_id = f"lamer_{dataset_tag}"
        try:
            self.prompts.render(prompt_id, query="test", contexts="test")
        except KeyError:
            prompt_id = "lamer_msmarco"

        passages: List[str] = []
        for _ in range(num_gen):
            msgs = self.prompts.render(prompt_id, query=query.text, contexts=ctx_blob)
            passage = self.llm.generate_one(msgs)
            passages.append(passage.strip().strip('"').strip("'"))

        reformulated = self.concat_interleave(query.text, passages)

        return ReformulatedQuery(
            qid=query.qid,
            original=query.text,
            reformulated=reformulated,
            metadata={
                "generated_passages": passages,
                "n_contexts_used": len(ctxs),
            },
        )

