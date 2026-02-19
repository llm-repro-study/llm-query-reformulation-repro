"""CSQE — corpus-grounded expansion combining knowledge-based and context-based signals."""

from __future__ import annotations

import re
from typing import List, Optional

from .base import BaseMethod, Query, ReformulatedQuery, _clean


class CSQE(BaseMethod):
    """Two-pronged expansion that combines:
      1. *KEQE* — LLM generates passages from parametric knowledge alone.
      2. *CSQE* — LLM extracts key sentences from initially retrieved documents.

    Both sets of expansions are concatenated with the repeated query.

    Reference: Mao et al., *CSQE*, 2024.
    """

    NAME = "csqe"
    REQUIRES_CONTEXTS = True

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        n_gen = int(self.params.get("n_expansions", 2))
        ctx_k = int(self.params.get("context_k", 10))

        # ── 1. Knowledge-based passages (KEQE) ──────────────────────────
        msgs_keqe = self.prompts.render("keqe", query=query.text)
        keqe_passages = self.llm.generate(msgs_keqe, n=n_gen)

        # ── 2. Context-based sentence extraction (CSQE) ─────────────────
        ctxs = (contexts or [])[:ctx_k]
        ctx_blob = "\n".join(f"{i+1}. {p}" for i, p in enumerate(ctxs))

        msgs_csqe = self.prompts.render("csqe", query=query.text, contexts=ctx_blob)
        csqe_raw = self.llm.generate(msgs_csqe, n=n_gen)
        csqe_sentences = [self._extract_sentences(r) for r in csqe_raw]

        # ── 3. Concatenate: query × n + KEQE passages + CSQE sentences ──
        parts = [query.text] * n_gen + keqe_passages + csqe_sentences
        reformulated = _clean(" ".join(parts)).lower()

        return ReformulatedQuery(
            qid=query.qid,
            original=query.text,
            reformulated=reformulated,
            metadata={
                "keqe_passages": keqe_passages,
                "csqe_sentences": csqe_sentences,
                "n_contexts_used": len(ctxs),
            },
        )

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_sentences(text: str) -> str:
        """Pull quoted sentences from the CSQE response; fall back to
        numbered-document extraction."""
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return " ".join(quoted)

        # fallback: grab content after numbered markers
        cleaned = re.sub(
            r"^Relevant Documents?:?\s*\n?", "", text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        chunks = re.findall(r"\d+[.:]\s*(.+?)(?=\d+[.:]|$)", cleaned, re.DOTALL)
        if chunks:
            return " ".join(" ".join(c.split()) for c in chunks if c.strip())
        return ""

