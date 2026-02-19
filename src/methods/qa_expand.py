"""QA-Expand — sub-question decomposition, answer generation, and filtering."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .base import BaseMethod, Query, ReformulatedQuery, _clean


class QAExpand(BaseMethod):
    """Three-stage pipeline: (1) generate sub-questions, (2) produce answers,
    (3) filter/refine answers via LLM, then concatenate retained answers
    with repeated query.

    Reference: Jagerman et al., *QA-Expand*, 2023.
    """

    NAME = "qa_expand"

    def reformulate_query(
        self,
        query: Query,
        contexts: Optional[List[str]] = None,
    ) -> ReformulatedQuery:
        num_subq = int(self.params.get("num_subquestions", 3))
        query_repeats = int(self.params.get("query_repeats", 3))

        # ── Step 1: generate sub-questions ──────────────────────────────
        msgs_sq = self.prompts.render("qa_expand_subq", query=query.text)
        raw_sq = self.llm.generate_one(msgs_sq)
        subquestions = self._parse_list(raw_sq, num_subq, prefix="question")

        # ── Step 2: generate answers ────────────────────────────────────
        questions_json = json.dumps(
            {f"question{i+1}": q for i, q in enumerate(subquestions)}
        )
        msgs_ans = self.prompts.render("qa_expand_answer", questions=questions_json)
        raw_ans = self.llm.generate_one(msgs_ans)
        answers = self._parse_list(raw_ans, num_subq, prefix="answer")

        # ── Step 3: filter / refine answers ─────────────────────────────
        answers_json = json.dumps(
            {f"answer{i+1}": a for i, a in enumerate(answers)}
        )
        msgs_ref = self.prompts.render(
            "qa_expand_refine", query=query.text, answers=answers_json
        )
        raw_ref = self.llm.generate_one(msgs_ref)
        kept = self._extract_kept_answers(raw_ref, num_subq)

        # ── Construct expanded query ────────────────────────────────────
        selected = [answers[i] for i in kept if i < len(answers)]
        cleaned = [_clean(a) for a in selected if a.strip()]
        reformulated = self.concat_repeat(
            query.text, " ".join(cleaned), query_repeats
        )

        return ReformulatedQuery(
            qid=query.qid,
            original=query.text,
            reformulated=reformulated,
            metadata={
                "subquestions": subquestions,
                "answers": answers,
                "kept_indices": kept,
            },
        )

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_list(raw: str, n: int, prefix: str) -> List[str]:
        """Try to extract a JSON dict like ``{prefix1: ..., prefix2: ...}``
        falling back to line-split if JSON parsing fails."""
        try:
            data = _robust_json(raw)
            return [data.get(f"{prefix}{i+1}", "") for i in range(n)]
        except Exception:
            lines = [l.strip("-•* \t") for l in raw.splitlines() if l.strip()]
            return (lines + [""] * n)[:n]

    @staticmethod
    def _extract_kept_answers(raw: str, n: int) -> List[int]:
        """Return indices of answers the refine step kept."""
        try:
            data = _robust_json(raw)
            return [
                i for i in range(n)
                if data.get(f"answer{i+1}", "").strip()
            ]
        except Exception:
            return list(range(n))


def _robust_json(text: str) -> Dict:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    # strip markdown fences
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    return json.loads(text)

