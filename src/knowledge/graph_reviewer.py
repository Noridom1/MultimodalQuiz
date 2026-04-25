from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from src.knowledge.concept_normalizer import CandidateMerge, CanonicalConcept
from src.utils.llm import LLMClient


@dataclass
class MergeProposal:
    concept_ids: list[str]
    canonical_label: str
    confidence_score: float
    reason: str


@dataclass
class GraphReviewReport:
    enabled: bool
    method: str
    merge_proposals: list[MergeProposal] = field(default_factory=list)
    do_not_merge: list[dict[str, Any]] = field(default_factory=list)
    needs_review: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "merge_proposals": [asdict(item) for item in self.merge_proposals],
            "do_not_merge": self.do_not_merge,
            "needs_review": self.needs_review,
            "warnings": self.warnings,
        }


def review_graph_for_merges(
    concepts: list[CanonicalConcept],
    candidate_merges: list[CandidateMerge],
    *,
    llm_client: LLMClient | None = None,
    enabled: bool | None = None,
) -> GraphReviewReport:
    if enabled is None:
        enabled = os.getenv("QUIZGEN_KG_ENABLE_LLM_REVIEW", "0").strip().lower() in {"1", "true", "yes"}

    if not candidate_merges:
        return GraphReviewReport(enabled=enabled, method="none")

    if not enabled:
        return GraphReviewReport(
            enabled=False,
            method="disabled",
            needs_review=[
                {
                    "concept_ids": candidate.concept_ids,
                    "labels": candidate.labels,
                    "score": candidate.score,
                    "reason": candidate.reason,
                }
                for candidate in candidate_merges
            ],
        )

    reviewer = llm_client or LLMClient()
    concept_index = {concept.id: concept for concept in concepts}
    prompt_payload = {
        "concepts": [
            {
                "id": concept.id,
                "label": concept.label,
                "aliases": concept.aliases,
                "mention_count": concept.mention_count,
                "definitions": concept.definitions[:3],
                "source_chunk_ids": concept.source_chunk_ids[:10],
                "section_paths": concept.section_paths[:5],
            }
            for concept in concepts
        ],
        "candidate_merges": [
            {
                "concept_ids": candidate.concept_ids,
                "labels": candidate.labels,
                "heuristic_score": candidate.score,
                "heuristic_reason": candidate.reason,
            }
            for candidate in candidate_merges[:50]
        ],
    }

    system_prompt = (
        "You review a knowledge graph and decide whether candidate concept nodes should be merged. "
        "Only judge identity, not topical relatedness. Return strict JSON."
    )
    user_prompt = (
        "Review the candidate concept merges. "
        "Prefer conservative decisions. "
        "Return JSON with keys merge_proposals, do_not_merge, needs_review. "
        "Each merge proposal must contain concept_ids, canonical_label, confidence_score, and reason.\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )

    try:
        raw = reviewer.complete(user_prompt, system_prompt=system_prompt)
        parsed = json.loads(_clean_json(raw))
    except Exception as exc:
        return GraphReviewReport(
            enabled=True,
            method="llm",
            warnings=[f"LLM graph review failed: {exc}"],
            needs_review=[
                {
                    "concept_ids": candidate.concept_ids,
                    "labels": candidate.labels,
                    "score": candidate.score,
                    "reason": candidate.reason,
                }
                for candidate in candidate_merges
            ],
        )

    proposals: list[MergeProposal] = []
    for item in parsed.get("merge_proposals", []):
        if not isinstance(item, dict):
            continue
        concept_ids = [str(value).strip() for value in item.get("concept_ids", []) if str(value).strip()]
        if len(concept_ids) < 2 or any(concept_id not in concept_index for concept_id in concept_ids):
            continue
        proposals.append(
            MergeProposal(
                concept_ids=concept_ids,
                canonical_label=str(item.get("canonical_label", "")).strip() or concept_index[concept_ids[0]].label,
                confidence_score=float(item.get("confidence_score", 0.0)),
                reason=str(item.get("reason", "")).strip() or "llm_review",
            )
        )

    return GraphReviewReport(
        enabled=True,
        method="llm",
        merge_proposals=proposals,
        do_not_merge=list(parsed.get("do_not_merge", [])) if isinstance(parsed.get("do_not_merge"), list) else [],
        needs_review=list(parsed.get("needs_review", [])) if isinstance(parsed.get("needs_review"), list) else [],
    )


def _clean_json(raw: object) -> str:
    text = str(raw).strip()
    if text.startswith("```"):
        closing = text.rfind("```")
        if closing > 3:
            text = text[3:closing].strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text
