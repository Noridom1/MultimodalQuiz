"""Question format profiles for quiz planning and generation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

# Internal canonical identifiers (snake_case)
CANONICAL_QUESTION_TYPES = frozenset(
    {"multiple_choice", "true_false", "fill_in_blank", "matching"}
)

_TYPE_ALIASES: dict[str, str] = {
    "multiple_choice": "multiple_choice",
    "multiple-choice": "multiple_choice",
    "multiplechoice": "multiple_choice",
    "mcq": "multiple_choice",
    "true_false": "true_false",
    "true-false": "true_false",
    "truefalse": "true_false",
    "t/f": "true_false",
    "fill_in_blank": "fill_in_blank",
    "fill-in-blank": "fill_in_blank",
    "fill_in_the_blank": "fill_in_blank",
    "fib": "fill_in_blank",
    "matching": "matching",
    "match": "matching",
}


def normalize_question_type(raw: str | None) -> str:
    """Map user/LLM question_type strings to a canonical value."""
    if raw is None:
        return ""
    key = str(raw).strip().lower().replace(" ", "_")
    if not key:
        return ""
    return _TYPE_ALIASES.get(key, key if key in CANONICAL_QUESTION_TYPES else key)


def _round_allocations(weights: dict[str, float], n: int) -> dict[str, int]:
    """Largest remainder method so counts sum to n."""
    if n <= 0:
        return {k: 0 for k in weights}
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("distribution weights must sum to a positive value")
    norm = {k: v / total_w for k, v in weights.items()}
    raw = {k: n * v for k, v in norm.items()}
    floors = {k: int(math.floor(raw[k])) for k in raw}
    rem = n - sum(floors.values())
    fracs = sorted(((raw[k] - floors[k], k) for k in raw), reverse=True)
    for i in range(rem):
        floors[fracs[i % len(fracs)][1]] += 1
    return floors


@dataclass(frozen=True)
class QuestionFormatProfile:
    """Distribution over canonical question types; values should sum to 1.0."""

    distribution: dict[str, float] = field(
        default_factory=lambda: {"multiple_choice": 1.0}
    )

    def __post_init__(self) -> None:
        merged: dict[str, float] = {}
        for k, v in self.distribution.items():
            ck = normalize_question_type(str(k))
            if ck not in CANONICAL_QUESTION_TYPES:
                raise ValueError(f"Unknown question format key: {k!r}")
            fv = float(v)
            if fv < 0 or not math.isfinite(fv):
                raise ValueError(f"Invalid weight for {k!r}: {v}")
            merged[ck] = merged.get(ck, 0.0) + fv
        s = sum(merged.values())
        if s <= 0:
            raise ValueError("distribution must have at least one positive weight")
        object.__setattr__(self, "distribution", {k: v / s for k, v in merged.items()})

    @property
    def allowed_types(self) -> frozenset[str]:
        return frozenset(self.distribution.keys())

    def is_mcq_only(self) -> bool:
        return self.allowed_types == frozenset({"multiple_choice"})

    def expected_counts(self, num_questions: int) -> dict[str, int]:
        return _round_allocations(self.distribution, num_questions)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {"question_format_distribution": dict(self.distribution)}

    def distribution_within_tolerance(
        self,
        counts: Mapping[str, int],
        num_questions: int,
        *,
        tolerance: float = 0.35,
    ) -> bool:
        """True if empirical counts are close enough to expected (for LLM slack)."""
        if num_questions <= 0:
            return True
        expected = self.expected_counts(num_questions)
        for t, exp_n in expected.items():
            got = int(counts.get(t, 0))
            max_dev = max(1, int(math.ceil(num_questions * tolerance)))
            if abs(got - exp_n) > max_dev:
                return False
        for t in counts:
            if counts[t] and t not in expected:
                return False
        return True


def default_format_profile() -> QuestionFormatProfile:
    return QuestionFormatProfile(distribution={"multiple_choice": 1.0})


def coerce_format_profile(
    value: QuestionFormatProfile | Mapping[str, float] | str | None,
) -> QuestionFormatProfile:
    if value is None:
        return default_format_profile()
    if isinstance(value, QuestionFormatProfile):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default_format_profile()
        canon = normalize_question_type(s)
        if canon in CANONICAL_QUESTION_TYPES and ":" not in s and "{" not in s:
            return QuestionFormatProfile(distribution={canon: 1.0})
        return parse_question_formats_json(s)
    if isinstance(value, Mapping):
        d = {str(k): float(v) for k, v in value.items()}
        return QuestionFormatProfile(distribution=d)
    raise TypeError(f"Cannot coerce format profile from {type(value)}")


def parse_question_formats_json(s: str) -> QuestionFormatProfile:
    """Parse JSON object like {\"multiple_choice\":0.7,\"true_false\":0.3}."""
    data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("question formats JSON must be an object")
    return QuestionFormatProfile(distribution={str(k): float(v) for k, v in data.items()})


def parse_question_formats_cli_arg(s: str) -> QuestionFormatProfile:
    """CLI: either JSON object or single canonical type name."""
    s = s.strip()
    if not s:
        return default_format_profile()
    if s.startswith("{"):
        return parse_question_formats_json(s)
    return coerce_format_profile(s)


def load_format_profile_from_pipeline_config(pipeline: Any) -> QuestionFormatProfile | None:
    """Read profile from configs/default.yaml ``pipeline`` section."""
    if not isinstance(pipeline, dict):
        return None
    raw = pipeline.get("question_format_distribution")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("pipeline.question_format_distribution must be a mapping")
    return QuestionFormatProfile(distribution={str(k): float(v) for k, v in raw.items()})


def merge_question_metadata(
    base: dict[str, Any],
    payload: dict[str, object],
    *,
    question_type: str,
) -> dict[str, Any]:
    """Copy type-specific keys from LLM payload into question metadata."""
    out = dict(base)
    qt = normalize_question_type(question_type)
    if qt == "matching":
        for key in (
            "matching_left",
            "matching_right",
            "matching_solution",
            "matching_pairs",
        ):
            if key in payload and payload[key] is not None:
                out[key] = payload[key]
    return out
