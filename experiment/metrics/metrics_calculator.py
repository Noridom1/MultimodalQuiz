from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _cosine_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)

    if not left_tokens or not right_tokens:
        return 0.0

    left_count = Counter(left_tokens)
    right_count = Counter(right_tokens)

    dot = 0.0
    for token, count in left_count.items():
        dot += float(count * right_count.get(token, 0))

    left_norm = math.sqrt(sum(float(v * v) for v in left_count.values()))
    right_norm = math.sqrt(sum(float(v * v) for v in right_count.values()))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


def _best_match(source: str, candidates: list[str]) -> tuple[int, float]:
    if not candidates:
        return -1, 0.0

    best_idx = 0
    best_score = -1.0
    for idx, candidate in enumerate(candidates):
        score = _cosine_similarity(source, candidate)
        if score > best_score:
            best_idx = idx
            best_score = score

    return best_idx, max(0.0, best_score)


def compute_coverage(
    document_concepts: list[str],
    quiz_concepts: list[str],
    *,
    threshold: float = 0.75,
) -> dict[str, Any]:
    """
    Soft coverage: mean over document concepts of best similarity to any quiz concept.
    Thresholded coverage: fraction of document concepts whose best similarity >= threshold.
    """
    if not document_concepts:
        return {
            "soft_coverage": 0.0,
            "thresholded_coverage": 0.0,
            "threshold": threshold,
            "matched_doc_count": 0,
            "total_doc_count": 0,
            "doc_to_quiz_matches": [],
        }

    matches: list[dict[str, Any]] = []
    best_scores: list[float] = []

    for doc_concept in document_concepts:
        quiz_idx, best_score = _best_match(doc_concept, quiz_concepts)
        quiz_concept = quiz_concepts[quiz_idx] if quiz_idx >= 0 else None
        matches.append(
            {
                "document_concept": doc_concept,
                "matched_quiz_concept": quiz_concept,
                "similarity": round(best_score, 6),
            }
        )
        best_scores.append(best_score)

    matched_doc_count = sum(1 for score in best_scores if score >= threshold)
    soft = sum(best_scores) / float(len(best_scores))
    thresholded = matched_doc_count / float(len(best_scores))

    return {
        "soft_coverage": round(soft, 6),
        "thresholded_coverage": round(thresholded, 6),
        "threshold": threshold,
        "matched_doc_count": matched_doc_count,
        "total_doc_count": len(document_concepts),
        "doc_to_quiz_matches": matches,
    }


def compute_duplication(document_concepts: list[str], quiz_concepts: list[str]) -> dict[str, Any]:
    """
    Duplication (redundancy) as:
    1 - unique matched document concepts / total quiz concepts
    """
    if not quiz_concepts:
        return {
            "duplication": 0.0,
            "unique_matched_document_concepts": 0,
            "total_quiz_concepts": 0,
            "quiz_to_doc_matches": [],
        }

    matched_indices: list[int] = []
    quiz_to_doc: list[dict[str, Any]] = []

    for quiz_concept in quiz_concepts:
        doc_idx, score = _best_match(quiz_concept, document_concepts)
        doc_concept = document_concepts[doc_idx] if doc_idx >= 0 else None
        matched_indices.append(doc_idx)
        quiz_to_doc.append(
            {
                "quiz_concept": quiz_concept,
                "matched_document_concept": doc_concept,
                "similarity": round(score, 6),
            }
        )

    unique_matches = len({idx for idx in matched_indices if idx >= 0})
    duplication = 1.0 - (unique_matches / float(len(quiz_concepts)))

    return {
        "duplication": round(max(0.0, duplication), 6),
        "unique_matched_document_concepts": unique_matches,
        "total_quiz_concepts": len(quiz_concepts),
        "quiz_to_doc_matches": quiz_to_doc,
    }


def compute_breadth(document_concepts: list[str], quiz_concepts: list[str]) -> dict[str, Any]:
    """
    Breadth proxy based on the distribution entropy of nearest document-concept matches.
    Normalized entropy is in [0, 1]. Higher means broader coverage over document concepts.
    """
    if not quiz_concepts or not document_concepts:
        return {
            "breadth_entropy": 0.0,
            "breadth_normalized_entropy": 0.0,
            "matched_document_concepts": 0,
            "total_document_concepts": len(document_concepts),
        }

    buckets: Counter[int] = Counter()
    for quiz_concept in quiz_concepts:
        doc_idx, _score = _best_match(quiz_concept, document_concepts)
        if doc_idx >= 0:
            buckets[doc_idx] += 1

    total = sum(buckets.values())
    if total == 0:
        return {
            "breadth_entropy": 0.0,
            "breadth_normalized_entropy": 0.0,
            "matched_document_concepts": 0,
            "total_document_concepts": len(document_concepts),
        }

    entropy = 0.0
    for count in buckets.values():
        p = count / float(total)
        entropy -= p * math.log(p)

    max_entropy = math.log(len(buckets)) if len(buckets) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "breadth_entropy": round(entropy, 6),
        "breadth_normalized_entropy": round(normalized_entropy, 6),
        "matched_document_concepts": len(buckets),
        "total_document_concepts": len(document_concepts),
    }


def compute_metrics(
    document_concepts: list[str],
    quiz_concepts: list[str],
    *,
    coverage_threshold: float = 0.75,
) -> dict[str, Any]:
    coverage = compute_coverage(document_concepts, quiz_concepts, threshold=coverage_threshold)
    breadth = compute_breadth(document_concepts, quiz_concepts)
    duplication = compute_duplication(document_concepts, quiz_concepts)

    return {
        "coverage": coverage,
        "breadth": breadth,
        "duplication": duplication,
        "counts": {
            "document_concepts": len(document_concepts),
            "quiz_concepts": len(quiz_concepts),
        },
    }
