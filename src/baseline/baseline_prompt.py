from __future__ import annotations

from textwrap import dedent


def _format_difficulty_distribution(difficulty_distribution: dict[str, float]) -> str:
    parts = [f"{key}={value:.2f}" for key, value in difficulty_distribution.items()]
    return ", ".join(parts)


def compute_difficulty_targets(num_questions: int, difficulty_distribution: dict[str, float]) -> dict[str, int]:
    ordered_keys = ["easy", "medium", "hard"]
    weights = {key: float(difficulty_distribution.get(key, 0.0)) for key in ordered_keys}

    raw_counts = {key: num_questions * weights[key] for key in ordered_keys}
    counts = {key: int(raw_counts[key]) for key in ordered_keys}
    remainder = num_questions - sum(counts.values())

    fractions = sorted(
        ((raw_counts[key] - counts[key], key) for key in ordered_keys),
        reverse=True,
    )
    for _, key in fractions:
        if remainder <= 0:
            break
        counts[key] += 1
        remainder -= 1

    if remainder > 0:
        counts["medium"] += remainder

    return counts


def build_baseline_prompt(
    document_text: str,
    *,
    num_questions: int,
    difficulty_distribution: dict[str, float],
) -> str:
    difficulty_targets = compute_difficulty_targets(num_questions, difficulty_distribution)
    difficulty_text = _format_difficulty_distribution(difficulty_distribution)

    return dedent(
        f"""
        You are generating a baseline quiz directly from a document.

        Return strict JSON only with this shape:
        {{
          "questions": [
            {{
              "question_text": "string",
              "options": ["string", "string", "string", "string"],
              "correct_answer": "string",
              "explanation": "string",
              "target_concept": "string",
              "difficulty": "easy|medium|hard",
              "question_type": "multiple_choice",
              "image_grounded": false
            }}
          ]
        }}

        Requirements:
        - Generate exactly {num_questions} questions.
        - Use the raw document as the only source of information.
        - Do not use graph, planner, or image-generation context.
        - Keep all questions in multiple_choice format.
        - Include exactly 4 non-empty options per question.
        - Make the correct answer one of the option texts or A/B/C/D.
        - Keep explanations concise and evidence-based.
        - Do not output markdown fences or extra commentary.

        Difficulty target distribution:
        - easy: {difficulty_targets['easy']}
        - medium: {difficulty_targets['medium']}
        - hard: {difficulty_targets['hard']}

        Difficulty distribution hint: {difficulty_text}

        Document:
        {document_text}
        """
    ).strip()