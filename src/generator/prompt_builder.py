from __future__ import annotations


def build_question_prompt(question_plan: dict[str, object]) -> str:
    """Build the prompt used to generate a question from a plan."""
    raise NotImplementedError("Question prompt construction is not implemented yet.")


def build_image_prompt(question_plan: dict[str, object]) -> str:
    """Build the prompt used to generate an image for a question."""
    raise NotImplementedError("Image prompt construction is not implemented yet.")
