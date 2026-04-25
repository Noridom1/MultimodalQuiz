from typing import List


def scan_prompt_for_forbidden_tokens(prompt: str, forbidden: List[str]) -> List[str]:
    """Return a list of forbidden tokens found in the prompt (case-insensitive).

    This helper is lightweight and intended for quick pre-submit checks.
    """
    found: List[str] = []
    lower = prompt.lower()
    for token in forbidden:
        if token.lower() in lower:
            found.append(token)
    return found
