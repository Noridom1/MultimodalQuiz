from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod

import requests
from pathlib import Path
import importlib.util
import re

# Prefer package import when running inside the project, but allow running
# this file directly by falling back to loading the local settings.py file.
try:
    from src.utils import settings
except Exception:
    settings_path = Path(__file__).resolve().parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", str(settings_path))
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)


# =========================================================
# PROVIDER INTERFACE
# =========================================================

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        pass


# =========================================================
# OPENAI PROVIDER
# =========================================================

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.model = settings.OPENAI.get("MODEL")
        self.endpoint = settings.OPENAI.get("ENDPOINT")
        self.timeout_seconds = settings.TIMEOUT_SECONDS

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        api_key = settings.OPENAI.get("API_KEY")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()


# =========================================================
# GOOGLE GEMINI PROVIDER
# =========================================================

class GeminiProvider(LLMProvider):
    """
    Uses Google Generative Language API (Gemini)
    Docs: https://ai.google.dev
    """

    def __init__(self):
        self.model = settings.GEMINI.get("MODEL")

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        api_key = settings.GEMINI.get("API_KEY")
        if not api_key:
            raise RuntimeError("Missing Google Gemini API key")

        # Lazily import genai to avoid hard dependency until used
        try:
            from google import genai
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("The `genai` package is required for GeminiProvider. "
                               "Install it with `pip install genai`.") from exc

        # Gemini SDK doesn't use a separate system role; prepend if provided
        contents = f"SYSTEM: {system_prompt}\n\n{prompt}" if system_prompt else prompt

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=self.model, contents=contents)

        # SDK response shapes vary by version; try common attributes
        text = getattr(response, "text", None)
        if text is not None:
            text = text.strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            return text

        # Try candidate-based shapes
        try:
            candidate_text = response.candidates[0].content[0].text
            if candidate_text:
                return candidate_text.strip()
        except Exception:
            pass

        try:
            candidate_text = response.candidates[0]["content"]["parts"][0]["text"]
            if candidate_text:
                return candidate_text.strip()
        except Exception:
            pass

        raise RuntimeError("Could not extract text from Gemini response")

# =========================================================
# MistralAI PROVIDER
# =========================================================

class MistralAIProvider(LLMProvider):
    def __init__(self):
        self.model = settings.MISTRAL.get("MODEL")

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_mistralai import ChatMistralAI
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "Mistral support requires 'langchain-core' and 'langchain-mistralai'."
            ) from exc

        api_key = settings.MISTRAL.get("API_KEY")
        if not api_key:
            raise RuntimeError("Missing MistralAI API key")

        # Initialize the MistralAI client
        client = ChatMistralAI(
            model=self.model,
            mistral_api_key=api_key,
            temperature=0.2
        )

        # Prepare the messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Get the response
        response = client.invoke(messages)
        return response.content

# =========================================================
# GROQAI PROVIDER (PLACEHOLDER)
# =========================================================

class GroqAIProvider(LLMProvider):
    def __init__(self):
        self.model = settings.GROQ.get("MODEL")

        from groq import Groq

        api_key = settings.GROQ.get("API_KEY")
        if not api_key:
            raise RuntimeError("Missing Groq API key")

        self.client = Groq(api_key=api_key)

    def extract_answer(self, response: dict) -> str:
        # Normalize and extract the final answer; ignore any <think> analysis.
        try:
            text = None

            # dict-like responses
            if isinstance(response, dict):
                try:
                    text = response["choices"][0]["message"]["content"]
                except Exception:
                    pass
                if not text:
                    try:
                        text = response["choices"][0]["text"]
                    except Exception:
                        pass
                if not text:
                    text = response.get("text") or response.get("content")

            # object-like responses
            else:
                text = getattr(response, "text", None)
                if not text:
                    try:
                        text = response.choices[0].message.content
                    except Exception:
                        try:
                            text = response.choices[0].text
                        except Exception:
                            text = None

            if text is None:
                raise RuntimeError("Could not extract text from Groq response")

            text = str(text).strip()

            # Remove <think>...</think> blocks which contain chain-of-thought
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

            # Prefer explicit <answer> tags
            m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()

            # Prefer bolded markdown like **Paris**
            m = re.search(r"\*\*(.*?)\*\*", text)
            if m:
                return m.group(1).strip()

            # Fallback: pattern like '... is X.' or 'Answer: X'
            m = re.search(r"\b(?:is|are)\s+([A-Z][^\.\n!?]{0,80})[\.\n!?]", text)
            if m:
                return m.group(1).strip()

            m = re.search(r"Answer[:\-]\s*(.+)$", text, re.IGNORECASE)
            if m:
                return m.group(1).strip()

            # Last resort: return cleaned text
            return text
        except Exception as exc:
            raise RuntimeError("Unexpected Groq response format") from exc
        
    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )

        return self.extract_answer(chat_completion)
  
# =========================================================
# FACTORY
# =========================================================

def build_provider() -> LLMProvider:
    provider = settings.PROVIDER
    print(f"Using LLM provider: {provider}")

    if provider == "openai":
        return OpenAIProvider()

    if provider in {"google", "gemini"}:
        return GeminiProvider()
    
    if provider in {"mistral", "mistralai"}:
        return MistralAIProvider()

    if provider in {"groq", "groqai"}:
        return GroqAIProvider()
    
    raise ValueError(f"Unsupported provider: {provider}")


# =========================================================
# CLIENT (CLEAN CORE INTERFACE)
# =========================================================

class LLMClient:
    def __init__(self):
        self.provider = build_provider()

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return self.provider.complete(prompt, system_prompt)

if __name__ == "__main__":
    # Simple test to verify provider works
    client = LLMClient()
    response = client.complete("Hi", system_prompt="You are a helpful assistant.")
    print("LLM Response:", response)