import os
from dotenv import load_dotenv

# Load .env early so subsequent os.getenv calls see values
load_dotenv()

# Top-level provider selection
PROVIDER = os.getenv("QUIZGEN_LLM_PROVIDER", "openai").strip().lower()

# Global defaults
GLOBAL_MODEL = os.getenv("QUIZGEN_LLM_MODEL", "gpt-5.4-mini")
TIMEOUT_SECONDS = int(os.getenv("QUIZGEN_LLM_TIMEOUT_SECONDS", "60"))

# OpenAI settings
OPENAI = {
    "API_KEY": os.getenv("OPENAI_API_KEY"),
    "MODEL": os.getenv("QUIZGEN_OPENAI_MODEL", GLOBAL_MODEL),
    "ENDPOINT": os.getenv("QUIZGEN_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
}

# Google Gemini settings
GEMINI = {
    "API_KEY": os.getenv("GOOGLE_API_KEY"),
    "MODEL": os.getenv("QUIZGEN_GEMINI_MODEL", "gemini-1.5-flash"),
}

# Mistral settings
MISTRAL = {
    "API_KEY": os.getenv("MISTRAL_API_KEY"),
    "MODEL": os.getenv("QUIZGEN_MISTRAL_MODEL", "mistral-small-latest"),
}

# Groq settings
GROQ = {
    "API_KEY": os.getenv("GROQ_API_KEY"),
    "MODEL": os.getenv("QUIZGEN_GROQ_MODEL", "qwen/qwen3-32b"),
}

# Pipeline / Extraction settings
EXTRACTOR_BACKEND = os.getenv("QUIZGEN_EXTRACTOR_BACKEND", "langchain")
EXTRACTION_GRANULARITY = os.getenv("QUIZGEN_EXTRACTION_GRANULARITY", "balanced")

# Knowledge Graph settings
KG_MAX_TOKENS = int(os.getenv("QUIZGEN_KG_MAX_TOKENS", "280"))
KG_OVERLAP_BLOCKS = int(os.getenv("QUIZGEN_KG_OVERLAP_BLOCKS", "1"))

def model_for(provider: str) -> str:
    p = provider.strip().lower()
    if p in {"openai"}:
        return OPENAI.get("MODEL") or GLOBAL_MODEL
    if p in {"google", "gemini"}:
        return GEMINI.get("MODEL") or GLOBAL_MODEL
    if p in {"mistral", "mistralai"}:
        return MISTRAL.get("MODEL") or GLOBAL_MODEL
    if p in {"groq", "groqai"}:
        return GROQ.get("MODEL") or GLOBAL_MODEL
    return GLOBAL_MODEL
