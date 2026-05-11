# Planner & Generator: Agentic vs Legacy

This document explains the agentic `TopicAgenticPlanner` and the modern `GenerationOrchestrator`/`LLMQuestionGenerator` flow, and contrasts it with the legacy quiz generation approach used previously in this codebase.

## Quick summary

- Agentic planner: topic-driven, LLM-assisted planning that explicitly maximizes concept coverage, attaches grounded knowledge, and produces structured `QuestionPlan` objects.
- Generator (modern): builds multi-chunk question prompts, generates LLM-driven natural-language image prompts, submits images in parallel with retries, and produces validated question JSON.
- Legacy flow: older `QuizPlanner`/legacy generator (``generation_mode=legacy``) used simpler templates, single-chunk contexts, and synchronous image+question generation with limited grounding guarantees.

## Key components (agentic)

- `TopicAgenticPlanner` (`src/planner/topic_planner.py`)
  - Input: `MultimodalDocumentGraph` (topics, concepts, chunks, images)
  - Outputs: list of `QuestionPlan` objects with fields like `target_concept`, `question_type`, `difficulty`, `reasoning_type`, `image_role`, `image_description`, `tested_fact_block_id`, and `metadata` (including `knowledge_context`).
  - Behaviors:
    - Retrieve rich `TopicContext` via `TopicContextRetriever`.
    - Greedy allocation across topics to maximize unique concept coverage.
    - Enforce mandatory `tested_fact_block_id` to ensure each question cites a grounded fact.
    - Attach top-N chunks per plan into `plan.metadata['knowledge_context']` (1/3/5 depending on reasoning type).

- `GenerationOrchestrator` (`src/generator/orchestrator.py`)
  - Builds question prompts using `PromptBuilder.build_question_prompt` (formats multi-chunk contexts).
  - Uses `PromptBuilder.build_image_prompt_via_llm` to ask the LLM for a concise natural-language image instruction.
  - Submits image generation tasks concurrently (ThreadPoolExecutor), with retries and resilient polling (configured in `configs/model_config.yaml`).
  - Calls `LLMQuestionGenerator.inference` to produce validated question JSON and ensures image grounding when required.

- `PromptBuilder` (`src/generator/prompt_builder.py`)
  - Produces LLM prompts for question generation and image generation.
  - Question prompts include an explicit `Tested fact block ID` and the `knowledge_context` block (multi-chunk text).
  - Image prompts are natural-language instructions returned by the LLM, not JSON.

## How the agentic workflow differs from legacy

- Grounding and provenance
  - Legacy: often relied on looser retrieval or single-chunk context; less strict citation of the fact used in the question.
  - Agentic: enforces `tested_fact_block_id`, attaches explicit chunk text in `knowledge_context`, and includes the cited block ID in the question prompt to reduce hallucinations and increase traceability.

- Coverage and allocation
  - Legacy: static or proportion-based allocation per section/topic, risk of repeated concepts.
  - Agentic: greedy allocation across topics that tracks `used_concepts` to maximize unique concept coverage under a fixed question budget.

- Prompt richness
  - Legacy: shorter prompts and fewer context chunks (often 1 chunk per concept).
  - Agentic: planner exposes up to 5 chunks per concept to the LLM, and the prompt builder formats multi-chunk contexts with headers and confidence metadata.

- Image prompting and generation
  - Legacy: static, rule-based image prompt templates and synchronous image generation.
  - Agentic: the image prompt is created by asking the LLM for a short natural-language instruction (1–3 sentences); image submission is retried and polled with resilience; tasks are run in parallel to speed up runs.

- Observability and audit
  - Legacy: fewer logs and less prompt-level recording.
  - Agentic: per-question prompt logs (question prompt, image prompt, raw LLM responses, knowledge_context) are written to `outputs/<run_id>/generation/prompt_logs/` for offline inspection and audit.

- Error handling and robustness
  - Legacy: failures in image generation often aborted runs without retries.
  - Agentic: image submission has exponential backoff retries; polling tolerates transient failures up to a threshold, and the orchestrator aggregates results from concurrent tasks.

## Migration notes

- Config
  - Use `configs/model_config.yaml` to centralize LLM and image provider settings. Ensure API keys are set in environment variables (e.g., `MISTRAL_API_KEY`, `FREEPIK_API_KEY`).

- Testing
  - Use the flags `--mock-image --mock-question` for local testing without provider calls.
  - For integration tests, inject a mock `LLMClient` into `TopicAgenticPlanner` and `LLMQuestionGenerator`.

- Extensibility
  - The `LLMClient` abstraction allows supporting multiple providers; share a single `LLMClient` instance across planner and generator for consistent behavior.
  - The `ImageGenerator` supports pluggable providers; retries and worker pool make it possible to add rate-limit-aware pooling later.

## Where to look in the code

- Planner: `src/planner/topic_planner.py`
- Prompt templates: `src/planner/topic_prompt_templates.py`
- Prompt builder: `src/generator/prompt_builder.py`
- Question generator: `src/generator/question_gen.py`
- Image generator: `src/generator/image_gen.py`
- Orchestrator: `src/generator/orchestrator.py`
- LLM client: `src/utils/llm.py`
- Validator & retriever: `src/knowledge/validator.py`, `src/knowledge/retriever.py`

---

This file was generated to document the planner+generator architecture and how it improves upon the legacy pipeline.
