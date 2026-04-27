# Workflow: End-to-End Data Flow and Module Interactions

This document explains how data moves through the project from entry point to final outputs, and how core modules collaborate at runtime.

## 1) Entry Points

### Primary end-to-end entry point
- scripts/run_pipeline.py
- Calls QuizGenerationPipeline.run in src/pipeline.py
- This is the canonical path for a full run: parse -> extract -> graph -> plan -> generate

### Generation-only entry point
- scripts/orchestrate_generation.py
- Calls GenerationOrchestrator.run in src/generator/orchestrator.py
- Used when a quiz plan already exists and only image/question generation is needed

### Supporting CLIs
- scripts/evaluate.py is a placeholder evaluation entry point (no scoring logic yet)
- src/knowledge/kg_builder.py can be run directly to build/export graph artifacts

## 2) High-Level Runtime Flow

Document path input
-> Document parsing and structure extraction
-> Semantic chunking and knowledge extraction
-> Knowledge graph construction, consolidation, topic induction, validation
-> Quiz planning from graph context
-> Image + question generation per plan item
-> Manifest + logs + artifact package

## 3) Full Pipeline Stage-by-Stage

## Stage A: Run Context and Output Layout Setup

Module:
- src/pipeline.py

Key logic:
- QuizGenerationPipeline.run creates RunContext via RunContext.create
- Creates output tree for a run_id under outputs/<run_id>/ with folders:
  - document
  - extraction
  - graph
  - planning
  - generation
  - logs

Artifacts initialized:
- outputs/<run_id>/manifest.json (written at end)
- outputs/<run_id>/logs/pipeline.log (JSONL event log written throughout)

## Stage B: Document Parsing

Modules:
- src/document_understanding/parser.py
- invoked by src/pipeline.py

Function path:
- parse_document(document_path)

Behavior:
- For PDF:
  - tries MinerU CLI extraction first
  - falls back to PyMuPDF
  - then falls back to pypdf
- For markdown/text:
  - reads file as UTF-8 text
- Normalizes lines and extracts:
  - sections
  - paragraphs
  - figures
  - captions

Data output in memory:
- ParsedDocument dataclass
  - markdown
  - sections
  - paragraphs
  - figures
  - captions

Persisted artifact:
- outputs/<run_id>/document/parsed_document.json

## Stage C: Chunking + Semantic Extraction

Modules:
- src/document_understanding/chunking.py
- src/document_understanding/extractor.py
- src/document_understanding/normalizer.py
- invoked by src/pipeline.py

Function path:
- parse_markdown_blocks(parsed_document.markdown)
- build_semantic_chunks(blocks, max_tokens, overlap_blocks)
- DocumentExtractor.extract_chunks(chunks)

Behavior:
- Markdown is converted into typed blocks (heading, paragraph, table, code, image, details, etc.)
- Blocks are grouped into semantic chunks with token budget and overlap
- Extractor backend:
  - langchain mode (LLM structured output batches), or
  - rule mode fallback
- Extracted schema includes:
  - concepts
  - definitions
  - relations
  - examples
  - chunk_extractions
  - summary
- Concepts are normalized by src/document_understanding/normalizer.py

Persisted artifact:
- outputs/<run_id>/extraction/extracted.json

## Stage D: Knowledge Graph Build Workflow

Modules:
- src/knowledge/kg_builder.py
- src/knowledge/concept_normalizer.py
- src/knowledge/graph_reviewer.py
- src/knowledge/merge_resolver.py
- src/knowledge/schema_consolidator.py
- src/knowledge/topic_inducer.py
- src/knowledge/validator.py
- src/knowledge/schema.py

Function path:
- build_knowledge_graph_workflow(document_understanding, extracted)
- export_graph_bundle(graph, checkpoints)

Behavior inside build_knowledge_graph_workflow:
1. Re-parse markdown blocks and semantic chunks for graph materialization
2. Build hierarchy nodes/edges:
   - document node
   - section nodes
   - contains edges
3. Materialize chunk nodes and sequence/containment edges
4. Attach artifact nodes from atomic blocks (table/code/image/details/raw_html)
5. Normalize concept mentions across chunks into canonical concept nodes
6. Add semantic edges:
   - mentions, defines, supports
   - relation edges from extracted relations (depends_on, part_of, causes, related_to, etc.)
7. Review merge candidates and apply merge proposals
8. Consolidate schema
9. Induce topic nodes and topic edges
10. Validate graph and attach validation metadata

Export behavior:
- Graph exported as project JSON and NetworkX node-link JSON
- Optional HTML visualization
- Checkpoint artifacts exported when available

In pipeline post-processing:
- Export filenames are normalized/renamed into stable names in outputs/<run_id>/graph

Persisted artifacts (typical):
- outputs/<run_id>/graph/graph.json
- outputs/<run_id>/graph/graph_networkx.json
- outputs/<run_id>/graph/graph.html (if enabled)
- outputs/<run_id>/graph/hierarchy.json
- outputs/<run_id>/graph/chunks.json
- outputs/<run_id>/graph/artifact_links.json
- outputs/<run_id>/graph/extraction_raw.json
- outputs/<run_id>/graph/canonicalization.json
- outputs/<run_id>/graph/merge_review.json
- outputs/<run_id>/graph/merge_application.json
- outputs/<run_id>/graph/graph_consolidated.json
- outputs/<run_id>/graph/topic_candidates.json
- outputs/<run_id>/graph/topic_consolidation.json
- outputs/<run_id>/graph/topics.json
- outputs/<run_id>/graph/graph_validation.json

Failure gate:
- If graph validation fails, pipeline raises RuntimeError and stops

## Stage E: Quiz Planning

Modules:
- src/planner/planner.py
- src/planner/prompt_templates.py
- src/utils/llm.py

Function path:
- QuizPlanner(knowledge_graph=document_graph)
- planner.plan(num_questions, difficulty_distribution)
- planner.save_plan(plans, plan_path)

Behavior:
- Planner builds graph context from concept/image nodes and edges
- Prompt created via render_planner_prompt
- LLM call through LLMClient (provider configured in configs/model_config.yaml)
- Parses JSON response and validates:
  - exact question count
  - difficulty values
  - reasoning_type values
  - image_role and non-empty image_description
  - concept coverage uniqueness
  - difficulty balance with tolerance
- Best-effort duplicate concept repair is applied before final coverage check

Persisted artifact:
- outputs/<run_id>/planning/quiz_plan.json

## Stage F: Multimodal Generation (Images + Questions)

Modules:
- src/generator/orchestrator.py
- src/generator/prompt_builder.py
- src/generator/image_gen.py
- src/generator/question_gen.py
- src/knowledge/schema.py (Question model validation)
- src/utils/llm.py

Function path:
- GenerationOrchestrator.run(plan_path, ...)

Per-plan-item flow:
1. Build prompts from plan:
   - question_prompt via PromptBuilder.build_question_prompt
   - image_prompt via PromptBuilder.build_image_prompt
2. Generate image (required):
   - ImageGenerator.generate calls provider endpoint (Freepik Gemini by default)
   - polls task status until completed
   - downloads image into generation/images
   - if image prompt is missing or image generation fails, run fails
3. Generate question:
   - LLMQuestionGenerator.inference(question_prompt, question_plan, image_path)
   - parses strict JSON
   - validates with Question.validate:
     - MCQ needs exactly 4 options
     - valid correct_answer
     - difficulty in easy/medium/hard
     - associated_image must be non-empty
   - retries with repair prompt when validation fails
4. Attach associated image reference and run metadata

Persisted artifacts:
- outputs/<run_id>/generation/questions.json
- outputs/<run_id>/generation/quiz_package.json
- outputs/<run_id>/generation/image_artifacts.json
- outputs/<run_id>/generation/images/*

## Stage G: Manifest and Final Return

Module:
- src/pipeline.py

Behavior:
- Writes manifest.json with:
  - run_id
  - source_document
  - stage completion status
  - artifact relative paths
  - effective config used for this run
- Writes structured events to logs/pipeline.log across all stages
- Returns an in-memory result object containing parsed document, extracted payload, graph JSON, plans, and generation results

Final persisted outputs to consume:
- outputs/<run_id>/manifest.json
- outputs/<run_id>/generation/quiz_package.json
- outputs/<run_id>/generation/questions.json
- outputs/<run_id>/generation/image_artifacts.json

## 4) Core Data Contracts

## Parsed document contract
- ParsedDocument (src/document_understanding/parser.py)
- markdown, sections, paragraphs, figures, captions

## Extraction payload contract
- extracted.json contains
  - concepts: list
  - definitions: map
  - relations: list of relation objects
  - examples: list
  - chunk_extractions: per chunk extraction rows
  - summary

## Graph contract
- MultimodalDocumentGraph (src/knowledge/schema.py)
  - nodes: GraphNode
  - edges: GraphEdge
  - metadata
- Node kinds include document/section/chunk/concept/topic/artifact
- Edge relations include structural and semantic links

## Plan contract
- QuestionPlan (src/planner/planner.py)
  - target_concept
  - question_type
  - difficulty
  - reasoning_type
  - image_role
  - image_description
  - learning_objective
  - metadata

## Generated question contract
- Question (src/knowledge/schema.py)
  - question_text, options, correct_answer, explanation
  - target_concept, difficulty, question_type
  - associated_image (required)
  - image_grounded flag
  - metadata

## 5) Configuration and Environment Influence

Primary config file:
- configs/model_config.yaml

Affects:
- LLM provider/model (planner + question generation via LLMClient)
- image generation provider endpoint/model (ImageGenerator)

Environment variables used in pipeline:
- QUIZGEN_EXTRACTOR_BACKEND
- QUIZGEN_LLM_PROVIDER
- QUIZGEN_EXTRACTION_GRANULARITY
- QUIZGEN_LLM_MODEL
- QUIZGEN_KG_MAX_TOKENS
- QUIZGEN_KG_OVERLAP_BLOCKS

API key env variables:
- provider-specific key from llm.api_key_env in model_config.yaml
- FREEPIK_API_KEY (default for image generation)

## 6) Module Interaction Summary

Execution ownership:
- scripts/run_pipeline.py owns CLI argument parsing and logging setup
- src/pipeline.py owns stage orchestration, artifact tracking, and manifest creation

Cross-module handoffs:
- parser output -> chunking/extractor
- parser + extraction output -> KG workflow
- graph output -> planner
- plan output -> generation orchestrator
- generated images + LLM output -> validated Question objects -> packaged outputs

Shared infrastructure:
- src/utils/llm.py provides provider-agnostic LLM calls
- src/utils/io.py standardizes JSON/JSONL writes and relative path handling
- src/knowledge/schema.py defines core graph and question models

## 7) Current Gaps and Non-Wired Components

- src/verifier/critic.py and src/verifier/scorer.py are placeholders (NotImplementedError)
- Verification stage described in README is not yet wired into src/pipeline.py
- scripts/evaluate.py exists as an entry point scaffold without implemented evaluation logic

In practice, the current runnable pipeline ends at generation packaging and manifest export.