# Multimodal Quiz Generation Framework

## 1. Problem Statement

Quiz-based learning is a widely adopted approach in education, supporting knowledge assessment, reinforcement, and active learning. With the advancement of natural language processing, automatic quiz generation has emerged as an important research area, enabling scalable and efficient creation of educational content. Existing approaches primarily focus on generating textual questions, such as multiple-choice questions (MCQs), short-answer questions, and fill-in-the-blank formats, directly from textual documents.

However, these methods suffer from a critical limitation: they rely almost exclusively on **text-only representations**, neglecting the role of visual information in learning. Prior research in cognitive science and multimedia learning suggests that incorporating visual elements significantly improves comprehension, engagement, and retention. This limitation is particularly evident in **text-heavy educational materials**, where the absence of visual aids can hinder effective learning.

Recent work in multimodal learning has attempted to address this issue by incorporating images into question answering and question generation tasks. Nevertheless, these approaches typically depend on **pre-existing images** available in the source documents or datasets. As a result, they are constrained by the availability, quality, and relevance of such visual content. Many real-world documents (e.g., lecture notes, textbooks, and articles) are inherently **visual-sparse**, limiting the applicability of existing multimodal quiz generation methods.

To address these challenges, we propose a novel framework for **multimodal quiz generation augmented with AI-generated images**. Instead of passively relying on existing visuals, our approach actively generates **contextually relevant images** to support quiz questions. Furthermore, we introduce an **agent-based planning mechanism** that explicitly determines what concepts to assess, how questions should be constructed, and whether visual support is required.

---

## 2. Framework Overview

The system is designed as an **agent-based pipeline**:

```
Document → Understanding → Knowledge → Planning → Generation → Verification
```

### Modules:

* **Document Understanding (U)**
* **Knowledge Construction (K)**
* **Quiz Planning (P)**
* **Multimodal Generation (G)**
* **Verification (V)**

---

## 3. Module Details

### 3.1 Document Understanding

#### Objective

Transform raw document (PDF or text) into structured semantic representation.

#### Steps

1. **Layout-aware Parsing**

   * Extract sections, paragraphs, figures, captions
2. **Semantic Chunking**

   * Group sentences using embedding similarity
3. **LLM-based Extraction**

   * Concepts
   * Definitions
   * Relations
   * Examples
4. **Concept Normalization**

   * Merge duplicates using embeddings + LLM

#### Output

```json
{
  "concepts": [...],
  "definitions": {...},
  "relations": [...],
  "examples": [...]
}
```

---

### 3.2 Knowledge Construction

#### Objective

Build structured knowledge representation.

#### Approach

* Construct **Knowledge Graph (KG)**:

  * Nodes = concepts
  * Edges = relations

#### Optional

* Add embeddings for retrieval
* Store source references

#### Purpose

* Improve reasoning
* Ensure concept coverage

---

### 3.3 Quiz Planning (Core Module)

#### Objective

Plan quiz before generating content.

#### Input

* Knowledge graph
* Number of questions
* Difficulty distribution

#### Output

```json
{
  "questions": [
    {
      "target_concept": "...",
      "question_type": "MCQ",
      "difficulty": "medium",
      "reasoning_type": "causal",
      "image_role": "illustrative",
      "image_description": "...",
      "learning_objective": "..."
    }
  ]
}
```

#### Key Features

* Concept coverage
* Difficulty balancing
* Image-aware planning

---

### 3.4 Multimodal Generation

#### Image Generation

* Convert description → prompt
* Use text-to-image models

#### Question Generation

* Generate:

  * Question
  * Options
  * Answer
  * Explanation

#### Constraint

* If image exists → question must depend on it

---

### 3.5 Verification Module

#### Objective

Ensure quality and consistency

#### Checks

* Answer correctness
* Clarity
* Image alignment
* Distractor quality

#### Process

* LLM critic evaluates each question
* Score + feedback
* Regenerate if needed

---

## 4. Codebase Structure

```
MultimodalQuiz/
│
├── configs/
│   ├── default.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/                        # input PDFs
│   ├── processed/                  # MinerU parsing outputs (per-document)
│   └── outputs/                    # legacy output location
│
├── outputs/                        # pipeline run outputs (see §5.3)
│
├── src/
│   ├── document_understanding/
│   │   ├── parser.py
│   │   ├── chunking.py
│   │   ├── extractor.py
│   │   └── normalizer.py
│   │
│   ├── knowledge/
│   │   ├── kg_builder.py
│   │   ├── schema.py
│   │   ├── schema_consolidator.py
│   │   ├── concept_normalizer.py
│   │   ├── topic_inducer.py
│   │   ├── merge_resolver.py
│   │   ├── graph_reviewer.py
│   │   ├── validator.py
│   │   └── retriever.py
│   │
│   ├── planner/
│   │   ├── planner.py
│   │   ├── prompt_templates.py
│   │   ├── topic_planner.py
│   │   └── topic_prompt_templates.py
│   │
│   ├── generator/
│   │   ├── question_gen.py
│   │   ├── image_gen.py
│   │   ├── prompt_builder.py
│   │   ├── prompt_checks.py
│   │   └── orchestrator.py
│   │
│   ├── verifier/
│   │   ├── critic.py
│   │   └── scorer.py
│   │
│   ├── utils/
│   │   ├── llm.py
│   │   ├── embedding.py
│   │   └── io.py
│   │
│   └── pipeline.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── scripts/
│   ├── run_pipeline.py             # full pipeline entry point
│   ├── orchestrate_generation.py  # generation-only orchestration
│   ├── evaluate.py                # evaluation utilities
│   ├── visualize_questions.py     # interactive quiz HTML generator
│   └── test_extractor.py
│
├── lib/                            # vendored JS libs for graph viewer
│
├── requirements.txt
└── README.md
```

## 5. Running Guide

### 5.1 Run the Full Pipeline

The full pipeline is executed through `scripts/run_pipeline.py`.

#### Prerequisites

* Install dependencies:

```bash
pip install -r requirements.txt
```

* Configure your environment variables (for LLM/image providers) in `.env` if needed.
* Prepare an input document path (PDF, Markdown, or text-like source supported by the parser).

#### Basic command

```bash
python scripts/run_pipeline.py <document_path>
```

Example:

```bash
python scripts/run_pipeline.py data/raw/book-riscv-rev5-chap8.pdf
```

By default this creates one run folder under `outputs/`.

#### Useful options

```bash
python scripts/run_pipeline.py <document_path> --num-questions 8 --easy 0.25 --medium 0.5 --hard 0.25 --generation-mode topic_agentic --log-level INFO
```

Main arguments:

* `--output-root`: root directory for run outputs (default: `outputs/`)
* `--run-id`: custom run folder name (otherwise auto-generated)
* `--num-questions`: number of quiz items to generate
* `--easy --medium --hard`: difficulty ratios used by planning
* `--generation-mode`: `topic_agentic` (default) or `legacy`
* `--mock-image`: skip real image generation and use mock image refs
* `--mock-question`: skip real question LLM generation and use mock questions
* `--log-level`: `DEBUG|INFO|WARNING|ERROR|CRITICAL`

To see all CLI options:

```bash
python scripts/run_pipeline.py --help
```

### 5.2 Visualize Questions from `questions.json`

After a successful run, you can generate an interactive quiz from the generated questions file:

```bash
python scripts/visualize_questions.py --json_path outputs/<run_id>/generation/questions.json --open
```

This creates `index.html` next to `questions.json` and opens it in your default browser. The quiz displays one question at a time, tracks time per question, and lets participants download a `user_study_results.json` file at the end.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--json_path` | *(required)* | Path to `questions.json` |
| `--image_dir` | `data/images/` | Fallback directory when image ref is a bare filename |
| `--output` | `index.html` next to JSON | Custom output HTML path |
| `--open` | off | Open the HTML in the default browser after writing |

If your images live in a custom directory:

```bash
python scripts/visualize_questions.py \
  --json_path outputs/<run_id>/generation/questions.json \
  --image_dir outputs/<run_id>/generation/images \
  --open
```

### 5.3 Understand the `outputs/` Folder

Each execution produces one run directory:

```text
outputs/
  <run_id>/
    manifest.json
    document/
    extraction/
    graph/
    planning/
    generation/
    logs/
```

`<run_id>` is either your explicit `--run-id` or an auto-generated ID like:
`YYYYMMDD_HHMMSS_<document_stem>_<short_hash>`.

#### What each subfolder contains

* `manifest.json`
  * Run metadata: source document, stage status, artifact paths, and runtime config.
  * Start here to inspect or debug a run.

* `document/`
  * `parsed_document.json`: structured parsing output (sections, paragraphs, figures, markdown).

* `extraction/`
  * `extracted.json`: semantic extraction results (concepts, definitions, relations, examples).

* `graph/`
  * `graph.json`: consolidated knowledge graph.
  * `graph_networkx.json`: NetworkX-friendly graph serialization.
  * `graph.html`: interactive graph visualization (when HTML export is enabled).
  * Additional intermediate/debug artifacts may appear (topic candidates, merge review, validation, etc.).

* `planning/`
  * `quiz_plan.json`: planned questions (target concepts, difficulty, image role, objectives).

* `generation/`
  * `questions.json`: final generated question objects.
  * `quiz_package.json`: packaged run payload (includes run_id, image dir, indexed results).
  * `image_artifacts.json`: image generation metadata and references.
  * `images/`: generated image files for the run.
  * `prompt_logs/`: saved prompt/context logs per question.

* `logs/`
  * `pipeline.log`: JSONL stage/event log for tracing and troubleshooting.

#### Quick inspection flow

1. Open `outputs/<run_id>/manifest.json`.
2. Check stage completion and artifact paths.
3. Review `generation/questions.json` for final quiz outputs.
4. Run `python scripts/visualize_questions.py --json_path outputs/<run_id>/generation/questions.json --open` for visual QA.
5. If something failed, inspect `logs/pipeline.log` and corresponding stage artifacts.

---

## 6. Key Contributions

* Planning-based quiz generation (not direct generation)
* Integration of AI-generated images
* Explicit modeling of image roles
* Multimodal verification mechanism

---

## 7. Future Extensions

* Adaptive quizzes based on learner level
* Reinforcement learning for planning optimization
* Feedback loop from student responses
* Difficulty prediction models

---

## 8. Notes

* Focus on **planner + image role modeling** as core novelty
* Avoid building a simple pipeline without structure
* Ensure reproducibility with clear prompts and intermediate outputs

## 9. User Study Guide

### Pre-test

Use one of the following topic files:

* `eco.json`
* `astronomy.json`
* `neuroscience.json`

```bash
python scripts/visualize_questions.py --no_score --json_path post-test/{topic}.json --share
```

Example:

```bash
python scripts/visualize_questions.py --no_score --json_path post-test/eco.json --share
```

### Review-test

```bash
python scripts/visualize_questions.py --json_path outputs/20260426_000329_eco_12bf04/generation/questions.json --share
```

### Post-test

```bash
python scripts/visualize_questions.py --json_path post-test/{topic}.json --share
```

---
