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
      "requires_image": true,
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
quizgen/
│
├── configs/
│   ├── default.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
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
│   │   └── schema.py
│   │
│   ├── planner/
│   │   ├── planner.py
│   │   └── prompt_templates.py
│   │
│   ├── generator/
│   │   ├── question_gen.py
│   │   ├── image_gen.py
│   │   └── prompt_builder.py
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
│   ├── run_pipeline.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

## 5. Key Contributions

* Planning-based quiz generation (not direct generation)
* Integration of AI-generated images
* Explicit modeling of image roles
* Multimodal verification mechanism

---

## 6. Future Extensions

* Adaptive quizzes based on learner level
* Reinforcement learning for planning optimization
* Feedback loop from student responses
* Difficulty prediction models

---

## 7. Notes

* Focus on **planner + image role modeling** as core novelty
* Avoid building a simple pipeline without structure
* Ensure reproducibility with clear prompts and intermediate outputs

---
