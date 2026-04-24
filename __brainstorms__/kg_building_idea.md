# KG Building Strategy for Multimodal Quiz Generation

Date: 2026-04-24

## 1. Goal and Scope

This document consolidates:
- Your target KG design
- Current project reality
- External best-practice findings (schema, relations, normalization, extraction styles)
- A practical integration strategy for this project pipeline

Project context from [__documents__/pipeline.md](__documents__/pipeline.md):
- Parse -> Extract -> Graph -> Plan -> Generate
- Supports one or many source documents
- KG must improve quiz planning quality, grounding, and multimodal traceability

## 2. Final Target KG Shape

### 2.1 Node layers

1. Structural layer
- document
- section (hierarchical)
- chunk (semantic chunk, first-class node)

2. Semantic layer
- concept (canonical entity)
- optional event/claim nodes for non-binary facts

3. Evidence layer
- image/figure nodes
- definition/example evidence blocks when needed

### 2.2 Core edges

Hierarchy:
- document -> section : contains
- section(parent) -> section(child) : contains
- section -> chunk : contains
- chunk -> chunk(next) : follows

Grounding:
- chunk -> concept : mentions or explains
- image -> concept : illustrates (when visual grounding exists)

Semantic:
- concept -> concept : related_to, depends_on, part_of, causes, semantically_similar_to

All semantic edges should keep:
- confidence (EXTRACTED/INFERRED/AMBIGUOUS)
- confidence_score (numeric)
- source_file
- source_chunk_id
- extraction_method

## 3. Current Gaps in This Repository

Observed from current implementation:
- No explicit document node in graph
- Chunk exists as metadata but not as node
- Concept extraction currently sentence-chunk oriented in extractor, not bound to graph chunk nodes
- Concept normalization is mostly lowercase dedup and does not do true canonicalization

Impact:
- Weaker grounding from quiz question back to specific chunk
- Harder cross-document entity consolidation
- Higher duplicate concept risk

## 4. External Findings to Incorporate

### 4.1 Schema and relation design (industry pattern)

Common production pattern is hybrid:
- Start with a controlled schema and relation vocabulary
- Allow limited schema evolution
- Run post-processing consolidation to reduce label/relation noise

Practical implication for this project:
- Keep a controlled relation set for planning/generation stability
- Preserve open relation text as metadata if needed, but map to controlled labels for core graph

### 4.2 Concept normalization pattern

Most robust pipelines use staged entity resolution:
1. surface normalization
2. alias heuristics (abbreviations, parentheses)
3. candidate generation
4. candidate ranking/disambiguation
5. canonical id assignment
6. ambiguity handling (keep separate if uncertain)

Practical implication for this project:
- Do not hard-merge aggressively
- Keep alias lists and merge confidence
- Preserve reversible provenance from mention to canonical concept

### 4.3 Extraction paradigm (LLM vs rule vs hybrid)

1. LLM-only full-document extraction
- Fast to prototype
- Lower determinism and harder auditing

2. Rule-only extraction
- Deterministic and cheap
- Brittle and lower recall on complex text

3. Hybrid extraction (recommended)
- Deterministic structure/chunking
- LLM extraction at chunk level under schema constraints
- Rule/ER normalization and validation post-processing

This aligns best with multimodal quiz generation because traceability and consistency matter as much as recall.

## 5. Recommended Strategy for This Project

### 5.1 Architecture decision

Use a hybrid pipeline:
1. deterministic structural graph build
2. chunk-scoped semantic extraction
3. canonical concept normalization
4. graph validation and consolidation

### 5.2 Why this fits quiz generation

- Quiz planning needs reliable concept frequency, connectivity, and section context
- Question generation needs grounded evidence (chunk/image provenance)
- Multi-document mode needs concept reuse across documents (canonical entities)
- Explainability requires stable source links for each generated question

## 6. Project Integration Plan

### Phase 1: Structural graph refactor

Changes:
- Add NodeKind.document and NodeKind.chunk
- Create one document node per source
- Materialize semantic chunks as chunk nodes
- Add contains/follows edges explicitly

Acceptance checks:
- Exactly one document node per input document
- Each chunk has one parent section
- Hierarchy traversal is deterministic across runs

### Phase 2: Extraction alignment to chunk nodes

Changes:
- Extract concepts and relations per graph chunk
- Store chunk_id on extracted relations
- Link chunk -> concept explicitly

Acceptance checks:
- Every extracted concept edge has source_chunk_id
- Planning can filter concepts by section/chunk

### Phase 3: Canonicalization module

Changes:
- Introduce canonical concept registry
- Add alias heuristics and threshold-based merge policy
- Rewrite concept references to canonical ids

Recommended policy:
- auto-merge only for high confidence
- keep ambiguous mentions separate with review flag

Acceptance checks:
- alias coverage improves on sample corpora
- false merge rate stays low

### Phase 4: Quality and governance

Changes:
- Add KG validators
- Add regression tests on representative docs
- Add extraction audit report (counts, merges, ambiguities)

Acceptance checks:
- no orphan chunk
- no concept without grounding unless explicitly global
- stable relation vocabulary compliance

## 7. Relation Vocabulary Recommendation

Use controlled core relations for stability:
- contains
- follows
- mentions
- explains
- defines
- illustrates
- related_to
- depends_on
- part_of
- causes
- semantically_similar_to

If extractor outputs other relation labels:
- map to controlled set for main graph
- keep original_relation in metadata

## 8. Canonicalization Decision Rules

Merge confidence tiers:
- >= 0.90: auto-merge
- 0.75 to < 0.90: merge if abbreviation or strong lexical/context evidence
- < 0.75: keep separate and flag ambiguous

Canonical naming preference:
1. long-form technical name in document
2. domain dictionary override
3. most frequent high-quality surface form

Always store:
- aliases
- canonicalization_confidence
- supporting_mentions

## 9. Immediate Next Build Step

Implement Phase 1 first in [src/knowledge/kg_builder.py](src/knowledge/kg_builder.py) and [src/knowledge/schema.py](src/knowledge/schema.py).

Reason:
- It creates the stable backbone needed for Phase 2 to Phase 4
- It directly improves downstream planning and quiz evidence grounding in [src/planner/planner.py](src/planner/planner.py) and generation stages

## 10. Success Criteria for Quiz Pipeline Impact

Track these metrics before and after rollout:
- % questions with explicit chunk-level provenance
- % questions with concept grounding and supporting relation path
- duplicate concept rate across multi-document runs
- planner topic coverage across section hierarchy
- human review score for factual grounding and relevance
