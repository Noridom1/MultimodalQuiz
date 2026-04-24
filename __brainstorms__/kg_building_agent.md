# KG Building as an Agent for Multimodal Quiz Pipeline

Date: 2026-04-24

## 1. Problem Framing

Idea: turn KG construction into an agent workflow with explicit tools such as:
- extract hierarchy
- semantic chunking
- extract concepts
- normalize concepts
- consolidate schema
- validate graph

Goal: improve modularity, observability, and controllability for one-document and multi-document quiz generation.

This aligns with the project pipeline in [__documents__/pipeline.md](__documents__/pipeline.md):
- Parse -> Extract -> Graph -> Plan -> Generate

## 2. Is Agentizing the KG Stage a Good Idea?

Short answer: yes, if we use a hybrid architecture.

Recommended split:
1. Deterministic tools for structure and chunking.
2. LLM-assisted tools for concept/relation extraction under schema constraints.
3. Rule-based/entity-resolution tools for canonicalization and validation.

Why this is better than one-shot LLM extraction:
- Better reproducibility and lower hallucination risk.
- Easier debugging when output quality drops.
- Clear provenance from question back to chunk and source artifact.
- Better support for multi-document entity consolidation.

## 3. Proposed Agent Toolset

### 3.1 Tool: extract_hierarchy

Input:
- parsed markdown blocks (with heading levels)

Output:
- document node
- section nodes
- edges: document->section, section->section (contains)

Deterministic rules:
- heading level defines parent section
- stable order index for reproducible ids

### 3.2 Tool: semantic_chunking

Input:
- section-scoped non-heading blocks

Output:
- chunk nodes
- edges: section->chunk (contains), chunk->chunk (follows)

Rules:
- preserve atomic blocks: table, image, code, details, raw_html
- keep process continuity for list-step sequences
- keep overlap where needed

### 3.3 Tool: extract_concepts

Input:
- chunk text + chunk metadata

Output:
- concept mentions
- chunk->concept grounding edges
- concept->concept candidate relations

Rules:
- extraction is chunk-local first
- every relation must include source_chunk_id and source_file
- allow confidence + confidence_score on all semantic relations

### 3.4 Tool: normalize_concepts

Input:
- concept mentions from all chunks/documents

Output:
- canonical concepts
- alias map
- rewritten concept references

Pipeline:
1. surface normalization
2. alias heuristics (abbrev, parenthetical)
3. candidate generation
4. ranking/disambiguation
5. canonical id assignment
6. ambiguity handling

### 3.5 Tool: attach_artifacts

Input:
- artifact nodes (table/image/code/etc.) with chunk context

Output:
- typed artifact nodes connected to chunk and concepts

Policy:
- artifacts stay as single nodes when they represent indivisible evidence units
- connect artifact to parent chunk and to related concepts
- keep provenance metadata (source span/path)

### 3.6 Tool: consolidate_schema

Input:
- raw nodes/edges from previous tools

Output:
- normalized relation vocabulary
- cleaned labels
- original values preserved in metadata

### 3.7 Tool: validate_graph

Input:
- full graph

Output:
- validation report + pass/fail

Checks:
- one document node per input doc
- each chunk has one parent section
- semantic edges have provenance and confidence
- no orphan concept unless explicitly global

## 4. Artifact Modeling Decision (Your New Suggestion)

Suggestion accepted: keep artifacts such as table/image as single nodes and connect them to nodes in the chunk.

Refined modeling:

1. Keep node kind and content type separate
- kind controls graph role (document/section/chunk/concept/artifact)
- content_type controls modality (text/table/image/code/formula/list)

2. Recommended node behavior
- table/image/code remain artifact nodes (not concept nodes)
- concept nodes represent canonical semantic entities
- chunk grounds both concept nodes and artifact nodes

3. Recommended edges
- chunk->artifact : contains
- artifact->concept : illustrates or supports or mentions
- concept->concept : semantic relations

Why not use concept node for table/image directly:
- concept and evidence are different abstraction levels
- mixing them increases noise for quiz planning logic
- keeping artifact separate helps multimodal explainability

## 5. content_type Design

Use content_type for both chunk and artifact nodes.

Allowed values:
- text
- table
- image
- code
- formula
- list
- mixed

Optional metadata:
- modality_weight
- extraction_method
- parser_confidence

Use in quiz generation:
- prioritize image/table-grounded concepts for multimodal questions
- balance question types by modality coverage

## 6. Suggested Schema (Agent-Friendly)

Node classes:
- document
- section
- chunk
- concept
- artifact

Node common fields:
- id
- label
- kind
- content_type
- source_file
- source_location
- section_path
- metadata

Edge common fields:
- source
- target
- relation
- confidence
- confidence_score
- source_file
- source_chunk_id
- metadata

Core relation vocabulary:
- contains
- follows
- mentions
- explains
- illustrates
- supports
- defines
- related_to
- depends_on
- part_of
- causes
- semantically_similar_to

## 7. Agent Orchestration Pattern

Recommended execution order:
1. extract_hierarchy
2. semantic_chunking
3. attach_artifacts
4. extract_concepts
5. normalize_concepts
6. consolidate_schema
7. validate_graph

Execution mode:
- sequential by default for deterministic behavior
- optional parallelism only for chunk-level extraction

Checkpoint artifacts per tool:
- hierarchy.json
- chunks.json
- artifact_links.json
- extraction_raw.json
- canonicalization.json
- graph_consolidated.json
- graph_validation.json

## 8. Integration into Current Project

Near-term integration path:

Phase 1 (low risk):
- add document and chunk nodes in graph builder
- add artifact node class + content_type support

Phase 2:
- switch concept extraction input from sentence-only chunks to graph chunk nodes
- attach source_chunk_id on semantic edges

Phase 3:
- add canonicalization module with merge thresholds

Phase 4:
- add validation gate before planning stage

Pipeline impact:
- [src/planner/planner.py](src/planner/planner.py) can rank concepts with stronger grounding
- generation can intentionally produce table/image-aware questions
- end-to-end provenance becomes clearer for audit and feedback

## 9. Trade-offs and Risks

1. More components means more engineering effort.
2. Canonicalization can introduce false merges if thresholds are too aggressive.
3. Agent orchestration requires strict contracts to avoid tool drift.

Mitigations:
- keep deterministic schemas and validators
- conservative merge thresholds initially
- preserve original mentions for rollback

## 10. Decision Proposal

Recommended decisions for this project now:

1. Adopt agentized KG build with hybrid toolchain.
2. Keep artifacts as separate single nodes.
3. Add content_type but keep concept separate from artifact.
4. Add validate_graph gate before quiz planning.

This keeps the KG reliable for quiz planning while improving multimodal grounding quality.
