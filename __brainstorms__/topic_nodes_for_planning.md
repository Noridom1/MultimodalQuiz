# Topic Nodes For Planning

Date: 2026-04-24

## Goal

Explore whether the planning phase should operate on a new high-level `topic` node type instead of selecting directly from raw concept nodes.

The intended use is:

- group relevant concepts under a higher-level topic
- allow grouping across multiple documents
- let planning select a topic first, then choose concepts/evidence inside that topic

This note is based on the current artifacts:

- [book-riscv-rev5-chap8_graph.html](/d:/projects/MultimodalQuiz/data/outputs/book-riscv-rev5-chap8_graph.html)
- [book-riscv-rev5-chap8_graph.json](/d:/projects/MultimodalQuiz/data/outputs/book-riscv-rev5-chap8_graph.json)
- [book-riscv-rev5-chap8_graph_networkx.json](/d:/projects/MultimodalQuiz/data/outputs/book-riscv-rev5-chap8_graph_networkx.json)
- [canonicalization.json](/d:/projects/MultimodalQuiz/data/outputs/canonicalization.json)

## Short Answer

Yes, a `topic` layer is a good idea.

But I would not use topic nodes as just another label for concept clusters. I would use them as a planning-oriented abstraction layer with stricter semantics:

- `concept` = atomic semantic unit that can be grounded to chunks and artifacts
- `topic` = planning unit that groups concepts into a teachable theme

That distinction matters because planning is not trying to find the smallest semantic unit. It is trying to choose a coherent assessment target.

## What I See In The Current Graph

The current KG is already good enough to motivate a topic layer:

- It has chunk grounding and section structure, so concepts can be grouped with evidence.
- It already contains local relation structure such as `depends_on`, `part_of`, and `related_to`.
- It has artifact links, which means topic nodes could later support multimodal planning.

However, the current graph also shows why planning should not consume raw concepts directly:

- Some concepts are too fine-grained for planning, such as `ra`, `sp`, `sd`, `ld`, `a0`, `a1`.
- Some concepts are clearly implementation tokens rather than quiz-worthy top-level targets.
- Some canonicalization outcomes are suspicious:
  - `struct context` has aliases `struct cpu` and `system call`
  - `callee-saved registers` has alias `caller-saved registers`
  - `scheduling policy` has alias `struct proc`
- Some sections are much more code-heavy than idea-heavy, so direct concept sampling will bias the planner toward low-level symbols.

This is the strongest argument for topic nodes: they can absorb noisy concept variation and give planning a more stable target space.

## Proposed Meaning Of A Topic Node

A topic node should represent a coherent instructional theme, not just a high-degree cluster.

For this chapter, likely topic examples are:

- CPU multiplexing
- context switching
- scheduler flow
- process state transitions
- per-CPU / per-process state
- round-robin scheduling
- scheduler correctness and locking

Each of those topics can contain multiple concepts. For example:

- `context switching`
  - `swtch`
  - `struct context`
  - `kernel thread`
  - `scheduler thread`
  - `callee-saved registers`
  - `RISC-V calling convention`

That is much closer to how a human would plan questions.

## Recommended Schema

Add a new node kind:

- `topic`

Suggested fields:

- `id`
- `label`
- `description`
- `topic_type`
- `source_documents`
- `section_paths`
- `metadata`

Suggested edge types:

- `topic -> concept : groups`
- `topic -> chunk : grounded_by`
- `topic -> artifact : illustrated_by`
- `topic -> topic : prerequisite_of`
- `document/section -> topic : covers`

I would keep `topic` separate from `section`.

Reason:

- `section` is author structure
- `topic` is semantic/planning structure

Sometimes they align. Often they do not.

## How To Build Topic Nodes

I would not create topic nodes in one shot from the full graph immediately. Use a staged strategy.

### Option 1: Section-seeded topics

Initial topic candidates come from section titles and section-local concept neighborhoods.

Example:

- section `8.1 Multiplexing` seeds topic `CPU multiplexing`
- section `8.3 Code: Context switching` seeds topic `context switching implementation`

Pros:

- simple
- easy to explain
- stable

Cons:

- topics may become too tied to author structure
- cross-section themes are missed

### Option 2: Relation-neighborhood topics

Build concept communities from concept-concept and chunk-concept connectivity.

Signals:

- shared chunk grounding
- shared section occurrence
- semantic relation density
- shared artifact support

Pros:

- more semantic
- better for cross-document grouping

Cons:

- noisier
- vulnerable to concept extraction/canonicalization errors

### Option 3: Hybrid topic induction

This is the best option.

Workflow:

1. Seed topic candidates from section titles and high-value concept labels.
2. Expand each seed with neighboring concepts from the graph.
3. Use an LLM or clustering pass to consolidate overlapping topic candidates.
4. Validate topic coherence.

This fits the rest of the architecture better than either pure section-based or pure clustering-based topics.

## How Topic Nodes Should Help Planning

A better planning pipeline would be:

1. choose topic
2. choose subtarget concept inside topic
3. choose supporting chunks / artifacts
4. choose difficulty and reasoning type

That is better than:

1. choose concept directly

because it gives planning a hierarchy:

- topic decides what the question is broadly about
- concept decides what exact knowledge point is tested
- chunk/artifact decides the evidence used

This improves:

- coverage balance
- question diversity
- coherence of distractors
- multimodal grounding

## Cross-Document Value

This idea becomes much stronger when you have multiple documents.

Without topic nodes:

- planning sees many document-local concept nodes
- the same broad subject appears as many fragmented concept neighborhoods

With topic nodes:

- concepts from multiple documents can be grouped under a shared instructional theme
- planning can ask for:
  - a topic only covered in one document
  - a topic reinforced across many documents
  - a topic with both textual and visual evidence
  - a topic with contrasting implementations across documents

That is probably the biggest long-term upside.

## My Main Recommendation

Do not make topic nodes purely graph-theoretic clusters.

Instead, make them planning-oriented semantic groupers with three requirements:

1. A topic must have a human-readable name.
2. A topic must contain multiple grounded concepts.
3. A topic must have enough evidence density to support question generation.

In other words, topic nodes should exist because they are useful for planning, not because a clustering algorithm can produce them.

## How To Score Topic Quality

Before using topics for planning, I would score each topic on:

- coherence
  - do member concepts belong together semantically?
- grounding
  - how many chunks/artifacts support the topic?
- breadth
  - does the topic cover multiple related concepts?
- distinctness
  - is it meaningfully different from other topics?
- pedagogical usefulness
  - could a teacher plausibly ask 1-3 questions from it?

A topic that fails pedagogical usefulness should not exist as a topic node.

## Suggested Planning Heuristics Once Topic Nodes Exist

When planning, prefer topics that:

- have at least 2-3 concept members
- have at least one defining or explanatory chunk
- have strong internal relation structure
- are not dominated by raw code symbols only
- have artifact support if multimodal questions are desired

Then, inside each topic, prefer concepts that:

- are central to the topic
- have clear grounding
- are not obviously noisy tokens
- have prerequisite or contrast relationships to sibling concepts

## Current Risks If You Add Topic Nodes Now

The current graph suggests a few risks:

1. Topic quality will inherit concept noise.

Low-level symbols like `ra`, `sp`, `sd`, `ld` may get pulled into topics too aggressively.

2. Some canonical concepts are clearly wrong.

If the concept layer is wrong, topic induction may group around bad anchors.

3. Section structure is currently very shallow.

The graph has many top-level sections but not much deeper semantic hierarchy, so section-seeded topics may be coarse.

4. Topic overlap will be common.

For example:

- `context switching`
- `scheduler`
- `kernel thread`

These are distinct, but strongly overlapping. You will need explicit overlap policy.

## Practical Design Suggestion

I would introduce topic nodes in two passes.

### Pass 1: Conservative topic layer

Only create topics from:

- major section themes
- high-confidence, repeated concepts
- manually or LLM-labeled grouped neighborhoods

Rules:

- minimum 2 concept members
- minimum 2 supporting chunks
- no topic built mostly from register names or code tokens

### Pass 2: Cross-document topic consolidation

Once you have multiple document graphs:

- merge similar local topics into global topics
- keep document-local topic instances linked to a global topic node

This is likely cleaner than trying to create global topics directly from raw concept nodes.

## One Structural Pattern I Like

Use two topic layers instead of one:

- `local_topic`
  - document-specific
  - grounded directly in section/chunk neighborhoods
- `global_topic`
  - cross-document
  - groups similar local topics

Why this is useful:

- it preserves provenance
- it avoids premature cross-document merging
- it gives planning both local and global selection modes

Example:

- local topic in this chapter: `context switching in xv6`
- global topic across corpus: `context switching`

That is a strong design if you expect many documents later.

## Concrete Next Step

If you decide to pursue this, I would prototype topic nodes as a post-processing layer over the current graph, not by changing concept extraction first.

Prototype pipeline:

1. load graph
2. filter low-value concepts
3. build section-seeded topic candidates
4. expand candidates via concept/chunk neighborhoods
5. score and prune topics
6. write `topics.json`
7. optionally materialize `topic` nodes into the graph

That will let you evaluate whether topics actually improve planning before baking them into the KG schema permanently.

## My Opinion

This is a good direction.

The current KG is already rich enough that planning directly on raw concept nodes will probably be too brittle and too low-level. A `topic` layer is the right abstraction if you want planning to behave more like curriculum design than entity sampling.

But I would keep the topic layer deliberately conservative at first, because the current concept layer still contains enough noise that aggressive topic induction would amplify mistakes rather than hide them.
