# Methodology: Quiz Planning and Generation

The MultimodalQuiz system employs a decoupled, agentic workflow for the automated creation of quiz questions from multimodal document graphs. This process is divided into two primary stages: **Topic-Driven Quiz Planning** and **Grounded Content Generation**.

## 1. Topic-Driven Agentic Planning

The planning phase determines *what* to ask before *how* to ask it. This prevents redundant questions and ensures systematic coverage of the source material's taxonomy.

### 1.1 Topic Identification and Context Retrieval
The `TopicAgenticPlanner` iterates through "topic nodes" (high-level concept nodes) in the Multimodal Document Graph. For each topic, a `TopicContextRetriever` performs a graph traversal to gather:
- **Associated Concepts:** Sub-concepts grouped under or mentioned by the topic.
- **Textual Evidence:** Text chunks related to these concepts, providing grounding facts.
- **Visual Assets:** Images linked to the concepts, used for multimodal question types.

### 1.2 Resource Allocation and Budgeting
To manage question distribution across diverse topics, the planner calculates a "proportional budget." It uses a greedy allocation strategy that prioritizes topics with the highest density of unique, uncovered concepts. This ensures that the generated quiz maximizes breadth across the document's knowledge base while respecting user-defined constraints on total question count and difficulty distribution (e.g., 40% Easy, 40% Medium, 20% Hard).

### 1.3 Question Schema Planning
For each allocated slot, the planner calls a Large Language Model (LLM) to produce a `QuestionPlan`. This plan is a metadata-rich specification that includes:
- **Target Concept:** The specific node being tested.
- **Reasoning Type:** Categorized as *factoid*, *causal*, or *multi-hop*.
- **Image Role:** Defines if an image is *illustrative* (supportive) or *reasoning* (mandatory for answering).
- **Source Grounding:** A `tested_fact_block_id` mapping the question to a specific source block in the original document.

---

## 2. Grounded Content Generation

The generation phase translates abstract plans into concrete, validated quiz items. This is orchestrated by the `GenerationOrchestrator`, which manages the parallel execution of image and question generation.

### 2.1 Multimodal Prompt Engineering
A `PromptBuilder` transforms the `QuestionPlan` and retrieved graph context into detailed instructions for the generators. 
- **Image Generation:** If the plan specifies an image-based question, an image prompt is synthesized (or a relevant extracted image is selected) based on the `image_description` in the plan.
- **Question Generation:** The prompt includes the "Knowledge Context" (raw text chunks from the graph) and strict formatting constraints. It enforces citation of the `tested_fact_block_id` within the explanation.

### 2.2 Grounded LLM Generation
The `LLMQuestionGenerator` uses the synthesized prompts to generate JSON-formatted questions. This stage implements **Image Grounding check**: if the plan requires "reasoning" from an image, the generator is instructed to reference concrete visual cues (e.g., "the label in the upper right," "the red arrow pointing to X").

### 2.3 Verification and Self-Correction
The generator includes an internal retry loop with a "Repair Prompt" mechanism. If the initial LLM output fails schema validation or lacks proper grounding evidence, the system produces a feedback prompt containing the validation error and the previous invalid attempt, allowing the agent to self-correct and ensure high-fidelity outputs.
