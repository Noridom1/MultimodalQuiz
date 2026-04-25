"""Topic-driven agentic quiz planner for dynamic, concept-rich question generation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from src.knowledge.retriever import TopicContextRetriever, calculate_topic_budget
from src.knowledge.schema import MultimodalDocumentGraph, NodeKind
from src.planner.planner import QuestionPlan
from src.planner.topic_prompt_templates import render_topic_plan_prompt
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class TopicAgenticPlanner:
    """Planner that iterates through topics and generates questions proportionally.
    
    This planner:
    1. Extracts all topic nodes from the graph
    2. For each topic, retrieves rich context (concepts, chunks, images)
    3. Allocates questions proportionally based on context density
    4. Calls LLM to generate plans grounded in topic context
    5. Enforces tested_fact_block_id citation for all questions
    """

    _VALID_DIFFICULTIES = {"easy", "medium", "hard"}
    _VALID_REASONING = {"factoid", "causal", "multi-hop"}
    _VALID_IMAGE_ROLES = {"illustrative", "reasoning", "distractor"}

    def __init__(
        self,
        knowledge_graph: MultimodalDocumentGraph | None = None,
        *,
        graph_json_path: str | Path | None = None,
        llm_client: LLMClient | Any | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize the topic-agentic planner.
        
        Args:
            knowledge_graph: The MultimodalDocumentGraph to plan from.
            graph_json_path: Optional path to load graph from JSON if knowledge_graph not provided.
            llm_client: LLMClient instance; if None, creates a default one.
            max_retries: Maximum number of LLM call retries per topic.
        """
        self._knowledge_graph = self._load_graph(knowledge_graph, graph_json_path)
        self._llm = llm_client or LLMClient()
        self._max_retries = max_retries
        self._retriever = TopicContextRetriever(self._knowledge_graph)

    @staticmethod
    def _load_graph(
        knowledge_graph: MultimodalDocumentGraph | None,
        graph_json_path: str | Path | None,
    ) -> MultimodalDocumentGraph:
        """Load the graph from provided object or JSON file."""
        if knowledge_graph is not None:
            return knowledge_graph
        if graph_json_path is not None:
            with open(graph_json_path) as f:
                data = json.load(f)
            return MultimodalDocumentGraph.model_validate(data)
        raise ValueError("Either knowledge_graph or graph_json_path must be provided.")

    def plan(
        self,
        *,
        total_questions: int | None = None,
        num_questions: int | None = None,
        difficulty_distribution: Optional[dict[str, float]] = None,
        max_per_topic: int = 10,
    ) -> list[QuestionPlan]:
        """Generate a quiz plan by iterating through topics.
        
        Args:
            total_questions: Total number of questions to generate.
            difficulty_distribution: Optional dict mapping difficulty to proportion.
            max_per_topic: Maximum questions per topic (prevents one topic from dominating).
            
        Returns:
            List of QuestionPlan objects.
            
        Raises:
            ValueError: If total_questions <= 0 or no topics found in graph.
            RuntimeError: If question generation fails after all retries.
        """
        # support legacy callers using `num_questions` keyword
        effective_total = total_questions if total_questions is not None else num_questions
        if effective_total is None or effective_total <= 0:
            raise ValueError("total_questions (or num_questions) must be greater than zero.")

        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.4, "medium": 0.4, "hard": 0.2}

        self._validate_difficulty_distribution(difficulty_distribution)

        # Extract all topic nodes from graph
        # Note: Topics are typically nodes with kind == "concept" that have special metadata
        # For now, we treat all concept nodes as potential topics
        # In a real system, topics might be marked with metadata["is_topic"] = True
        topic_ids = [node.id for node in self._knowledge_graph.nodes if node.kind == NodeKind.concept]

        if not topic_ids:
            raise ValueError("No topic nodes found in graph. Cannot proceed with topic-agentic planning.")

        logger.info(f"Found {len(topic_ids)} topic nodes. Planning {effective_total} questions.")

        # Calculate total resources for proportional allocation, but prioritize uncovered concepts
        topic_contexts = {}
        for topic_id in topic_ids:
            try:
                context = self._retriever.retrieve_context(topic_id)
                topic_contexts[topic_id] = context
            except ValueError as e:
                logger.warning(f"Failed to retrieve context for topic {topic_id}: {e}")
                continue

        if not topic_contexts:
            raise RuntimeError("Could not retrieve context for any topics.")

        # Greedy allocation to maximize unique concept coverage across topics
        all_plans = []
        remaining_budget = effective_total
        used_concepts: set[str] = set()

        # Precompute uncovered counts per topic (by concept labels)
        def topic_uncovered_concepts(ctx):
            return [c for c in ctx.associated_concepts if c.label.lower() not in used_concepts]

        # Loop through topics and allocate proportionally by uncovered concept counts
        for topic_id in topic_ids:
            if topic_id not in topic_contexts:
                continue

            context = topic_contexts[topic_id]
            uncovered = topic_uncovered_concepts(context)
            # Skip topics with no uncovered concepts
            if not uncovered:
                logger.debug(f"Skipping topic {topic_id} ({context.topic_label}) - no uncovered concepts")
                continue

            # Compute total uncovered across remaining topics
            total_uncovered = sum(len(topic_uncovered_concepts(topic_contexts[t])) for t in topic_ids if t in topic_contexts)
            if total_uncovered <= 0:
                break

            # Allocate proportionally to uncovered concept counts
            allocated = int(round(remaining_budget * len(uncovered) / total_uncovered))
            allocated = max(1, min(allocated, remaining_budget, max_per_topic))

            if allocated == 0:
                logger.debug(f"Skipping topic {topic_id} (0 allocated)")
                continue

            logger.info(f"Generating {allocated} questions for topic {context.topic_label} (uncovered concepts: {len(uncovered)})")

            # Generate plans for this topic, restricting to uncovered concept labels
            try:
                only_concept_labels = [c.label for c in uncovered]
                topic_plans = self._generate_topic_plans(
                    context,
                    allocated,
                    difficulty_distribution,
                    only_concepts=only_concept_labels,
                )

                all_plans.extend(topic_plans)
                # Mark produced target concepts as used
                for p in topic_plans:
                    if p.target_concept:
                        used_concepts.add(p.target_concept.lower())

                remaining_budget -= len(topic_plans)
                if remaining_budget <= 0:
                    break
            except RuntimeError as e:
                logger.error(f"Failed to generate plans for topic {context.topic_label}: {e}")
                continue

        if not all_plans:
            raise RuntimeError("No question plans were generated.")

        logger.info(f"Generated {len(all_plans)} question plans (requested {effective_total})")

        return all_plans

    def _generate_topic_plans(
        self,
        context,
        num_questions: int,
        difficulty_distribution: dict[str, float],
        *,
        only_concepts: list[str] | None = None,
    ) -> list[QuestionPlan]:
        """Generate question plans for a specific topic.
        
        Args:
            context: TopicContext with concepts, chunks, images.
            num_questions: Number of questions to generate for this topic.
            difficulty_distribution: Difficulty distribution as dict.
            
        Returns:
            List of QuestionPlan objects.
            
        Raises:
            RuntimeError: If generation fails after max_retries.
        """
        prompt = render_topic_plan_prompt(
            context,
            num_questions,
            difficulty_distribution,
            only_concepts=only_concepts,
        )

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                raw_output = self._llm.complete(prompt)

                if raw_output is None or not str(raw_output).strip():
                    raise RuntimeError("LLM returned empty response")

                cleaned = self._clean_llm_output(raw_output)
                if not cleaned:
                    logger.debug(f"[Topic {context.topic_id}] LLM returned no JSON. Raw: {repr(raw_output)}")
                    raise RuntimeError("LLM returned no JSON content")

                payload = json.loads(cleaned)
                plans = self._parse_topic_plans(payload, num_questions, context.topic_id)

                # Attach the grounding text for the tested_fact_block_id into each plan's metadata
                # so the generator sees the actual fact text (knowledge_context) when building prompts.
                try:
                    self._attach_knowledge_context(plans, context)
                except Exception:
                    logger.exception("Failed to attach knowledge context to plans; continuing without it")

                # Validate tested_fact presence (mandatory)
                self._validate_tested_facts(plans, context.topic_id)

                return plans

            except json.JSONDecodeError as exc:
                logger.debug(f"[Topic {context.topic_id}] Attempt {attempt + 1}: JSON parse error: {exc}")
                last_error = exc
                continue
            except RuntimeError as exc:
                logger.debug(f"[Topic {context.topic_id}] Attempt {attempt + 1}: {exc}")
                last_error = exc
                if "tested_fact" in str(exc).lower():
                    # If it's a tested_fact issue, retry with stronger prompt
                    continue
                break

        raise RuntimeError(
            f"Failed to generate valid topic plans for {context.topic_id} after {self._max_retries + 1} attempts"
        ) from last_error

    def _parse_topic_plans(
        self,
        payload: Any,
        expected_count: int,
        topic_id: str,
    ) -> list[QuestionPlan]:
        """Parse LLM output into QuestionPlan objects.
        
        Args:
            payload: Parsed JSON from LLM.
            expected_count: Expected number of questions.
            topic_id: ID of the topic for context in errors.
            
        Returns:
            List of QuestionPlan objects.
            
        Raises:
            RuntimeError: If parsing fails validation.
        """
        if not isinstance(payload, dict):
            raise RuntimeError("Payload must be a JSON object.")

        rows = payload.get("questions")
        if not isinstance(rows, list):
            raise RuntimeError("Payload must include 'questions' list.")

        if len(rows) != expected_count:
            raise RuntimeError(f"Expected {expected_count} questions, got {len(rows)}")

        plans: list[QuestionPlan] = []

        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"Question {idx} must be a JSON object")

            target_concept = self._normalized_text(row.get("target_concept"))
            question_type = self._normalized_text(row.get("question_type"))
            difficulty = self._normalized_text(row.get("difficulty")).lower()
            reasoning_type = self._normalized_text(row.get("reasoning_type")).lower()
            image_role_raw = row.get("image_role", "illustrative")
            image_description = self._normalized_text(row.get("image_description") or target_concept)
            learning_objective = self._normalized_text(row.get("learning_objective", ""))
            tested_fact_block_id = self._normalized_text(row.get("tested_fact_block_id"))

            # Validate fields
            if not target_concept:
                raise RuntimeError(f"Question {idx}: target_concept is required")
            if not question_type:
                raise RuntimeError(f"Question {idx}: question_type is required")
            if difficulty not in self._VALID_DIFFICULTIES:
                raise RuntimeError(f"Question {idx}: invalid difficulty '{difficulty}'")
            if reasoning_type not in self._VALID_REASONING:
                raise RuntimeError(f"Question {idx}: invalid reasoning_type '{reasoning_type}'")
            if not image_description:
                raise RuntimeError(f"Question {idx}: image_description is required")
            if not tested_fact_block_id:
                raise RuntimeError(f"Question {idx}: tested_fact_block_id is MANDATORY (cite the block ID from chunks)")

            image_role = self._normalize_image_role(image_role_raw)

            plans.append(
                QuestionPlan(
                    target_concept=target_concept,
                    question_type=question_type,
                    difficulty=difficulty,
                    reasoning_type=reasoning_type,
                    image_role=image_role,
                    image_description=image_description,
                    learning_objective=learning_objective,
                    tested_fact_block_id=tested_fact_block_id,
                )
            )

        return plans

    def _validate_tested_facts(self, plans: list[QuestionPlan], topic_id: str) -> None:
        """Validate that all plans have tested_fact_block_id set.
        
        Args:
            plans: List of question plans.
            topic_id: Topic ID for logging.
            
        Raises:
            RuntimeError: If any plan lacks tested_fact_block_id.
        """
        missing = [i for i, p in enumerate(plans) if not p.tested_fact_block_id]
        if missing:
            raise RuntimeError(
                f"Topic {topic_id}: {len(missing)} questions missing tested_fact_block_id "
                f"(questions {missing}). Every question must cite a block ID."
            )

    @staticmethod
    def _clean_llm_output(raw: object) -> str:
        """Extract JSON from LLM output, handling markdown and fences."""
        if raw is None:
            return ""

        s = str(raw).strip()

        # Remove fenced code blocks
        if s.startswith("```"):
            closing = s.rfind("```")
            if closing > 3:
                inner = s[3:closing].strip()
                if inner.lower().startswith("json"):
                    inner = inner[4:].lstrip()
                s = inner

        s = s.strip().strip("`").strip()

        # Extract outermost JSON object
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            s = s[first:last + 1]

        return s

    def _normalize_image_role(self, value: object) -> str:
        """Normalize and validate image_role."""
        normalized = self._normalized_text(value).lower()
        if normalized not in self._VALID_IMAGE_ROLES:
            raise RuntimeError(f"Invalid image_role: {normalized}. Must be one of {self._VALID_IMAGE_ROLES}")
        return normalized

    @staticmethod
    def _normalized_text(value: object) -> str:
        """Normalize text values (strip, null-safe)."""
        if value is None:
            return ""
        return str(value).strip()

    def _validate_difficulty_distribution(self, dist: dict[str, float]) -> None:
        """Validate difficulty distribution sums to 1.0 and uses valid keys."""
        if not isinstance(dist, dict):
            raise ValueError("difficulty_distribution must be a dict")
        if not dist:
            raise ValueError("difficulty_distribution cannot be empty")

        total = 0.0
        for difficulty, proportion in dist.items():
            if difficulty not in self._VALID_DIFFICULTIES:
                raise ValueError(f"Invalid difficulty: {difficulty}")
            if not isinstance(proportion, (int, float)) or proportion < 0:
                raise ValueError(f"Proportion for {difficulty} must be a positive number")
            total += proportion

        if not (0.99 <= total <= 1.01):  # Allow for floating-point rounding
            raise ValueError(f"Difficulty distribution must sum to 1.0 (got {total})")

    def _attach_knowledge_context(self, plans: list[QuestionPlan], context) -> None:
        """Attach grounding text for each plan's `tested_fact_block_id` into plan.metadata['knowledge_context'].

        The function searches `context.concept_chunks` for a TextChunk whose `source_block_id` or
        `id` matches the `tested_fact_block_id` and copies the chunk text into the plan metadata.
        """
        if not plans:
            return

        # Build lookups: chunk id and source_block_id -> chunk object
        chunk_by_id: dict[str, object] = {}
        chunk_by_block: dict[str, object] = {}
        try:
            for concept_id, chunks in getattr(context, "concept_chunks", {}).items():
                for chunk in chunks:
                    # chunk may be a dataclass or dict-like
                    chunk_id = getattr(chunk, "id", None) or (chunk.get("id") if isinstance(chunk, dict) else None)
                    source_block = getattr(chunk, "source_block_id", None) or (chunk.get("source_block_id") if isinstance(chunk, dict) else None)
                    if chunk_id:
                        chunk_by_id[str(chunk_id)] = chunk
                    if source_block:
                        chunk_by_block[str(source_block)] = chunk
        except Exception:
            logger.exception("Error building chunk lookups from context; skipping knowledge_context attachment")

        # Helper to pick top-N chunks for a given concept id
        def pick_top_chunks_for_concept(concept_id: str, n: int) -> list[object]:
            try:
                chunks = getattr(context, "concept_chunks", {}).get(concept_id, [])
                # Assume chunks already ordered by relevance; fallback to first-n
                return chunks[:n]
            except Exception:
                return []

        for plan in plans:
            tfid = getattr(plan, "tested_fact_block_id", None)
            if not tfid:
                continue

            # Find the chunk that matches tfid (either chunk id or source_block_id)
            matched_chunk = chunk_by_id.get(str(tfid)) or chunk_by_block.get(str(tfid))

            # Determine number of chunks to attach by reasoning type
            reasoning = getattr(plan, "reasoning_type", "factoid") or "factoid"
            reasoning = reasoning.lower()
            if reasoning == "factoid":
                top_n = 1
            elif reasoning == "causal":
                top_n = 3
            elif reasoning == "multi-hop":
                top_n = 5
            else:
                top_n = 3

            attached_chunks: list[dict] = []

            # If we found the matched chunk, try to find its concept and pick siblings
            if matched_chunk:
                # Find concept id that contains this chunk
                found_concept_id = None
                for concept_id, chunks in getattr(context, "concept_chunks", {}).items():
                    for c in chunks:
                        cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                        sb = getattr(c, "source_block_id", None) or (c.get("source_block_id") if isinstance(c, dict) else None)
                        if str(cid) == str(tfid) or str(sb) == str(tfid):
                            found_concept_id = concept_id
                            break
                    if found_concept_id:
                        break

                if found_concept_id:
                    top_chunks = pick_top_chunks_for_concept(found_concept_id, top_n)
                    for c in top_chunks:
                        cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                        sb = getattr(c, "source_block_id", None) or (c.get("source_block_id") if isinstance(c, dict) else None)
                        text = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                        conf = getattr(c, "confidence", None) or (c.get("confidence") if isinstance(c, dict) else None)
                        attached_chunks.append({"id": cid, "source_block_id": sb, "confidence": conf, "text": text})

            # Fallback: if no matched chunk found, try direct lookup by tfid
            if not attached_chunks:
                direct = chunk_by_id.get(str(tfid)) or chunk_by_block.get(str(tfid))
                if direct:
                    c = direct
                    cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                    sb = getattr(c, "source_block_id", None) or (c.get("source_block_id") if isinstance(c, dict) else None)
                    text = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                    conf = getattr(c, "confidence", None) or (c.get("confidence") if isinstance(c, dict) else None)
                    attached_chunks.append({"id": cid, "source_block_id": sb, "confidence": conf, "text": text})

            # As a last resort, attach nothing
            if attached_chunks:
                if not isinstance(plan.metadata, dict):
                    plan.metadata = {}
                # Only set if not already present
                if not plan.metadata.get("knowledge_context"):
                    plan.metadata["knowledge_context"] = attached_chunks

    def save_plan(
        self,
        plans: list[QuestionPlan],
        output_path: str | Path,
    ) -> None:
        """Save plans to a JSON file for consumption by the generator.
        
        Args:
            plans: List of QuestionPlan objects to save.
            output_path: Path to write the JSON plan file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(p) for p in plans]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved topic agentic plan ({len(plans)} questions) → {output_path}")
