"""Topic-driven agentic quiz planner for dynamic, concept-rich question generation."""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

from src.knowledge.retriever import TopicContextRetriever, calculate_topic_budget
from src.knowledge.schema import MultimodalDocumentGraph, NodeKind
from src.planner.planner import QuestionPlan
from src.question_formats import (
    QuestionFormatProfile,
    coerce_format_profile,
    normalize_question_type,
)
from src.planner.topic_prompt_templates import render_topic_plan_prompt
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)


def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, object]) -> None:
    # region agent log
    try:
        with open("debug-c02dfd.log", "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "c02dfd",
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(dt.datetime.utcnow().timestamp() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        pass
    # endregion


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
    _TYPE_ORDER = ("multiple_choice", "true_false", "fill_in_blank", "matching")

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
        format_profile: QuestionFormatProfile | Mapping[str, float] | str | None = None,
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
        profile = coerce_format_profile(format_profile)
        target_type_counts = profile.expected_counts(effective_total)
        produced_type_counts: Counter[str] = Counter()

        # Prefer explicit topic nodes from topic induction. Fall back to concept nodes for older graphs.
        topic_ids = [node.id for node in self._knowledge_graph.nodes if node.kind == NodeKind.topic]
        topic_source = "topic"
        if not topic_ids:
            topic_ids = [node.id for node in self._knowledge_graph.nodes if node.kind == NodeKind.concept]
            topic_source = "concept-fallback"

        if not topic_ids:
            raise ValueError("No topic nodes found in graph. Cannot proceed with topic-agentic planning.")

        logger.info("Found %d %s candidates. Planning %d questions.", len(topic_ids), topic_source, effective_total)

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

        logger.debug(
            "Initial topic contexts retrieved: %d. Starting planning loop with budget %d.",
            len(topic_contexts),
            remaining_budget,
        )
        # Loop through topics and allocate proportionally by uncovered concept counts
        for topic_id in topic_ids:
            logger.info("Evaluating topic %s for question generation (remaining budget: %d)", topic_id, remaining_budget)
            if topic_id not in topic_contexts:
                continue

            context = topic_contexts[topic_id]
            uncovered = topic_uncovered_concepts(context)
            # Skip topics with no uncovered concepts
            if not uncovered:
                logger.debug("Skipping topic %s (%s) - no uncovered concepts", topic_id, context.topic_label)
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
            _debug_log(
                run_id="planner-loop",
                hypothesis_id="H5",
                location="src/planner/topic_planner.py:169",
                message="topic allocation before generation",
                data={
                    "topic_id": topic_id,
                    "topic_label": context.topic_label,
                    "allocated": allocated,
                    "remaining_budget": remaining_budget,
                    "allowed_types": sorted(profile.allowed_types),
                    "distribution": dict(profile.distribution),
                    "target_type_counts": dict(target_type_counts),
                    "produced_type_counts": dict(produced_type_counts),
                },
            )

            # Generate plans for this topic, restricting to uncovered concept labels
            try:
                only_concept_labels = [c.label for c in uncovered]
                required_types = self._build_required_types_for_batch(
                    profile=profile,
                    target_counts=target_type_counts,
                    produced_counts=produced_type_counts,
                    batch_size=allocated,
                )
                topic_plans = self._generate_topic_plans(
                    context,
                    allocated,
                    difficulty_distribution,
                    only_concepts=only_concept_labels,
                    format_profile=profile,
                    required_types=required_types,
                )

                all_plans.extend(topic_plans)
                produced_type_counts.update(p.question_type for p in topic_plans)
                # Mark produced target concepts as used
                for p in topic_plans:
                    if p.target_concept:
                        used_concepts.add(p.target_concept.lower())

                remaining_budget -= len(topic_plans)
                if remaining_budget <= 0:
                    break
            except RuntimeError as e:
                logger.error("Failed to generate plans for topic %s: %s", context.topic_label, e)
                continue

        if not all_plans:
            raise RuntimeError("No question plans were generated.")

        logger.info(f"Generated {len(all_plans)} question plans (requested {effective_total})")
        all_plans = self._finalize_global_type_distribution(
            plans=all_plans,
            profile=profile,
            requested_total=effective_total,
        )
        _all_counts = Counter(p.question_type for p in all_plans)
        _debug_log(
            run_id="planner-final",
            hypothesis_id="H6",
            location="src/planner/topic_planner.py:199",
            message="final topic planner question type counts",
            data={
                "requested_total": effective_total,
                "produced_total": len(all_plans),
                "counts": dict(_all_counts),
            },
        )

        return all_plans

    def _generate_topic_plans(
        self,
        context,
        num_questions: int,
        difficulty_distribution: dict[str, float],
        *,
        only_concepts: list[str] | None = None,
        format_profile: QuestionFormatProfile | None = None,
        required_types: list[str] | None = None,
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
        profile = format_profile or coerce_format_profile(None)
        prompt = render_topic_plan_prompt(
            context,
            num_questions,
            difficulty_distribution,
            only_concepts=only_concepts,
            format_profile=profile,
            required_types=required_types,
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
                plans = self._parse_topic_plans(
                    payload, num_questions, context.topic_id, profile, required_types=required_types
                )

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
                lowered = str(exc).lower()
                if "tested_fact" in lowered or "question format" in lowered:
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
        format_profile: QuestionFormatProfile,
        required_types: list[str] | None = None,
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
        if required_types is not None and len(required_types) != expected_count:
            raise RuntimeError(
                f"Topic {topic_id}: required_types must have {expected_count} entries, got {len(required_types)}"
            )

        plans: list[QuestionPlan] = []

        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"Question {idx} must be a JSON object")

            target_concept = self._normalized_text(row.get("target_concept"))
            question_type_raw = self._normalized_text(row.get("question_type"))
            question_type = normalize_question_type(question_type_raw)
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
            if question_type not in format_profile.allowed_types:
                raise RuntimeError(
                    f"Question {idx}: question_type {question_type_raw!r} (normalized {question_type!r}) "
                    f"not allowed by format profile {sorted(format_profile.allowed_types)}. question format retry."
                )
            if required_types is not None and question_type != required_types[idx]:
                raise RuntimeError(
                    f"Question {idx}: expected question_type {required_types[idx]!r}, got {question_type!r}. "
                    "question format retry."
                )
            if difficulty not in self._VALID_DIFFICULTIES:
                raise RuntimeError(f"Question {idx}: invalid difficulty '{difficulty}'")
            if reasoning_type not in self._VALID_REASONING:
                raise RuntimeError(f"Question {idx}: invalid reasoning_type '{reasoning_type}'")
            if not image_description:
                raise RuntimeError(f"Question {idx}: image_description is required")
            if not tested_fact_block_id:
                raise RuntimeError(f"Question {idx}: tested_fact_block_id is MANDATORY (cite the block ID from chunks)")

            image_role = self._normalize_image_role(image_role_raw)

            row_meta = row.get("metadata")
            plan_meta: dict[str, object] = dict(row_meta) if isinstance(row_meta, dict) else {}
            if question_type == "matching":
                mp = row.get("matching_pairs")
                if isinstance(mp, list) and "matching_pairs" not in plan_meta:
                    plan_meta["matching_pairs"] = mp

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
                    metadata=plan_meta,
                )
            )

        _check_applies = (
            required_types is None
            and (not format_profile.is_mcq_only())
            and (len(format_profile.allowed_types) > 1)
        )
        _debug_log(
            run_id=f"topic-{topic_id}",
            hypothesis_id="H7",
            location="src/planner/topic_planner.py:367",
            message="distribution check gate",
            data={
                "expected_count": expected_count,
                "allowed_types_count": len(format_profile.allowed_types),
                "is_mcq_only": format_profile.is_mcq_only(),
                "required_types": required_types or [],
                "check_applies": _check_applies,
                "counts": dict(Counter(p.question_type for p in plans)),
                "expected_counts": format_profile.expected_counts(expected_count),
            },
        )

        if _check_applies:
            tallies = Counter(p.question_type for p in plans)
            count_map = {t: int(tallies.get(t, 0)) for t in format_profile.allowed_types}
            if not format_profile.distribution_within_tolerance(count_map, expected_count):
                raise RuntimeError(
                    f"Topic {topic_id}: question format distribution off-target "
                    f"(counts={count_map}, expected≈{format_profile.expected_counts(expected_count)}). "
                    "question format retry."
                )

        return plans

    def _build_required_types_for_batch(
        self,
        *,
        profile: QuestionFormatProfile,
        target_counts: Mapping[str, int],
        produced_counts: Mapping[str, int],
        batch_size: int,
    ) -> list[str]:
        """Build deterministic required type sequence for the next batch."""
        if batch_size <= 0:
            return []

        remaining = {
            t: max(0, int(target_counts.get(t, 0)) - int(produced_counts.get(t, 0)))
            for t in profile.allowed_types
        }
        required: list[str] = []
        type_order = [t for t in self._TYPE_ORDER if t in profile.allowed_types]
        fallback_type = type_order[0] if type_order else "multiple_choice"

        for _ in range(batch_size):
            positive = [t for t in type_order if remaining.get(t, 0) > 0]
            if positive:
                choice = max(
                    positive,
                    key=lambda t: (remaining[t], profile.distribution.get(t, 0.0), -type_order.index(t)),
                )
                remaining[choice] -= 1
                required.append(choice)
            else:
                required.append(fallback_type)
        return required

    def _finalize_global_type_distribution(
        self,
        *,
        plans: list[QuestionPlan],
        profile: QuestionFormatProfile,
        requested_total: int,
    ) -> list[QuestionPlan]:
        """Ensure final plan list matches global type targets as closely as possible."""
        if requested_total <= 0:
            return plans
        trimmed = plans[:requested_total]
        target_counts = profile.expected_counts(len(trimmed))
        current_counts = Counter(p.question_type for p in trimmed)
        deficits = {
            t: max(0, int(target_counts.get(t, 0)) - int(current_counts.get(t, 0)))
            for t in profile.allowed_types
        }
        surpluses = {
            t: max(0, int(current_counts.get(t, 0)) - int(target_counts.get(t, 0)))
            for t in profile.allowed_types
        }
        if not any(deficits.values()):
            return trimmed

        donor_indices: dict[str, list[int]] = {t: [] for t in profile.allowed_types}
        for idx, plan in enumerate(trimmed):
            t = plan.question_type
            if surpluses.get(t, 0) > 0:
                donor_indices[t].append(idx)
                surpluses[t] -= 1

        type_order = [t for t in self._TYPE_ORDER if t in profile.allowed_types]
        for needed in type_order:
            while deficits.get(needed, 0) > 0:
                donor = next((t for t in type_order if donor_indices.get(t)), None)
                if donor is None:
                    raise RuntimeError(
                        "Unable to repair global question-type distribution after planning. "
                        f"Current={dict(Counter(p.question_type for p in trimmed))}, "
                        f"target={target_counts}."
                    )
                idx = donor_indices[donor].pop()
                trimmed[idx].question_type = needed
                deficits[needed] -= 1

        return trimmed

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
