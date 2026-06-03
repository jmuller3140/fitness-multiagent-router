from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from fitness_router.models import RouteCandidate, RouteDecision, RouteName
from fitness_router.route_registry import build_route_catalog, clarification_question
from fitness_router.router_artifact import (
    artifact_to_prompt_messages,
    default_router_prompt_artifact,
    load_router_prompt_artifact,
)
from fitness_router.settings import Settings, get_settings


class IntentRouter(Protocol):
    def route(self, user_input: str) -> RouteDecision:
        """Return a typed route decision for a user request."""


class DemoIntentRouter:
    """Offline router for local demos and tests when no DSPy LM is configured."""

    def route(self, user_input: str) -> RouteDecision:
        text = user_input.casefold().strip()

        missing_context_markers = (
            "adjust it",
            "make it better",
            "this plan",
            "what about",
            "that was too",
        )
        if any(marker in text for marker in missing_context_markers):
            return RouteDecision(
                route="FALLBACK",
                confidence=0.57,
                reason="The request refers to prior context that is not available.",
                clarification_question=clarification_question(user_input),
            )

        generate_markers = (
            "build",
            "create",
            "make",
            "give me",
            "plan",
            "session",
            "workout",
            "design",
            "generate",
            "warmup",
        )
        log_markers = (
            "i did",
            "just did",
            "logged",
            "completed",
            "finished",
            "log",
            "record",
            "save",
            "mark",
            "add",
            "hit",
            "sets",
        )

        has_generate_intent = any(marker in text for marker in generate_markers)
        has_log_marker = any(marker in text for marker in log_markers) or bool(
            re.search(r"\d+\s*x\s*\d+", text)
        )
        has_completed_phrase = any(
            marker in text
            for marker in (
                "i did",
                "i just did",
                "just did",
                "completed",
                "finished",
                "logged",
            )
        )
        has_log_intent = has_log_marker and (
            any(char.isdigit() for char in text) or "done" in text or has_completed_phrase
        )
        has_multi_intent = has_generate_intent and (
            has_log_marker or "logging" in text or "changing my plan" in text
        )
        if has_multi_intent:
            return RouteDecision(
                route="FALLBACK",
                confidence=0.58,
                reason="The request contains both workout generation and logging intent.",
                clarification_question=clarification_question(user_input),
            )

        if len(text.split()) <= 2:
            return RouteDecision(
                route="FALLBACK",
                confidence=0.55,
                reason=(
                    "The request names a fitness concept but does not say whether "
                    "to explain, plan, or log it."
                ),
                candidate_routes=[
                    RouteCandidate(route="COACH", score=0.33),
                    RouteCandidate(route="WORKOUT_GENERATE", score=0.33),
                    RouteCandidate(route="WORKOUT_LOG", score=0.34),
                ],
                clarification_question=clarification_question(user_input),
            )

        question_starts = ("what", "why", "how", "which", "should", "is ", "explain")
        if text.startswith(question_starts):
            return RouteDecision(
                route="COACH",
                confidence=0.87,
                reason="The request asks for coaching or fitness information.",
            )

        if has_log_intent:
            return RouteDecision(
                route="WORKOUT_LOG",
                confidence=0.9,
                reason="The request describes completed exercise work with quantities.",
            )

        if has_generate_intent:
            return RouteDecision(
                route="WORKOUT_GENERATE",
                confidence=0.88,
                reason="The request asks for a workout plan from constraints.",
            )

        return RouteDecision(
            route="FALLBACK",
            confidence=0.52,
            reason="The intent is underspecified.",
            clarification_question=clarification_question(user_input),
        )


class StaticIntentRouter:
    def __init__(self, decision: RouteDecision):
        self.decision = decision

    def route(self, user_input: str) -> RouteDecision:
        return self.decision


class LangChainStructuredIntentRouter:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._chain = None

    def _load_artifact(self) -> dict[str, Any]:
        artifact_path = self.settings.dspy_router_artifact_file()
        if not artifact_path.exists():
            return default_router_prompt_artifact()
        return load_router_prompt_artifact(artifact_path.read_text(encoding="utf-8"))

    def _load_chain(self):
        if self._chain is not None:
            return self._chain

        if not self.settings.dspy_model:
            raise RuntimeError("DSPY_MODEL must be set to use the structured LLM router.")

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LangChain OpenAI integration is not installed. Run uv sync in apps/api."
            ) from exc

        artifact = self._load_artifact()
        prompt = ChatPromptTemplate.from_messages(artifact_to_prompt_messages(artifact))
        llm = ChatOpenAI(
            model=self.settings.langchain_model_name(),
            temperature=0,
            **self.settings.langchain_chat_kwargs(),
        )
        self._chain = prompt | llm.with_structured_output(
            RouteDecision,
            method="function_calling",
        )
        return self._chain

    def route(self, user_input: str) -> RouteDecision:
        raw = self._load_chain().invoke(
            {
                "route_catalog": build_route_catalog(),
                "user_input": user_input,
            }
        )
        if isinstance(raw, RouteDecision):
            decision = raw
        else:
            decision = RouteDecision.model_validate(raw)

        if decision.route == "FALLBACK" and not decision.clarification_question:
            decision = decision.model_copy(
                update={"clarification_question": clarification_question(user_input)}
            )
        elif decision.clarification_question == "":
            decision = decision.model_copy(update={"clarification_question": None})
        return decision


class DSPyIntentRouter:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._program = None

    def _load_program(self):
        if self._program is not None:
            return self._program

        try:
            import dspy
        except ImportError as exc:
            raise RuntimeError("DSPy is not installed. Run uv sync in apps/api.") from exc

        if not self.settings.dspy_model:
            raise RuntimeError("DSPY_MODEL must be set to use the DSPy router.")

        from fitness_router.router_program import IntentRouterProgram

        dspy.configure(lm=dspy.LM(self.settings.dspy_model, **self.settings.dspy_lm_kwargs()))
        program = IntentRouterProgram()
        artifact_path = self.settings.dspy_router_artifact_file()
        if artifact_path.exists() and not _is_langchain_runtime_artifact(artifact_path):
            program.load(str(artifact_path))
        self._program = program
        return program

    def route(self, user_input: str) -> RouteDecision:
        program = self._load_program()
        prediction = program(route_catalog=build_route_catalog(), user_input=user_input)
        return coerce_prediction_to_route_decision(prediction, user_input)


def coerce_prediction_to_route_decision(prediction: object, user_input: str) -> RouteDecision:
    values: dict[str, object] = {}
    if hasattr(prediction, "toDict"):
        values = prediction.toDict()
    elif hasattr(prediction, "__dict__"):
        values = dict(prediction.__dict__)
    elif isinstance(prediction, dict):
        values = prediction

    candidate_routes = values.get("candidate_routes") or []
    parsed_candidates = _parse_candidate_routes(candidate_routes)

    route = str(values.get("route", "FALLBACK")).strip().upper()
    valid_routes: set[RouteName] = {"COACH", "WORKOUT_GENERATE", "WORKOUT_LOG", "FALLBACK"}
    if route not in valid_routes:
        route = "FALLBACK"

    confidence = _float_or_default(values.get("confidence"), 0.0)
    clarification = values.get("clarification_question")
    if not clarification and route == "FALLBACK":
        clarification = clarification_question(user_input)

    return RouteDecision(
        route=route,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, confidence)),
        reason=str(values.get("reason") or "DSPy router returned a structured route decision."),
        candidate_routes=parsed_candidates,
        clarification_question=str(clarification) if clarification else None,
    )


def make_intent_router(settings: Settings | None = None) -> IntentRouter:
    resolved = settings or get_settings()
    if resolved.dspy_model:
        return LangChainStructuredIntentRouter(resolved)
    if resolved.allow_demo_router_without_llm:
        return DemoIntentRouter()
    raise RuntimeError("No intent router configured. Set DSPY_MODEL or allow demo routing.")


def route_decision_from_json(path: str | Path) -> RouteDecision:
    return RouteDecision.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _is_langchain_runtime_artifact(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(
            "kind"
        ) == "langchain_structured_router_prompt"
    except (OSError, ValueError):
        return False


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _parse_candidate_routes(raw: object) -> list[RouteCandidate]:
    if isinstance(raw, str):
        candidates = []
        for item in raw.split(","):
            route = item.strip().upper()
            if route in {"COACH", "WORKOUT_GENERATE", "WORKOUT_LOG", "FALLBACK"}:
                candidates.append(RouteCandidate(route=route, score=0.0))  # type: ignore[arg-type]
        return candidates

    if isinstance(raw, list):
        parsed = []
        for item in raw:
            if isinstance(item, dict):
                try:
                    parsed.append(RouteCandidate.model_validate(item))
                except ValueError:
                    continue
            elif isinstance(item, str):
                route = item.strip().upper()
                if route in {"COACH", "WORKOUT_GENERATE", "WORKOUT_LOG", "FALLBACK"}:
                    parsed.append(RouteCandidate(route=route, score=0.0))  # type: ignore[arg-type]
        return parsed

    return []
