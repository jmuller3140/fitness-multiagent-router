from __future__ import annotations

import json
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from fitness_router.models import (
    BuildWorkoutInput,
    ExerciseSearchResult,
    GeneratedWorkout,
    HubState,
)
from fitness_router.settings import Settings, get_settings
from fitness_router.tools import (
    WORKOUT_TOOLS,
    ToolExecutionError,
    build_workout,
    parse_generation_constraints,
    search_exercises,
    search_exercises_tool,
)

WORKOUT_GENERATOR_SYSTEM_PROMPT = """You are a workout generator tool-calling agent.

Use the available tools instead of inventing exercises.
First call search_exercises from the user's constraints.
If search_exercises returns no matches, stop and let the application explain the supported options.
Only call build_workout with exercise_ids that came back from search_exercises.
The build_workout call must include duration_minutes, focus, and one or more valid exercise_ids.
"""

TOOL_CALL_LIMIT = 4
WORKOUT_TOOL_BY_NAME = {tool.name: tool for tool in WORKOUT_TOOLS}


class WorkoutGeneratorAgent(Protocol):
    def generate(self, state: HubState) -> HubState:
        """Generate a workout response from hub state."""


class DeterministicWorkoutGeneratorAgent:
    """Offline tool executor for demos and tests when no LLM is configured."""

    def generate(self, state: HubState) -> HubState:
        user_input = state["user_input"]
        search_input, duration, focus = parse_generation_constraints(user_input)
        tool_calls = [_tool_call_event("search_exercises", search_input.model_dump(mode="json"))]
        search_result = search_exercises(search_input)

        if not search_result.matches:
            return _no_results_state(search_result, tool_calls)

        selected_ids = [exercise.id for exercise in search_result.matches[:5]]
        build_input = BuildWorkoutInput(
            duration_minutes=duration,
            focus=focus,
            exercise_ids=selected_ids,
        )
        tool_calls.append(_tool_call_event("build_workout", build_input.model_dump(mode="json")))
        try:
            workout = build_workout(build_input)
        except ToolExecutionError as exc:
            return _tool_error_state(state, search_result, str(exc), tool_calls)

        return _workout_state(search_result, workout, tool_calls)


class LangChainWorkoutGeneratorAgent:
    def __init__(self, settings: Settings | None = None, chain: Any | None = None):
        self.settings = settings or get_settings()
        self._chain = chain

    def _load_chain(self):
        if self._chain is not None:
            return self._chain
        if not self.settings.dspy_model:
            raise RuntimeError("DSPY_MODEL must be set to use the LangChain workout generator.")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - dependency is installed in the API env.
            raise RuntimeError(
                "LangChain OpenAI integration is not installed. Run uv sync in apps/api."
            ) from exc

        llm = ChatOpenAI(
            model=self.settings.langchain_model_name(),
            temperature=0,
            **self.settings.langchain_chat_kwargs(),
        )
        self._chain = llm.bind_tools(WORKOUT_TOOLS)
        return self._chain

    def generate(self, state: HubState) -> HubState:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        except ImportError as exc:  # pragma: no cover - dependency is installed in the API env.
            raise RuntimeError("LangChain core messages are not installed.") from exc

        chain = self._load_chain()
        messages: list[Any] = [
            SystemMessage(content=WORKOUT_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=state["user_input"]),
        ]
        tool_calls: list[dict[str, Any]] = []
        search_result: ExerciseSearchResult | None = None

        for _ in range(TOOL_CALL_LIMIT):
            response = chain.invoke(messages)
            messages.append(response)

            invalid_calls = list(getattr(response, "invalid_tool_calls", []) or [])
            if invalid_calls:
                return _tool_error_state(
                    state,
                    search_result,
                    f"Invalid tool call schema: {invalid_calls}",
                    tool_calls,
                )

            requested_calls = list(getattr(response, "tool_calls", []) or [])
            if not requested_calls:
                if search_result and search_result.matches:
                    return _build_from_search_result(state, search_result, tool_calls)
                return _tool_error_state(
                    state,
                    search_result,
                    "LLM did not call the required workout tools.",
                    tool_calls,
                )

            for requested in requested_calls:
                name = str(requested.get("name") or "")
                args = requested.get("args") if isinstance(requested.get("args"), dict) else {}
                call_id = str(requested.get("id") or f"call_{len(tool_calls) + 1}")
                tool_calls.append(_tool_call_event(name, args))

                tool = WORKOUT_TOOL_BY_NAME.get(name)
                if tool is None:
                    return _tool_error_state(
                        state,
                        search_result,
                        f"Unknown workout tool: {name}",
                        tool_calls,
                    )

                try:
                    payload = tool.invoke(args)
                except Exception as exc:
                    return _tool_error_state(state, search_result, str(exc), tool_calls)

                messages.append(
                    ToolMessage(content=_json_tool_content(payload), tool_call_id=call_id)
                )

                if name == search_exercises_tool.name:
                    search_result = ExerciseSearchResult.model_validate(payload)
                    if not search_result.matches:
                        return _no_results_state(search_result, tool_calls)
                else:
                    workout = GeneratedWorkout.model_validate(payload)
                    return _workout_state(search_result, workout, tool_calls)

        assert search_result is not None
        return _build_from_search_result(state, search_result, tool_calls)


def make_workout_generator_agent(settings: Settings | None = None) -> WorkoutGeneratorAgent:
    resolved = settings or get_settings()
    if resolved.dspy_model:
        return LangChainWorkoutGeneratorAgent(resolved)
    return DeterministicWorkoutGeneratorAgent()


def workout_generator_node(
    state: HubState,
    agent: WorkoutGeneratorAgent | None = None,
) -> HubState:
    return (agent or make_workout_generator_agent()).generate(state)


def _workout_state(
    search_result: ExerciseSearchResult | None,
    workout: GeneratedWorkout,
    tool_calls: list[dict[str, Any]],
) -> HubState:
    section_lines = []
    for section in workout.sections:
        names = ", ".join(item.name for item in section.exercises)
        section_lines.append(f"{section.name.title()}: {names}")
    response = f"{workout.title}\n" + "\n".join(section_lines)

    return {
        "final_response": response,
        "structured_output": {
            "search": search_result.model_dump() if search_result else None,
            "workout": workout.model_dump(),
            "tool_calls": tool_calls,
        },
    }


def _build_from_search_result(
    state: HubState,
    search_result: ExerciseSearchResult,
    tool_calls: list[dict[str, Any]],
) -> HubState:
    _, duration, focus = parse_generation_constraints(state["user_input"])
    build_input = BuildWorkoutInput(
        duration_minutes=duration,
        focus=focus,
        exercise_ids=[exercise.id for exercise in search_result.matches[:5]],
    )
    tool_calls.append(_tool_call_event("build_workout", build_input.model_dump(mode="json")))
    try:
        workout = build_workout(build_input)
    except ToolExecutionError as exc:
        return _tool_error_state(state, search_result, str(exc), tool_calls)
    return _workout_state(search_result, workout, tool_calls)


def _no_results_state(
    search_result: ExerciseSearchResult,
    tool_calls: list[dict[str, Any]],
) -> HubState:
    response = (
        f"I could not build that workout because {search_result.no_results_reason}. "
        "I will not invent exercises outside the dataset. Supported equipment includes: "
        f"{', '.join(search_result.supported_equipment[:10])}."
    )
    return {
        "final_response": response,
        "structured_output": {
            "search": search_result.model_dump(),
            "workout": None,
            "tool_calls": tool_calls,
        },
    }


def _tool_error_state(
    state: HubState,
    search_result: ExerciseSearchResult | None,
    error: str,
    tool_calls: list[dict[str, Any]],
) -> HubState:
    return {
        "final_response": f"The workout tool call was invalid: {error}",
        "structured_output": {
            "search": search_result.model_dump() if search_result else None,
            "workout": None,
            "tool_calls": tool_calls,
            "tool_error": error,
        },
        "errors": [*state.get("errors", []), error],
    }


def _tool_call_event(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args}


def _json_tool_content(payload: object) -> str:
    return json.dumps(payload, sort_keys=True)


def build_workout_generator_graph(
    settings: Settings | None = None,
    agent: WorkoutGeneratorAgent | None = None,
):
    resolved_agent = agent or make_workout_generator_agent(settings)
    graph = StateGraph(HubState)
    graph.add_node("workout_generator", lambda state: workout_generator_node(state, resolved_agent))
    graph.add_edge(START, "workout_generator")
    graph.add_edge("workout_generator", END)
    return graph.compile()
