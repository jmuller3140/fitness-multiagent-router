import sys
from types import ModuleType, SimpleNamespace

import pytest

from fitness_router.graphs.workout_generator import (
    DeterministicWorkoutGeneratorAgent,
    LangChainWorkoutGeneratorAgent,
    make_workout_generator_agent,
)
from fitness_router.models import SearchExercisesInput
from fitness_router.settings import Settings
from fitness_router.tools import ToolExecutionError, search_exercises


class FakeToolCallingChain:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return SimpleNamespace(tool_calls=[], invalid_tool_calls=[])


def _tool_response(name, args, call_id):
    return SimpleNamespace(
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        invalid_tool_calls=[],
    )


def test_langchain_workout_generator_executes_search_then_build_tools():
    search_result = search_exercises(SearchExercisesInput(equipment=["Dumbbell"], limit=1))
    exercise_id = search_result.matches[0].id
    chain = FakeToolCallingChain(
        [
            _tool_response(
                "search_exercises",
                {"equipment": ["Dumbbell"], "limit": 1},
                "search-call",
            ),
            _tool_response(
                "build_workout",
                {
                    "duration_minutes": 30,
                    "focus": "upper body",
                    "exercise_ids": [exercise_id],
                },
                "build-call",
            ),
        ]
    )
    agent = LangChainWorkoutGeneratorAgent(Settings(dspy_model="openai/test"), chain=chain)

    state = agent.generate({"user_input": "Build an upper body dumbbell workout", "errors": []})

    assert state["structured_output"]["workout"]["duration_minutes"] == 30
    assert [call["name"] for call in state["structured_output"]["tool_calls"]] == [
        "search_exercises",
        "build_workout",
    ]
    assert len(chain.messages) == 2


def test_langchain_workout_generator_recovers_from_no_results_and_tool_errors():
    no_results = LangChainWorkoutGeneratorAgent(
        Settings(dspy_model="openai/test"),
        chain=FakeToolCallingChain(
            [
                _tool_response(
                    "search_exercises",
                    {"equipment": ["Cable Machine", "Sled"], "limit": 3},
                    "search-call",
                )
            ]
        ),
    ).generate({"user_input": "Build me a cable and sled workout", "errors": []})
    invalid_id = LangChainWorkoutGeneratorAgent(
        Settings(dspy_model="openai/test"),
        chain=FakeToolCallingChain(
            [
                _tool_response(
                    "build_workout",
                    {
                        "duration_minutes": 30,
                        "focus": "bad ids",
                        "exercise_ids": ["not-a-real-exercise-id"],
                    },
                    "build-call",
                )
            ]
        ),
    ).generate({"user_input": "Build a workout", "errors": ["prior"]})

    assert no_results["structured_output"]["workout"] is None
    assert no_results["structured_output"]["search"]["matches"] == []
    assert "will not invent exercises outside the dataset" in no_results["final_response"]
    assert invalid_id["structured_output"]["tool_error"] == (
        "Unknown exercise ids: not-a-real-exercise-id"
    )
    assert invalid_id["errors"] == ["prior", "Unknown exercise ids: not-a-real-exercise-id"]


def test_langchain_workout_generator_handles_bad_llm_tool_boundaries():
    invalid_schema = LangChainWorkoutGeneratorAgent(
        Settings(dspy_model="openai/test"),
        chain=FakeToolCallingChain(
            [SimpleNamespace(tool_calls=[], invalid_tool_calls=[{"args": "bad"}])]
        ),
    ).generate({"user_input": "Build a workout", "errors": []})
    no_tool_call = LangChainWorkoutGeneratorAgent(
        Settings(dspy_model="openai/test"),
        chain=FakeToolCallingChain([SimpleNamespace(tool_calls=[], invalid_tool_calls=[])]),
    ).generate({"user_input": "Build a workout", "errors": []})
    unknown_tool = LangChainWorkoutGeneratorAgent(
        Settings(dspy_model="openai/test"),
        chain=FakeToolCallingChain([_tool_response("unknown_tool", {}, "bad-call")]),
    ).generate({"user_input": "Build a workout", "errors": []})

    assert "Invalid tool call schema" in invalid_schema["structured_output"]["tool_error"]
    assert no_tool_call["structured_output"]["tool_error"] == (
        "LLM did not call the required workout tools."
    )
    assert unknown_tool["structured_output"]["tool_error"] == "Unknown workout tool: unknown_tool"


def test_langchain_workout_generator_auto_builds_after_search_without_build_call():
    state = LangChainWorkoutGeneratorAgent(
        Settings(dspy_model="openai/test"),
        chain=FakeToolCallingChain(
            [
                _tool_response(
                    "search_exercises",
                    {"equipment": ["Dumbbell"], "limit": 1},
                    "search-call",
                ),
                SimpleNamespace(tool_calls=[], invalid_tool_calls=[]),
            ]
        ),
    ).generate({"user_input": "Build a 20 minute dumbbell workout", "errors": []})

    assert state["structured_output"]["workout"]["duration_minutes"] == 20
    assert [call["name"] for call in state["structured_output"]["tool_calls"]] == [
        "search_exercises",
        "build_workout",
    ]


def test_langchain_workout_generator_auto_build_reports_build_errors(monkeypatch):
    def fail_build_workout(input):
        raise ToolExecutionError("auto build failed")

    monkeypatch.setattr("fitness_router.graphs.workout_generator.build_workout", fail_build_workout)
    state = LangChainWorkoutGeneratorAgent(
        Settings(dspy_model="openai/test"),
        chain=FakeToolCallingChain(
            [
                _tool_response(
                    "search_exercises",
                    {"equipment": ["Dumbbell"], "limit": 1},
                    "search-call",
                ),
                SimpleNamespace(tool_calls=[], invalid_tool_calls=[]),
            ]
        ),
    ).generate({"user_input": "Build a dumbbell workout", "errors": ["prior"]})

    assert state["structured_output"]["tool_error"] == "auto build failed"
    assert state["errors"] == ["prior", "auto build failed"]
    assert [call["name"] for call in state["structured_output"]["tool_calls"]] == [
        "search_exercises",
        "build_workout",
    ]


def test_langchain_workout_generator_auto_builds_after_repeated_search_calls():
    chain = FakeToolCallingChain(
        [
            _tool_response("search_exercises", {"equipment": ["Dumbbell"], "limit": 1}, "call-1"),
            _tool_response("search_exercises", {"equipment": ["Dumbbell"], "limit": 1}, "call-2"),
            _tool_response("search_exercises", {"equipment": ["Dumbbell"], "limit": 1}, "call-3"),
            _tool_response("search_exercises", {"equipment": ["Dumbbell"], "limit": 1}, "call-4"),
        ]
    )
    agent = LangChainWorkoutGeneratorAgent(Settings(dspy_model="openai/test"), chain=chain)
    state = agent.generate({"user_input": "Build a workout", "errors": []})

    assert state["structured_output"]["workout"]["duration_minutes"] == 30
    assert [call["name"] for call in state["structured_output"]["tool_calls"]] == [
        "search_exercises",
        "search_exercises",
        "search_exercises",
        "search_exercises",
        "build_workout",
    ]


def test_workout_generator_agent_factory_and_langchain_loader(monkeypatch):
    events = []
    fake_openai = ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            events.append(("llm", kwargs))

        def bind_tools(self, tools):
            events.append(("bind_tools", [tool.name for tool in tools]))
            return "bound-chain"

    fake_openai.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai)

    deterministic = make_workout_generator_agent(Settings(dspy_model=None))
    langchain = make_workout_generator_agent(
        Settings(dspy_model="deepseek/deepseek-chat", deepseek_api_key="key")
    )
    loaded = langchain._load_chain()

    assert isinstance(deterministic, DeterministicWorkoutGeneratorAgent)
    assert isinstance(langchain, LangChainWorkoutGeneratorAgent)
    assert loaded == "bound-chain"
    assert langchain._load_chain() == "bound-chain"
    assert events == [
        (
            "llm",
            {
                "model": "deepseek-chat",
                "temperature": 0,
                "api_key": "key",
                "base_url": "https://api.deepseek.com",
            },
        ),
        ("bind_tools", ["search_exercises", "build_workout"]),
    ]


def test_langchain_workout_generator_requires_model():
    with pytest.raises(RuntimeError, match="DSPY_MODEL"):
        LangChainWorkoutGeneratorAgent(Settings(dspy_model=None))._load_chain()
