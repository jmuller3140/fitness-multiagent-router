import builtins
import json
import sys
from types import ModuleType

import pytest

from fitness_router.models import RouteCandidate, RouteDecision
from fitness_router.paths import ROOT_DIR
from fitness_router.router_artifact import (
    LANGCHAIN_ROUTER_ARTIFACT_KIND,
    artifact_to_prompt_messages,
    load_router_prompt_artifact,
    runtime_artifact_from_dspy_state,
)
from fitness_router.routing import (
    DemoIntentRouter,
    DSPyIntentRouter,
    LangChainStructuredIntentRouter,
    StaticIntentRouter,
    _float_or_default,
    _is_langchain_runtime_artifact,
    _parse_candidate_routes,
    coerce_prediction_to_route_decision,
    make_intent_router,
    route_decision_from_json,
)
from fitness_router.settings import Settings


@pytest.mark.parametrize(
    ("message", "route"),
    [
        ("Can you adjust it?", "FALLBACK"),
        ("Build a workout and log 3x10 bench press", "FALLBACK"),
        ("Bench press", "FALLBACK"),
        ("What muscles does a deadlift work?", "COACH"),
        ("I completed bench press", "WORKOUT_LOG"),
        ("I just did 3x10 bench press", "WORKOUT_LOG"),
        ("Build me a 30 minute workout", "WORKOUT_GENERATE"),
        ("train harder please", "FALLBACK"),
    ],
)
def test_demo_intent_router_classifies_supported_and_fallback_cases(message, route):
    decision = DemoIntentRouter().route(message)

    assert decision.route == route
    if route == "FALLBACK":
        assert decision.clarification_question


def test_static_intent_router_returns_configured_decision():
    decision = RouteDecision(route="COACH", confidence=1.0, reason="fixed")

    assert StaticIntentRouter(decision).route("anything") is decision


def test_dspy_router_reports_missing_import(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dspy":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="DSPy is not installed"):
        DSPyIntentRouter(Settings(dspy_model="fake"))._load_program()


def test_dspy_router_requires_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "dspy", ModuleType("dspy"))

    with pytest.raises(RuntimeError, match="DSPY_MODEL must be set"):
        DSPyIntentRouter(Settings(dspy_model=None))._load_program()


def test_dspy_router_loads_artifact_and_routes(monkeypatch, tmp_path):
    artifact = tmp_path / "router.json"
    artifact.write_text("{}", encoding="utf-8")
    events = []

    fake_dspy = ModuleType("dspy")

    class FakeLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    def configure(*, lm):
        events.append(("configure", lm.model, lm.kwargs))

    fake_dspy.LM = FakeLM
    fake_dspy.configure = configure

    fake_program_module = ModuleType("fitness_router.router_program")

    class FakeProgram:
        def __init__(self):
            self.loaded = []

        def load(self, path):
            self.loaded.append(path)
            events.append(("load", path))

        def __call__(self, *, route_catalog, user_input):
            events.append(("call", bool(route_catalog), user_input))
            return {"route": "COACH", "confidence": "0.93", "reason": "Question"}

    fake_program_module.IntentRouterProgram = FakeProgram
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    monkeypatch.setitem(sys.modules, "fitness_router.router_program", fake_program_module)

    router = DSPyIntentRouter(
        Settings(
            dspy_model="deepseek/deepseek-chat",
            deepseek_api_key="test-key",
            dspy_router_artifact=str(artifact),
        )
    )

    decision = router.route("What is a deadlift?")

    assert decision.route == "COACH"
    assert events == [
        ("configure", "deepseek/deepseek-chat", {"api_key": "test-key"}),
        ("load", str(artifact)),
        ("call", True, "What is a deadlift?"),
    ]
    assert router._load_program() is router._program


def test_dspy_router_skips_missing_artifact(monkeypatch, tmp_path):
    fake_dspy = ModuleType("dspy")
    fake_dspy.LM = lambda model: model
    fake_dspy.configure = lambda *, lm: None

    fake_program_module = ModuleType("fitness_router.router_program")

    class FakeProgram:
        def load(self, path):
            raise AssertionError("missing artifact should not load")

    fake_program_module.IntentRouterProgram = FakeProgram
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    monkeypatch.setitem(sys.modules, "fitness_router.router_program", fake_program_module)

    program = DSPyIntentRouter(
        Settings(dspy_model="fake-model", dspy_router_artifact=str(tmp_path / "missing.json"))
    )._load_program()

    assert isinstance(program, FakeProgram)


def test_langchain_router_uses_structured_output_and_runtime_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "router.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": LANGCHAIN_ROUTER_ARTIFACT_KIND,
                "instruction": "Optimized instruction",
                "few_shots": [
                    {
                        "user_input": "Bench press",
                        "route_decision": {
                            "route": "FALLBACK",
                            "confidence": 0.95,
                            "reason": "Ambiguous",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    events = []

    fake_prompts = ModuleType("langchain_core.prompts")

    class FakePrompt:
        @classmethod
        def from_messages(cls, messages):
            events.append(("prompt", messages))
            return cls()

        def __or__(self, structured):
            events.append(("pipe", structured))
            return FakeChain()

    class FakeChain:
        def invoke(self, payload):
            events.append(("invoke", payload["user_input"], bool(payload["route_catalog"])))
            return {
                "route": "COACH",
                "confidence": 0.91,
                "reason": "Question",
                "candidate_routes": [],
                "clarification_question": "",
            }

    fake_prompts.ChatPromptTemplate = FakePrompt
    fake_openai = ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            events.append(("llm", kwargs))

        def with_structured_output(self, schema, *, method):
            events.append(("structured", schema, method))
            return "structured-chain"

    fake_openai.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", fake_prompts)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai)

    router = LangChainStructuredIntentRouter(
        Settings(
            dspy_model="deepseek/deepseek-chat",
            deepseek_api_key="test-key",
            dspy_router_artifact=str(artifact),
        )
    )
    decision = router.route("What is a deadlift?")

    assert decision == RouteDecision(route="COACH", confidence=0.91, reason="Question")
    assert ("structured", RouteDecision, "function_calling") in events
    assert any(
        event[0] == "prompt" and "Optimized instruction" in event[1][0][1]
        for event in events
    )
    assert (
        "llm",
        {
            "model": "deepseek-chat",
            "temperature": 0,
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com",
        },
    ) in events


def test_langchain_router_caches_missing_artifact_and_normalizes_fallback(monkeypatch, tmp_path):
    events = []
    fake_prompts = ModuleType("langchain_core.prompts")

    class FakePrompt:
        @classmethod
        def from_messages(cls, messages):
            events.append(("prompt", messages))
            return cls()

        def __or__(self, structured):
            return FakeChain()

    class FakeChain:
        def invoke(self, payload):
            events.append(("invoke", payload["user_input"]))
            return RouteDecision(route="FALLBACK", confidence=0.6, reason="Ambiguous")

    fake_prompts.ChatPromptTemplate = FakePrompt
    fake_openai = ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, *, method):
            return object()

    fake_openai.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", fake_prompts)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai)

    router = LangChainStructuredIntentRouter(
        Settings(
            dspy_model="openai/test",
            dspy_router_artifact=str(tmp_path / "missing.json"),
        )
    )

    assert router._load_chain() is router._load_chain()
    decision = router.route("Bench press")

    assert decision.route == "FALLBACK"
    assert decision.clarification_question
    assert len([event for event in events if event[0] == "prompt"]) == 1

    class CoachChain:
        def invoke(self, payload):
            return RouteDecision(route="COACH", confidence=0.9, reason="Question")

    router._chain = CoachChain()
    assert router.route("Should I add rows?").clarification_question is None


def test_langchain_router_reports_missing_model_and_imports(monkeypatch):
    with pytest.raises(RuntimeError, match="DSPY_MODEL"):
        LangChainStructuredIntentRouter(Settings(dspy_model=None))._load_chain()

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_openai":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="LangChain OpenAI integration"):
        LangChainStructuredIntentRouter(Settings(dspy_model="openai/test"))._load_chain()


def test_router_artifact_converts_dspy_state_and_builds_prompt_messages():
    state = {
        "predict.predict": {
            "signature": {"instructions": "Optimized by DSPy."},
            "demos": [
                {
                    "user_input": "Bench press",
                    "route": "FALLBACK",
                    "confidence": 0.9,
                    "reason": "Ambiguous fragment",
                }
            ],
        }
    }

    artifact = runtime_artifact_from_dspy_state(state, example_count=1, train_count=1)
    loaded = load_router_prompt_artifact(json.dumps(state))
    messages = artifact_to_prompt_messages(artifact)

    assert artifact["kind"] == LANGCHAIN_ROUTER_ARTIFACT_KIND
    assert artifact["instruction"] == "Optimized by DSPy."
    assert artifact["few_shots"][0]["route_decision"]["route"] == "FALLBACK"
    assert loaded["instruction"] == "Optimized by DSPy."
    assert messages[0][0] == "system"
    assert "Route catalog" in messages[0][1]
    assert messages[-1] == ("human", "{user_input}")


def test_router_artifact_handles_malformed_inputs(tmp_path):
    state = {
        "bad": "value",
        "empty": {"signature": {}, "demos": "bad"},
        "examples": {
            "train": [
                "bad",
                {"route": "COACH"},
                {"kwargs": {"user_input": "Bad route", "route": "NOPE"}},
            ]
        },
    }
    runtime = {
        "kind": LANGCHAIN_ROUTER_ARTIFACT_KIND,
        "instruction": "",
        "routing_policy": "",
        "few_shots": [
            "bad",
            {"user_input": "", "route_decision": {}},
            {"user_input": "Bad decision", "route_decision": {"route": "NOPE"}},
        ],
    }
    runtime_with_bad_few_shots = {
        "kind": LANGCHAIN_ROUTER_ARTIFACT_KIND,
        "few_shots": {"bad": "value"},
    }
    invalid_path = tmp_path / "bad.json"
    invalid_path.write_text("{", encoding="utf-8")

    assert runtime_artifact_from_dspy_state("bad")["instruction"]
    assert runtime_artifact_from_dspy_state(state)["few_shots"] == []
    assert load_router_prompt_artifact(json.dumps(runtime))["few_shots"] == []
    assert load_router_prompt_artifact(json.dumps(runtime_with_bad_few_shots))["few_shots"] == []
    assert not _is_langchain_runtime_artifact(invalid_path)


class ToDictPrediction:
    def toDict(self):
        return {
            "route": "unknown",
            "confidence": "-5",
            "candidate_routes": "COACH, nope, WORKOUT_LOG",
        }


class ObjectPrediction:
    def __init__(self):
        self.route = "fallback"
        self.confidence = "not-a-number"
        self.reason = ""
        self.candidate_routes = ["FALLBACK", "bad"]
        self.clarification_question = ""


def test_coerce_prediction_accepts_todict_objects_and_clamps_values():
    decision = coerce_prediction_to_route_decision(ToDictPrediction(), "Bench press")

    assert decision.route == "FALLBACK"
    assert decision.confidence == 0.0
    assert decision.reason == "DSPy router returned a structured route decision."
    assert [candidate.route for candidate in decision.candidate_routes] == [
        "COACH",
        "WORKOUT_LOG",
    ]
    assert decision.clarification_question


def test_coerce_prediction_accepts_dicts_and_attribute_objects():
    dict_decision = coerce_prediction_to_route_decision(
        {
            "route": "workout_generate",
            "confidence": "1.7",
            "reason": "Generate",
            "candidate_routes": [
                {"route": "WORKOUT_GENERATE", "score": 0.9},
                {"route": "NOPE", "score": 1.0},
            ],
            "clarification_question": "ignored",
        },
        "Build a workout",
    )
    object_decision = coerce_prediction_to_route_decision(ObjectPrediction(), "Bench")
    empty_decision = coerce_prediction_to_route_decision(object(), "Bench")

    assert dict_decision.route == "WORKOUT_GENERATE"
    assert dict_decision.confidence == 1.0
    assert dict_decision.candidate_routes == [
        RouteCandidate(route="WORKOUT_GENERATE", score=0.9)
    ]
    assert object_decision.route == "FALLBACK"
    assert object_decision.confidence == 0.0
    assert object_decision.clarification_question
    assert empty_decision.route == "FALLBACK"


def test_make_intent_router_selects_configured_backend_or_errors():
    assert isinstance(
        make_intent_router(Settings(dspy_model="fake")),
        LangChainStructuredIntentRouter,
    )
    assert isinstance(
        make_intent_router(Settings(dspy_model=None, allow_demo_router_without_llm=True)),
        DemoIntentRouter,
    )

    with pytest.raises(RuntimeError, match="No intent router configured"):
        make_intent_router(Settings(dspy_model=None, allow_demo_router_without_llm=False))


def test_settings_passes_deepseek_api_key_only_to_deepseek_models():
    deepseek = Settings(
        dspy_model="deepseek/deepseek-chat",
        deepseek_api_key="key",
    )

    assert deepseek.dspy_lm_kwargs() == {"api_key": "key"}
    assert deepseek.langchain_model_name() == "deepseek-chat"
    assert deepseek.langchain_chat_kwargs() == {
        "api_key": "key",
        "base_url": "https://api.deepseek.com",
    }
    openai = Settings(
        dspy_model="openai/gpt-4o-mini",
        deepseek_api_key="key",
    )
    assert openai.dspy_lm_kwargs() == {}
    assert openai.langchain_model_name() == "openai/gpt-4o-mini"
    assert openai.langchain_chat_kwargs() == {}
    assert (
        Settings(dspy_model="deepseek/deepseek-chat", deepseek_api_key=None).dspy_lm_kwargs()
        == {}
    )
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        Settings(dspy_model="deepseek/deepseek-chat", deepseek_api_key="").langchain_chat_kwargs()
    with pytest.raises(RuntimeError, match="DSPY_MODEL"):
        Settings(dspy_model=None).langchain_model_name()


def test_settings_resolves_relative_paths_from_repo_root():
    settings = Settings(
        exercise_data_path="data/exercises.json",
        dspy_router_artifact="apps/api/artifacts/router_optimized.json",
    )

    assert settings.exercise_data_file() == ROOT_DIR / "data/exercises.json"
    assert settings.dspy_router_artifact_file() == (
        ROOT_DIR / "apps/api/artifacts/router_optimized.json"
    )


def test_route_decision_from_json_and_private_parse_helpers(tmp_path):
    path = tmp_path / "decision.json"
    path.write_text(
        json.dumps({"route": "COACH", "confidence": 0.8, "reason": "Question"}),
        encoding="utf-8",
    )

    parsed = _parse_candidate_routes(
        [{"route": "COACH", "score": 0.5}, {"route": "BAD", "score": 0.5}, "FALLBACK", 123]
    )

    assert route_decision_from_json(path).route == "COACH"
    assert _float_or_default(None, 0.4) == 0.4
    assert [candidate.route for candidate in parsed] == ["COACH", "FALLBACK"]
    assert _parse_candidate_routes({"route": "COACH"}) == []
