from collections import Counter
from types import ModuleType

from fitness_router.evals.dataset import (
    as_dspy_examples,
    load_router_eval_examples,
    selected_route_for_eval,
)
from fitness_router.evals.evaluate_router import evaluate
from fitness_router.evals.evaluate_router import main as evaluate_main
from fitness_router.models import RouteDecision, RouterEvalExample


def test_router_eval_dataset_has_many_examples_and_full_route_coverage():
    examples = load_router_eval_examples()
    counts = Counter(example.expected_route for example in examples)

    assert len(examples) >= 40
    assert counts["COACH"] >= 8
    assert counts["WORKOUT_GENERATE"] >= 8
    assert counts["WORKOUT_LOG"] >= 8
    assert counts["FALLBACK"] >= 8


def test_router_eval_dataset_helpers_load_custom_file_and_convert_to_dspy(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        (
            '{"user_input":"What is a squat?","expected_route":"COACH",'
            '"should_fallback":false,"reason_category":"question"}\n\n'
        ),
        encoding="utf-8",
    )
    fake_dspy = ModuleType("dspy")

    class FakeExample:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def with_inputs(self, *inputs):
            self.inputs = inputs
            return self

    fake_dspy.Example = FakeExample
    monkeypatch.setitem(__import__("sys").modules, "dspy", fake_dspy)

    examples = load_router_eval_examples(path)
    dspy_examples = as_dspy_examples(examples)

    assert len(examples) == 1
    assert selected_route_for_eval(
        RouteDecision(route="WORKOUT_GENERATE", confidence=0.2, reason="low confidence")
    ) == "FALLBACK"
    assert selected_route_for_eval(
        RouteDecision(route="WORKOUT_LOG", confidence=0.9, reason="high confidence")
    ) == "WORKOUT_LOG"
    assert dspy_examples[0].user_input == "What is a squat?"


class FakeRouter:
    def __init__(self, decisions):
        self.decisions = decisions

    def route(self, user_input):
        return self.decisions[user_input]


def test_evaluate_router_scores_routes_and_empty_dataset(monkeypatch):
    examples = [
        RouterEvalExample(
            user_input="coach",
            expected_route="COACH",
            should_fallback=False,
            reason_category="question",
        ),
        RouterEvalExample(
            user_input="vague",
            expected_route="FALLBACK",
            should_fallback=True,
            reason_category="ambiguous",
        ),
    ]
    router = FakeRouter(
        {
            "coach": RouteDecision(route="COACH", confidence=0.95, reason="ok"),
            "vague": RouteDecision(route="WORKOUT_LOG", confidence=0.3, reason="low"),
        }
    )

    monkeypatch.setattr(
        "fitness_router.evals.evaluate_router.load_router_eval_examples",
        lambda: examples,
    )
    scored = evaluate(router)
    monkeypatch.setattr(
        "fitness_router.evals.evaluate_router.load_router_eval_examples",
        lambda: [],
    )
    empty = evaluate(router)

    assert scored["accuracy"] == 1.0
    assert scored["fallback_accuracy"] == 1.0
    assert scored["example_count"] == 2
    assert empty["accuracy"] == 0
    assert empty["fallback_accuracy"] == 0


def test_evaluate_router_main_selects_demo_langchain_dspy_and_auto_backends(monkeypatch, capsys):
    created = []

    class FakeDemo:
        pass

    class FakeLangChain:
        def __init__(self, settings):
            created.append(("langchain", settings))

    class FakeDSPy:
        def __init__(self, settings):
            created.append(("dspy", settings))

    class FakeAuto:
        pass

    monkeypatch.setattr("fitness_router.evals.evaluate_router.DemoIntentRouter", FakeDemo)
    monkeypatch.setattr(
        "fitness_router.evals.evaluate_router.LangChainStructuredIntentRouter",
        FakeLangChain,
    )
    monkeypatch.setattr("fitness_router.evals.evaluate_router.DSPyIntentRouter", FakeDSPy)
    monkeypatch.setattr(
        "fitness_router.evals.evaluate_router.make_intent_router",
        lambda settings: FakeAuto(),
    )
    monkeypatch.setattr(
        "fitness_router.evals.evaluate_router.evaluate",
        lambda router: {"router": type(router).__name__},
    )

    monkeypatch.setattr("sys.argv", ["evaluate_router", "--backend", "demo"])
    evaluate_main()
    assert '"router": "FakeDemo"' in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["evaluate_router", "--backend", "langchain"])
    evaluate_main()
    assert '"router": "FakeLangChain"' in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["evaluate_router", "--backend", "dspy"])
    evaluate_main()
    assert '"router": "FakeDSPy"' in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["evaluate_router"])
    evaluate_main()
    assert '"router": "FakeAuto"' in capsys.readouterr().out
    assert created
