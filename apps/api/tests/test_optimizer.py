import json
from types import ModuleType, SimpleNamespace

import pytest

from fitness_router.evals import optimize_router
from fitness_router.models import RouteDecision
from fitness_router.settings import Settings


class FakeExample:
    def __init__(self, user_input, expected_route, should_fallback=False, reason_category="test"):
        self.user_input = user_input
        self.expected_route = expected_route
        self.should_fallback = should_fallback
        self.reason_category = reason_category


def test_route_metric_rewards_selected_route_and_fallback_precision(monkeypatch):
    monkeypatch.setattr(
        optimize_router,
        "validate_router_prediction",
        lambda pred, user_input: pred,
    )

    correct = optimize_router.route_metric(
        FakeExample("coach", "COACH"),
        RouteDecision(route="COACH", confidence=0.9, reason="ok"),
    )
    low_confidence = optimize_router.route_metric(
        FakeExample("coach", "COACH"),
        RouteDecision(route="COACH", confidence=0.2, reason="low"),
    )
    fallback = optimize_router.route_metric(
        FakeExample("vague", "FALLBACK", should_fallback=True),
        RouteDecision(route="WORKOUT_LOG", confidence=0.2, reason="low"),
    )
    wrong_fallback = optimize_router.route_metric(
        FakeExample("vague", "FALLBACK", should_fallback=True),
        RouteDecision(route="WORKOUT_LOG", confidence=0.95, reason="wrong"),
    )

    assert correct == 1.0
    assert low_confidence == 0.2
    assert fallback == 1.0
    assert wrong_fallback == 0.0


def test_optimizer_lazy_router_program_helpers(monkeypatch):
    fake_program_module = ModuleType("fitness_router.router_program")

    class FakeProgram:
        pass

    fake_program_module.IntentRouterProgram = FakeProgram
    fake_program_module.validate_prediction = lambda pred, user_input: (
        pred,
        user_input,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "fitness_router.router_program",
        fake_program_module,
    )

    assert isinstance(optimize_router.make_router_program(), FakeProgram)
    assert optimize_router.validate_router_prediction("pred", "input") == ("pred", "input")


def test_split_golden_examples_is_deterministic_stratified_and_validated():
    examples = [
        FakeExample(f"coach-{index}", "COACH") for index in range(4)
    ] + [
        FakeExample(f"log-{index}", "WORKOUT_LOG") for index in range(4)
    ]

    train_a, val_a = optimize_router.split_golden_examples(examples, seed=7)
    train_b, val_b = optimize_router.split_golden_examples(examples, seed=7)

    assert [example.user_input for example in train_a] == [
        example.user_input for example in train_b
    ]
    assert [example.user_input for example in val_a] == [example.user_input for example in val_b]
    assert {example.expected_route for example in val_a} == {"COACH", "WORKOUT_LOG"}
    assert len(train_a) == 6
    assert len(val_a) == 2

    with pytest.raises(ValueError, match="val_fraction"):
        optimize_router.split_golden_examples(examples, val_fraction=1.0)
    with pytest.raises(ValueError, match="At least two"):
        optimize_router.split_golden_examples([examples[0]])


def test_split_golden_examples_handles_singleton_group_with_fallback_val():
    trainset, valset = optimize_router.split_golden_examples(
        [FakeExample("coach", "COACH"), FakeExample("log", "WORKOUT_LOG")]
    )

    assert len(trainset) == 1
    assert len(valset) == 1


def test_optimizer_main_requires_model(monkeypatch):
    monkeypatch.setattr(optimize_router, "get_settings", lambda: Settings(dspy_model=None))
    monkeypatch.setitem(__import__("sys").modules, "dspy", ModuleType("dspy"))
    monkeypatch.setattr("sys.argv", ["optimize_router"])

    with pytest.raises(SystemExit, match="Set DSPY_MODEL"):
        optimize_router.main()


def test_optimizer_main_trains_on_golden_split_and_saves(monkeypatch, tmp_path, capsys):
    events = []
    output = tmp_path / "router.json"
    goldens = tmp_path / "goldens.jsonl"
    goldens.write_text("{}", encoding="utf-8")

    fake_dspy = ModuleType("dspy")

    class FakeLM:
        def __init__(self, model):
            self.model = model

    class FakeOptimizer:
        def __init__(self, *, metric, auto, num_threads):
            events.append(("optimizer", metric is optimize_router.route_metric, auto, num_threads))

        def compile(self, program, *, trainset, valset):
            events.append(("compile", type(program).__name__, len(trainset), len(valset)))
            return SimpleNamespace(
                save=lambda path: (
                    events.append(("save", path)),
                    output.write_text(
                        json.dumps(
                            {
                                "predict.predict": {
                                    "signature": {"instructions": "Optimized instruction"}
                                }
                            }
                        ),
                        encoding="utf-8",
                    ),
                )
            )

    fake_dspy.LM = FakeLM
    fake_dspy.MIPROv2 = FakeOptimizer
    fake_dspy.configure = lambda *, lm: events.append(("configure", lm.model))

    class FakeProgram:
        pass

    examples = [
        FakeExample("coach-1", "COACH"),
        FakeExample("coach-2", "COACH"),
        FakeExample("log-1", "WORKOUT_LOG"),
        FakeExample("log-2", "WORKOUT_LOG"),
    ]

    monkeypatch.setitem(__import__("sys").modules, "dspy", fake_dspy)
    monkeypatch.setattr(
        optimize_router,
        "get_settings",
        lambda: Settings(
            dspy_model="openai/test",
            dspy_router_artifact=str(tmp_path / "default.json"),
        ),
    )
    monkeypatch.setattr(
        optimize_router,
        "load_router_eval_examples",
        lambda path=None: events.append(("load", path)) or examples,
    )
    monkeypatch.setattr(optimize_router, "as_dspy_examples", lambda raw: raw)
    monkeypatch.setattr(optimize_router, "make_router_program", FakeProgram)
    monkeypatch.setattr(
        "sys.argv",
        [
            "optimize_router",
            "--goldens",
            str(goldens),
            "--output",
            str(output),
            "--auto",
            "medium",
            "--num-threads",
            "2",
            "--val-fraction",
            "0.5",
            "--seed",
            "99",
        ],
    )

    optimize_router.main()

    assert ("configure", "openai/test") in events
    assert ("load", str(goldens)) in events
    assert ("optimizer", True, "medium", 2) in events
    assert ("compile", "FakeProgram", 2, 2) in events
    assert ("save", str(output)) in events
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["kind"] == "langchain_structured_router_prompt"
    assert saved["instruction"] == "Optimized instruction"
    assert saved["metrics"] == {
        "example_count": 4,
        "train_count": 2,
        "validation_count": 2,
    }
    assert "4 golden examples (2 train, 2 validation)" in capsys.readouterr().out


def test_optimizer_main_uses_default_goldens_and_output(monkeypatch, tmp_path):
    events = []
    fake_dspy = ModuleType("dspy")
    fake_dspy.LM = lambda model: model
    fake_dspy.configure = lambda *, lm: None
    fake_dspy.MIPROv2 = lambda **kwargs: SimpleNamespace(
        compile=lambda program, *, trainset, valset: SimpleNamespace(
            save=lambda path: events.append(("save", path))
        )
    )

    monkeypatch.setitem(__import__("sys").modules, "dspy", fake_dspy)
    monkeypatch.setattr(
        optimize_router,
        "get_settings",
        lambda: Settings(
            dspy_model="openai/test",
            dspy_router_artifact=str(tmp_path / "default.json"),
        ),
    )
    monkeypatch.setattr(
        optimize_router,
        "load_router_eval_examples",
        lambda path=None: [FakeExample("one", "COACH"), FakeExample("two", "COACH")],
    )
    monkeypatch.setattr(optimize_router, "as_dspy_examples", lambda raw: raw)
    monkeypatch.setattr(optimize_router, "make_router_program", lambda: object())
    monkeypatch.setattr("sys.argv", ["optimize_router"])

    optimize_router.main()

    assert events == [("save", str(tmp_path / "default.json"))]
    saved = json.loads((tmp_path / "default.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "langchain_structured_router_prompt"
