from fitness_router.data import load_exercises
from fitness_router.evals.dataset import load_workout_generator_eval_examples
from fitness_router.evals.evaluate_workout_generator import (
    classify_outcome,
    evaluate,
    workout_uses_valid_exercise_ids,
)
from fitness_router.evals.evaluate_workout_generator import (
    main as evaluate_generator_main,
)
from fitness_router.models import WorkoutGeneratorEvalExample


class FakeGeneratorAgent:
    def __init__(self, states):
        self.states = states

    def generate(self, state):
        return self.states[state["user_input"]]


def test_workout_generator_eval_dataset_loads_packaged_and_custom_examples(tmp_path):
    packaged = load_workout_generator_eval_examples()
    path = tmp_path / "generator.jsonl"
    path.write_text(
        (
            '{"user_input":"Build a test workout","expected_outcome":"WORKOUT_BUILT",'
            '"required_tool_names":["search_exercises","build_workout"],'
            '"reason_category":"test"}\n\n'
        ),
        encoding="utf-8",
    )

    custom = load_workout_generator_eval_examples(path)

    assert len(packaged) >= 5
    assert {example.expected_outcome for example in packaged} >= {"WORKOUT_BUILT", "NO_RESULTS"}
    assert custom[0].required_tool_names == ["search_exercises", "build_workout"]


def test_evaluate_workout_generator_scores_outcomes_tools_and_empty_dataset():
    valid_exercise_id = next(iter(load_exercises())).id
    examples = [
        WorkoutGeneratorEvalExample(
            user_input="built",
            expected_outcome="WORKOUT_BUILT",
            required_tool_names=["search_exercises", "build_workout"],
            reason_category="success",
        ),
        WorkoutGeneratorEvalExample(
            user_input="empty",
            expected_outcome="NO_RESULTS",
            required_tool_names=["search_exercises"],
            reason_category="no_results",
        ),
        WorkoutGeneratorEvalExample(
            user_input="bad",
            expected_outcome="TOOL_ERROR",
            required_tool_names=["build_workout"],
            reason_category="invalid_tool",
        ),
    ]
    agent = FakeGeneratorAgent(
        {
            "built": {
                "structured_output": {
                    "workout": {
                        "sections": [
                            {
                                "exercises": [
                                    {"exercise_id": valid_exercise_id}
                                ]
                            }
                        ]
                    },
                    "tool_calls": [
                        {"name": "search_exercises"},
                        {"name": "build_workout"},
                    ],
                }
            },
            "empty": {
                "structured_output": {
                    "search": {"matches": []},
                    "workout": None,
                    "tool_calls": [{"name": "search_exercises"}],
                }
            },
            "bad": {
                "structured_output": {
                    "workout": None,
                    "tool_error": "bad",
                    "tool_calls": [{"name": "build_workout"}],
                }
            },
        }
    )

    scored = evaluate(agent, examples)
    empty = evaluate(agent, [])

    assert scored["accuracy"] == 1.0
    assert scored["required_tool_accuracy"] == 1.0
    assert scored["valid_workout_accuracy"] == 1.0
    assert empty["accuracy"] == 0
    assert empty["required_tool_accuracy"] == 0
    assert empty["valid_workout_accuracy"] == 0


def test_workout_generator_eval_helpers_cover_invalid_workout_shapes():
    valid_exercise_id = next(iter(load_exercises())).id
    assert classify_outcome({"workout": {"sections": []}}) == "WORKOUT_BUILT"
    assert classify_outcome({"search": {"matches": []}, "workout": None}) == "NO_RESULTS"
    assert classify_outcome({"tool_error": "bad", "workout": None}) == "TOOL_ERROR"

    assert not workout_uses_valid_exercise_ids(None)
    assert not workout_uses_valid_exercise_ids({"sections": "bad"})
    assert not workout_uses_valid_exercise_ids({"sections": [{"exercises": []}]})
    assert not workout_uses_valid_exercise_ids(
        {"sections": [{"exercises": [{"exercise_id": "missing"}]}]}
    )
    assert workout_uses_valid_exercise_ids(
        {"sections": [{"exercises": [{"exercise_id": valid_exercise_id}]}]}
    )


def test_evaluate_workout_generator_main_selects_backends(monkeypatch, capsys):
    created = []

    class FakeAuto:
        pass

    class FakeLangChain:
        def __init__(self, settings):
            created.append(("langchain", settings))

    class FakeDemo:
        pass

    monkeypatch.setattr(
        "fitness_router.evals.evaluate_workout_generator.make_workout_generator_agent",
        lambda settings: FakeAuto(),
    )
    monkeypatch.setattr(
        "fitness_router.evals.evaluate_workout_generator.LangChainWorkoutGeneratorAgent",
        FakeLangChain,
    )
    monkeypatch.setattr(
        "fitness_router.evals.evaluate_workout_generator.DeterministicWorkoutGeneratorAgent",
        FakeDemo,
    )
    monkeypatch.setattr(
        "fitness_router.evals.evaluate_workout_generator.evaluate",
        lambda agent: {"agent": type(agent).__name__},
    )

    monkeypatch.setattr("sys.argv", ["evaluate_workout_generator"])
    evaluate_generator_main()
    assert '"agent": "FakeAuto"' in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["evaluate_workout_generator", "--backend", "langchain"])
    evaluate_generator_main()
    assert '"agent": "FakeLangChain"' in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["evaluate_workout_generator", "--backend", "demo"])
    evaluate_generator_main()
    assert '"agent": "FakeDemo"' in capsys.readouterr().out
    assert created
