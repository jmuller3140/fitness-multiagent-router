from __future__ import annotations

import argparse
import json
from typing import Any

from fitness_router.data import exercises_by_id
from fitness_router.evals.dataset import load_workout_generator_eval_examples
from fitness_router.graphs.workout_generator import (
    DeterministicWorkoutGeneratorAgent,
    LangChainWorkoutGeneratorAgent,
    WorkoutGeneratorAgent,
    make_workout_generator_agent,
)
from fitness_router.models import WorkoutGeneratorEvalExample, WorkoutGeneratorOutcome
from fitness_router.settings import get_settings


def evaluate(
    agent: WorkoutGeneratorAgent,
    examples: list[WorkoutGeneratorEvalExample] | None = None,
) -> dict[str, Any]:
    rows = []
    correct = 0
    required_tools_correct = 0
    valid_workouts = 0
    built_workouts = 0
    eval_examples = examples if examples is not None else load_workout_generator_eval_examples()

    for example in eval_examples:
        state = agent.generate({"user_input": example.user_input, "errors": []})
        structured = state.get("structured_output") or {}
        outcome = classify_outcome(structured)
        tool_names = [call.get("name") for call in structured.get("tool_calls", [])]
        has_required_tools = all(name in tool_names for name in example.required_tool_names)
        required_tools_correct += int(has_required_tools)

        workout_is_valid = True
        if outcome == "WORKOUT_BUILT":
            built_workouts += 1
            workout_is_valid = workout_uses_valid_exercise_ids(structured.get("workout"))
            valid_workouts += int(workout_is_valid)

        is_correct = (
            outcome == example.expected_outcome
            and has_required_tools
            and workout_is_valid
        )
        correct += int(is_correct)
        rows.append(
            {
                "user_input": example.user_input,
                "expected_outcome": example.expected_outcome,
                "outcome": outcome,
                "tool_names": tool_names,
                "tool_calls": structured.get("tool_calls", []),
                "required_tools_present": has_required_tools,
                "workout_uses_valid_exercise_ids": workout_is_valid,
                "correct": is_correct,
                "reason_category": example.reason_category,
            }
        )

    total = len(eval_examples)
    return {
        "accuracy": correct / total if total else 0,
        "required_tool_accuracy": required_tools_correct / total if total else 0,
        "valid_workout_accuracy": valid_workouts / built_workouts if built_workouts else 0,
        "example_count": total,
        "results": rows,
    }


def classify_outcome(structured: dict[str, Any]) -> WorkoutGeneratorOutcome:
    if structured.get("workout") is not None:
        return "WORKOUT_BUILT"
    search = structured.get("search")
    if isinstance(search, dict) and search.get("matches") == []:
        return "NO_RESULTS"
    return "TOOL_ERROR"


def workout_uses_valid_exercise_ids(workout: object) -> bool:
    if not isinstance(workout, dict):
        return False
    valid_ids = exercises_by_id()
    sections = workout.get("sections")
    if not isinstance(sections, list):
        return False
    exercise_ids = [
        exercise.get("exercise_id")
        for section in sections
        if isinstance(section, dict)
        for exercise in section.get("exercises", [])
        if isinstance(exercise, dict)
    ]
    return bool(exercise_ids) and all(exercise_id in valid_ids for exercise_id in exercise_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the workout generator on labeled tool-use examples."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "langchain", "demo"],
        default="auto",
        help="Generator backend. auto uses configured runtime, demo uses offline tools.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.backend == "auto":
        agent = make_workout_generator_agent(settings)
    elif args.backend == "langchain":
        agent = LangChainWorkoutGeneratorAgent(settings)
    else:
        agent = DeterministicWorkoutGeneratorAgent()

    print(json.dumps(evaluate(agent), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
