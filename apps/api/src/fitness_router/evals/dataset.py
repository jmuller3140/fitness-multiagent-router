from __future__ import annotations

import json
from pathlib import Path

from fitness_router.models import RouteDecision, RouterEvalExample, WorkoutGeneratorEvalExample
from fitness_router.paths import DEFAULT_ROUTER_EVALS_PATH, DEFAULT_WORKOUT_GENERATOR_EVALS_PATH
from fitness_router.route_registry import build_route_catalog


def load_router_eval_examples(
    path: str | Path = DEFAULT_ROUTER_EVALS_PATH,
) -> list[RouterEvalExample]:
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(RouterEvalExample.model_validate(json.loads(line)))
    return examples


def load_workout_generator_eval_examples(
    path: str | Path = DEFAULT_WORKOUT_GENERATOR_EVALS_PATH,
) -> list[WorkoutGeneratorEvalExample]:
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(WorkoutGeneratorEvalExample.model_validate(json.loads(line)))
    return examples


def selected_route_for_eval(decision: RouteDecision, threshold: float = 0.8) -> str:
    if decision.route == "FALLBACK" or decision.confidence < threshold:
        return "FALLBACK"
    return decision.route


def as_dspy_examples(examples: list[RouterEvalExample]):
    import dspy

    catalog = build_route_catalog()
    return [
        dspy.Example(
            route_catalog=catalog,
            user_input=example.user_input,
            expected_route=example.expected_route,
            should_fallback=example.should_fallback,
            reason_category=example.reason_category,
        ).with_inputs("route_catalog", "user_input")
        for example in examples
    ]
