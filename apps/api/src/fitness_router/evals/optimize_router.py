from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from fitness_router.evals.dataset import (
    as_dspy_examples,
    load_router_eval_examples,
    selected_route_for_eval,
)
from fitness_router.router_artifact import runtime_artifact_from_dspy_state
from fitness_router.settings import get_settings


def validate_router_prediction(pred: object, user_input: str):
    from fitness_router.router_program import validate_prediction

    return validate_prediction(pred, user_input)


def make_router_program():
    from fitness_router.router_program import IntentRouterProgram

    return IntentRouterProgram()


def route_metric(example, pred, trace=None) -> float:
    decision = validate_router_prediction(pred, example.user_input)
    selected_route = selected_route_for_eval(decision)
    route_score = 1.0 if selected_route == example.expected_route else 0.0
    fallback_score = 1.0
    if bool(example.should_fallback):
        fallback_score = 1.0 if selected_route == "FALLBACK" else 0.0
    return 0.8 * route_score + 0.2 * fallback_score


def split_golden_examples(
    examples: list,
    *,
    val_fraction: float = 0.25,
    seed: int = 13,
) -> tuple[list, list]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be greater than 0 and less than 1.")
    if len(examples) < 2:
        raise ValueError("At least two golden examples are required for optimization.")

    rng = random.Random(seed)
    grouped: dict[str, list] = defaultdict(list)
    for example in examples:
        grouped[str(example.expected_route)].append(example)

    trainset = []
    valset = []
    for route in sorted(grouped):
        group = grouped[route][:]
        rng.shuffle(group)
        val_count = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val_count = min(val_count, len(group) - 1)
        valset.extend(group[:val_count])
        trainset.extend(group[val_count:])

    if not valset:
        valset.append(trainset.pop())

    rng.shuffle(trainset)
    rng.shuffle(valset)
    return trainset, valset


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the DSPy intent router.")
    parser.add_argument(
        "--goldens",
        default=None,
        help="Path to JSONL golden router eval examples. Defaults to the packaged eval set.",
    )
    parser.add_argument("--output", default=None, help="Path to save the optimized DSPy program.")
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import dspy

    settings = get_settings()
    if not settings.dspy_model:
        raise SystemExit("Set DSPY_MODEL before optimizing the router.")

    dspy.configure(lm=dspy.LM(settings.dspy_model, **settings.dspy_lm_kwargs()))
    if args.goldens:
        raw_examples = load_router_eval_examples(args.goldens)
    else:
        raw_examples = load_router_eval_examples()
    examples = as_dspy_examples(raw_examples)
    trainset, valset = split_golden_examples(
        examples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        "Optimizing router with "
        f"{len(examples)} golden examples "
        f"({len(trainset)} train, {len(valset)} validation)."
    )

    program = make_router_program()
    optimizer = dspy.MIPROv2(metric=route_metric, auto=args.auto, num_threads=args.num_threads)
    compiled = optimizer.compile(program, trainset=trainset, valset=valset)

    output = Path(args.output) if args.output else settings.dspy_router_artifact_file()
    output.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(output))
    dspy_state = _load_json_object(output)
    artifact = runtime_artifact_from_dspy_state(
        dspy_state,
        example_count=len(examples),
        train_count=len(trainset),
        validation_count=len(valset),
    )
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Saved optimized router to {output}")


def _load_json_object(path: Path) -> dict:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return values if isinstance(values, dict) else {}


if __name__ == "__main__":  # pragma: no cover
    main()
