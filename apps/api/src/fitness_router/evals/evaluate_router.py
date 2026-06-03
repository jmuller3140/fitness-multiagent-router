from __future__ import annotations

import argparse
import json

from fitness_router.evals.dataset import load_router_eval_examples, selected_route_for_eval
from fitness_router.routing import (
    DemoIntentRouter,
    DSPyIntentRouter,
    IntentRouter,
    LangChainStructuredIntentRouter,
    make_intent_router,
)
from fitness_router.settings import get_settings


def evaluate(router: IntentRouter) -> dict:
    examples = load_router_eval_examples()
    rows = []
    correct = 0
    fallback_correct = 0
    fallback_total = 0

    for example in examples:
        decision = router.route(example.user_input)
        selected_route = selected_route_for_eval(decision)
        is_correct = selected_route == example.expected_route
        correct += int(is_correct)

        if example.should_fallback:
            fallback_total += 1
            fallback_correct += int(selected_route == "FALLBACK")

        rows.append(
            {
                "user_input": example.user_input,
                "expected_route": example.expected_route,
                "selected_route": selected_route,
                "raw_route": decision.route,
                "confidence": decision.confidence,
                "correct": is_correct,
                "reason_category": example.reason_category,
            }
        )

    total = len(examples)
    return {
        "accuracy": correct / total if total else 0,
        "fallback_accuracy": fallback_correct / fallback_total if fallback_total else 0,
        "example_count": total,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the intent router on labeled examples.")
    parser.add_argument(
        "--backend",
        choices=["auto", "langchain", "dspy", "demo"],
        default="auto",
        help=(
            "Router backend. auto uses the configured production router, "
            "otherwise the offline demo router."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.backend == "auto":
        router = make_intent_router(settings)
    elif args.backend == "langchain":
        router = LangChainStructuredIntentRouter(settings)
    elif args.backend == "dspy":
        router: IntentRouter = DSPyIntentRouter(settings)
    else:
        router = DemoIntentRouter()

    print(json.dumps(evaluate(router), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
