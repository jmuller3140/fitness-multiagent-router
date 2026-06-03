from __future__ import annotations

import argparse
import json

from fitness_router.graphs import build_hub_graph
from fitness_router.models import RouteDecision


def run(message: str) -> dict:
    graph = build_hub_graph()
    state = graph.invoke({"user_input": message, "errors": []})
    decision = state["route_decision"]
    if not isinstance(decision, RouteDecision):
        decision = RouteDecision.model_validate(decision)

    return {
        "input": message,
        "selected_route": state.get("selected_route", decision.route),
        "route_decision": decision.model_dump(mode="json"),
        "final_response": state.get("final_response"),
        "structured_output": state.get("structured_output"),
        "errors": state.get("errors", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fitness multi-agent graph once.")
    parser.add_argument("message", nargs="+", help="Natural-language fitness request.")
    args = parser.parse_args()
    print(json.dumps(run(" ".join(args.message)), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
