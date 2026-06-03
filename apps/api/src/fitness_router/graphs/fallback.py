from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from fitness_router.models import HubState
from fitness_router.route_registry import clarification_question


def fallback_node(state: HubState) -> HubState:
    decision = state.get("route_decision")
    question = decision.clarification_question if decision else None
    if not question:
        question = clarification_question(state["user_input"])

    return {
        "final_response": question,
        "structured_output": {
            "fallback": True,
            "original_route": decision.route if decision else None,
            "reason": decision.reason if decision else "No route decision was available.",
        },
    }


def build_fallback_graph():
    graph = StateGraph(HubState)
    graph.add_node("fallback", fallback_node)
    graph.add_edge(START, "fallback")
    graph.add_edge("fallback", END)
    return graph.compile()
