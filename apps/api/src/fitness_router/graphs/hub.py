from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from fitness_router.graphs.coach import build_coach_graph
from fitness_router.graphs.fallback import build_fallback_graph
from fitness_router.graphs.workout_generator import build_workout_generator_graph
from fitness_router.graphs.workout_logger import build_workout_logger_graph
from fitness_router.models import HubState, RouteDecision, RouteName
from fitness_router.route_registry import clarification_question
from fitness_router.routing import IntentRouter, make_intent_router
from fitness_router.settings import Settings, get_settings


def build_hub_graph(router: IntentRouter | None = None, settings: Settings | None = None):
    resolved_settings = settings or get_settings()
    resolved_router = router or make_intent_router(resolved_settings)

    graph = StateGraph(HubState)
    graph.add_node("route_intent", _route_intent_node(resolved_router))
    graph.add_node("normalize_route", _normalize_route_node(resolved_settings))
    graph.add_node("coach_graph", build_coach_graph())
    graph.add_node(
        "workout_generator_graph",
        build_workout_generator_graph(settings=resolved_settings),
    )
    graph.add_node("workout_logger_graph", build_workout_logger_graph())
    graph.add_node("fallback_graph", build_fallback_graph())

    graph.add_edge(START, "route_intent")
    graph.add_edge("route_intent", "normalize_route")
    graph.add_conditional_edges(
        "normalize_route",
        _select_route,
        {
            "COACH": "coach_graph",
            "WORKOUT_GENERATE": "workout_generator_graph",
            "WORKOUT_LOG": "workout_logger_graph",
            "FALLBACK": "fallback_graph",
        },
    )
    graph.add_edge("coach_graph", END)
    graph.add_edge("workout_generator_graph", END)
    graph.add_edge("workout_logger_graph", END)
    graph.add_edge("fallback_graph", END)
    return graph.compile()


def _route_intent_node(router: IntentRouter):
    def node(state: HubState) -> HubState:
        decision = router.route(state["user_input"])
        return {"route_decision": decision, "errors": state.get("errors", [])}

    return node


def _normalize_route_node(settings: Settings):
    def node(state: HubState) -> HubState:
        decision = state.get("route_decision")
        if decision is None:
            decision = RouteDecision(
                route="FALLBACK",
                confidence=0.0,
                reason="Router did not return a decision.",
                clarification_question=clarification_question(state["user_input"]),
            )

        selected_route: RouteName = decision.route
        if (
            decision.route == "FALLBACK"
            or decision.confidence < settings.router_confidence_threshold
            or _has_close_candidate_margin(decision)
        ):
            selected_route = "FALLBACK"
            if not decision.clarification_question:
                decision = decision.model_copy(
                    update={"clarification_question": clarification_question(state["user_input"])}
                )

        return {"route_decision": decision, "selected_route": selected_route}

    return node


def _select_route(state: HubState) -> RouteName:
    return state.get("selected_route", "FALLBACK")


def _has_close_candidate_margin(decision: RouteDecision) -> bool:
    candidates = [candidate for candidate in decision.candidate_routes if candidate.score > 0]
    if len(candidates) < 2:
        return False
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    return ranked[0].route != "FALLBACK" and ranked[0].score - ranked[1].score < 0.12
