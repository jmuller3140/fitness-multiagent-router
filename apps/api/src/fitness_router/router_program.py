from __future__ import annotations

from fitness_router.models import RouteDecision

try:
    import dspy
except ImportError as exc:  # pragma: no cover - surfaced only when DSPy backend is requested.
    raise RuntimeError("DSPy is required for router_program. Run uv sync in apps/api.") from exc


class RouteIntentSignature(dspy.Signature):
    """Classify a fitness request into exactly one route."""

    route_catalog: str = dspy.InputField(desc="Route descriptions and examples.")
    user_input: str = dspy.InputField(desc="The user's raw natural-language request.")
    route: str = dspy.OutputField(desc="One of COACH, WORKOUT_GENERATE, WORKOUT_LOG, FALLBACK.")
    confidence: float = dspy.OutputField(desc="A number from 0.0 to 1.0.")
    reason: str = dspy.OutputField(desc="Short explanation for the selected route.")
    candidate_routes: list[str] = dspy.OutputField(
        desc="Ranked plausible route names, most likely first."
    )
    clarification_question: str = dspy.OutputField(
        desc="Question to ask if ambiguous; empty string otherwise."
    )


class IntentRouterProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(RouteIntentSignature)

    def forward(self, *, route_catalog: str, user_input: str):
        return self.predict(route_catalog=route_catalog, user_input=user_input)


def validate_prediction(values: object, user_input: str) -> RouteDecision:
    from fitness_router.routing import coerce_prediction_to_route_decision

    return coerce_prediction_to_route_decision(values, user_input)
