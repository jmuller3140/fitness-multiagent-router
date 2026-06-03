from __future__ import annotations

from dataclasses import dataclass

from fitness_router.models import RouteName


@dataclass(frozen=True)
class RouteSpec:
    route: RouteName
    responsibility: str
    examples: tuple[str, ...]
    clarification_label: str


ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(
        route="COACH",
        responsibility=(
            "Answer fitness, training, exercise, muscle, equipment, "
            "or programming questions."
        ),
        examples=(
            "What muscles does a deadlift work?",
            "Is a goblet squat good for beginners?",
            "What is the difference between vertical and horizontal pulling?",
        ),
        clarification_label="get coaching information",
    ),
    RouteSpec(
        route="WORKOUT_GENERATE",
        responsibility=(
            "Build a workout plan from constraints such as duration, muscles, "
            "equipment, or goal."
        ),
        examples=(
            "Build me a 30 min upper body session with dumbbells",
            "Create a lower body workout using a barbell",
            "Give me a quick core finisher",
        ),
        clarification_label="generate a workout",
    ),
    RouteSpec(
        route="WORKOUT_LOG",
        responsibility="Parse completed workout activity into structured log entries.",
        examples=(
            "I just did 3x10 bench press at 185 lbs",
            "Logged 4 sets of 12 goblet squats with 45 lb",
            "Yesterday I completed 5x5 deadlifts at 225",
        ),
        clarification_label="log a completed workout",
    ),
    RouteSpec(
        route="FALLBACK",
        responsibility=(
            "Ask for clarification when intent is ambiguous, unsupported, "
            "or too underspecified."
        ),
        examples=(
            "Bench press",
            "Can you adjust it?",
            "Workout yesterday",
        ),
        clarification_label="clarify the request",
    ),
)


def build_route_catalog() -> str:
    rows = []
    for spec in ROUTES:
        examples = "; ".join(f'"{example}"' for example in spec.examples)
        rows.append(f"- {spec.route}: {spec.responsibility} Examples: {examples}")
    return "\n".join(rows)


def clarification_question(user_input: str) -> str:
    labels = ", ".join(spec.clarification_label for spec in ROUTES if spec.route != "FALLBACK")
    return f'Do you want to {labels}? I need a little more detail for "{user_input}".'
