from __future__ import annotations

import re
from difflib import SequenceMatcher

from langgraph.graph import END, START, StateGraph

from fitness_router.data import load_exercises
from fitness_router.models import Exercise, HubState, WorkoutLogEntry

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - dependency is declared but this keeps import graceful.
    fuzz = None


def workout_logger_node(state: HubState) -> HubState:
    user_input = state["user_input"]
    sets, reps = _extract_sets_reps(user_input)
    weight, unit = _extract_weight(user_input)
    exercise_phrase = _extract_exercise_phrase(user_input)
    match, score, alternatives = _best_exercise_match(exercise_phrase)

    if match is None:
        entry = WorkoutLogEntry(
            exercise_id=None,
            exercise_name=exercise_phrase or user_input,
            matched_exercise_name=None,
            sets=sets,
            reps=reps,
            weight=weight,
            weight_unit=unit,
            confidence=0.35,
        )
        response = (
            "I parsed the workout quantities, but I could not confidently match the exercise "
            "to the dataset. Please provide the exercise name more specifically."
        )
    elif alternatives:
        entry = WorkoutLogEntry(
            exercise_id=None,
            exercise_name=exercise_phrase,
            matched_exercise_name=None,
            sets=sets,
            reps=reps,
            weight=weight,
            weight_unit=unit,
            confidence=score,
        )
        response = "I found multiple plausible exercise matches: " + ", ".join(
            exercise.name for exercise in alternatives
        )
    else:
        entry = WorkoutLogEntry(
            exercise_id=match.id,
            exercise_name=exercise_phrase,
            matched_exercise_name=match.name,
            sets=sets,
            reps=reps,
            weight=weight,
            weight_unit=unit,
            confidence=score,
        )
        set_text = f"{sets}x{reps}" if sets and reps else "the set details"
        weight_text = f" at {weight:g} {unit}" if weight and unit else ""
        response = f"Logged {set_text} {match.name}{weight_text}."

    return {
        "final_response": response,
        "structured_output": {"log_entries": [entry.model_dump()]},
    }


def _extract_sets_reps(text: str) -> tuple[int | None, int | None]:
    compact = text.casefold().replace(" ", "")
    match = re.search(r"(\d+)x(\d+)", compact)
    if match:
        return int(match.group(1)), int(match.group(2))

    words = text.casefold()
    match = re.search(r"(\d+)\s+sets?\s+(?:of\s+)?(\d+)", words)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def _extract_weight(text: str) -> tuple[float | None, str | None]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(lbs?|pounds?|kg|kilograms?)", text.casefold())
    if not match:
        return None, None
    unit = "kg" if match.group(2).startswith("kg") or match.group(2).startswith("kilo") else "lb"
    return float(match.group(1)), unit


def _extract_exercise_phrase(text: str) -> str:
    cleaned = text.casefold()
    cleaned = re.sub(r"\bi\s+(?:just\s+)?(?:did|completed|finished|logged)\b", "", cleaned)
    cleaned = re.sub(
        r"\b(?:at|with)\s+\d+(?:\.\d+)?\s*(?:lbs?|pounds?|kg|kilograms?)\b",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\d+\s*x\s*\d+", "", cleaned)
    cleaned = re.sub(r"\d+\s+sets?\s+(?:of\s+)?\d+", "", cleaned)
    cleaned = re.sub(r"[^a-z\s-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text.strip()


def _best_exercise_match(phrase: str) -> tuple[Exercise | None, float, list[Exercise]]:
    scored = sorted(
        ((_similarity(phrase, exercise.name), exercise) for exercise in load_exercises()),
        key=lambda item: item[0],
        reverse=True,
    )
    if not scored:
        return None, 0.0, []

    best_score, best_exercise = scored[0]
    confidence = best_score / 100.0
    if best_score < 55:
        return None, confidence, []

    close = [exercise for score, exercise in scored[1:4] if best_score - score <= 3 and score >= 55]
    if close and best_score < 76:
        return best_exercise, confidence, [best_exercise, *close]

    return best_exercise, confidence, []


def _similarity(left: str, right: str) -> float:
    if fuzz is not None:
        return float(
            max(
                fuzz.token_set_ratio(left, right),
                fuzz.partial_token_set_ratio(left, right),
                fuzz.WRatio(left, right),
            )
        )
    return 100.0 * SequenceMatcher(a=left.casefold(), b=right.casefold()).ratio()


def build_workout_logger_graph():
    graph = StateGraph(HubState)
    graph.add_node("workout_logger", workout_logger_node)
    graph.add_edge(START, "workout_logger")
    graph.add_edge("workout_logger", END)
    return graph.compile()
