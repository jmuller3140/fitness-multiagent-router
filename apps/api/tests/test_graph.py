from fitness_router.graphs import build_hub_graph
from fitness_router.graphs.coach import coach_node
from fitness_router.graphs.fallback import fallback_node
from fitness_router.graphs.hub import (
    _has_close_candidate_margin,
    _normalize_route_node,
    _select_route,
)
from fitness_router.graphs.workout_generator import (
    DeterministicWorkoutGeneratorAgent,
    workout_generator_node,
)
from fitness_router.graphs.workout_logger import (
    _best_exercise_match,
    _extract_exercise_phrase,
    _extract_sets_reps,
    _extract_weight,
    _similarity,
    workout_logger_node,
)
from fitness_router.models import BuildWorkoutInput, Exercise, RouteDecision
from fitness_router.routing import StaticIntentRouter
from fitness_router.settings import Settings
from fitness_router.tools import ToolExecutionError


def _graph_for(decision: RouteDecision):
    return build_hub_graph(
        router=StaticIntentRouter(decision),
        settings=Settings(dspy_model=None, router_confidence_threshold=0.8),
    )


def test_ambiguous_low_confidence_input_falls_back_to_clarification():
    graph = _graph_for(
        RouteDecision(
            route="WORKOUT_LOG",
            confidence=0.42,
            reason="Bench press could be a question, generation constraint, or log entry.",
        )
    )

    state = graph.invoke({"user_input": "Bench press", "errors": []})

    assert state["selected_route"] == "FALLBACK"
    assert "Do you want to" in state["final_response"]
    assert state["structured_output"]["fallback"] is True


def test_workout_generation_recovers_when_search_has_no_dataset_matches():
    graph = _graph_for(
        RouteDecision(
            route="WORKOUT_GENERATE",
            confidence=0.93,
            reason="The user asks to build a workout from equipment constraints.",
        )
    )

    state = graph.invoke(
        {"user_input": "Build me a workout with a cable machine and sled", "errors": []}
    )

    assert state["selected_route"] == "WORKOUT_GENERATE"
    assert "will not invent exercises outside the dataset" in state["final_response"]
    assert state["structured_output"]["workout"] is None
    assert state["structured_output"]["search"]["matches"] == []


def test_workout_logger_returns_valid_structured_log_entry():
    graph = _graph_for(
        RouteDecision(
            route="WORKOUT_LOG",
            confidence=0.96,
            reason="The user reports completed sets, reps, and load.",
        )
    )

    state = graph.invoke({"user_input": "I just did 3x10 bench press at 185 lbs", "errors": []})
    entry = state["structured_output"]["log_entries"][0]

    assert state["selected_route"] == "WORKOUT_LOG"
    assert entry["sets"] == 3
    assert entry["reps"] == 10
    assert entry["weight"] == 185.0
    assert entry["weight_unit"] == "lb"
    assert entry["matched_exercise_name"]


def test_coach_node_returns_dataset_match_and_no_match_responses():
    matched = coach_node({"user_input": "What does Barbell Decline Bench Press train?"})
    unmatched = coach_node({"user_input": "How should I warm up for a handstand?"})

    assert matched["structured_output"]["matched_exercise"]["name"] == "Barbell Decline Bench Press"
    assert "primarily trains" in matched["final_response"]
    assert unmatched["structured_output"]["matched_exercise"] is None
    assert "does not contain an exact named exercise match" in unmatched["final_response"]


def test_fallback_node_works_without_route_decision():
    state = fallback_node({"user_input": "Bench press"})

    assert state["structured_output"] == {
        "fallback": True,
        "original_route": None,
        "reason": "No route decision was available.",
    }
    assert "Do you want to" in state["final_response"]


def test_normalize_route_handles_missing_decision_existing_question_and_close_margin():
    normalize = _normalize_route_node(Settings(dspy_model=None, router_confidence_threshold=0.8))

    missing = normalize({"user_input": "Bench press"})
    fallback_with_question = normalize(
        {
            "user_input": "Bench press",
            "route_decision": RouteDecision(
                route="FALLBACK",
                confidence=0.99,
                reason="Already fallback",
                clarification_question="Choose one?",
            ),
        }
    )
    close_margin = normalize(
        {
            "user_input": "Build or log?",
            "route_decision": RouteDecision(
                route="WORKOUT_GENERATE",
                confidence=0.95,
                reason="Close decision",
                candidate_routes=[
                    {"route": "WORKOUT_GENERATE", "score": 0.51},
                    {"route": "WORKOUT_LOG", "score": 0.46},
                ],
            ),
        }
    )

    assert missing["selected_route"] == "FALLBACK"
    assert missing["route_decision"].reason == "Router did not return a decision."
    assert fallback_with_question["route_decision"].clarification_question == "Choose one?"
    assert close_margin["selected_route"] == "FALLBACK"
    assert _select_route({}) == "FALLBACK"
    assert not _has_close_candidate_margin(
        RouteDecision(
            route="WORKOUT_GENERATE",
            confidence=1.0,
            reason="Not close",
            candidate_routes=[
                {"route": "FALLBACK", "score": 0.6},
                {"route": "WORKOUT_LOG", "score": 0.58},
            ],
        )
    )


def test_workout_generator_success_and_invalid_tool_call(monkeypatch):
    agent = DeterministicWorkoutGeneratorAgent()
    success = workout_generator_node(
        {"user_input": "Build me a 45 minute upper body dumbbell push workout", "errors": []},
        agent,
    )

    assert success["structured_output"]["workout"]["duration_minutes"] == 45
    assert "Warmup:" in success["final_response"]
    assert [call["name"] for call in success["structured_output"]["tool_calls"]] == [
        "search_exercises",
        "build_workout",
    ]

    def fail_build_workout(input: BuildWorkoutInput):
        raise ToolExecutionError("bad ids")

    monkeypatch.setattr("fitness_router.graphs.workout_generator.build_workout", fail_build_workout)
    failed = workout_generator_node(
        {"user_input": "Build me an upper body dumbbell workout", "errors": ["prior"]},
        agent,
    )

    assert failed["structured_output"]["workout"] is None
    assert failed["errors"] == ["prior", "bad ids"]
    assert "tool call was invalid" in failed["final_response"]


def test_workout_logger_handles_unmatched_alternative_and_missing_quantity_paths(monkeypatch):
    unmatched = workout_logger_node({"user_input": "I did 2 sets of 8 mystery lift with 10 kg"})
    assert unmatched["structured_output"]["log_entries"][0]["matched_exercise_name"] is None
    assert unmatched["structured_output"]["log_entries"][0]["weight_unit"] == "kg"

    exercise = Exercise.model_validate(
        {
            "id": "ex1",
            "name": "Bench Press",
            "muscle_groups": ["chest"],
            "joints_loaded": ["shoulder"],
            "movement_patterns": ["upper push"],
            "equipment_required": ["Barbell"],
            "is_bilateral": True,
            "priority_tier": 1,
            "is_reps": True,
            "is_duration": True,
            "supports_weight": True,
        }
    )
    alternative = exercise.model_copy(update={"id": "ex2", "name": "Incline Bench Press"})
    monkeypatch.setattr(
        "fitness_router.graphs.workout_logger._best_exercise_match",
        lambda phrase: (exercise, 0.7, [exercise, alternative]),
    )
    ambiguous = workout_logger_node({"user_input": "I logged 3 sets of 10 press"})

    monkeypatch.setattr(
        "fitness_router.graphs.workout_logger._best_exercise_match",
        lambda phrase: (exercise, 0.95, []),
    )
    no_quantities = workout_logger_node({"user_input": "Bench Press"})

    assert "multiple plausible" in ambiguous["final_response"]
    assert ambiguous["structured_output"]["log_entries"][0]["exercise_id"] is None
    assert no_quantities["final_response"] == "Logged the set details Bench Press."


def test_workout_logger_helpers_cover_no_match_empty_dataset_and_sequence_matcher(monkeypatch):
    assert _extract_sets_reps("4 sets of 12 goblet squats") == (4, 12)
    assert _extract_sets_reps("no quantities") == (None, None)
    assert _extract_weight("bodyweight only") == (None, None)
    assert _extract_weight("20 kilograms") == (20.0, "kg")
    assert _extract_exercise_phrase("!!!") == "!!!"

    monkeypatch.setattr("fitness_router.graphs.workout_logger.load_exercises", lambda: ())
    assert _best_exercise_match("anything") == (None, 0.0, [])

    first = Exercise.model_validate(
        {
            "id": "first",
            "name": "First Press",
            "muscle_groups": ["chest"],
            "joints_loaded": ["shoulder"],
            "movement_patterns": ["upper push"],
            "equipment_required": [],
            "is_bilateral": True,
            "priority_tier": 1,
            "is_reps": True,
            "is_duration": True,
            "supports_weight": False,
        }
    )
    second = first.model_copy(update={"id": "second", "name": "Second Press"})
    monkeypatch.setattr(
        "fitness_router.graphs.workout_logger.load_exercises",
        lambda: (first, second),
    )
    monkeypatch.setattr(
        "fitness_router.graphs.workout_logger._similarity",
        lambda phrase, name: 70.0 if name == "First Press" else 68.0,
    )
    match, confidence, alternatives = _best_exercise_match("press")
    assert match == first
    assert confidence == 0.7
    assert alternatives == [first, second]

    monkeypatch.setattr("fitness_router.graphs.workout_logger.fuzz", None)
    assert _similarity("bench press", "bench press") == 100.0
