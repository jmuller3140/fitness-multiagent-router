import pytest

from fitness_router.data import load_exercises
from fitness_router.models import BuildWorkoutInput, SearchExercisesInput
from fitness_router.tools import (
    ToolExecutionError,
    build_workout,
    build_workout_tool,
    infer_focus_from_exercises,
    parse_generation_constraints,
    search_exercises,
    search_exercises_tool,
)


def test_search_exercises_returns_empty_result_with_supported_alternatives():
    result = search_exercises(SearchExercisesInput(equipment=["Cable Machine", "Sled"]))

    assert result.matches == []
    assert result.no_results_reason is not None
    assert result.supported_equipment


def test_build_workout_rejects_unknown_exercise_id():
    with pytest.raises(ToolExecutionError):
        build_workout(BuildWorkoutInput(exercise_ids=["not-a-real-exercise-id"]))


def test_search_exercises_matches_filters_and_default_no_result_reason(monkeypatch):
    match = search_exercises(
        SearchExercisesInput(
            muscle_groups=["chest"],
            equipment=["Dumbbell"],
            movement_patterns=["upper push"],
            limit=2,
        )
    )

    assert 0 < len(match.matches) <= 2
    assert match.no_results_reason is None

    aliased = search_exercises(
        SearchExercisesInput(muscle_groups=["lower body"], equipment=["barbell"], limit=2)
    )
    relaxed = search_exercises(
        SearchExercisesInput(
            muscle_groups=["lower body"],
            equipment=["barbell"],
            movement_patterns=["squat", "hinge"],
            limit=2,
        )
    )
    assert aliased.matches
    assert aliased.query.muscle_groups == ["glutes", "hamstrings", "quadriceps"]
    assert aliased.query.equipment == ["Barbell"]
    assert relaxed.matches
    assert relaxed.query.movement_patterns == []

    monkeypatch.setattr("fitness_router.tools.load_exercises", lambda: ())
    monkeypatch.setattr("fitness_router.tools.supported_equipment", lambda: ())
    monkeypatch.setattr("fitness_router.tools.supported_muscle_groups", lambda: ())
    empty = search_exercises(SearchExercisesInput())

    assert empty.matches == []
    assert empty.no_results_reason == "No dataset exercises matched the request"

    described = search_exercises(
        SearchExercisesInput(muscle_groups=["imaginary"], movement_patterns=["spiral"])
    )
    assert "muscle groups=['imaginary']" in described.no_results_reason
    assert "movement patterns=['spiral']" in described.no_results_reason


def test_build_workout_success_duration_variants_and_constructed_empty_input():
    exercises = list(load_exercises())
    weighted = next(
        exercise for exercise in exercises if exercise.supports_weight and exercise.is_reps
    )
    duration_only = next(exercise for exercise in exercises if not exercise.is_reps)
    unweighted = next(exercise for exercise in exercises if not exercise.supports_weight)

    workout = build_workout(
        BuildWorkoutInput(
            duration_minutes=45,
            focus=None,
            exercise_ids=[weighted.id, duration_only.id, unweighted.id],
        )
    )

    assert workout.title.endswith("workout")
    assert workout.sections[1].exercises[0].sets == 4
    assert workout.sections[1].exercises[1].reps is None
    assert workout.sections[1].exercises[1].duration_seconds == 45
    assert workout.sections[1].exercises[2].rest_seconds == 45
    assert infer_focus_from_exercises([]) == "general fitness"

    with pytest.raises(ToolExecutionError, match="At least one valid exercise id"):
        build_workout(
            BuildWorkoutInput.model_construct(duration_minutes=30, focus=None, exercise_ids=[])
        )


def test_parse_generation_constraints_and_structured_tools():
    parsed, duration, focus = parse_generation_constraints(
        "Create a 20 minute lower body barbell hinge workout"
    )
    defaulted, default_duration, default_focus = parse_generation_constraints(
        "Create a bodyweight workout"
    )
    combined, _, combined_focus = parse_generation_constraints("upper body lower body core workout")

    search_payload = search_exercises_tool.invoke({"equipment": ["Dumbbell"], "limit": 1})
    workout_payload = build_workout_tool.invoke(
        {
            "duration_minutes": 30,
            "focus": "test",
            "exercise_ids": [search_payload["matches"][0]["id"]],
        }
    )

    assert duration == 20
    assert focus == "lower body"
    assert parsed.muscle_groups == ["glutes", "hamstrings", "quadriceps"]
    assert parsed.equipment == ["Barbell"]
    assert parsed.movement_patterns == ["hinge"]
    assert default_duration == 30
    assert default_focus is None
    assert defaulted.equipment == []
    assert combined_focus == "upper body"
    assert "core" in combined.muscle_groups
    assert workout_payload["focus"] == "test"
