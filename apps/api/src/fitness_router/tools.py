from __future__ import annotations

import re
from collections import Counter

from langchain_core.tools import StructuredTool

from fitness_router.data import (
    exercises_by_id,
    load_exercises,
    match_any_text_term,
    match_text_terms,
    supported_equipment,
    supported_muscle_groups,
)
from fitness_router.models import (
    BuildWorkoutInput,
    Exercise,
    ExerciseSearchResult,
    GeneratedWorkout,
    SearchExercisesInput,
    WorkoutExercise,
    WorkoutSection,
)


class ToolExecutionError(ValueError):
    """Raised when a tool call is well-formed but semantically invalid."""


def search_exercises(input: SearchExercisesInput) -> ExerciseSearchResult:
    normalized_input = input.model_copy(
        update={
            "muscle_groups": _expand_query_terms(input.muscle_groups, _MUSCLE_TERMS),
            "equipment": _expand_query_terms(input.equipment, _EQUIPMENT_TERMS),
            "movement_patterns": _expand_query_terms(input.movement_patterns, _MOVEMENT_TERMS),
        }
    )
    matches = _matching_exercises(normalized_input)
    effective_input = normalized_input
    if not matches and normalized_input.movement_patterns:
        relaxed_input = normalized_input.model_copy(update={"movement_patterns": []})
        relaxed_matches = _matching_exercises(relaxed_input)
        if relaxed_matches:
            matches = relaxed_matches
            effective_input = relaxed_input

    ranked_matches = sorted(matches, key=lambda item: (item.priority_tier, item.name.casefold()))
    limited_matches = ranked_matches[: effective_input.limit]

    no_results_reason = None
    if not limited_matches:
        parts = []
        if effective_input.muscle_groups:
            parts.append(f"muscle groups={effective_input.muscle_groups}")
        if effective_input.equipment:
            parts.append(f"equipment={effective_input.equipment}")
        if effective_input.movement_patterns:
            parts.append(f"movement patterns={effective_input.movement_patterns}")
        no_results_reason = "No dataset exercises matched " + ", ".join(parts or ["the request"])

    return ExerciseSearchResult(
        query=effective_input,
        matches=limited_matches,
        no_results_reason=no_results_reason,
        supported_equipment=list(supported_equipment())[:20],
        supported_muscle_groups=list(supported_muscle_groups()),
    )


def build_workout(input: BuildWorkoutInput) -> GeneratedWorkout:
    by_id = exercises_by_id()
    missing = [exercise_id for exercise_id in input.exercise_ids if exercise_id not in by_id]
    if missing:
        raise ToolExecutionError(f"Unknown exercise ids: {', '.join(missing)}")

    selected = [by_id[exercise_id] for exercise_id in input.exercise_ids]
    if not selected:
        raise ToolExecutionError("At least one valid exercise id is required.")

    warmup = WorkoutSection(
        name="warmup",
        exercises=[
            WorkoutExercise(
                exercise_id=selected[0].id,
                name=selected[0].name,
                duration_seconds=180,
                rest_seconds=30,
                notes="Move through a lighter version of the main pattern.",
            )
        ],
    )

    main_exercises = [
        WorkoutExercise(
            exercise_id=exercise.id,
            name=exercise.name,
            sets=3 if input.duration_minutes <= 35 else 4,
            reps=10 if exercise.is_reps else None,
            duration_seconds=None if exercise.is_reps else 45,
            rest_seconds=75 if exercise.supports_weight else 45,
        )
        for exercise in selected[:6]
    ]
    main = WorkoutSection(name="main", exercises=main_exercises)

    cooldown = WorkoutSection(
        name="cooldown",
        exercises=[
            WorkoutExercise(
                exercise_id=selected[-1].id,
                name=selected[-1].name,
                duration_seconds=120,
                rest_seconds=20,
                notes="Slow the tempo and keep effort easy.",
            )
        ],
    )

    title_focus = input.focus or infer_focus_from_exercises(selected)
    return GeneratedWorkout(
        title=f"{input.duration_minutes} minute {title_focus} workout",
        duration_minutes=input.duration_minutes,
        focus=title_focus,
        sections=[warmup, main, cooldown],
    )


def infer_focus_from_exercises(exercises: list[Exercise]) -> str:
    groups = Counter(group for exercise in exercises for group in exercise.muscle_groups)
    if not groups:
        return "general fitness"
    top_groups = [group for group, _ in groups.most_common(2)]
    return " and ".join(top_groups)


def _matching_exercises(input: SearchExercisesInput) -> list[Exercise]:
    matches: list[Exercise] = []
    for exercise in load_exercises():
        if not match_any_text_term(input.muscle_groups, exercise.muscle_groups):
            continue
        if not match_text_terms(input.equipment, exercise.equipment_required):
            continue
        if not match_any_text_term(input.movement_patterns, exercise.movement_patterns):
            continue
        matches.append(exercise)
    return matches


_MUSCLE_TERMS = {
    "upper body": ["chest", "back", "deltoids", "triceps", "biceps"],
    "lower body": ["quadriceps", "hamstrings", "glutes"],
    "core": ["core"],
    "chest": ["chest"],
    "back": ["back"],
    "shoulders": ["deltoids"],
    "shoulder": ["deltoids"],
    "arms": ["biceps", "triceps"],
    "legs": ["quadriceps", "hamstrings", "glutes"],
    "quads": ["quadriceps"],
    "quad": ["quadriceps"],
    "hamstring": ["hamstrings"],
    "glutes": ["glutes"],
}

_EQUIPMENT_TERMS = {
    "dumbbell": ["Dumbbell"],
    "dumbbells": ["Dumbbell"],
    "barbell": ["Barbell"],
    "kettlebell": ["Kettlebell"],
    "bench": ["Bench"],
    "rack": ["Rack"],
    "bodyweight": [],
    "cable": ["Cable Machine"],
    "machine": ["Machine"],
    "sled": ["Sled"],
}

_MOVEMENT_TERMS = {
    "push": ["upper push"],
    "pull": ["pull"],
    "squat": ["squat"],
    "hinge": ["hinge"],
    "carry": ["carry"],
    "core": ["core"],
}


def _expand_query_terms(values: list[str], aliases: dict[str, list[str]]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        mapped = aliases.get(value.casefold().strip())
        if mapped is None:
            expanded.append(value)
        else:
            expanded.extend(mapped)
    return sorted(set(expanded), key=str.casefold)


def parse_generation_constraints(user_input: str) -> tuple[SearchExercisesInput, int, str | None]:
    text = user_input.casefold()
    duration_match = re.search(r"(\d+)\s*(?:min|minute|minutes)", text)
    duration = int(duration_match.group(1)) if duration_match else 30

    muscle_groups: list[str] = []
    focus: str | None = None
    for term, groups in _MUSCLE_TERMS.items():
        if term in text:
            muscle_groups.extend(groups)
            if focus is None and term in {"upper body", "lower body", "core"}:
                focus = term

    equipment: list[str] = []
    for term, mapped in _EQUIPMENT_TERMS.items():
        if term in text:
            equipment.extend(mapped)

    movement_patterns: list[str] = []
    for term, mapped in _MOVEMENT_TERMS.items():
        if term in text:
            movement_patterns.extend(mapped)

    search_input = SearchExercisesInput(
        muscle_groups=sorted(set(muscle_groups)),
        equipment=sorted(set(equipment)),
        movement_patterns=sorted(set(movement_patterns)),
        limit=8,
    )
    return search_input, duration, focus


search_exercises_tool = StructuredTool.from_function(
    name="search_exercises",
    description="Search the exercise dataset by muscle groups, equipment, and movement patterns.",
    func=lambda **kwargs: search_exercises(
        SearchExercisesInput.model_validate(kwargs)
    ).model_dump(),
    args_schema=SearchExercisesInput,
)

build_workout_tool = StructuredTool.from_function(
    name="build_workout",
    description="Build a structured workout from valid exercise ids in the dataset.",
    func=lambda **kwargs: build_workout(BuildWorkoutInput.model_validate(kwargs)).model_dump(),
    args_schema=BuildWorkoutInput,
)

WORKOUT_TOOLS = [search_exercises_tool, build_workout_tool]
