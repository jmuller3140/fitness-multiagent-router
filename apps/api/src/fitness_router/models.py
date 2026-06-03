from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

RouteName = Literal["COACH", "WORKOUT_GENERATE", "WORKOUT_LOG", "FALLBACK"]


class RouteCandidate(BaseModel):
    route: RouteName = Field(description="Candidate route.")
    score: float = Field(ge=0.0, le=1.0, description="Relative score for this candidate route.")


class RouteDecision(BaseModel):
    route: RouteName = Field(description="Best route for this user request.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that the selected route is correct.",
    )
    reason: str = Field(description="Short explanation of the routing decision.")
    candidate_routes: list[RouteCandidate] = Field(
        default_factory=list,
        description="Ranked candidate routes when the router exposes alternatives.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Clarifying question for ambiguous or unsupported requests.",
    )

    model_config = ConfigDict(extra="forbid")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User request to route.")
    session_id: str = Field(default="demo", description="Client-provided conversation/session id.")


class ChatResponse(BaseModel):
    session_id: str
    route: RouteName
    confidence: float
    reason: str
    final_response: str
    structured_output: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class Exercise(BaseModel):
    id: str
    name: str
    muscle_groups: list[str]
    joints_loaded: list[str]
    movement_patterns: list[str]
    equipment_required: list[str]
    is_bilateral: bool
    side: str | None = None
    priority_tier: int
    is_reps: bool
    is_duration: bool
    supports_weight: bool
    estimated_rep_duration: float | None = None
    bilateral_pair_id: str | None = None


class SearchExercisesInput(BaseModel):
    muscle_groups: list[str] = Field(
        default_factory=list,
        description="Requested muscle groups, such as chest, back, deltoids, glutes, or core.",
    )
    equipment: list[str] = Field(
        default_factory=list,
        description="Requested equipment terms, such as dumbbell, barbell, bench, or rack.",
    )
    movement_patterns: list[str] = Field(
        default_factory=list,
        description="Requested movement pattern terms, such as squat, hinge, upper push, or pull.",
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of matching exercise records to return.",
    )


class ExerciseSearchResult(BaseModel):
    query: SearchExercisesInput
    matches: list[Exercise]
    no_results_reason: str | None = None
    supported_equipment: list[str] = Field(default_factory=list)
    supported_muscle_groups: list[str] = Field(default_factory=list)


class BuildWorkoutInput(BaseModel):
    duration_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Target total workout length in minutes.",
    )
    focus: str | None = Field(
        default=None,
        description="Short workout focus label, such as upper body or lower body.",
    )
    exercise_ids: list[str] = Field(
        min_length=1,
        description="Exercise ids selected from the exercise dataset.",
    )


class WorkoutExercise(BaseModel):
    exercise_id: str
    name: str
    sets: int | None = None
    reps: int | None = None
    duration_seconds: int | None = None
    rest_seconds: int = 60
    notes: str | None = None


class WorkoutSection(BaseModel):
    name: Literal["warmup", "main", "cooldown"]
    exercises: list[WorkoutExercise]


class GeneratedWorkout(BaseModel):
    title: str
    duration_minutes: int
    focus: str | None = None
    sections: list[WorkoutSection]


class WorkoutLogEntry(BaseModel):
    exercise_id: str | None
    exercise_name: str
    matched_exercise_name: str | None
    sets: int | None
    reps: int | None
    weight: float | None
    weight_unit: Literal["lb", "kg"] | None
    confidence: float = Field(ge=0.0, le=1.0)


class RouterEvalExample(BaseModel):
    user_input: str
    expected_route: RouteName
    should_fallback: bool = False
    reason_category: str


WorkoutGeneratorOutcome = Literal["WORKOUT_BUILT", "NO_RESULTS", "TOOL_ERROR"]


class WorkoutGeneratorEvalExample(BaseModel):
    user_input: str
    expected_outcome: WorkoutGeneratorOutcome
    required_tool_names: list[Literal["search_exercises", "build_workout"]] = Field(
        default_factory=list,
        description="Tool names that should be invoked for this generator case.",
    )
    reason_category: str


class HubState(TypedDict, total=False):
    user_input: str
    route_decision: RouteDecision
    selected_route: RouteName
    final_response: str
    structured_output: dict[str, Any] | None
    errors: list[str]
