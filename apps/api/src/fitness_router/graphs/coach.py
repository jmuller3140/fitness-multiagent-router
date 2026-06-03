from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from fitness_router.data import load_exercises
from fitness_router.models import Exercise, HubState


def _matching_exercises(user_input: str) -> list[Exercise]:
    text = user_input.casefold()
    return [exercise for exercise in load_exercises() if exercise.name.casefold() in text]


def coach_node(state: HubState) -> HubState:
    user_input = state["user_input"]
    matches = _matching_exercises(user_input)

    if matches:
        exercise = matches[0]
        response = (
            f"{exercise.name} primarily trains {', '.join(exercise.muscle_groups)}. "
            f"It loads {', '.join(exercise.joints_loaded)} and uses "
            f"{', '.join(exercise.equipment_required) or 'no listed equipment'}. "
            f"The dataset classifies it as {', '.join(exercise.movement_patterns)}."
        )
        structured = {"matched_exercise": exercise.model_dump()}
    else:
        response = (
            "This is a coaching request. I can answer the training question, but the exercise "
            "dataset does not contain an exact named exercise match for the prompt. For best "
            "results, ask about a specific movement from the dataset or include the equipment."
        )
        structured = {"matched_exercise": None}

    return {"final_response": response, "structured_output": structured}


def build_coach_graph():
    graph = StateGraph(HubState)
    graph.add_node("coach", coach_node)
    graph.add_edge(START, "coach")
    graph.add_edge("coach", END)
    return graph.compile()
