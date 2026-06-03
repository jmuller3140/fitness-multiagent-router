from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fitness_router.models import Exercise
from fitness_router.settings import get_settings


def _norm(value: str) -> str:
    return value.strip().casefold()


@lru_cache(maxsize=4)
def load_exercises(path: str | None = None) -> tuple[Exercise, ...]:
    settings = get_settings()
    data_path = Path(path) if path else settings.exercise_data_file()
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return tuple(Exercise.model_validate(item) for item in raw)


@lru_cache(maxsize=4)
def exercises_by_id(path: str | None = None) -> dict[str, Exercise]:
    return {exercise.id: exercise for exercise in load_exercises(path)}


@lru_cache(maxsize=4)
def supported_equipment(path: str | None = None) -> tuple[str, ...]:
    equipment = {
        equipment_item
        for exercise in load_exercises(path)
        for equipment_item in exercise.equipment_required
    }
    return tuple(sorted(equipment, key=str.casefold))


@lru_cache(maxsize=4)
def supported_muscle_groups(path: str | None = None) -> tuple[str, ...]:
    groups = {group for exercise in load_exercises(path) for group in exercise.muscle_groups}
    return tuple(sorted(groups, key=str.casefold))


def match_text_terms(needles: list[str], haystack: list[str]) -> bool:
    if not needles:
        return True
    normalized_haystack = [_norm(item) for item in haystack]
    for needle in needles:
        normalized_needle = _norm(needle)
        if not any(
            normalized_needle in value or value in normalized_needle
            for value in normalized_haystack
        ):
            return False
    return True


def match_any_text_term(needles: list[str], haystack: list[str]) -> bool:
    if not needles:
        return True
    normalized_haystack = [_norm(item) for item in haystack]
    return any(
        any(_norm(needle) in value or value in _norm(needle) for value in normalized_haystack)
        for needle in needles
    )
