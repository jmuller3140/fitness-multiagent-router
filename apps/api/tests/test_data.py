import json

from fitness_router import data


def test_loaders_support_explicit_dataset_path(tmp_path):
    data.load_exercises.cache_clear()
    data.exercises_by_id.cache_clear()
    data.supported_equipment.cache_clear()
    data.supported_muscle_groups.cache_clear()

    sample = data.load_exercises()[0]
    path = tmp_path / "exercises.json"
    path.write_text(json.dumps([sample.model_dump(mode="json")]), encoding="utf-8")

    loaded = data.load_exercises(str(path))

    assert loaded == (sample,)
    assert data.exercises_by_id(str(path)) == {sample.id: sample}
    assert data.supported_equipment(str(path)) == tuple(sorted(sample.equipment_required))
    assert data.supported_muscle_groups(str(path)) == tuple(sorted(sample.muscle_groups))


def test_text_term_matching_handles_empty_exact_partial_and_missing_terms():
    assert data.match_text_terms([], ["Dumbbell"])
    assert data.match_text_terms(["dumbbell"], ["Dumbbell"])
    assert data.match_text_terms(["Cable"], ["Cable Machine"])
    assert not data.match_text_terms(["barbell"], ["Dumbbell"])

    assert data.match_any_text_term([], ["upper push - horizontal"])
    assert data.match_any_text_term(["upper push"], ["upper push - horizontal"])
    assert data.match_any_text_term(["Cable Machine"], ["Cable"])
    assert not data.match_any_text_term(["hinge"], ["upper push"])
