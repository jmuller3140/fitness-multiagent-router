import json
import sys
from types import SimpleNamespace

from fitness_router import db
from fitness_router.models import RouteDecision
from fitness_router.settings import Settings


def _decision() -> RouteDecision:
    return RouteDecision(route="WORKOUT_LOG", confidence=0.9, reason="Log request")


def test_persist_conversation_returns_none_without_database_url(monkeypatch):
    monkeypatch.setattr(db, "get_settings", lambda: Settings(database_url=None))

    assert (
        db.persist_conversation(
            session_id="s1",
            user_input="input",
            selected_route="COACH",
            route_decision=_decision(),
            final_response="response",
            structured_output=None,
        )
        is None
    )


def test_persist_conversation_writes_conversation_and_log_entries(monkeypatch):
    executed = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, sql, params):
            executed.append((sql, params))

        def fetchone(self):
            return ("conversation-1",)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(db, "get_settings", lambda: Settings(database_url="postgresql://local"))
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda database_url: FakeConnection()),
    )

    conversation_id = db.persist_conversation(
        session_id="s1",
        user_input="I did 3x10 bench press",
        selected_route="WORKOUT_LOG",
        route_decision=_decision(),
        final_response="Logged",
        structured_output={
            "log_entries": [
                {
                    "exercise_id": "bench",
                    "exercise_name": "bench press",
                    "matched_exercise_name": "Bench Press",
                    "sets": 3,
                    "reps": 10,
                    "weight": 185.0,
                    "weight_unit": "lb",
                    "confidence": 0.95,
                }
            ]
        },
    )

    assert conversation_id == "conversation-1"
    assert len(executed) == 2
    conversation_params = executed[0][1]
    assert conversation_params[0] == "s1"
    assert json.loads(conversation_params[3])["route"] == "WORKOUT_LOG"
    assert json.loads(conversation_params[5])["log_entries"][0]["sets"] == 3
    assert executed[1][1][0] == "conversation-1"


def test_persist_conversation_writes_empty_payload_without_log_entries(monkeypatch):
    executed = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, sql, params):
            executed.append((sql, params))

        def fetchone(self):
            return ("conversation-2",)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(db, "get_settings", lambda: Settings(database_url="postgresql://local"))
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=lambda database_url: FakeConnection()),
    )

    assert db.persist_conversation(
        session_id="s2",
        user_input="input",
        selected_route="COACH",
        route_decision=_decision(),
        final_response="response",
        structured_output=None,
    ) == "conversation-2"
    assert len(executed) == 1
    assert json.loads(executed[0][1][5]) == {}
