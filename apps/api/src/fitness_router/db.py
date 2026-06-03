from __future__ import annotations

import json
from typing import Any

from fitness_router.models import RouteDecision
from fitness_router.settings import get_settings


def persist_conversation(
    *,
    session_id: str,
    user_input: str,
    selected_route: str,
    route_decision: RouteDecision,
    final_response: str,
    structured_output: dict[str, Any] | None,
) -> str | None:
    settings = get_settings()
    if not settings.database_url:
        return None

    import psycopg

    payload = structured_output or {}
    with psycopg.connect(settings.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (
                  session_id,
                  user_input,
                  selected_route,
                  route_decision,
                  final_response,
                  structured_output
                )
                VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb)
                RETURNING id
                """,
                (
                    session_id,
                    user_input,
                    selected_route,
                    json.dumps(route_decision.model_dump(mode="json")),
                    final_response,
                    json.dumps(payload),
                ),
            )
            conversation_id = str(cursor.fetchone()[0])

            for entry in payload.get("log_entries", []):
                cursor.execute(
                    """
                    INSERT INTO workout_log_entries (
                      conversation_id,
                      exercise_id,
                      exercise_name,
                      matched_exercise_name,
                      sets,
                      reps,
                      weight,
                      weight_unit,
                      confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        entry.get("exercise_id"),
                        entry.get("exercise_name"),
                        entry.get("matched_exercise_name"),
                        entry.get("sets"),
                        entry.get("reps"),
                        entry.get("weight"),
                        entry.get("weight_unit"),
                        entry.get("confidence"),
                    ),
                )

    return conversation_id
