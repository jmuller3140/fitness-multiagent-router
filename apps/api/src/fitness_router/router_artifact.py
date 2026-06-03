from __future__ import annotations

import json
from typing import Any

from fitness_router.models import RouteDecision

LANGCHAIN_ROUTER_ARTIFACT_KIND = "langchain_structured_router_prompt"
ARTIFACT_SCHEMA_VERSION = 1

DEFAULT_ROUTER_INSTRUCTION = "Classify a fitness request into exactly one route."
ROUTER_POLICY = (
    "Use these routing rules:\n"
    "- COACH: fitness, exercise, muscle, equipment, programming, or how-to questions.\n"
    "- WORKOUT_GENERATE: requests to create, build, design, generate, or plan a workout.\n"
    "- WORKOUT_LOG: reports or commands to record completed exercise activity.\n"
    "- FALLBACK: ambiguous fragments, missing prior context, or multi-intent requests.\n"
    "- A how-to question about recording or logging remains COACH unless the user directly "
    "commands you to save/log/record completed work.\n"
    "- Ambiguous fragments like 'Bench press' should be FALLBACK, not a guessed route.\n\n"
    "Return only a JSON object matching the RouteDecision schema. candidate_routes must be "
    "a list of objects with route and score fields."
)


def default_router_prompt_artifact() -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": LANGCHAIN_ROUTER_ARTIFACT_KIND,
        "optimizer": "default",
        "instruction": DEFAULT_ROUTER_INSTRUCTION,
        "routing_policy": ROUTER_POLICY,
        "few_shots": [],
        "dspy_state": None,
    }


def load_router_prompt_artifact(raw: str) -> dict[str, Any]:
    values = json.loads(raw)
    if values.get("kind") == LANGCHAIN_ROUTER_ARTIFACT_KIND:
        return _normalize_runtime_artifact(values)
    return runtime_artifact_from_dspy_state(values)


def runtime_artifact_from_dspy_state(
    state: dict[str, Any],
    *,
    optimizer: str = "dspy.MIPROv2",
    example_count: int | None = None,
    train_count: int | None = None,
    validation_count: int | None = None,
) -> dict[str, Any]:
    return _normalize_runtime_artifact(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "kind": LANGCHAIN_ROUTER_ARTIFACT_KIND,
            "optimizer": optimizer,
            "instruction": _extract_instruction(state),
            "routing_policy": ROUTER_POLICY,
            "few_shots": _extract_few_shots(state),
            "metrics": {
                "example_count": example_count,
                "train_count": train_count,
                "validation_count": validation_count,
            },
            "dspy_state": state or None,
        }
    )


def artifact_to_prompt_messages(artifact: dict[str, Any]) -> list[tuple[str, str]]:
    normalized = _normalize_runtime_artifact(artifact)
    system = (
        f"{normalized['instruction']}\n\n"
        f"{normalized['routing_policy']}\n\n"
        "Route catalog:\n{route_catalog}"
    )
    messages = [("system", system)]
    for example in normalized["few_shots"]:
        messages.append(("human", _escape_template_text(str(example["user_input"]))))
        messages.append(
            (
                "ai",
                _escape_template_text(json.dumps(example["route_decision"], sort_keys=True)),
            )
        )
    messages.append(("human", "{user_input}"))
    return messages


def _normalize_runtime_artifact(values: dict[str, Any]) -> dict[str, Any]:
    artifact = default_router_prompt_artifact()
    artifact.update(values)
    artifact["schema_version"] = ARTIFACT_SCHEMA_VERSION
    artifact["kind"] = LANGCHAIN_ROUTER_ARTIFACT_KIND
    artifact["instruction"] = str(artifact.get("instruction") or DEFAULT_ROUTER_INSTRUCTION)
    artifact["routing_policy"] = str(artifact.get("routing_policy") or ROUTER_POLICY)
    artifact["few_shots"] = _normalize_few_shots(artifact.get("few_shots"))
    return artifact


def _normalize_few_shots(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []

    examples = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        user_input = item.get("user_input")
        decision = item.get("route_decision")
        if not user_input or not isinstance(decision, dict):
            continue
        try:
            parsed = RouteDecision.model_validate(decision)
        except ValueError:
            continue
        examples.append(
            {
                "user_input": str(user_input),
                "route_decision": parsed.model_dump(mode="json"),
            }
        )
    return examples


def _extract_instruction(state: dict[str, Any]) -> str:
    if not isinstance(state, dict):
        return DEFAULT_ROUTER_INSTRUCTION
    for value in state.values():
        if not isinstance(value, dict):
            continue
        signature = value.get("signature")
        if isinstance(signature, dict) and signature.get("instructions"):
            return str(signature["instructions"])
    return DEFAULT_ROUTER_INSTRUCTION


def _extract_few_shots(state: dict[str, Any]) -> list[dict[str, Any]]:
    examples = []
    if not isinstance(state, dict):
        return examples

    for value in state.values():
        if not isinstance(value, dict):
            continue
        for key in ("demos", "train"):
            raw_examples = value.get(key)
            if not isinstance(raw_examples, list):
                continue
            examples.extend(_dspy_examples_to_few_shots(raw_examples))
    return examples


def _dspy_examples_to_few_shots(raw_examples: list[object]) -> list[dict[str, Any]]:
    examples = []
    for raw in raw_examples:
        if not isinstance(raw, dict):
            continue
        values = raw.get("kwargs") if isinstance(raw.get("kwargs"), dict) else raw
        user_input = values.get("user_input")
        if not user_input:
            continue
        decision = {
            "route": values.get("route") or values.get("expected_route"),
            "confidence": values.get("confidence", 1.0),
            "reason": values.get("reason") or "Few-shot routing example.",
            "candidate_routes": values.get("candidate_routes") or [],
            "clarification_question": values.get("clarification_question"),
        }
        try:
            parsed = RouteDecision.model_validate(decision)
        except ValueError:
            continue
        examples.append(
            {
                "user_input": str(user_input),
                "route_decision": parsed.model_dump(mode="json"),
            }
        )
    return examples


def _escape_template_text(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")
