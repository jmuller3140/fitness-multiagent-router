from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fitness_router.db import persist_conversation
from fitness_router.graphs import build_hub_graph
from fitness_router.models import ChatRequest, ChatResponse, RouteDecision

app = FastAPI(title="Fitness Multi-Agent Router", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _graph():
    return build_hub_graph()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state = _graph().invoke({"user_input": request.message, "errors": []})
    decision = state["route_decision"]
    if not isinstance(decision, RouteDecision):
        decision = RouteDecision.model_validate(decision)

    errors = list(state.get("errors", []))
    try:
        persist_conversation(
            session_id=request.session_id,
            user_input=request.message,
            selected_route=state.get("selected_route", decision.route),
            route_decision=decision,
            final_response=state.get("final_response", ""),
            structured_output=state.get("structured_output"),
        )
    except Exception as exc:  # pragma: no cover - persistence must not break the demo path.
        errors.append(f"Persistence failed: {exc}")

    return ChatResponse(
        session_id=request.session_id,
        route=state.get("selected_route", decision.route),
        confidence=decision.confidence,
        reason=decision.reason,
        final_response=state.get("final_response", ""),
        structured_output=state.get("structured_output"),
        errors=errors,
    )
