from fitness_router import api
from fitness_router.models import ChatRequest, RouteDecision


class FakeGraph:
    def __init__(self, state):
        self.state = state
        self.invocations = []

    def invoke(self, state):
        self.invocations.append(state)
        return self.state


def test_graph_factory_caches_built_graph(monkeypatch):
    api._graph.cache_clear()
    built = object()
    calls = []

    def build():
        calls.append("called")
        return built

    monkeypatch.setattr(api, "build_hub_graph", build)

    assert api._graph() is built
    assert api._graph() is built
    assert calls == ["called"]
    api._graph.cache_clear()


def test_health_returns_ok():
    assert api.health() == {"status": "ok"}


def test_chat_coerces_dict_decision_and_persists(monkeypatch):
    state = {
        "route_decision": {
            "route": "COACH",
            "confidence": 0.91,
            "reason": "Question",
        },
        "selected_route": "COACH",
        "final_response": "Answer",
        "structured_output": {"matched_exercise": None},
        "errors": [],
    }
    graph = FakeGraph(state)
    persisted = {}

    monkeypatch.setattr(api, "_graph", lambda: graph)
    monkeypatch.setattr(api, "persist_conversation", lambda **kwargs: persisted.update(kwargs))

    response = api.chat(ChatRequest(session_id="s1", message="What is a squat?"))

    assert graph.invocations == [{"user_input": "What is a squat?", "errors": []}]
    assert response.session_id == "s1"
    assert response.route == "COACH"
    assert response.final_response == "Answer"
    assert persisted["selected_route"] == "COACH"
    assert isinstance(persisted["route_decision"], RouteDecision)


def test_chat_uses_decision_route_default_and_keeps_persistence_errors(monkeypatch):
    decision = RouteDecision(route="FALLBACK", confidence=0.2, reason="Too vague")
    graph = FakeGraph({"route_decision": decision})

    def fail_persist(**kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(api, "_graph", lambda: graph)
    monkeypatch.setattr(api, "persist_conversation", fail_persist)

    response = api.chat(ChatRequest(session_id="s2", message="Bench"))

    assert response.route == "FALLBACK"
    assert response.final_response == ""
    assert response.structured_output is None
    assert response.errors == ["Persistence failed: db unavailable"]
