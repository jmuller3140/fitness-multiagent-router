import json

from fitness_router import main
from fitness_router.models import RouteDecision


class FakeGraph:
    def __init__(self, decision):
        self.decision = decision

    def invoke(self, state):
        return {
            "route_decision": self.decision,
            "final_response": "response",
            "structured_output": {"ok": True},
        }


def test_run_coerces_route_decision_dict(monkeypatch):
    monkeypatch.setattr(
        main,
        "build_hub_graph",
        lambda: FakeGraph({"route": "COACH", "confidence": 0.9, "reason": "Question"}),
    )

    result = main.run("What is a squat?")

    assert result["selected_route"] == "COACH"
    assert result["route_decision"]["route"] == "COACH"
    assert result["errors"] == []


def test_run_uses_selected_route_and_main_prints_json(monkeypatch, capsys):
    decision = RouteDecision(route="WORKOUT_LOG", confidence=0.9, reason="Log")

    class SelectedGraph(FakeGraph):
        def invoke(self, state):
            result = super().invoke(state)
            result["selected_route"] = "FALLBACK"
            result["errors"] = ["err"]
            return result

    monkeypatch.setattr(main, "build_hub_graph", lambda: SelectedGraph(decision))
    assert main.run("Log this")["selected_route"] == "FALLBACK"

    monkeypatch.setattr("sys.argv", ["fitness_router", "Log", "this"])
    main.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["input"] == "Log this"
