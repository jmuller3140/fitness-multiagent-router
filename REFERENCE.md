# Reference: OpenEMR HealthlyBot Intent Router

## Source Reference

The transferable local precedent is the HealthlyBot agent API inside the OpenEMR repo:

- `/Users/jimmy/Development/jimmy-muller-openemr/healthlybot/agent-api/src/healthlybot_agent_api/router.py`
- `/Users/jimmy/Development/jimmy-muller-openemr/healthlybot/agent-api/src/healthlybot_agent_api/routing/resolver.py`
- `/Users/jimmy/Development/jimmy-muller-openemr/healthlybot/agent-api/src/healthlybot_agent_api/routing/registry.py`
- `/Users/jimmy/Development/jimmy-muller-openemr/healthlybot/agent-api/src/healthlybot_agent_api/schemas.py`
- `/Users/jimmy/Development/jimmy-muller-openemr/healthlybot/agent-api/src/healthlybot_agent_api/service.py`
- `/Users/jimmy/Development/jimmy-muller-openemr/healthlybot/agent-api/src/healthlybot_agent_api/dag.py`
- `/Users/jimmy/Development/jimmy-muller-openemr/healthlybot/agent-api/tests/test_router.py`
- `/Users/jimmy/Development/jimmy-muller-openemr/healthlybot/agent-api/src/healthlybot_agent_api/evals/router_examples.jsonl`

## Reference Architecture Pattern

HealthlyBot uses a clear control-plane pattern:

```text
ChatRequest
  -> router condition chain / semantic predictor
  -> RouteDecision
  -> workflow runner lookup
  -> selected workflow DAG
  -> StreamEvent / WorkflowOutput
```

The key product decision is that routing is a separate control-plane concern. The router does not execute workflow logic. It chooses a route, explains the decision, and records confidence and fallback metadata. The service layer then dispatches to a workflow runner.

## Transferable Decisions

The following decisions should be adapted into the fitness take-home:

1. **Typed route decision**
   - Keep route, confidence, reason, candidates, and clarification options as explicit structured data.
   - For this project, the route enum should be `COACH`, `WORKOUT_GENERATE`, `WORKOUT_LOG`, and `FALLBACK`.

2. **Confidence and abstention policy**
   - Do not silently dispatch ambiguous input.
   - If the top route is below threshold, route to clarification.
   - If competing routes are too close, route to clarification.
   - Include the model's reason in state for demo/debug output.

3. **Workflow registry**
   - Keep route descriptions, examples, and clarification labels in one registry-like module.
   - Use that registry to build the router prompt so the route catalog and fallback language stay synchronized.

4. **Workflow runner lookup**
   - Keep a map from route ID to graph builder.
   - In the LangGraph implementation, this becomes conditional edges from the hub graph to compiled subgraphs.

5. **Critical-path tests**
   - Test successful routing to generation and logging.
   - Test ambiguous input falling back to clarification.
   - Test generator recovery when exercise search returns no results.
   - Test logger fuzzy-match behavior for casual exercise names.

6. **Evaluation mindset**
   - Treat routing as a measurable classifier, not incidental prompt behavior.
   - Maintain a small JSONL or pytest case set with expected route, confidence expectation, and reason category.

## Non-Transferable Details

These parts should not be copied directly:

1. **Deterministic keyword router**
   - HealthlyBot has deterministic conditions and regex-like term matching for some flows.
   - The take-home explicitly requires LLM structured output for routing.
   - This project should use `llm.with_structured_output(RouteDecision)` or equivalent.

2. **DSPy artifact pipeline**
   - HealthlyBot has an optional DSPy optimizer and artifact loader.
   - That is heavier than the 2-3 hour assignment and not the requested stack.
   - The right transfer is the typed-output idea, not the DSPy implementation.

3. **Custom DAG runner**
   - HealthlyBot has a custom `DagRunner`.
   - The assessment requires LangGraph `StateGraph` with typed state and explicit edges.
   - Workflow step ideas can transfer, but the runtime should be LangGraph-native.

4. **Clinical safety domain**
   - HealthlyBot's refusal policy is clinical/billing-specific.
   - This project's fallback policy should focus on ambiguity, unsupported fitness requests, unavailable exercise data, and invalid tool calls.

## Adapted Router Contract

The router should produce a Pydantic model like:

```python
class RouteDecision(BaseModel):
    route: Literal["COACH", "WORKOUT_GENERATE", "WORKOUT_LOG", "FALLBACK"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    candidate_routes: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
```

The hub graph should then normalize this decision:

```text
if route == FALLBACK:
  -> fallback
elif confidence < MIN_ROUTE_CONFIDENCE:
  -> fallback
elif route margin is uncertain, if candidate scores are present:
  -> fallback
else:
  -> selected subgraph
```

## Reference Decision Ledger

| Decision | Source | Transfer Type | Project Decision |
|---|---|---|---|
| Router returns typed route decision with confidence/reason | HealthlyBot `RouteDecision` | Referenced | Use Pydantic `RouteDecision` from LangChain structured output |
| Low-confidence routes abstain | HealthlyBot `RouteResolver` | Referenced | Send to `FALLBACK` clarification node below threshold |
| Workflow catalog is centralized | HealthlyBot workflow registry | Referenced | Create a small route catalog for coach/generator/logger |
| Route maps to workflow runner | HealthlyBot service runner map | Referenced | Use LangGraph conditional edges to subgraphs |
| Workflow execution is a DAG | HealthlyBot `DagRunner` | Adapted | Use LangGraph `StateGraph`, not custom runner |
| Router can be evaluated independently | HealthlyBot router tests/evals | Referenced | Add pytest cases for route decisions and fallback behavior |
