# Preliminary Architecture

## Objective

Build a small end-to-end fitness coaching multi-agent system that satisfies the take-home requirements:

- A LangGraph hub agent routes requests to specialized sub-agents.
- Routing uses LLM structured output, not regex or keyword matching.
- Ambiguous inputs route to an explicit fallback or clarification path.
- Sub-agents are separate graphs composed into the hub.
- Tools use Pydantic input schemas with field descriptions.
- The system handles empty exercise search results and invalid tool calls gracefully.

## Architecture Summary

The system should be a composed LangGraph application:

```text
START
  -> route_intent
  -> normalize_route
  -> conditional edge:
       COACH            -> coach_graph
       WORKOUT_GENERATE -> workout_generator_graph
       WORKOUT_LOG      -> workout_logger_graph
       FALLBACK         -> fallback_graph
  -> END
```

The top-level graph is the hub. It owns request state, routing, fallback, and dispatch. The specialized agents own their domain-specific work.

## Architecture Anchors

### ARCH-001: Hub Graph Owns Routing

The hub is a LangGraph `StateGraph` with typed state and explicit edges. It must contain:

- `route_intent`: calls the LLM with structured output.
- `normalize_route`: applies confidence/fallback policy.
- Conditional edges to subgraphs.
- A fallback path for ambiguity and unsupported requests.

The hub should not inline the generator or logger logic. It should call compiled subgraphs.

### ARCH-002: Router Uses Structured Output

The router should use a Pydantic model with `with_structured_output()`:

```python
class RouteDecision(BaseModel):
    route: Literal["COACH", "WORKOUT_GENERATE", "WORKOUT_LOG", "FALLBACK"] = Field(
        description="Best route for this user request."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that the selected route is correct."
    )
    reason: str = Field(description="Short explanation of the routing decision.")
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask when the request is ambiguous or unsupported."
    )
```

The prompt should include the route catalog:

| Route | Responsibility | Example |
|---|---|---|
| `COACH` | Answer fitness/exercise questions | "What muscles does a deadlift work?" |
| `WORKOUT_GENERATE` | Build a workout plan from constraints | "Build me a 30 min upper body session with dumbbells" |
| `WORKOUT_LOG` | Parse completed workout activity into structured logs | "I just did 3x10 bench press at 185 lbs" |
| `FALLBACK` | Ask clarification or explain unsupported request | "Bench press" |

### ARCH-003: Fallback Policy Is Explicit

The router should not silently misroute. The normalization node should route to `FALLBACK` when:

- The model selects `FALLBACK`.
- Confidence is below `0.80`.
- The request is too short or underspecified and the model marks it ambiguous.
- The route reason indicates multiple plausible intents.

Fallback output should ask a specific clarifying question, such as:

```text
"Do you want information about bench press, a workout containing bench press, or to log a bench press set?"
```

### ARCH-004: Sub-Agents Are Separate Graphs

Each sub-agent should be constructed in its own module and return a compiled graph:

```text
app/
  graphs/
    hub.py
    coach.py
    workout_generator.py
    workout_logger.py
    fallback.py
```

The hub imports graph builders, not implementation functions.

### ARCH-005: Workout Generator Is Tool-Calling

The workout generator graph should use two tools:

1. `search_exercises`
   - Searches `exercises.json` by muscle groups, equipment, and movement patterns.
   - Returns matching exercise records and an empty result reason when no matches exist.

2. `build_workout`
   - Assembles warmup, main, and cooldown sections.
   - Uses selected exercise IDs.
   - Validates exercise IDs against the loaded dataset.

Both tools need Pydantic input schemas with field descriptions.

Failure behavior:

- If search returns no results, the generator should explain what was unavailable and suggest supported alternatives from the dataset.
- If `build_workout` receives an invalid exercise ID or bad schema, catch the error and return a controlled recovery message.

### ARCH-006: Workout Logger Produces Structured JSON

The workout logger graph should extract log entries:

```python
class WorkoutLogEntry(BaseModel):
    exercise_id: str | None
    exercise_name: str
    matched_exercise_name: str | None
    sets: int | None
    reps: int | None
    weight: float | None
    weight_unit: Literal["lb", "kg"] | None
    confidence: float
```

It should fuzzy-match user text against exercise names in `exercises.json`, so "bench press" can match a canonical bench press variation. If multiple close matches exist, return clarification rather than inventing a match.

### ARCH-007: Coach Is Retrieval-Light And Dataset-Grounded

The coach graph can be simple but should still use the exercise dataset when useful. It should answer questions about muscles, equipment, movement patterns, and training concepts. For the take-home scope, it does not need a separate vector index.

### ARCH-008: State Is Typed And Auditable

The hub state should include:

```python
class HubState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    route_decision: RouteDecision | None
    selected_route: str | None
    final_response: str | None
    structured_output: dict[str, Any] | None
    errors: list[str]
```

Subgraphs can extend their own state with tool results, matched exercises, and structured outputs.

## Data Flow

1. User input enters the hub graph.
2. `route_intent` asks the LLM for a structured `RouteDecision`.
3. `normalize_route` applies confidence and fallback rules.
4. Conditional edge dispatches to one subgraph.
5. Subgraph performs domain work and writes `final_response` and optional `structured_output`.
6. Hub returns the final response plus route metadata for the demo/transcript.

## Module Plan

```text
app/
  __init__.py
  main.py
  data.py
  models.py
  routing.py
  tools.py
  graphs/
    __init__.py
    hub.py
    coach.py
    workout_generator.py
    workout_logger.py
    fallback.py
tests/
  test_router.py
  test_workout_generator.py
  test_workout_logger.py
examples/
  demo_transcript.md
data/
  exercises.json
```

## Critical Tests

The minimum two critical paths should be:

1. **Ambiguous input falls back**
   - Input: `"Bench press"`
   - Expected: route is `FALLBACK`, response asks whether the user wants coaching, generation, or logging.
   - Reason: prevents silent misrouting, explicitly required by the prompt.

2. **Workout generation recovers from no exercise results**
   - Input: `"Build me a workout with a cable machine and sled"`
   - If the dataset lacks those constraints, `search_exercises` returns no results.
   - Expected: no crash, no hallucinated exercises, helpful alternatives.
   - Reason: resilience requirement is explicit and easy to regress.

Recommended additional tests:

- `"I just did 3x10 bench press at 185 lbs"` routes to `WORKOUT_LOG` and returns structured JSON.
- `"Build me a 30 min upper body session with dumbbells"` routes to `WORKOUT_GENERATE`.
- Invalid tool call with an unknown exercise ID returns a controlled error.

## Demo Shape

A CLI demo is enough:

```bash
python -m app.main "Build me a 30 min upper body session with dumbbells"
```

The output should include:

- Selected route
- Confidence
- Final answer
- Structured JSON for generated workouts or log entries

## Production Evaluation Plan

Production evaluation should track:

- Router accuracy on labeled examples.
- Fallback rate and fallback resolution rate.
- Low-confidence route rate.
- Tool error rate by tool name and error type.
- Empty exercise search rate.
- Invalid tool-call recovery rate.
- JSON schema validation success rate for workout logs.
- User correction rate after routing or logging.

Key failure modes:

- Ambiguous requests routed with high confidence.
- Workout generator hallucinating exercises outside the dataset.
- Logger incorrectly matching the wrong exercise variant.
- Invalid tool calls escaping as crashes.
- Route prompts drifting as new routes are added.

The system is working when router accuracy is high on held-out examples, fallback is common for genuinely ambiguous inputs, generated workouts only use valid dataset exercises, and structured log outputs validate consistently.

## Open Questions

- Which LLM provider will be used for the demo?
- Should the logger prefer exact canonical exercise names or return multiple candidate matches for review?
- Should coach answers be restricted to the exercise dataset, or can they include general fitness knowledge with dataset citations when available?
