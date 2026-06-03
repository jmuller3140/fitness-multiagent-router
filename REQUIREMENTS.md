# Assessment Requirements Coverage

Source prompt copied from `/Users/jimmy/Development/candidate-assessment/1-multi-agent`:

- [assignment/README.md](./assignment/README.md)
- [assignment/ASSESSMENT.md](./assignment/ASSESSMENT.md)
- [data/exercises.json](./data/exercises.json)

## Core Task

Build a multi-agent fitness coaching system where a hub agent routes user requests to specialized sub-agents using LangGraph.

Implemented in [apps/api/src/fitness_router/graphs/hub.py](./apps/api/src/fitness_router/graphs/hub.py). The FastAPI server invokes the compiled hub graph from [apps/api/src/fitness_router/api.py](./apps/api/src/fitness_router/api.py).

## Routes

| Required route | Implementation |
| --- | --- |
| `COACH` | [apps/api/src/fitness_router/graphs/coach.py](./apps/api/src/fitness_router/graphs/coach.py) |
| `WORKOUT_GENERATE` | [apps/api/src/fitness_router/graphs/workout_generator.py](./apps/api/src/fitness_router/graphs/workout_generator.py) |
| `WORKOUT_LOG` | [apps/api/src/fitness_router/graphs/workout_logger.py](./apps/api/src/fitness_router/graphs/workout_logger.py) |
| Fallback / clarification | [apps/api/src/fitness_router/graphs/fallback.py](./apps/api/src/fitness_router/graphs/fallback.py) |

## Routing

Requirement: routing must use LLM structured output, not only regex or keyword matching.

Implemented with a LangChain runtime router in [apps/api/src/fitness_router/routing.py](./apps/api/src/fitness_router/routing.py) that calls DeepSeek with `with_structured_output(RouteDecision)`. DSPy remains in [apps/api/src/fitness_router/router_program.py](./apps/api/src/fitness_router/router_program.py) and [apps/api/src/fitness_router/evals/optimize_router.py](./apps/api/src/fitness_router/evals/optimize_router.py) to optimize/export the router prompt artifact from golden evals. The offline demo router is retained for local runs without `DSPY_MODEL`.

Requirement: ambiguous inputs should not silently misroute.

Implemented with route confidence, candidate route margins, and explicit fallback normalization in [apps/api/src/fitness_router/graphs/hub.py](./apps/api/src/fitness_router/graphs/hub.py). Low-confidence, direct fallback, and close-candidate decisions dispatch to the fallback graph.

## Sub-Agents

Requirement: workout generator should use `search_exercises` and `build_workout` tools.

Implemented in [apps/api/src/fitness_router/tools.py](./apps/api/src/fitness_router/tools.py), with Pydantic schemas in [apps/api/src/fitness_router/models.py](./apps/api/src/fitness_router/models.py). When `DSPY_MODEL` is configured, [apps/api/src/fitness_router/graphs/workout_generator.py](./apps/api/src/fitness_router/graphs/workout_generator.py) uses a LangChain chat model bound to `search_exercises` and `build_workout`. Without an LLM, it uses a deterministic offline executor over the same tool contract for local demos and tests.

Requirement: workout logger should parse exercise name, sets, reps, and weight, fuzzy-match the dataset, and return structured JSON.

Implemented in [apps/api/src/fitness_router/graphs/workout_logger.py](./apps/api/src/fitness_router/graphs/workout_logger.py). Structured log entries use `WorkoutLogEntry` from [apps/api/src/fitness_router/models.py](./apps/api/src/fitness_router/models.py).

## Resilience

Requirement: recover when exercise search returns no results.

Implemented in [apps/api/src/fitness_router/graphs/workout_generator.py](./apps/api/src/fitness_router/graphs/workout_generator.py). The system returns a controlled no-results response and supported equipment metadata instead of inventing exercises.

Requirement: catch invalid tool calls and respond meaningfully.

Implemented with LangChain tool-call schema handling, unknown-tool recovery, Pydantic validation errors, and `ToolExecutionError` in [apps/api/src/fitness_router/tools.py](./apps/api/src/fitness_router/tools.py), all handled in the workout generator graph.

## Required Deliverables

| Requirement | Status |
| --- | --- |
| Hub is a LangGraph `StateGraph` with typed state and explicit edges | Implemented in `graphs/hub.py` using `HubState` |
| Sub-agents are separate graphs composed into the hub | Implemented in `graphs/coach.py`, `fallback.py`, `workout_generator.py`, `workout_logger.py` |
| Tools have Pydantic input schemas with field descriptions | Implemented in `models.py` for `SearchExercisesInput` and `BuildWorkoutInput` |
| Test at least 2 critical paths | Exceeded: API test suite enforces 100% statement and branch coverage |
| Include runnable demo or transcript | Web app in `apps/web`; CLI demo via `python -m fitness_router`; transcript in [examples/demo_transcript.md](./examples/demo_transcript.md) |
| README includes production evaluation section | Included in [README.md](./README.md) |
| Submit as public GitHub repo | Repository packaging is ready; public visibility is an external GitHub setting |

## Stretch Goals Covered

| Stretch goal | Status |
| --- | --- |
| Streaming support | Not implemented |
| Multi-turn conversation memory | Not implemented beyond session id persistence |
| Injury avoidance using `joints_loaded` | Partially supported by coach responses using dataset fields |
| Bilateral exercise pairing | Not implemented |
| Observability | Basic persistence/eval artifacts only; no tracing backend |

## Verification

Run:

```bash
pnpm test
pnpm lint
pnpm build
```

The API test command enforces 100% statement and branch coverage.

The current live DeepSeek/LangChain router eval scores 47/48 examples, 97.92% normalized route accuracy, and 100% fallback accuracy.

The current live DeepSeek/LangChain workout-generator eval scores 5/5 examples, 100% required-tool accuracy, and 100% valid-workout-ID accuracy.
