# Fitness Multi-Agent Router

Fitness coaching multi-agent take-home implemented as a Turborepo workspace:

- `apps/web`: Vite + React chat UI for exercising the router.
- `apps/api`: FastAPI workspace app with a LangGraph hub and composed subgraphs.
- `infra/postgres`: Postgres schema for conversations, workout logs, and router eval runs.
- `data/exercises.json`: exercise dataset used by the generator, logger, and coach.

The architecture follows [REFERENCE.md](./REFERENCE.md): routing is a typed control-plane decision with confidence, reason, candidates, and fallback metadata. The difference from the HealthlyBot precedent is that the LangGraph `START` edge goes directly to the router node, then normalized routing dispatches to one compiled subgraph. In live LLM mode, DSPy optimizes the router prompt artifact and LangChain calls DeepSeek with `with_structured_output(RouteDecision)` for the runtime schema contract.

```text
START
  -> route_intent
  -> normalize_route
  -> COACH | WORKOUT_GENERATE | WORKOUT_LOG | FALLBACK subgraph
  -> END
```

## Run Locally

Install JavaScript dependencies:

```bash
pnpm install
```

Install Python dependencies with Python 3.13:

```bash
uv sync --project apps/api --extra dev --python python3.13
```

Start Postgres:

```bash
docker compose up -d postgres
```

Copy `.env.example` to `.env` and set `DSPY_MODEL` plus the provider key if you want live LLM routing. The current DeepSeek setup uses `DSPY_MODEL=deepseek/deepseek-chat` and `DEEPSEEK_API_KEY`. Without `DSPY_MODEL`, the API uses a deterministic demo router so the graph and UI still run locally.

Start the API and web app through Turbo:

```bash
pnpm dev
```

Or run either workspace app directly:

```bash
pnpm api:dev
pnpm web:dev
```

The web app calls `http://localhost:8000` by default. Set `VITE_API_URL` when starting `@fitness-router/web` if the FastAPI workspace app is running elsewhere.

Open `http://localhost:5173`.

Run the CLI demo:

```bash
uv run --project apps/api python -m fitness_router "Build me a 30 min upper body session with dumbbells"
```

Run tests:

```bash
pnpm api:test
```

The API test command enforces 100% statement and branch coverage for the FastAPI, LangGraph, routing, tool, data, persistence, CLI, and eval helper modules.

## DSPy + LangChain Router

Golden router examples live in [apps/api/src/fitness_router/evals/router_examples.jsonl](./apps/api/src/fitness_router/evals/router_examples.jsonl). They cover coaching, workout generation, workout logging, ambiguous fragments, unsupported generation constraints, and multi-intent fallback cases.

Evaluate the active router:

```bash
pnpm api:eval
```

Optimize the router prompt artifact with DSPy:

```bash
DSPY_MODEL=openai/gpt-4o-mini pnpm api:optimize
```

The optimizer trains a DSPy `IntentRouterProgram` against normalized subgraph route accuracy. Its metric rewards the final selected subgraph route after confidence fallback, not only the raw label emitted by the model. It uses a deterministic stratified train/validation split of the golden evals, then exports a LangChain runtime artifact at [apps/api/artifacts/router_optimized.json](./apps/api/artifacts/router_optimized.json). The production router loads that artifact, calls DeepSeek through LangChain, and validates the result with `with_structured_output(RouteDecision)`. To use a different golden file:

```bash
DSPY_MODEL=openai/gpt-4o-mini pnpm --filter @fitness-router/api optimize --goldens path/to/goldens.jsonl
```

## Workout Generator Tool Calls

When `DSPY_MODEL` is configured, the workout generator subgraph uses a LangChain chat model bound to the `search_exercises` and `build_workout` tools. The no-LLM demo path uses the same tool contract deterministically so local runs still work without provider credentials.

Evaluate the configured generator tool-calling path:

```bash
pnpm api:eval:generator
```

Evaluate the offline demo generator:

```bash
pnpm --filter @fitness-router/api eval:generator --backend demo
```

The generator eval checks expected outcome, required tool calls, no-results recovery, and whether generated workout exercise IDs all exist in the dataset.

## Critical Paths Tested

1. Ambiguous low-confidence input falls back to clarification.
2. Workout generation recovers when `search_exercises` returns no dataset matches.
3. Workout logging extracts structured JSON from conversational input.
4. The router eval set maintains broad coverage across all routes.
5. The workout generator eval set verifies tool-call behavior and valid generated exercise IDs.

## How I would evaluate this system in production

I would treat the intent router as a measurable classifier. The primary metric is normalized route accuracy on a continuously refreshed labeled set, segmented by route, ambiguity type, and source surface. The next most important metrics are fallback precision, fallback resolution rate, low-confidence dispatch rate, and user correction rate after routing.

For the workout generator, I would track empty exercise searches, invalid tool-call recovery, generated exercise IDs outside the dataset, and user edits to generated workouts. For the logger, I would track schema validation success, fuzzy-match confidence distribution, ambiguous match rate, and downstream corrections to exercise, sets, reps, or load.

The system is working when high-confidence routing is accurate, genuinely ambiguous requests fall back instead of being silently misrouted, generated workouts only use valid dataset exercises, invalid tool calls produce controlled recovery messages, and structured log entries are accepted without frequent user correction.

## Documents

- [assignment/ASSESSMENT.md](./assignment/ASSESSMENT.md) is the original take-home prompt.
- [REQUIREMENTS.md](./REQUIREMENTS.md) maps each assessment requirement to the implementation and verification coverage.
- [REFERENCE.md](./REFERENCE.md) captures the local OpenEMR/HealthlyBot intent-router precedent and what should or should not transfer.
- [ARCHITECTURE.md](./ARCHITECTURE.md) defines the preliminary LangGraph architecture.
