# AI Engineering — Take-Home Project: Fitness Coaching Multi-Agent System

**Time:** 2-3 hours | **Stack:** Python, LangGraph, LangChain, any LLM provider

A **small system that works end-to-end** is a great outcome. Focus on correctness over completeness — feel free to stub or mock anything that isn't core to the architecture.

---

## The Task

Build a multi-agent fitness coaching system where a **hub agent** routes user requests to specialized sub-agents using LangGraph.

**Three routes:**

| Route | Example |
|-------|---------|
| `COACH` | "What muscles does a deadlift work?" |
| `WORKOUT_GENERATE` | "Build me a 30 min upper body session with dumbbells" |
| `WORKOUT_LOG` | "I just did 3x10 bench press at 185 lbs" |

### Routing

Routing must use **LLM structured output** (e.g. `with_structured_output()`), not regex or keyword matching.

The router should handle **ambiguous inputs** gracefully. For example:
- "I did a workout yesterday, can you adjust it?" — is this `COACH`, `WORKOUT_LOG`, or something else?
- "Bench press" — no clear intent

Include a **confidence score or fallback route** so the agent doesn't silently misroute. How you handle low-confidence is up to you (ask for clarification, default route, etc.) — make the decision explicit.

### Sub-Agents

**Workout Generator** — a tool-calling agent with:
- `search_exercises` — searches `exercises.json` by muscle groups, equipment, movement patterns
- `build_workout` — assembles a structured workout (warmup/main/cooldown) from selected exercises with sets, reps, rest

**Workout Logger** — extracts structured data from natural language:
- Parses exercise name, sets, reps, weight from conversational input
- Fuzzy-matches against the exercise dataset (user says "bench press", not "Barbell Flat Bench Press")
- Returns structured JSON log entries

### Resilience

The agent should handle failure gracefully. At minimum:
- If `search_exercises` returns **no results** (e.g. user asks for equipment not in the dataset), the agent should recover — not crash or hallucinate exercises
- If the LLM produces an **invalid tool call** (wrong exercise ID, bad schema), the system should catch it and respond meaningfully

---

## Requirements

1. Hub is a LangGraph `StateGraph` with typed state and explicit edges
2. Sub-agents are separate graphs composed into the hub (not inlined functions)
3. Tools have Pydantic input schemas with field descriptions
4. **Test at least 2 critical paths** — pick the ones that matter most and explain why you chose them
5. Include a runnable demo or transcript showing the system in action (simple web view is fine)
6. **In your README, include a short section: "How I would evaluate this system in production"** — what metrics, what failure modes to monitor, how to know it's working. A few paragraphs is fine.
7. Submit as a **public GitHub repo**


---

## Stretch Goals

- Streaming support
- Multi-turn conversation memory
- Injury avoidance using `joints_loaded` from the exercise data
- Bilateral exercise pairing (auto-include other side)
- Observability (tracing LLM calls and tool invocations — e.g. Langfuse, OpenTelemetry, or structured logging)

---

## Data

See `exercises.json` (50 exercises). Key fields: `muscle_groups`, `joints_loaded`, `movement_patterns`, `equipment_required`, `priority_tier`, `is_bilateral`, `bilateral_pair_id`.
