# Multi-Agent System — Take-Home

**Time:** 2–3 hours · **Stack:** Python, LangGraph, LangChain, any LLM provider

Build a hub agent that routes user requests to specialized sub-agents using LangGraph. The router classifies intent with LLM structured output and dispatches to a coach, a workout generator, or a workout logger.

| Route | Example |
|-------|---------|
| `COACH` | "What muscles does a deadlift work?" |
| `WORKOUT_GENERATE` | "Build me a 30 min upper body session with dumbbells" |
| `WORKOUT_LOG` | "I just did 3x10 bench press at 185 lbs" |

## Files

- **[`ASSESSMENT.md`](./ASSESSMENT.md)** — the full prompt: task, requirements, stretch goals
- **[`exercises.json`](./exercises.json)** — the exercise dataset (50 exercises)

## Submitting

Build in a **public** GitHub repo. Include a runnable demo or transcript and a README covering setup. See [`ASSESSMENT.md`](./ASSESSMENT.md) for the complete requirements.
