# Demo Transcript

These examples use the local graph with the demo router. With `DSPY_MODEL` set, the same graph calls the DSPy router program.

## Workout Generation

Input:

```text
Build me a 30 min upper body session with dumbbells
```

Expected route metadata:

```json
{
  "selected_route": "WORKOUT_GENERATE",
  "confidence": 0.88
}
```

Expected response shape:

```text
30 minute upper body workout
Warmup: ...
Main: ...
Cooldown: ...
```

The structured output includes the exercise search result and generated workout sections.

## Workout Logging

Input:

```text
I just did 3x10 bench press at 185 lbs
```

Expected route metadata:

```json
{
  "selected_route": "WORKOUT_LOG",
  "confidence": 0.9
}
```

Expected structured output:

```json
{
  "log_entries": [
    {
      "sets": 3,
      "reps": 10,
      "weight": 185.0,
      "weight_unit": "lb",
      "matched_exercise_name": "Barbell Decline Bench Press"
    }
  ]
}
```

## Ambiguous Input

Input:

```text
Bench press
```

Expected route metadata:

```json
{
  "selected_route": "FALLBACK"
}
```

Expected response:

```text
Do you want to get coaching information, generate a workout, log a completed workout?
```

## Unsupported Equipment Recovery

Input:

```text
Build me a workout with a cable machine and sled
```

Expected route metadata:

```json
{
  "selected_route": "WORKOUT_GENERATE"
}
```

Expected response:

```text
I could not build that workout because no dataset exercises matched the requested equipment. I will not invent exercises outside the dataset.
```
