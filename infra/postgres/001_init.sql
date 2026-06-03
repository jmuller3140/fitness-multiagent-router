CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id text NOT NULL,
  user_input text NOT NULL,
  selected_route text NOT NULL,
  route_decision jsonb NOT NULL,
  final_response text NOT NULL,
  structured_output jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_session_created_idx
  ON conversations (session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS workout_log_entries (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid REFERENCES conversations(id) ON DELETE CASCADE,
  exercise_id text,
  exercise_name text NOT NULL,
  matched_exercise_name text,
  sets integer,
  reps integer,
  weight numeric,
  weight_unit text,
  confidence numeric NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS router_eval_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model text NOT NULL,
  artifact_path text,
  accuracy numeric NOT NULL,
  fallback_accuracy numeric NOT NULL,
  example_count integer NOT NULL,
  results jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
