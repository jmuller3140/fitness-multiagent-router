from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
API_DIR = PACKAGE_DIR.parents[1]
APPS_DIR = API_DIR.parent
ROOT_DIR = APPS_DIR.parent
DEFAULT_EXERCISE_DATA_PATH = ROOT_DIR / "data" / "exercises.json"
DEFAULT_ROUTER_EVALS_PATH = PACKAGE_DIR / "evals" / "router_examples.jsonl"
DEFAULT_WORKOUT_GENERATOR_EVALS_PATH = PACKAGE_DIR / "evals" / "workout_generator_examples.jsonl"
DEFAULT_ROUTER_ARTIFACT_PATH = API_DIR / "artifacts" / "router_optimized.json"
