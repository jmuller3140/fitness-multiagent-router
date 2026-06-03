from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from fitness_router.paths import DEFAULT_EXERCISE_DATA_PATH, DEFAULT_ROUTER_ARTIFACT_PATH, ROOT_DIR


class Settings(BaseSettings):
    database_url: str | None = Field(default=None, description="Postgres connection URL.")
    exercise_data_path: str = Field(
        default=str(DEFAULT_EXERCISE_DATA_PATH),
        description="Path to the exercise dataset JSON file.",
    )
    dspy_model: str | None = Field(
        default=None,
        description="DSPy LM identifier, for example deepseek/deepseek-chat.",
    )
    deepseek_api_key: str | None = Field(
        default=None,
        description="DeepSeek API key used when DSPY_MODEL is a DeepSeek LiteLLM model.",
    )
    dspy_router_artifact: str = Field(
        default=str(DEFAULT_ROUTER_ARTIFACT_PATH),
        description="Path to a compiled DSPy router artifact.",
    )

    def resolve_repo_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return ROOT_DIR / path

    def exercise_data_file(self) -> Path:
        return self.resolve_repo_path(self.exercise_data_path)

    def dspy_router_artifact_file(self) -> Path:
        return self.resolve_repo_path(self.dspy_router_artifact)

    def dspy_lm_kwargs(self) -> dict[str, str]:
        if (
            self.deepseek_api_key
            and self.dspy_model
            and self.dspy_model.casefold().startswith("deepseek/")
        ):
            return {"api_key": self.deepseek_api_key}
        return {}

    def langchain_model_name(self) -> str:
        if not self.dspy_model:
            raise RuntimeError("DSPY_MODEL must be set to use the structured LLM router.")
        if self.dspy_model.casefold().startswith("deepseek/"):
            return self.dspy_model.split("/", 1)[1]
        return self.dspy_model

    def langchain_chat_kwargs(self) -> dict[str, str]:
        if self.dspy_model and self.dspy_model.casefold().startswith("deepseek/"):
            if not self.deepseek_api_key:
                raise RuntimeError("DEEPSEEK_API_KEY must be set for DeepSeek routing.")
            return {
                "api_key": self.deepseek_api_key,
                "base_url": "https://api.deepseek.com",
            }
        return {}

    router_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required to dispatch to a non-fallback subgraph.",
    )
    allow_demo_router_without_llm: bool = Field(
        default=True,
        description="Use an offline router when DSPy is not configured.",
    )

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
