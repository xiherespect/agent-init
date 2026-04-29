from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SETTINGS_FILE = Path(__file__).resolve().parents[2] / ".env"


class ProjectSettings(BaseModel):
    api_version: str = "v1"


class LLMSettings(BaseModel):
    api_key: SecretStr = SecretStr("")
    api_base: str = ""
    model: str = ""


class LangSmithSettings(BaseModel):
    api_key: SecretStr = SecretStr("")
    project: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_SETTINGS_FILE),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    zhipu: LLMSettings = Field(default_factory=LLMSettings)
    deepseek: LLMSettings = Field(default_factory=LLMSettings)
    qwen: LLMSettings = Field(default_factory=LLMSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
