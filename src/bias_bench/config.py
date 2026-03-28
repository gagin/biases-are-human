"""Configuration models for the Bias Dissociation Benchmark."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    name: str = Field(..., description="Human-readable model name")
    openrouter_model_id: str = Field(..., description="OpenRouter model ID")
    family: str = Field(..., description="Model family (e.g., 'gpt', 'claude', 'llama', 'gemini')")
    capability_tier: Literal["small", "medium", "large"] = Field(
        ..., description="Model capability tier"
    )


class RunConfig(BaseModel):
    """Configuration for a benchmark run."""

    models: list[ModelConfig] = Field(..., description="List of models to run")
    temperature: float = Field(default=0.0, description="Temperature for model sampling")
    num_runs: int = Field(default=5, description="Number of times to run each item")
    max_concurrent: int = Field(default=3, description="Maximum concurrent requests")
    timeout_seconds: int = Field(default=30, description="Timeout for individual requests in seconds")


def load_config(path: str) -> RunConfig:
    """Load a RunConfig from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        RunConfig: Parsed configuration object

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration is invalid
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError("Configuration file is empty")

    return RunConfig(**data)
