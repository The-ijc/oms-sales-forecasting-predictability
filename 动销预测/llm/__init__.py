"""Controlled LLM explanation layer for OMS sales forecasting."""

from .config import LLMConfig
from .grounded_service import run_grounded_llm_analysis

__all__ = ["LLMConfig", "run_grounded_llm_analysis"]
