from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class LLMConfig:
    mode: str = "rule"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout_seconds: float = 20.0
    max_output_chars: int = 2400
    max_tokens: int = 700

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "LLMConfig":
        source = environ or os.environ
        return cls(
            mode=(source.get("LLM_MODE") or "rule").strip().lower() or "rule",
            api_key=(source.get("LLM_API_KEY") or "").strip(),
            base_url=(source.get("LLM_BASE_URL") or "").strip(),
            model=(source.get("LLM_MODEL") or "").strip(),
            timeout_seconds=_float_env(source.get("LLM_TIMEOUT_SECONDS"), 20.0),
        )

    @property
    def requested_llm(self) -> bool:
        return self.mode == "llm"

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.api_key:
            missing.append("LLM_API_KEY")
        if not self.base_url:
            missing.append("LLM_BASE_URL")
        if not self.model:
            missing.append("LLM_MODEL")
        return missing

    def is_complete(self) -> bool:
        return not self.missing_fields()


def _float_env(value: Optional[str], default: float) -> float:
    try:
        parsed = float(value) if value is not None and str(value).strip() else default
    except ValueError:
        return default
    return max(1.0, min(parsed, 120.0))
