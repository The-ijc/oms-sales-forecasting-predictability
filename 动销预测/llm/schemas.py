from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMStatus:
    requested: bool
    used: bool
    mode: str
    message: str
    fallback_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundedLLMResult:
    rule_response: Dict[str, Any]
    fact_pack: Dict[str, Any]
    llm_explanation: Optional[Dict[str, Any]] = None
    status: LLMStatus = field(default_factory=lambda: LLMStatus(False, False, "rule", "规则驱动模式"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_response": self.rule_response,
            "fact_pack": self.fact_pack,
            "llm_explanation": self.llm_explanation,
            "llm_status": self.status.to_dict(),
        }
