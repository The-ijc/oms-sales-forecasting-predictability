from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentContext:
    db_path: str = ""
    level: str = "渠道大类"
    channel: str = "全部"
    channel_subtype: str = "全部"
    product_model: str = "全部"
    horizon: int = 3
    min_train_periods: int = 12
    backtest_windows: int = 6
    limit: int = 50

    @classmethod
    def from_dict(cls, context: Optional[Dict[str, Any]]) -> "AgentContext":
        context = context or {}
        return cls(
            db_path=str(context.get("db_path") or context.get("database_path") or ""),
            level=str(context.get("level") or "渠道大类"),
            channel=str(context.get("channel") or "全部"),
            channel_subtype=str(
                context.get("channel_subtype")
                or context.get("channel_detail")
                or context.get("channel_")
                or "全部"
            ),
            product_model=str(context.get("product_model") or "全部"),
            horizon=int(context.get("horizon") or 3),
            min_train_periods=int(context.get("min_train_periods") or 12),
            backtest_windows=int(context.get("backtest_windows") or 6),
            limit=int(context.get("limit") or 50),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    intent: str
    tool_calls: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    answer: str = ""
    disclaimer: str = ""
    analysis_context: Dict[str, Any] = field(default_factory=dict)
    tool_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolError:
    tool: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": False, "tool": self.tool, "错误": self.message}


@dataclass
class ToolCallRecord:
    tool_call_id: str
    tool_name: str
    arguments: Any
    result: Dict[str, Any]
    ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallingAgentResponse:
    ok: bool
    answer: str = ""
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    error: Optional[str] = None
    analysis_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
