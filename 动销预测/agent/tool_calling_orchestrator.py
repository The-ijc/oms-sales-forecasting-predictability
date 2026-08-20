from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union

from ..llm.client import OpenAICompatibleChatClient
from ..llm.config import LLMConfig
from .prompts import TOOL_CALLING_AGENT_SYSTEM_PROMPT
from .schemas import AgentContext, ToolCallRecord, ToolCallingAgentResponse
from .tools import TOOL_SCHEMAS, execute_tool


ContextInput = Union[AgentContext, Dict[str, Any]]


def _analysis_context(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "level": context.get("level"),
        "channel": context.get("channel"),
        "channel_subtype": context.get("channel_subtype"),
        "product_model": context.get("product_model"),
        "horizon": context.get("horizon"),
    }


def _normalize_context(context: ContextInput) -> Dict[str, Any]:
    if isinstance(context, AgentContext):
        return context.to_dict()
    if isinstance(context, dict):
        normalized = dict(context)
        if "db_path" not in normalized and context.get("database_path"):
            normalized["db_path"] = context["database_path"]
        if "channel_subtype" not in normalized:
            subtype = context.get("channel_detail") or context.get("channel_")
            if subtype is not None:
                normalized["channel_subtype"] = subtype
        return normalized
    raise TypeError("context 必须是 AgentContext 或 dict。")


def _safe_error_message(exc: Exception, config: LLMConfig) -> str:
    message = str(exc) or exc.__class__.__name__
    if config.api_key:
        message = message.replace(config.api_key, "[redacted]")
    return message


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    return str(value)


def _response_error(
    message: str,
    *,
    tool_calls: List[ToolCallRecord],
    iterations: int,
    analysis_context: Dict[str, Any],
) -> Dict[str, Any]:
    return ToolCallingAgentResponse(
        ok=False,
        tool_calls=tool_calls,
        iterations=iterations,
        error=message,
        analysis_context=analysis_context,
    ).to_dict()


def run_tool_calling_agent(
    question: str,
    context: ContextInput,
    client: OpenAICompatibleChatClient,
    config: LLMConfig,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """Run the native Chat Completions tool-calling loop."""
    records: List[ToolCallRecord] = []

    try:
        protected_context = _normalize_context(context)
    except (TypeError, ValueError) as exc:
        return _response_error(
            str(exc),
            tool_calls=records,
            iterations=0,
            analysis_context={},
        )

    try:
        public_context = _analysis_context(AgentContext.from_dict(protected_context).to_dict())
    except (TypeError, ValueError) as exc:
        return _response_error(
            f"context 参数错误：{exc}",
            tool_calls=records,
            iterations=0,
            analysis_context={},
        )
    if not isinstance(question, str) or not question.strip():
        return _response_error(
            "question 必须是非空的自然语言问题。",
            tool_calls=records,
            iterations=0,
            analysis_context=public_context,
        )
    if not isinstance(max_iterations, int) or max_iterations < 1:
        return _response_error(
            "max_iterations 必须是大于等于 1 的整数。",
            tool_calls=records,
            iterations=0,
            analysis_context=public_context,
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": TOOL_CALLING_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]

    for iteration in range(1, max_iterations + 1):
        try:
            assistant_message = client.complete_message(
                messages,
                config,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except Exception as exc:
            return _response_error(
                f"LLM 请求失败：{_safe_error_message(exc, config)}",
                tool_calls=records,
                iterations=iteration,
                analysis_context=public_context,
            )

        if not isinstance(assistant_message, dict):
            return _response_error(
                "LLM 返回的 assistant message 结构异常。",
                tool_calls=records,
                iterations=iteration,
                analysis_context=public_context,
            )

        messages.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls")
        if tool_calls:
            if not isinstance(tool_calls, list):
                return _response_error(
                    "assistant tool_calls 结构异常。",
                    tool_calls=records,
                    iterations=iteration,
                    analysis_context=public_context,
                )

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    return _response_error(
                        "assistant tool_call 结构异常。",
                        tool_calls=records,
                        iterations=iteration,
                        analysis_context=public_context,
                    )

                tool_call_id = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                    return _response_error(
                        "assistant tool_call 缺少 id。",
                        tool_calls=records,
                        iterations=iteration,
                        analysis_context=public_context,
                    )
                if not isinstance(function, dict):
                    return _response_error(
                        f"tool_call {tool_call_id} 缺少 function。",
                        tool_calls=records,
                        iterations=iteration,
                        analysis_context=public_context,
                    )

                tool_name = function.get("name")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    return _response_error(
                        f"tool_call {tool_call_id} 的 function 缺少 name。",
                        tool_calls=records,
                        iterations=iteration,
                        analysis_context=public_context,
                    )

                arguments = function.get("arguments", "{}")
                result = execute_tool(
                    tool_name,
                    arguments,
                    protected_context=protected_context,
                )
                if not isinstance(result, dict):
                    result = {
                        "ok": False,
                        "tool": tool_name,
                        "错误": "工具返回结构异常。",
                    }

                record = ToolCallRecord(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    ok=bool(result.get("ok")),
                )
                records.append(record)

                try:
                    observation = json.dumps(
                        result,
                        ensure_ascii=False,
                        default=_json_default,
                    )
                except (TypeError, ValueError, RecursionError) as exc:
                    return _response_error(
                        f"工具结果无法序列化：{exc}",
                        tool_calls=records,
                        iterations=iteration,
                        analysis_context=public_context,
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": observation,
                    }
                )

            if iteration == max_iterations:
                return _response_error(
                    "reached_max_iterations：未能在限制轮数内生成最终回答。",
                    tool_calls=records,
                    iterations=iteration,
                    analysis_context=public_context,
                )
            continue

        content = assistant_message.get("content")
        if isinstance(content, str) and content.strip():
            return ToolCallingAgentResponse(
                ok=True,
                answer=content.strip(),
                tool_calls=records,
                iterations=iteration,
                analysis_context=public_context,
            ).to_dict()

        return _response_error(
            "assistant message 既没有非空 content，也没有 tool_calls。",
            tool_calls=records,
            iterations=iteration,
            analysis_context=public_context,
        )

    return _response_error(
        "reached_max_iterations：未能在限制轮数内生成最终回答。",
        tool_calls=records,
        iterations=max_iterations,
        analysis_context=public_context,
    )
