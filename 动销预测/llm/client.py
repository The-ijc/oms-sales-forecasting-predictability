from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .config import LLMConfig


class LLMClientError(RuntimeError):
    """Readable LLM client error without exposing secrets."""


def build_chat_completions_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise LLMClientError("LLM_BASE_URL 为空。")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def sanitize_secret(message: str, secret: str) -> str:
    if not secret:
        return message
    return str(message).replace(secret, "[redacted]")


class OpenAICompatibleChatClient:
    def __init__(self, opener: Optional[Callable[..., Any]] = None):
        self._opener = opener or urllib.request.urlopen

    def complete(self, messages: List[Dict[str, Any]], config: LLMConfig) -> str:
        message = self._request_chat_completion(
            messages,
            config,
            response_format={"type": "json_object"},
        )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("LLM API 返回内容为空。")
        return content

    def complete_message(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return self._request_chat_completion(
            messages,
            config,
            tools=tools,
            tool_choice=tool_choice,
        )

    def _request_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = build_chat_completions_url(config.base_url)
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": config.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with self._opener(request, timeout=config.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _safe_http_detail(exc, config.api_key)
            raise LLMClientError(f"LLM API HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            detail = sanitize_secret(str(exc.reason), config.api_key)
            raise LLMClientError(f"LLM API 连接失败：{detail}") from exc
        except TimeoutError as exc:
            raise LLMClientError("LLM API 调用超时。") from exc
        except OSError as exc:
            detail = sanitize_secret(str(exc), config.api_key)
            raise LLMClientError(f"LLM API 调用失败：{detail}") from exc

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM API 返回不是合法 JSON。") from exc

        try:
            choices = body["choices"]
            first_choice = choices[0]
            message = first_choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM API 返回结构异常。") from exc
        if not isinstance(message, dict):
            raise LLMClientError("LLM API 返回 message 结构异常。")
        return message


def _safe_http_detail(exc: urllib.error.HTTPError, api_key: str) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    body = sanitize_secret(body, api_key)
    if not body:
        return "未返回错误详情"
    return body[:300]
