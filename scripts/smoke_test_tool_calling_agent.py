from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 动销预测.agent.schemas import AgentContext  # noqa: E402
from 动销预测.agent.tool_calling_orchestrator import run_tool_calling_agent  # noqa: E402
from 动销预测.llm.client import OpenAICompatibleChatClient  # noqa: E402
from 动销预测.llm.config import LLMConfig  # noqa: E402


DEFAULT_QUESTION = "请预测当前筛选范围未来三个月销量，并说明可测性和风险依据。"
SENSITIVE_KEYS = {
    "api_key",
    "llm_api_key",
    "db_path",
    "database_path",
    "system_prompt",
    "min_train_periods",
    "backtest_windows",
    "limit",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实 LLM Tool Calling Agent smoke test")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="发送给 Agent 的测试问题")
    parser.add_argument(
        "--level",
        choices=["渠道大类", "渠道细分类", "渠道型号"],
        default="渠道大类",
        help="受保护的预测层级",
    )
    parser.add_argument("--channel", default="全部", help="受保护的渠道大类筛选")
    parser.add_argument("--channel-subtype", default="全部", help="受保护的渠道细分类筛选")
    parser.add_argument("--product-model", default="全部", help="受保护的商品型号筛选")
    parser.add_argument("--horizon", type=int, choices=range(1, 13), default=3, metavar="1-12")
    parser.add_argument("--max-iterations", type=int, choices=range(1, 7), default=3, metavar="1-6")
    return parser


def _configuration_errors(config: LLMConfig, environ: Mapping[str, str]) -> List[str]:
    errors = []
    if config.mode != "llm":
        errors.append("LLM_MODE 必须设置为 llm")
    missing = config.missing_fields()
    if missing:
        errors.append("缺少 LLM 配置：" + "、".join(missing))
    if not str(environ.get("OMS_SQLITE_PATH") or "").strip():
        errors.append("缺少数据库配置：OMS_SQLITE_PATH")
    return errors


def _secret_values(config: LLMConfig, db_path: str) -> List[str]:
    values = {config.api_key, db_path}
    if db_path:
        path = Path(db_path).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        values.update({str(path), str(resolved), resolved.as_posix()})
    return sorted((value for value in values if value), key=len, reverse=True)


def _redact_text(text: str, secrets: Iterable[str]) -> str:
    redacted = str(text)
    for secret in secrets:
        redacted = redacted.replace(secret, "[redacted]")
    lowered = redacted.lower()
    if any(key in lowered for key in SENSITIVE_KEYS):
        return "[已隐藏受保护内容]"
    return redacted


def sanitize_output(value: Any, secrets: Iterable[str]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize_output(item, secrets)
            for key, item in value.items()
            if str(key).strip().lower() not in SENSITIVE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_output(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact_text(value, secrets)
    return value


def _decode_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def print_agent_result(response: Dict[str, Any], config: LLMConfig, db_path: str) -> None:
    secrets = _secret_values(config, db_path)
    print(f"Agent 是否成功：{'是' if response.get('ok') else '否'}")
    print(f"最终 answer：{_redact_text(response.get('answer') or '未返回最终回答', secrets)}")
    calls = response.get("tool_calls") if isinstance(response.get("tool_calls"), list) else []
    print(f"tool call 数量：{len(calls)}")
    print(f"iteration 数量：{response.get('iterations', 0)}")

    for index, call in enumerate(calls, start=1):
        if not isinstance(call, dict):
            continue
        print(f"\n工具调用 {index}")
        print(f"tool_name：{call.get('tool_name') or '未知工具'}")
        print("arguments：")
        _print_json(sanitize_output(_decode_arguments(call.get("arguments", {})), secrets))
        print(f"执行状态：{'成功' if call.get('ok') else '失败'}")
        print("observation/result：")
        _print_json(sanitize_output(call.get("result", {}), secrets))

    if response.get("error"):
        print(f"error：{_redact_text(response['error'], secrets)}")


def main(argv: Optional[List[str]] = None, environ: Optional[Mapping[str, str]] = None) -> int:
    args = _parser().parse_args(argv)
    source = environ if environ is not None else os.environ
    config = LLMConfig.from_env(source)
    errors = _configuration_errors(config, source)
    if errors:
        print("配置检查未通过：")
        for error in errors:
            print(f"- {error}")
        return 2

    db_path = str(source.get("OMS_SQLITE_PATH") or "").strip()
    if not Path(db_path).expanduser().is_file():
        print("配置检查未通过：OMS_SQLITE_PATH 指向的数据库不存在或不是文件。")
        return 2

    context = AgentContext(
        db_path=db_path,
        level=args.level,
        channel=args.channel,
        channel_subtype=args.channel_subtype,
        product_model=args.product_model,
        horizon=args.horizon,
    )
    response = run_tool_calling_agent(
        args.question,
        context,
        OpenAICompatibleChatClient(),
        config,
        max_iterations=args.max_iterations,
    )
    print_agent_result(response, config, db_path)
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
