from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from 动销预测.agent import analyze_sales_question
from src.utils import format_percent

from .client import LLMClientError, OpenAICompatibleChatClient, sanitize_secret
from .config import LLMConfig
from .prompts import build_messages
from .safety import SafetyValidationError, parse_and_validate_llm_json
from .schemas import GroundedLLMResult, LLMStatus


RuleAnalyzer = Callable[[str, Dict[str, Any]], Dict[str, Any]]


def run_grounded_llm_analysis(
    question: str,
    context: Dict[str, Any],
    *,
    config: Optional[LLMConfig] = None,
    client: Optional[Any] = None,
    rule_analyzer: RuleAnalyzer = analyze_sales_question,
) -> Dict[str, Any]:
    cfg = config or LLMConfig.from_env()
    rule_response = rule_analyzer(question, context)
    fact_pack = build_fact_pack(rule_response, context)

    if not cfg.requested_llm:
        return GroundedLLMResult(
            rule_response=rule_response,
            fact_pack=fact_pack,
            status=LLMStatus(False, False, cfg.mode, "LLM_MODE=rule，使用规则驱动分析。"),
        ).to_dict()

    missing = cfg.missing_fields()
    if missing:
        return _fallback(
            rule_response,
            fact_pack,
            cfg,
            "LLM 配置不完整，已回退到规则驱动分析。",
            f"缺少环境变量：{', '.join(missing)}",
        )

    llm_client = client or OpenAICompatibleChatClient()
    try:
        raw = llm_client.complete(build_messages(question, fact_pack), cfg)
        explanation = parse_and_validate_llm_json(raw, fact_pack, cfg.max_output_chars)
    except (LLMClientError, SafetyValidationError, ValueError) as exc:
        reason = sanitize_secret(str(exc), cfg.api_key)
        message = "LLM 输出未通过事实一致性校验，已回退到规则驱动分析。" if isinstance(exc, SafetyValidationError) else "LLM 调用失败，已回退到规则驱动分析。"
        return _fallback(rule_response, fact_pack, cfg, message, reason)

    return GroundedLLMResult(
        rule_response=rule_response,
        fact_pack=fact_pack,
        llm_explanation=explanation,
        status=LLMStatus(True, True, "llm", "LLM 增强解释已通过事实一致性校验。"),
    ).to_dict()


def _fallback(
    rule_response: Dict[str, Any],
    fact_pack: Dict[str, Any],
    config: LLMConfig,
    message: str,
    reason: str,
) -> Dict[str, Any]:
    return GroundedLLMResult(
        rule_response=rule_response,
        fact_pack=fact_pack,
        status=LLMStatus(True, False, config.mode, message, sanitize_secret(reason, config.api_key)),
    ).to_dict()


def build_fact_pack(rule_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    tool_outputs = rule_response.get("tool_outputs") or {}
    forecast = tool_outputs.get("get_forecast_result") or {}
    comparison = tool_outputs.get("get_model_comparison") or {}
    predictability = tool_outputs.get("get_predictability_evidence") or {}
    summary = tool_outputs.get("generate_business_summary") or {}

    analysis_context = rule_response.get("analysis_context") or {
        "level": context.get("level"),
        "channel": context.get("channel"),
        "channel_subtype": context.get("channel_subtype") or context.get("channel_detail"),
        "product_model": context.get("product_model"),
        "horizon": context.get("horizon"),
        "backtest_windows": context.get("backtest_windows"),
    }

    model_candidates = comparison.get("所有候选模型") or []
    allowed_models = _allowed_models(forecast, comparison, model_candidates)
    metrics = _collect_metrics(forecast, comparison, predictability)
    risks = list(predictability.get("风险提示") or [])
    forecast_values = list(forecast.get("未来预测值") or [])

    return {
        "actual_analysis_object": analysis_context,
        "forecast_horizon": analysis_context.get("horizon"),
        "best_model": forecast.get("最优模型") or comparison.get("最优模型"),
        "allowed_models": allowed_models,
        "forecast_values": forecast_values,
        "forecast_interval": forecast.get("预测区间") or [],
        "recent_complete_period": forecast.get("最近完整账期"),
        "model_metrics": metrics,
        "predictability": {
            "score": predictability.get("综合可测性得分"),
            "grade": predictability.get("可测性等级"),
            "evidence": list(predictability.get("可测性证据列表") or []),
        },
        "risk_warnings": risks,
        "tool_calls": list(rule_response.get("tool_calls") or []),
        "evidence": list(rule_response.get("evidence") or []),
        "business_scope": (
            "LLM 仅用于业务解释与报告表达；预测值、回测误差、模型选择和可测性结论均由本地工具生成。"
            "预测区间不代表准确率保证。不得输出客户明细、原始交易数据或敏感业务记录。"
        ),
        "rule_answer": rule_response.get("answer", ""),
        "business_summary": summary.get("业务摘要"),
    }


def _allowed_models(
    forecast: Dict[str, Any],
    comparison: Dict[str, Any],
    model_candidates: list[Dict[str, Any]],
) -> list[str]:
    models = []
    for value in (forecast.get("最优模型"), comparison.get("最优模型")):
        if value:
            models.append(str(value))
    for row in model_candidates:
        model = row.get("模型")
        if model:
            models.append(str(model))
    return list(dict.fromkeys(models))


def _collect_metrics(
    forecast: Dict[str, Any],
    comparison: Dict[str, Any],
    predictability: Dict[str, Any],
) -> Dict[str, Any]:
    selected = None
    for row in comparison.get("所有候选模型") or []:
        if row.get("是否最优"):
            selected = row
            break

    metrics: Dict[str, Any] = {
        "WAPE": predictability.get("回测 WAPE"),
        "WAPE_display": format_percent(predictability.get("回测 WAPE")),
    }
    if selected:
        for key in ("WAPE", "sMAPE", "MAE", "Bias"):
            metrics[key] = selected.get(key)
            if key in {"WAPE", "sMAPE", "Bias"}:
                metrics[f"{key}_display"] = format_percent(selected.get(key))

    if forecast.get("最优模型"):
        metrics["best_model_from_forecast"] = forecast.get("最优模型")
    return metrics
