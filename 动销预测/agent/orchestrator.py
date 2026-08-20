from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .prompts import RISK_DISCLAIMER
from .schemas import AgentContext, AgentResponse
from .tools import (
    compact_evidence_from_outputs,
    generate_business_summary,
    get_forecast_result,
    get_model_comparison,
    get_predictability_evidence,
)


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _detect_intent(question: str) -> str:
    text = question.strip().lower()
    if _contains_any(text, ["不建议", "强行预测", "不能预测", "不适合预测"]):
        return "not_recommend_forecast"
    if _contains_any(text, ["管理层摘要", "摘要", "简报", "报告"]):
        return "business_summary"
    if _contains_any(text, ["为什么"]) and _contains_any(text, ["模型", "算法", "选择", "选"]):
        return "model_choice"
    if _contains_any(text, ["比较", "候选算法", "不同算法", "算法对比"]):
        return "compare_algorithms"
    if _contains_any(text, ["数据质量", "质量", "缺失", "零值", "cv", "adi"]):
        return "data_quality"
    if _contains_any(text, ["可测", "靠谱吗", "可靠", "可信"]):
        return "predictability"
    if _contains_any(text, ["风险", "隐患", "注意"]):
        return "forecast_risk"
    if _contains_any(text, ["未来", "趋势", "预测", "动销怎么样", "下个月", "接下来"]):
        return "forecast_trend"
    return "forecast_trend"


def _question_horizon(question: str, fallback: int) -> int:
    match = re.search(r"未来\s*(\d+)\s*个?月", question)
    if not match:
        match = re.search(r"(\d+)\s*个?月", question)
    if match:
        return max(1, min(int(match.group(1)), 12))
    return fallback


def _question_context(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    ctx = AgentContext.from_dict(context).to_dict()
    ctx["horizon"] = _question_horizon(question, int(ctx.get("horizon") or 3))

    if "线下" in question and ctx.get("channel") in {"", "全部", None}:
        ctx["channel"] = "线下"
    if "线上" in question and ctx.get("channel") in {"", "全部", None}:
        ctx["channel"] = "线上"
    if "型号" in question and ctx.get("level") in {"", "渠道大类", None}:
        ctx["level"] = "渠道型号"
    return ctx


def _actual_object_text(context: Dict[str, Any]) -> str:
    return (
        f"预测层级={context.get('level') or '全部'}；"
        f"渠道大类={context.get('channel') or '全部'}；"
        f"渠道细分类={context.get('channel_subtype') or '全部'}；"
        f"型号={context.get('product_model') or '全部'}；"
        f"预测步长={context.get('horizon') or '-'}个月；"
        f"回测窗口={context.get('backtest_windows') or '-'}个月"
    )


def _prefix_actual_object(answer: str, context: Dict[str, Any]) -> str:
    if answer.startswith("实际分析对象："):
        return answer
    return f"实际分析对象：{_actual_object_text(context)}\n\n{answer}"


def _tool_outputs_by_name(outputs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for output in outputs:
        tool_name = str(output.get("tool") or "")
        if tool_name:
            result[tool_name] = output
    return result


def _forecast_answer(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return result.get("错误", "预测工具调用失败。")
    forecasts = result.get("未来预测值", [])
    forecast_text = "；".join(
        f"{row.get('月份')} 预测值 {row.get('预测值')}" for row in forecasts
    ) or "暂无未来预测值"
    interval_note = result.get("预测区间说明", "")
    return (
        f"{result.get('预测对象名称')} 的最优模型为 {result.get('最优模型')}，"
        f"最近完整账期为 {result.get('最近完整账期')}，{forecast_text}。{interval_note}"
    )


def _model_answer(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return result.get("错误", "模型比较工具调用失败。")
    rows = result.get("所有候选模型", [])
    ranked = [row for row in rows if row.get("排名")]
    ranked = sorted(ranked, key=lambda row: row["排名"])[:3]
    top_text = "；".join(
        f"第{row['排名']}名 {row.get('模型')}，WAPE {row.get('WAPE')}"
        for row in ranked
    ) or "没有形成有效回测排名"
    return f"当前最优模型是 {result.get('最优模型')}。{result.get('说明')} 候选模型排名：{top_text}。"


def _quality_answer(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return result.get("错误", "可测性工具调用失败。")
    return (
        f"{result.get('预测对象名称')} 的综合可测性得分为 {result.get('综合可测性得分')}，"
        f"等级为 {result.get('可测性等级')}。历史月份数 {result.get('历史月份数')}，"
        f"缺失比例 {result.get('缺失比例')}，零值比例 {result.get('零值比例')}，"
        f"CV {result.get('CV')}，ADI {result.get('ADI')}，回测 WAPE {result.get('回测 WAPE')}。"
    )


def _risk_answer(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return result.get("错误", "风险证据工具调用失败。")
    risks = result.get("风险提示", [])
    risk_text = "；".join(risks) if risks else "当前工具未发现明显预测风险。"
    return f"主要预测风险：{risk_text}"


def _not_recommend_answer(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return result.get("错误", "可测性工具调用失败。")
    risks = result.get("风险提示", [])
    if risks:
        return "不建议强行预测的依据包括：" + "；".join(risks)
    return "当前工具没有给出强烈的不建议预测信号，但仍需结合业务变化和数据完整性判断。"


def _call_summary(ctx: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], str]:
    forecast = get_forecast_result(context=ctx)
    comparison = get_model_comparison(context=ctx)
    evidence = get_predictability_evidence(context=ctx)
    summary = generate_business_summary(forecast, comparison, evidence)
    answer = summary.get("业务摘要") if summary.get("ok") else summary.get("错误", "摘要生成失败。")
    return (
        [
            "get_forecast_result",
            "get_model_comparison",
            "get_predictability_evidence",
            "generate_business_summary",
        ],
        [forecast, comparison, evidence, summary],
        answer,
    )


def analyze_sales_question(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """规则驱动智能分析入口，不调用真实 LLM API。"""
    intent = _detect_intent(question)
    ctx = _question_context(question, context or {})
    tool_calls: List[str]
    outputs: List[Dict[str, Any]]

    if intent == "business_summary":
        tool_calls, outputs, answer = _call_summary(ctx)
    elif intent in {"model_choice", "compare_algorithms"}:
        comparison = get_model_comparison(context=ctx)
        tool_calls = ["get_model_comparison"]
        outputs = [comparison]
        answer = _model_answer(comparison)
    elif intent in {"data_quality", "predictability"}:
        evidence = get_predictability_evidence(context=ctx)
        tool_calls = ["get_predictability_evidence"]
        outputs = [evidence]
        answer = _quality_answer(evidence)
    elif intent == "forecast_risk":
        evidence = get_predictability_evidence(context=ctx)
        tool_calls = ["get_predictability_evidence"]
        outputs = [evidence]
        answer = _risk_answer(evidence)
    elif intent == "not_recommend_forecast":
        evidence = get_predictability_evidence(context=ctx)
        tool_calls = ["get_predictability_evidence"]
        outputs = [evidence]
        answer = _not_recommend_answer(evidence)
    else:
        forecast = get_forecast_result(context=ctx)
        evidence = get_predictability_evidence(context=ctx)
        tool_calls = ["get_forecast_result", "get_predictability_evidence"]
        outputs = [forecast, evidence]
        answer = _forecast_answer(forecast)
        if evidence.get("ok") and evidence.get("风险提示"):
            answer += " 需要关注：" + "；".join(evidence.get("风险提示", [])[:3])

    response = AgentResponse(
        intent=intent,
        tool_calls=tool_calls,
        evidence=compact_evidence_from_outputs(outputs),
        answer=_prefix_actual_object(answer, ctx),
        disclaimer=RISK_DISCLAIMER,
        analysis_context={
            "level": ctx.get("level"),
            "channel": ctx.get("channel"),
            "channel_subtype": ctx.get("channel_subtype"),
            "product_model": ctx.get("product_model"),
            "horizon": ctx.get("horizon"),
            "backtest_windows": ctx.get("backtest_windows"),
        },
        tool_outputs=_tool_outputs_by_name(outputs),
    )
    return response.to_dict()
