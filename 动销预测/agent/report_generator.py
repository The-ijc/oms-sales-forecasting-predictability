from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .prompts import SUMMARY_TEMPLATE_NOTE


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "不可计算"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: Any) -> str:
    if value is None:
        return "不可计算"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _forecast_text(rows: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        month = row.get("月份") or row.get("month") or "未知账期"
        value = row.get("预测值", row.get("value"))
        parts.append(f"{month} 为 {_format_number(value)}")
    return "；".join(parts) if parts else "暂无未来预测值"


def _interval_text(rows: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        month = row.get("月份") or row.get("month") or "未知账期"
        lower = row.get("下界", row.get("lower"))
        upper = row.get("上界", row.get("upper"))
        parts.append(f"{month} 约为 {_format_number(lower)} 至 {_format_number(upper)}")
    return "；".join(parts) if parts else "当前回测窗口不足，未形成预测区间"


def _first_items(items: Iterable[Any], limit: int = 3) -> str:
    selected = [str(item) for item in list(items or [])[:limit]]
    return "；".join(selected) if selected else "暂无明显风险提示"


def build_business_summary(
    forecast_result: Dict[str, Any],
    model_comparison: Dict[str, Any],
    predictability_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """基于工具输入生成固定模板摘要，不自行创造预测数值。"""
    if not forecast_result or not model_comparison or not predictability_evidence:
        return {"ok": False, "业务摘要": "缺少预测、模型比较或可测性输入，无法生成业务摘要。"}

    object_name = forecast_result.get("预测对象名称", "当前预测对象")
    history_months = forecast_result.get("历史月份数")
    best_model = forecast_result.get("最优模型") or model_comparison.get("最优模型", "未识别")
    recent_period = forecast_result.get("最近完整账期") or "未识别"
    forecast_rows = forecast_result.get("未来预测值", [])
    interval_rows = forecast_result.get("预测区间", [])

    score = predictability_evidence.get("综合可测性得分")
    grade = predictability_evidence.get("可测性等级", "未评级")
    wape = predictability_evidence.get("回测 WAPE")
    risks = predictability_evidence.get("风险提示", [])

    summary = (
        f"{SUMMARY_TEMPLATE_NOTE}\n"
        f"分析对象为 {object_name}，最近完整账期为 {recent_period}，历史月份数为 {_format_number(history_months, 0)}。"
        f"当前最优模型为 {best_model}。未来预测值：{_forecast_text(forecast_rows)}。"
        f"经验预测区间：{_interval_text(interval_rows)}。"
        f"综合可测性得分为 {_format_number(score, 1)}，等级为 {grade}，回测 WAPE 为 {_format_percent(wape)}。"
        f"主要风险提示：{_first_items(risks)}。"
        "预测区间不是准确率保证，仅表示基于历史回测残差形成的经验范围。"
    )
    return {"ok": True, "业务摘要": summary}
