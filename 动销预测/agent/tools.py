from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.data_access import (
    DataAccessError,
    LEVEL_CONFIG,
    detect_global_latest_incomplete,
    get_aggregated_series,
    get_default_db_path,
    get_global_monthly_summary,
)
from src.model_selection import (
    DISPLAY_ALGORITHMS,
    algorithm_availability_for_series,
    select_model_and_forecast,
)
from src.predictability import assess_predictability
from src.utils import (
    ALGORITHM_LABELS,
    DEFAULT_BACKTEST_WINDOWS,
    MIN_TRAIN_PERIODS,
    MODEL_COMPLEXITY_RANK,
    SEASONAL_STRENGTH_THRESHOLD,
    format_percent,
    normalize_points,
    round_float,
)

from .report_generator import build_business_summary
from .knowledge_retriever import retrieve_knowledge
from .product_knowledge_retriever import retrieve_product_knowledge
from .schemas import AgentContext, ToolError


LEVEL_ALIASES = {
    "渠道大类": "channel",
    "channel": "channel",
    "渠道细分类": "channel_detail",
    "渠道细分": "channel_detail",
    "channel_detail": "channel_detail",
    "channel_subtype": "channel_detail",
    "渠道型号": "channel_model",
    "型号": "channel_model",
    "channel_model": "channel_model",
}

BUSINESS_LLM_TOOL_ARGUMENT_KEYS = {
    "level",
    "channel",
    "channel_subtype",
    "product_model",
    "horizon",
}
KNOWLEDGE_LLM_TOOL_ARGUMENT_KEYS = {"query", "top_k"}
PRODUCT_KNOWLEDGE_LLM_TOOL_ARGUMENT_KEYS = {"query", "product_model", "top_k"}
LLM_TOOL_ARGUMENT_KEYS = (
    BUSINESS_LLM_TOOL_ARGUMENT_KEYS
    | KNOWLEDGE_LLM_TOOL_ARGUMENT_KEYS
    | PRODUCT_KNOWLEDGE_LLM_TOOL_ARGUMENT_KEYS
)

TOOL_LLM_ARGUMENT_KEYS = {
    "get_forecast_result": BUSINESS_LLM_TOOL_ARGUMENT_KEYS,
    "get_model_comparison": BUSINESS_LLM_TOOL_ARGUMENT_KEYS,
    "get_predictability_evidence": BUSINESS_LLM_TOOL_ARGUMENT_KEYS,
    "search_knowledge_base": KNOWLEDGE_LLM_TOOL_ARGUMENT_KEYS,
    "search_product_knowledge": PRODUCT_KNOWLEDGE_LLM_TOOL_ARGUMENT_KEYS,
}

PROTECTED_CONTEXT_KEYS = {
    "db_path",
    "level",
    "channel",
    "channel_subtype",
    "product_model",
    "horizon",
    "min_train_periods",
    "backtest_windows",
    "limit",
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_forecast_result",
            "description": (
                "查询指定业务对象的动销预测结果，返回未来预测值、经验预测区间、"
                "最优模型和最近完整账期。不会返回原始明细。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "description": "预测层级，可用层级：渠道大类、渠道细分类、渠道型号。",
                    },
                    "channel": {
                        "type": "string",
                        "description": "渠道大类，例如线上、线下；未知时使用当前上下文。",
                    },
                    "channel_subtype": {
                        "type": "string",
                        "description": "渠道细分类；未知时使用当前上下文。",
                    },
                    "product_model": {
                        "type": "string",
                        "description": "产品型号；未知时使用当前上下文。",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "预测未来月份数，范围 1 到 12。",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_comparison",
            "description": (
                "比较指定业务对象下各候选预测模型的启用状态、回测指标、排名和最终模型选择原因。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "description": "预测层级，可用层级：渠道大类、渠道细分类、渠道型号。",
                    },
                    "channel": {
                        "type": "string",
                        "description": "渠道大类，例如线上、线下；未知时使用当前上下文。",
                    },
                    "channel_subtype": {
                        "type": "string",
                        "description": "渠道细分类；未知时使用当前上下文。",
                    },
                    "product_model": {
                        "type": "string",
                        "description": "产品型号；未知时使用当前上下文。",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "预测未来月份数，范围 1 到 12。",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_predictability_evidence",
            "description": (
                "获取指定业务对象的可测性评分、数据质量证据、回测 WAPE 和预测风险提示。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "description": "预测层级，可用层级：渠道大类、渠道细分类、渠道型号。",
                    },
                    "channel": {
                        "type": "string",
                        "description": "渠道大类，例如线上、线下；未知时使用当前上下文。",
                    },
                    "channel_subtype": {
                        "type": "string",
                        "description": "渠道细分类；未知时使用当前上下文。",
                    },
                    "product_model": {
                        "type": "string",
                        "description": "产品型号；未知时使用当前上下文。",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "预测未来月份数，范围 1 到 12。",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "检索项目内的指标定义、预测方法、业务规则、数据口径、操作说明和能力边界。"
                "不查询 SQLite，不返回当前销量、预测值或实时业务数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要从企业知识库检索的问题或关键词。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的相关知识片段数量，默认 3，最大 5。",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product_knowledge",
            "description": (
                "检索商品型号、商品名称、品牌、分类、产品介绍、主要规格、功能特点和资料来源。"
                "不查询 SQLite，不返回销量、预测值或可测性结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要检索的商品资料问题，可包含商品型号。",
                    },
                    "product_model": {
                        "type": "string",
                        "description": "可选商品型号；当前上下文已有具体型号时，以受保护上下文为准。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的商品资料数量，默认 3，最大 5。",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


@dataclass
class _AnalysisBundle:
    item: Dict[str, Any]
    completed: Any
    selection: Any
    report: Any
    latest_info: Dict[str, Any]


def _success(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "tool": tool_name, **payload}


def _error(tool_name: str, exc: Exception) -> Dict[str, Any]:
    return ToolError(tool=tool_name, message=str(exc)).to_dict()


def _level_key(level: str) -> str:
    key = LEVEL_ALIASES.get(str(level).strip())
    if not key:
        raise ValueError(f"不支持的预测层级：{level}")
    return key


def _level_label(level_key: str) -> str:
    return LEVEL_CONFIG.get(level_key, {}).get("label", level_key)


def _clean_filter_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "全部":
        return None
    return text


def _filters(channel: Optional[str], channel_subtype: Optional[str], product_model: Optional[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    channel_value = _clean_filter_value(channel)
    subtype_value = _clean_filter_value(channel_subtype)
    model_value = _clean_filter_value(product_model)
    if channel_value:
        result["channel"] = channel_value
    if subtype_value:
        result["channel_detail"] = subtype_value
    if model_value:
        result["product_model"] = model_value
    return result


def _db_path(db_path: Optional[str]) -> str:
    return str(db_path or get_default_db_path() or "")


def _bounded_horizon(horizon: int) -> int:
    try:
        value = int(horizon)
    except (TypeError, ValueError):
        value = 3
    return max(1, min(value, 12))


def _prepare_analysis(
    *,
    db_path: Optional[str] = None,
    level: str = "渠道大类",
    channel: Optional[str] = None,
    channel_subtype: Optional[str] = None,
    product_model: Optional[str] = None,
    horizon: int = 3,
    min_train_periods: int = MIN_TRAIN_PERIODS,
    backtest_windows: int = DEFAULT_BACKTEST_WINDOWS,
    limit: int = 50,
) -> _AnalysisBundle:
    db = _db_path(db_path)
    if not db:
        raise ValueError("数据库路径为空，请通过 context 或 OMS_SQLITE_PATH 提供只读数据库路径。")

    level_key = _level_key(level)
    query_filters = _filters(channel, channel_subtype, product_model)
    monthly_summary = get_global_monthly_summary(db)
    latest_info = detect_global_latest_incomplete(monthly_summary)
    series_list = get_aggregated_series(db, level_key, query_filters, limit=max(1, int(limit or 1)))
    if not series_list:
        raise ValueError("当前筛选条件下没有可分析的聚合序列。")

    item = series_list[0]
    completed = normalize_points(item["points"])
    selection = select_model_and_forecast(
        completed.months,
        completed.values,
        horizon=_bounded_horizon(horizon),
        min_train_periods=int(min_train_periods or MIN_TRAIN_PERIODS),
        backtest_windows=int(backtest_windows or DEFAULT_BACKTEST_WINDOWS),
        global_latest_info=latest_info,
    )
    report = assess_predictability(completed, selection, latest_info)
    return _AnalysisBundle(
        item=item,
        completed=completed,
        selection=selection,
        report=report,
        latest_info=latest_info,
    )


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _prediction_interval(selection: Any) -> tuple[List[Dict[str, Any]], str]:
    best = selection.best_backtest
    if best is None or len(best.fold_errors) < 3:
        return [], "历史回测不足，暂不提供预测区间。"

    residuals = [-error for error in best.fold_errors]
    lower_residual = _quantile(residuals, 0.10)
    upper_residual = _quantile(residuals, 0.90)
    if lower_residual is None or upper_residual is None:
        return [], "历史回测不足，暂不提供预测区间。"

    intervals: List[Dict[str, Any]] = []
    for month, forecast_value in zip(selection.forecast_months, selection.forecast_values):
        lower = max(0.0, forecast_value + lower_residual)
        upper = max(lower, forecast_value + upper_residual)
        intervals.append(
            {
                "月份": month,
                "下界": round_float(lower, 2),
                "上界": round_float(upper, 2),
            }
        )
    return intervals, "经验预测区间基于滚动回测残差 10% 和 90% 分位数，不代表准确率保证。"


def _valid_backtests(selection: Any) -> List[Any]:
    return [
        result
        for result in selection.backtest_results
        if result.fold_count > 0 and result.primary_score != float("inf")
    ]


def _ranked_backtests(selection: Any) -> List[Any]:
    return sorted(
        _valid_backtests(selection),
        key=lambda result: (
            result.primary_score,
            result.smape,
            MODEL_COMPLEXITY_RANK.get(result.algorithm, 999),
        ),
    )


def _rank_map(selection: Any) -> Dict[str, int]:
    return {result.algorithm: index + 1 for index, result in enumerate(_ranked_backtests(selection))}


def _metric_text(value: Optional[float], percent: bool = False) -> str:
    if percent:
        return format_percent(value)
    if value is None:
        return "不可计算"
    return f"{value:.2f}"


def _model_selection_note(selection: Any) -> str:
    best = selection.best_backtest
    if best is None:
        return "历史回测窗口不足，系统使用当前可用的保守算法生成预测，并在可测性中提示风险。"
    metric_name = "WAPE" if best.wape is not None else "MAE"
    metric_value = _metric_text(best.wape, True) if metric_name == "WAPE" else _metric_text(best.mae)
    return f"系统选择 {selection.selected_algorithm_label}，主比较指标 {metric_name} 为 {metric_value}。"


def _risk_list(bundle: _AnalysisBundle) -> List[str]:
    report = bundle.report
    selection = bundle.selection
    risks: List[str] = []

    if report.history_months < 12:
        risks.append("历史月份不足：历史少于 12 个月，回测支撑较弱。")
    if report.zero_ratio >= 0.5:
        risks.append("零值比例过高：零动销比例达到或超过 50%，建议关注间歇性需求。")
    if report.cv > 1.0:
        risks.append("波动过大：CV 超过 1，序列不稳定。")
    if report.best_wape is None and report.backtest_windows == 0:
        risks.append("回测不足：没有足够窗口评估模型误差。")
    elif report.best_wape is not None and report.best_wape > 0.8:
        risks.append("回测误差偏高：最佳模型 WAPE 超过 80%。")
    if selection.excluded_latest_month:
        risks.append(f"最新月份疑似不完整：{selection.excluded_latest_month} 已从训练和回测中排除。")
    if bundle.latest_info and bundle.latest_info.get("is_incomplete"):
        risks.extend(bundle.latest_info.get("reasons", []))
    if report.seasonality_strength < SEASONAL_STRENGTH_THRESHOLD:
        risks.append(
            f"季节性不足：季节性强度 {report.seasonality_strength:.2f} 低于 "
            f"{SEASONAL_STRENGTH_THRESHOLD:.2f}，季节模型可能不稳定。"
        )
    if report.valid_months < 12:
        risks.append("冷启动或新品风险：有效月份少于 12 个，可能缺少稳定历史模式。")
    if selection.logs:
        risks.extend(selection.logs)

    return list(dict.fromkeys(risks))


def _context_kwargs(context: Optional[Dict[str, Any]] = None, **overrides: Any) -> Dict[str, Any]:
    ctx = AgentContext.from_dict(context)
    kwargs = ctx.to_dict()
    kwargs.update({key: value for key, value in overrides.items() if value is not None})
    return kwargs


def get_forecast_result(
    level: Optional[str] = None,
    channel: Optional[str] = None,
    channel_subtype: Optional[str] = None,
    product_model: Optional[str] = None,
    horizon: Optional[int] = None,
    *,
    db_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    min_train_periods: Optional[int] = None,
    backtest_windows: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """查询聚合预测结果，不返回原始数据库明细。"""
    tool_name = "get_forecast_result"
    try:
        kwargs = _context_kwargs(
            context,
            db_path=db_path,
            level=level,
            channel=channel,
            channel_subtype=channel_subtype,
            product_model=product_model,
            horizon=horizon,
            min_train_periods=min_train_periods,
            backtest_windows=backtest_windows,
            limit=limit,
        )
        bundle = _prepare_analysis(**kwargs)
        interval_rows, interval_note = _prediction_interval(bundle.selection)
        forecasts = [
            {"月份": month, "预测值": round_float(value, 2)}
            for month, value in zip(bundle.selection.forecast_months, bundle.selection.forecast_values)
        ]
        return _success(
            tool_name,
            {
                "预测层级": _level_label(bundle.item["level"]),
                "预测对象名称": bundle.item["label"],
                "历史月份数": bundle.report.history_months,
                "最优模型": bundle.selection.selected_algorithm_label,
                "最优模型代码": bundle.selection.selected_algorithm,
                "未来预测值": forecasts,
                "预测区间": interval_rows,
                "预测区间说明": interval_note,
                "最近完整账期": bundle.report.recent_complete_month,
            },
        )
    except (DataAccessError, ValueError, RuntimeError) as exc:
        return _error(tool_name, exc)


def get_model_comparison(
    level: Optional[str] = None,
    channel: Optional[str] = None,
    channel_subtype: Optional[str] = None,
    product_model: Optional[str] = None,
    horizon: Optional[int] = None,
    *,
    db_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    min_train_periods: Optional[int] = None,
    backtest_windows: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """比较候选算法的启用状态和回测表现。"""
    tool_name = "get_model_comparison"
    try:
        kwargs = _context_kwargs(
            context,
            db_path=db_path,
            level=level,
            channel=channel,
            channel_subtype=channel_subtype,
            product_model=product_model,
            horizon=horizon,
            min_train_periods=min_train_periods,
            backtest_windows=backtest_windows,
            limit=limit,
        )
        bundle = _prepare_analysis(**kwargs)
        availability = algorithm_availability_for_series(bundle.selection.training_values)
        results_by_algorithm = {result.algorithm: result for result in bundle.selection.backtest_results}
        ranks = _rank_map(bundle.selection)
        algorithms = list(DISPLAY_ALGORITHMS)
        for algorithm in bundle.selection.candidate_algorithms:
            if algorithm not in algorithms:
                algorithms.append(algorithm)

        rows: List[Dict[str, Any]] = []
        for algorithm in algorithms:
            result = results_by_algorithm.get(algorithm)
            available = availability.get(
                algorithm,
                {
                    "enabled": algorithm in bundle.selection.candidate_algorithms,
                    "status": "已启用",
                    "reason": "保守兜底策略。",
                },
            )
            rows.append(
                {
                    "模型代码": algorithm,
                    "模型": ALGORITHM_LABELS.get(algorithm, algorithm),
                    "是否启用": bool(available.get("enabled")),
                    "状态": available.get("status"),
                    "排名": ranks.get(algorithm),
                    "WAPE": round_float(result.wape) if result else None,
                    "sMAPE": round_float(result.smape) if result else None,
                    "MAE": round_float(result.mae, 2) if result else None,
                    "Bias": round_float(result.bias) if result else None,
                    "回测窗口数": result.fold_count if result else 0,
                    "说明": (result.message if result and result.message else available.get("reason")),
                    "是否最优": algorithm == bundle.selection.selected_algorithm,
                }
            )

        return _success(
            tool_name,
            {
                "预测对象名称": bundle.item["label"],
                "所有候选模型": rows,
                "最优模型": bundle.selection.selected_algorithm_label,
                "最优模型代码": bundle.selection.selected_algorithm,
                "说明": _model_selection_note(bundle.selection),
            },
        )
    except (DataAccessError, ValueError, RuntimeError) as exc:
        return _error(tool_name, exc)


def get_predictability_evidence(
    level: Optional[str] = None,
    channel: Optional[str] = None,
    channel_subtype: Optional[str] = None,
    product_model: Optional[str] = None,
    horizon: Optional[int] = None,
    *,
    db_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    min_train_periods: Optional[int] = None,
    backtest_windows: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """获取可测性证据和风险提示，不返回原始明细。"""
    tool_name = "get_predictability_evidence"
    try:
        kwargs = _context_kwargs(
            context,
            db_path=db_path,
            level=level,
            channel=channel,
            channel_subtype=channel_subtype,
            product_model=product_model,
            horizon=horizon,
            min_train_periods=min_train_periods,
            backtest_windows=backtest_windows,
            limit=limit,
        )
        bundle = _prepare_analysis(**kwargs)
        risks = _risk_list(bundle)
        return _success(
            tool_name,
            {
                "预测对象名称": bundle.item["label"],
                "综合可测性得分": round_float(bundle.report.overall_score, 1),
                "可测性等级": bundle.report.level,
                "历史月份数": bundle.report.history_months,
                "缺失比例": round_float(bundle.report.missing_rate),
                "零值比例": round_float(bundle.report.zero_ratio),
                "CV": round_float(bundle.report.cv),
                "ADI": round_float(bundle.report.adi),
                "季节性强度": round_float(bundle.report.seasonality_strength),
                "趋势强度": round_float(bundle.report.trend_strength),
                "回测 WAPE": round_float(bundle.report.best_wape),
                "风险提示": risks,
                "可测性证据列表": list(bundle.report.evidence),
            },
        )
    except (DataAccessError, ValueError, RuntimeError) as exc:
        return _error(tool_name, exc)


def search_knowledge_base(
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search project documentation without accessing business data or SQLite."""
    tool_name = "search_knowledge_base"
    try:
        context = context or {}
        effective_query = str(query if query is not None else context.get("query") or "").strip()
        if not effective_query:
            raise ValueError("知识检索 query 不能为空。")
        effective_top_k = top_k if top_k is not None else context.get("top_k", 3)
        results = retrieve_knowledge(effective_query, top_k=effective_top_k)
        return _success(
            tool_name,
            {
                "query": effective_query,
                "results": results,
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        return _error(tool_name, exc)


def search_product_knowledge(
    query: Optional[str] = None,
    product_model: Optional[str] = None,
    top_k: Optional[int] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search structured product material without accessing SQLite business data."""
    tool_name = "search_product_knowledge"
    try:
        context = context or {}
        effective_query = str(query if query is not None else context.get("query") or "").strip()
        if not effective_query:
            raise ValueError("商品知识检索 query 不能为空。")
        effective_model = product_model if product_model is not None else context.get("product_model")
        effective_top_k = top_k if top_k is not None else context.get("top_k", 3)
        results = retrieve_product_knowledge(
            effective_query,
            product_model=effective_model,
            top_k=effective_top_k,
        )
        return _success(
            tool_name,
            {
                "query": effective_query,
                "product_model": str(effective_model or ""),
                "results": results,
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        return _error(tool_name, exc)


TOOL_REGISTRY = {
    "get_forecast_result": get_forecast_result,
    "get_model_comparison": get_model_comparison,
    "get_predictability_evidence": get_predictability_evidence,
    "search_knowledge_base": search_knowledge_base,
    "search_product_knowledge": search_product_knowledge,
}


def _parse_tool_arguments(tool_name: str, arguments: Optional[Any]) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具参数不是合法 JSON：{exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("工具参数 JSON 必须是对象。")
        return parsed
    raise ValueError(f"{tool_name} 的工具参数必须是 dict 或 JSON object string。")


def _filtered_dict(source: Optional[Dict[str, Any]], allowed_keys: set[str]) -> Dict[str, Any]:
    source = source or {}
    return {key: value for key, value in source.items() if key in allowed_keys and value is not None}


def execute_tool(
    tool_name: str,
    arguments: Optional[Any],
    protected_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a registered LLM tool with untrusted arguments and protected context."""
    if tool_name not in TOOL_REGISTRY:
        return _error(str(tool_name or ""), ValueError("未知工具"))

    try:
        parsed_arguments = _parse_tool_arguments(tool_name, arguments)
        allowed_argument_keys = TOOL_LLM_ARGUMENT_KEYS.get(tool_name, set())
        llm_context = _filtered_dict(parsed_arguments, allowed_argument_keys)
        protected = _filtered_dict(protected_context, PROTECTED_CONTEXT_KEYS)
        if tool_name == "search_product_knowledge" and protected.get("product_model") in {"", "全部"}:
            protected.pop("product_model", None)
        context = {**llm_context, **protected}
        return TOOL_REGISTRY[tool_name](context=context)
    except Exception as exc:
        return _error(tool_name, exc)


def generate_business_summary(
    forecast_result: Dict[str, Any],
    model_comparison: Dict[str, Any],
    predictability_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """固定模板业务摘要，输入必须来自工具返回结果。"""
    tool_name = "generate_business_summary"
    try:
        summary = build_business_summary(forecast_result, model_comparison, predictability_evidence)
        return _success(tool_name, summary)
    except (TypeError, ValueError, RuntimeError) as exc:
        return _error(tool_name, exc)


def compact_evidence_from_outputs(outputs: Iterable[Dict[str, Any]]) -> List[str]:
    evidence: List[str] = []
    for output in outputs:
        if not output.get("ok"):
            evidence.append(output.get("错误", "工具调用失败。"))
            continue
        if "预测对象名称" in output:
            evidence.append(f"预测对象：{output['预测对象名称']}")
        if "最优模型" in output:
            evidence.append(f"最优模型：{output['最优模型']}")
        if "综合可测性得分" in output:
            evidence.append(
                f"可测性：{output.get('综合可测性得分')}，等级 {output.get('可测性等级')}"
            )
        for item in output.get("可测性证据列表", [])[:3]:
            evidence.append(str(item))
        for item in output.get("风险提示", [])[:3]:
            evidence.append(str(item))
    return list(dict.fromkeys(evidence))
