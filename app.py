from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

try:
    import pandas as pd
except ImportError:  # pragma: no cover - Streamlit 环境通常会包含 pandas
    pd = None

from src.data_access import (
    DataAccessError,
    detect_global_latest_incomplete,
    ensure_database_readable,
    get_aggregated_series,
    get_default_db_path,
    get_distinct_values,
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
    MAX_OBJECTS_DEFAULT,
    MIN_TRAIN_PERIODS,
    MODEL_COMPLEXITY_RANK,
    SEASONAL_STRENGTH_THRESHOLD,
    WAPE_TIE_TOLERANCE,
    format_percent,
    normalize_points,
    round_float,
)
from 动销预测.agent import analyze_sales_question
from 动销预测.agent.tool_calling_orchestrator import run_tool_calling_agent
from 动销预测.llm import LLMConfig, run_grounded_llm_analysis
from 动销预测.llm.client import OpenAICompatibleChatClient


LEVEL_OPTIONS = {
    "渠道大类": "channel",
    "渠道细分类": "channel_detail",
    "渠道型号": "channel_model",
}

METRIC_HELP = {
    "WAPE": "整体加权绝对百分比误差，越低越好；真实值总和为 0 时不可计算。",
    "sMAPE": "对称平均绝对百分比误差，越低越好，适合辅助比较不同规模序列。",
    "MAE": "平均绝对误差，保留动销数量单位，越低越好。",
    "Bias": "预测偏差。正值表示整体高估，负值表示整体低估。",
}

AGENT_PRESET_QUESTIONS = {
    "分析未来趋势": "分析未来趋势，并说明预测结果、预测区间和主要风险。",
    "为什么选择该模型": "为什么选择该模型？请结合候选模型回测误差说明。",
    "查看数据质量": "查看数据质量和可测性证据。",
    "查看预测风险": "查看预测风险，并说明需要关注哪些业务边界。",
    "生成管理层摘要": "生成管理层摘要。",
    "为什么不建议强行预测": "为什么不建议强行预测？",
}

AGENT_MODE_TEXT = "当前模式：规则驱动智能分析（V2.1）"
AGENT_MODE_NOTE = "后续可接入 LLM 进行自然语言理解和报告润色，但预测数值仍必须来自本地工具。"
AGENT_SCOPE_NOTE = (
    "智能助手不直接生成预测数值。所有预测结果、模型选择、回测误差和可测性结论均来自本地预测工具与历史数据分析。"
)
CHANNEL_KEYWORDS = ("线上", "线下")
ANSWER_MODE_OPTIONS = {
    ":material/account_tree: Rule": "rule",
    ":material/fact_check: Grounded LLM": "llm",
    ":material/hub: Tool Calling Agent": "tool_calling",
}
ANSWER_MODE_DESCRIPTIONS = {
    "rule": "V2.1 · Python 关键词规则决定工具调用。",
    "llm": "V2.2 · 本地工具提供事实，LLM 负责受控解释。",
    "tool_calling": "LLM 根据问题自主选择并组合业务与知识工具。",
}
TOOL_UI_META = {
    "get_forecast_result": ("Forecast", "生成当前范围的销量预测结果", "analytics"),
    "get_model_comparison": ("Model Comparison", "比较候选模型与回测表现", "compare_arrows"),
    "get_predictability_evidence": ("Predictability", "读取数据质量与可测性证据", "fact_check"),
    "search_knowledge_base": ("Knowledge Base", "检索指标定义与业务规则", "menu_book"),
    "search_product_knowledge": ("Product Knowledge", "检索商品型号与产品资料", "inventory_2"),
}
EMPTY_ANALYSIS_ASSET = Path(__file__).resolve().parent / "assets" / "illustrations" / "empty-analysis.svg"
SENSITIVE_AGENT_TRACE_KEYS = {
    "db_path",
    "database_path",
    "api_key",
    "system_prompt",
    "min_train_periods",
    "backtest_windows",
    "limit",
}

APP_CSS = """
<style>
:root {
    --bg: #080D14;
    --sidebar: #0A111B;
    --surface: #0D1622;
    --surface-elevated: #111D2A;
    --border: rgba(148, 163, 184, 0.14);
    --text: #F1F5F9;
    --muted: #8B9BB0;
    --accent: #38BDF8;
    --accent-secondary: #7C8CFF;
    --success: #34D399;
    --warning: #FBBF24;
    --danger: #F87171;
}

[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1520px;
    padding: 1.4rem 2.25rem 4rem;
}

[data-testid="stSidebar"] {
    min-width: 272px;
    max-width: 272px;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

.workspace-header {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    min-height: 92px;
    gap: 1.5rem;
    padding: 0.2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

.workspace-eyebrow,
.section-kicker,
.sidebar-section-label {
    color: var(--accent);
    font-size: 0.75rem;
    line-height: 1.2;
    font-weight: 650;
    text-transform: uppercase;
}

.workspace-header h1 {
    margin: 0.3rem 0 0;
    color: var(--text);
    font-size: 2rem;
    line-height: 1.15;
    font-weight: 650;
    letter-spacing: 0;
}

.workspace-subtitle {
    margin-top: 0.45rem;
    color: var(--muted);
    font-size: 0.9rem;
}

.workspace-statuses {
    display: flex;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 0.45rem;
    max-width: 390px;
}

.status-badge,
.sidebar-status {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    color: #B7C5D5;
    background: rgba(148, 163, 184, 0.06);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 550;
}

.status-badge {
    min-height: 30px;
    padding: 0.3rem 0.68rem;
}

.sidebar-status {
    width: 100%;
    min-height: 34px;
    padding: 0.42rem 0.62rem;
    margin-bottom: 0.4rem;
    border-radius: 9px;
}

.status-dot {
    width: 7px;
    height: 7px;
    flex: 0 0 7px;
    border-radius: 50%;
    background: var(--muted);
}

.status-dot.ready { background: var(--success); }
.status-dot.waiting { background: var(--warning); }

.sidebar-brand {
    padding: 0 0 0.85rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.35rem;
}

.sidebar-brand strong {
    display: block;
    color: var(--text);
    font-size: 0.92rem;
    font-weight: 620;
}

.sidebar-brand span {
    display: block;
    color: var(--muted);
    font-size: 0.72rem;
    margin-top: 0.2rem;
}

.sidebar-section-label {
    color: #718399;
    margin: 1rem 0 0.48rem;
}

.section-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.2rem 0 0.7rem;
}

.section-heading h2 {
    margin: 0.22rem 0 0;
    color: var(--text);
    font-size: 1.25rem;
    line-height: 1.25;
    font-weight: 600;
}

.section-heading p {
    margin: 0;
    color: var(--muted);
    font-size: 0.78rem;
}

.assistant-boundary {
    color: var(--muted);
    font-size: 0.78rem;
    line-height: 1.6;
    margin-top: 0.45rem;
}

.context-strip,
.summary-strip,
.metadata-bar {
    display: grid;
    gap: 0;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
}

.context-strip { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.summary-strip { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.metadata-bar { grid-template-columns: repeat(4, minmax(0, 1fr)); }

.stat-cell,
.metadata-cell {
    min-width: 0;
    padding: 0.8rem 0.9rem;
    border-right: 1px solid var(--border);
}

.stat-cell:last-child,
.metadata-cell:last-child { border-right: 0; }

.stat-label,
.metadata-label {
    display: block;
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 500;
    margin-bottom: 0.28rem;
}

.stat-value {
    display: block;
    overflow: hidden;
    color: var(--text);
    font-size: 1.05rem;
    line-height: 1.3;
    font-weight: 600;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.metadata-value {
    color: #C8D4E1;
    font-size: 0.78rem;
    line-height: 1.45;
}

.answer-surface {
    padding: 1.05rem 1.15rem;
    background: var(--surface-elevated);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 14px;
}

.answer-surface .question-label {
    color: var(--muted);
    font-size: 0.75rem;
    margin-bottom: 0.35rem;
}

.activity-item {
    position: relative;
    display: grid;
    grid-template-columns: 30px minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.65rem;
    min-height: 52px;
    padding: 0.55rem 0.65rem;
    margin-bottom: 0.35rem;
    background: rgba(13, 22, 34, 0.72);
    border: 1px solid var(--border);
    border-radius: 10px;
}

.activity-icon {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    color: var(--accent);
    background: rgba(56, 189, 248, 0.09);
    border-radius: 8px;
    font-size: 0.9rem;
}

.activity-icon .material-symbols-rounded { font-size: 1rem; }

.activity-name {
    color: var(--text);
    font-size: 0.86rem;
    font-weight: 600;
}

.activity-purpose {
    color: var(--muted);
    font-size: 0.74rem;
    margin-top: 0.12rem;
}

.activity-state {
    color: var(--success);
    font-size: 0.72rem;
    font-weight: 600;
}

.activity-state.failed { color: var(--danger); }

.empty-state {
    display: grid;
    place-items: center;
    min-height: 220px;
    padding: 1.2rem;
    text-align: center;
    background: rgba(13, 22, 34, 0.45);
    border: 1px dashed rgba(148, 163, 184, 0.18);
    border-radius: 14px;
}

.empty-state h3 {
    margin: 0.6rem 0 0.25rem;
    color: var(--text);
    font-size: 1rem;
    border: 0;
}

.empty-state p {
    margin: 0;
    color: var(--muted);
    font-size: 0.8rem;
}

/* Stable Streamlit test IDs are used only where native APIs cannot express the layout. */
[data-testid="stRadio"] div[role="radiogroup"] {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.55rem;
}

[data-testid="stRadioOption"] {
    min-height: 84px;
    margin: 0 !important;
    padding: 0.78rem 0.82rem !important;
    align-items: flex-start !important;
    background: rgba(13, 22, 34, 0.7);
    border: 1px solid var(--border);
    border-radius: 10px;
}

[data-testid="stRadioOption"][data-selected="true"] {
    background: var(--surface-elevated) !important;
    border-color: rgba(56, 189, 248, 0.75) !important;
    box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.08);
}

[data-testid="stRadioOption"] p { line-height: 1.35; }
[data-testid="stRadioOption"] small { color: var(--muted); }

[data-testid="stTextArea"] textarea {
    min-height: 68px !important;
    background: var(--surface-elevated);
    border-radius: 10px;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.2) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    border-radius: 12px !important;
}

@media (max-width: 1100px) {
    .workspace-header { grid-template-columns: 1fr; }
    .workspace-statuses { justify-content: flex-start; max-width: none; }
    .context-strip { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .summary-strip,
    .metadata-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .stat-cell:nth-child(3),
    .metadata-cell:nth-child(3) { border-left: 0; }
}

@media (max-width: 700px) {
    [data-testid="stAppViewContainer"] > .main .block-container {
        padding: 1rem 0.9rem 3rem;
    }
    .workspace-header h1 { font-size: 1.65rem; }
    .context-strip,
    .summary-strip,
    .metadata-bar,
    [data-testid="stRadio"] div[role="radiogroup"] {
        grid-template-columns: 1fr;
    }
    .stat-cell,
    .metadata-cell { border-right: 0; border-bottom: 1px solid var(--border); }
    .stat-cell:last-child,
    .metadata-cell:last-child { border-bottom: 0; }
}
</style>
"""


def apply_ui_theme() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_page_header(*, db_ready: bool, knowledge_ready: bool, agent_ready: bool) -> None:
    statuses = (
        ("SQLite Connected" if db_ready else "SQLite Offline", db_ready),
        ("Knowledge Ready" if knowledge_ready else "Knowledge Unavailable", knowledge_ready),
        ("Agent Enabled" if agent_ready else "Agent Local", agent_ready),
    )
    badges = "".join(
        f'<span class="status-badge"><span class="status-dot {"ready" if ready else "waiting"}"></span>{label}</span>'
        for label, ready in statuses
    )
    st.markdown(
        f"""
        <div class="workspace-header">
          <div>
            <div class="workspace-eyebrow">OMS Intelligence</div>
            <h1>Sales Forecasting Workspace</h1>
            <div class="workspace-subtitle">结合本地预测工具、商品知识库与 Tool Calling Agent 的智能分析工作台</div>
          </div>
          <div class="workspace-statuses">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_label(label: str) -> None:
    st.markdown(f'<div class="sidebar-section-label">{label}</div>', unsafe_allow_html=True)


def render_sidebar_status(label: str, ready: bool) -> None:
    state = "ready" if ready else "waiting"
    st.markdown(
        f'<div class="sidebar-status"><span class="status-dot {state}"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def render_section_heading(kicker: str, title: str, description: str = "") -> None:
    description_html = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f"""
        <div class="section-heading">
          <div><div class="section-kicker">{escape(kicker)}</div><h2>{escape(title)}</h2></div>
          {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _filters_to_tuple(filters: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((key, value) for key, value in filters.items() if value and value != "全部"))


def _tuple_to_filters(filters_tuple: Tuple[Tuple[str, str], ...]) -> Dict[str, str]:
    return dict(filters_tuple)


@st.cache_data(ttl=600)
def cached_distinct_values(db_path: str, field_key: str, filters_tuple, limit: int = 200):
    return get_distinct_values(db_path, field_key, _tuple_to_filters(filters_tuple), limit=limit)


@st.cache_data(ttl=600)
def cached_global_monthly_summary(db_path: str):
    return get_global_monthly_summary(db_path)


@st.cache_data(ttl=600)
def cached_aggregated_series(db_path: str, level: str, filters_tuple, limit: int):
    return get_aggregated_series(db_path, level, _tuple_to_filters(filters_tuple), limit=limit)


def dataframe(data):
    if pd is not None:
        return pd.DataFrame(data)
    return data


def percent_text(value: Optional[float]) -> str:
    return format_percent(value)


def signed_percent_text(value: Optional[float]) -> str:
    if value is None:
        return "不可计算"
    return f"{value:+.2%}"


def number_text(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "不可计算"
    return f"{value:.{digits}f}"


def quantile(values: Sequence[float], q: float) -> Optional[float]:
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


def prediction_interval(selection):
    """用最佳模型回测残差的 10%/90% 分位数生成经验预测区间。"""
    best = selection.best_backtest
    if best is None or len(best.fold_errors) < 3:
        return None, "历史回测不足，暂不提供预测区间。"

    residuals = [-error for error in best.fold_errors]  # actual - predicted
    lower_residual = quantile(residuals, 0.10)
    upper_residual = quantile(residuals, 0.90)
    if lower_residual is None or upper_residual is None:
        return None, "历史回测不足，暂不提供预测区间。"

    intervals = []
    for month, forecast_value in zip(selection.forecast_months, selection.forecast_values):
        lower = max(0.0, forecast_value + lower_residual)
        upper = max(lower, forecast_value + upper_residual)
        intervals.append({"月份": month, "下界": lower, "上界": upper})
    return intervals, "经验预测区间基于滚动回测残差 10% 和 90% 分位数，不代表准确率保证。"


def render_intro():
    render_section_heading("Data Contract", "数据口径", "查看当前分析使用的数据定义与可信度边界")
    st.markdown(
        """
        <div class="metadata-bar">
          <div class="metadata-cell">
            <span class="metadata-label">数据来源</span>
            <span class="metadata-value">SQLite · adb_model_sales_summary</span>
          </div>
          <div class="metadata-cell">
            <span class="metadata-label">预测目标</span>
            <span class="metadata-value">qty_total</span>
          </div>
          <div class="metadata-cell">
            <span class="metadata-label">分析维度</span>
            <span class="metadata-value">账期 · 渠道 · 型号 · 品类</span>
          </div>
          <div class="metadata-cell">
            <span class="metadata-label">可测性边界</span>
            <span class="metadata-value">数据支撑程度，不代表准确率保证</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("查看完整数据口径说明", expanded=False):
        st.markdown(
            "- **数据来源**：SQLite 表 `adb_model_sales_summary`。\n"
            "- **字段映射**：`fin_year_month` 为时间字段，`channel` 为渠道大类，"
            "`channel_` 为渠道细分类，`product_model` 为型号，`product_category` 为品类，"
            "`product_class` 为细分类，`qty_total` 为预测目标。\n"
            "- **口径提醒**：当前以 `qty_total` 作为动销数量口径，最终项目交付前需要与业务方确认该字段是否与动销定义完全一致。\n"
            "- **可测性提醒**：可测性表示当前数据对预测的支撑程度，不等同于保证准确率；实际预测可信度应结合滚动回测误差判断。"
        )


def build_summary_row(item, selection, report):
    forecasts = selection.forecast_values + [0.0] * max(0, 6 - len(selection.forecast_values))
    return {
        "预测对象": item["label"],
        "最佳算法": selection.selected_algorithm_label,
        "可测性等级": report.level,
        "综合可测性得分": round_float(report.overall_score, 1),
        "历史月份数": report.history_months,
        "零值比例": percent_text(report.zero_ratio),
        "WAPE": percent_text(report.best_wape),
        "sMAPE": percent_text(report.best_smape),
        "未来 1 月预测": round_float(forecasts[0], 2),
        "未来 3 月预测": round_float(sum(forecasts[:3]), 2),
        "未来 6 月预测": round_float(sum(forecasts[:6]), 2),
    }


def valid_backtests(selection):
    return [
        result
        for result in selection.backtest_results
        if result.fold_count > 0 and result.primary_score != float("inf")
    ]


def ranked_backtests(selection):
    return sorted(
        valid_backtests(selection),
        key=lambda result: (
            result.primary_score,
            result.smape,
            MODEL_COMPLEXITY_RANK.get(result.algorithm, 999),
        ),
    )


def metric_name_for(result) -> str:
    return "WAPE" if result and result.wape is not None else "MAE"


def metric_value_text(result, metric_name: str) -> str:
    if result is None:
        return "不可计算"
    if metric_name == "WAPE":
        return percent_text(result.wape)
    return number_text(result.mae)


def model_selection_reason(selection) -> str:
    selected = selection.best_backtest
    if selected is None:
        return "历史回测窗口不足，系统使用当前可用的保守算法生成预测，并在可测性中提示风险。"

    ranked = ranked_backtests(selection)
    metric_name = metric_name_for(selected)
    score_text = metric_value_text(selected, metric_name)
    if not ranked:
        return "没有可比较的有效回测结果，使用当前可用模型。"

    metric_best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    selected_is_metric_best = metric_best.algorithm == selected.algorithm

    if second is None:
        return f"{selected.algorithm_label} 是唯一形成有效回测窗口的模型，{metric_name} 为 {score_text}。"

    if selected_is_metric_best:
        second_score = metric_value_text(second, metric_name_for(second))
        reason = (
            f"{selected.algorithm_label} 的 {metric_name} 为 {score_text}，"
            f"优于 {second.algorithm_label} 的 {metric_name_for(second)} {second_score}。"
        )
    else:
        best_score = metric_value_text(metric_best, metric_name_for(metric_best))
        reason = (
            f"{metric_best.algorithm_label} 的 {metric_name_for(metric_best)} 为 {best_score}，"
            f"{selected.algorithm_label} 的 {metric_name} 为 {score_text}。两者差异在容忍范围内，"
            f"因此按简单稳定优先原则选择 {selected.algorithm_label}。"
        )

    if selected.wape is not None and second.wape is not None:
        best_wape = min(selected.wape, second.wape)
        larger_wape = max(selected.wape, second.wape)
        if best_wape == 0:
            close = larger_wape <= WAPE_TIE_TOLERANCE
        else:
            close = larger_wape <= best_wape * (1 + WAPE_TIE_TOLERANCE)
        if close and selected.algorithm != second.algorithm:
            reason += " 最优与第二名 WAPE 差异较小，模型复杂度和稳定性被纳入最终判断。"
    return reason


def render_model_selection_card(detail):
    selection = detail["selection"]
    best = selection.best_backtest
    primary_metric = metric_name_for(best)

    st.markdown("### 自动模型选择说明")
    with st.container(border=True):
        cols = st.columns(5)
        cols[0].metric("最优算法", selection.selected_algorithm_label)
        cols[1].metric("主指标", primary_metric, help="优先使用 WAPE；若真实值总和为 0，则改用 MAE。")
        cols[2].metric("WAPE", percent_text(best.wape if best else None), help=METRIC_HELP["WAPE"])
        cols[3].metric("sMAPE", percent_text(best.smape if best else None), help=METRIC_HELP["sMAPE"])
        cols[4].metric("回测窗口", best.fold_count if best else 0)

        metric_cols = st.columns(2)
        metric_cols[0].metric("MAE", number_text(best.mae if best else None), help=METRIC_HELP["MAE"])
        metric_cols[1].metric("Bias", signed_percent_text(best.bias if best else None), help=METRIC_HELP["Bias"])

        st.markdown(f"**选择原因：** {model_selection_reason(selection)}")

    with st.expander("指标解释", expanded=False):
        st.markdown(
            "- **WAPE**：整体加权绝对百分比误差，越低越好。\n"
            "- **sMAPE**：对称平均绝对百分比误差，越低越好。\n"
            "- **MAE**：平均绝对误差，保留动销数量单位。\n"
            "- **Bias**：预测偏差；正值表示整体高估，负值表示整体低估。"
        )


def render_chart(detail):
    selection = detail["selection"]
    interval_rows, interval_message = prediction_interval(selection)

    chart_rows = []
    for month, value in zip(selection.training_months, selection.training_values):
        chart_rows.append({"月份": month, "系列": "历史动销", "数量": value})
    for month, value in zip(selection.forecast_months, selection.forecast_values):
        chart_rows.append({"月份": month, "系列": "点预测", "数量": value})
    if interval_rows:
        for row in interval_rows:
            chart_rows.append(
                {
                    "月份": row["月份"],
                    "系列": "经验预测区间",
                    "下界": row["下界"],
                    "上界": row["上界"],
                }
            )

    legend_html = """
    <div style="display:flex; gap:1.2rem; align-items:center; margin:0.4rem 0 0.8rem 0;">
      <span><span style="display:inline-block;width:24px;height:3px;background:#69a9ff;margin-right:6px;"></span>历史动销</span>
      <span><span style="display:inline-block;width:24px;border-top:3px dashed #45b8d4;margin-right:6px;"></span>点预测</span>
      <span><span style="display:inline-block;width:24px;height:10px;background:rgba(122,167,184,.28);margin-right:6px;"></span>经验预测区间</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    if pd is not None:
        chart_df = pd.DataFrame(chart_rows)
        sort_order = selection.training_months + selection.forecast_months
        spec = {
            "height": 360,
            "layer": [
                {
                    "transform": [{"filter": "datum['系列'] == '经验预测区间'"}],
                    "mark": {"type": "area", "opacity": 0.22, "color": "#7aa7b8"},
                    "encoding": {
                        "x": {"field": "月份", "type": "ordinal", "sort": sort_order, "title": "账期"},
                        "y": {"field": "下界", "type": "quantitative", "title": "动销数量"},
                        "y2": {"field": "上界"},
                    },
                },
                {
                    "transform": [{"filter": "datum['系列'] == '历史动销'"}],
                    "mark": {"type": "line", "strokeWidth": 3, "color": "#69a9ff"},
                    "encoding": {
                        "x": {"field": "月份", "type": "ordinal", "sort": sort_order, "title": "账期"},
                        "y": {"field": "数量", "type": "quantitative", "title": "动销数量"},
                    },
                },
                {
                    "transform": [{"filter": "datum['系列'] == '点预测'"}],
                    "mark": {
                        "type": "line",
                        "strokeWidth": 3,
                        "strokeDash": [8, 5],
                        "color": "#45b8d4",
                    },
                    "encoding": {
                        "x": {"field": "月份", "type": "ordinal", "sort": sort_order, "title": "账期"},
                        "y": {"field": "数量", "type": "quantitative", "title": "动销数量"},
                    },
                },
                {
                    "transform": [{"filter": "datum['系列'] == '点预测'"}],
                    "mark": {"type": "point", "filled": True, "size": 60, "color": "#45b8d4"},
                    "encoding": {
                        "x": {"field": "月份", "type": "ordinal", "sort": sort_order},
                        "y": {"field": "数量", "type": "quantitative"},
                    },
                },
            ],
            "resolve": {"scale": {"y": "shared"}},
        }
        st.vega_lite_chart(chart_df, spec, use_container_width=True)
    else:
        st.write(chart_rows)

    if interval_rows:
        st.caption(interval_message)
        st.dataframe(
            dataframe(
                [
                    {
                        "月份": row["月份"],
                        "点预测": round_float(value, 2),
                        "经验下界": round_float(row["下界"], 2),
                        "经验上界": round_float(row["上界"], 2),
                    }
                    for row, value in zip(interval_rows, selection.forecast_values)
                ]
            ),
            use_container_width=True,
        )
    else:
        st.info(interval_message)


def risk_list(detail, latest_info=None) -> List[str]:
    report = detail["report"]
    selection = detail["selection"]
    risks = []

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
    if latest_info and latest_info.get("is_incomplete"):
        risks.extend(latest_info.get("reasons", []))
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


def render_predictability_evidence(detail, latest_info=None):
    report = detail["report"]
    risks = risk_list(detail, latest_info)

    st.markdown("### 数据质量与可测性证据")
    score_cols = st.columns(4)
    score_cols[0].metric("综合可测性得分", f"{report.overall_score:.1f}")
    score_cols[1].metric("可测性等级", report.level)
    score_cols[2].metric("回测 WAPE", percent_text(report.best_wape), help=METRIC_HELP["WAPE"])
    score_cols[3].metric(
        "误差稳定性",
        number_text(report.error_stability, 3),
        help="不同回测窗口残差的波动程度，越低表示误差越稳定。",
    )

    rows = [
        {
            "历史月份数": report.history_months,
            "有效月份数": report.valid_months,
            "缺失月份比例": percent_text(report.missing_rate),
            "零值比例": percent_text(report.zero_ratio),
            "CV": number_text(report.cv, 3),
            "ADI": number_text(report.adi, 3),
            "季节性强度": number_text(report.seasonality_strength, 3),
            "趋势强度": number_text(report.trend_strength, 3),
            "最新完整账期": report.recent_complete_month or "无",
            "回测 WAPE": percent_text(report.best_wape),
            "回测误差稳定性": number_text(report.error_stability, 3),
        }
    ]
    st.dataframe(dataframe(rows), use_container_width=True)

    st.markdown("**证据列表**")
    for evidence in report.evidence:
        st.markdown(f"- {evidence}")

    st.markdown("**风险提示**")
    if risks:
        for risk in risks:
            st.warning(risk)
    else:
        st.success("未发现明显预测风险，但仍需结合业务节奏和回测误差判断。")


def backtest_rank_map(selection) -> Dict[str, int]:
    ranked = ranked_backtests(selection)
    return {result.algorithm: index + 1 for index, result in enumerate(ranked)}


def render_backtest_table(detail):
    selection = detail["selection"]
    availability = algorithm_availability_for_series(selection.training_values)
    results_by_algorithm = {result.algorithm: result for result in selection.backtest_results}
    ranks = backtest_rank_map(selection)
    display_algorithms = list(DISPLAY_ALGORITHMS)
    for algorithm in selection.candidate_algorithms:
        if algorithm not in display_algorithms:
            display_algorithms.append(algorithm)

    rows = []
    for algorithm in display_algorithms:
        result = results_by_algorithm.get(algorithm)
        available = availability.get(
            algorithm,
            {"enabled": algorithm in selection.candidate_algorithms, "status": "已启用", "reason": "保守兜底策略。"},
        )

        if result is None:
            status = available["status"]
            note = available["reason"]
            row = {
                "排名": "-",
                "选中": "是" if algorithm == selection.selected_algorithm else "",
                "算法": ALGORITHM_LABELS.get(algorithm, algorithm),
                "状态": status,
                "WAPE": "未计算",
                "sMAPE": "未计算",
                "MAE": "未计算",
                "Bias": "未计算",
                "回测窗口数": 0,
                "误差稳定性": "未计算",
                "说明": note or "未启用或不适用。",
            }
        else:
            if result.fold_count == 0:
                status = "不适用" if result.status in {"skipped", "failed"} else result.status
            else:
                status = "已计算" if result.status == "ok" else "部分成功"
            note = result.message or available["reason"] or "已完成滚动回测。"
            row = {
                "排名": ranks.get(algorithm, "-"),
                "选中": "是" if algorithm == selection.selected_algorithm else "",
                "算法": result.algorithm_label,
                "状态": status,
                "WAPE": percent_text(result.wape),
                "sMAPE": percent_text(result.smape),
                "MAE": number_text(result.mae),
                "Bias": signed_percent_text(result.bias),
                "回测窗口数": result.fold_count,
                "误差稳定性": number_text(result.error_stability, 3),
                "说明": note or "已完成滚动回测。",
            }
        rows.append(row)

    table_data = dataframe(rows)
    if pd is not None:
        styled = table_data.style.apply(
            lambda row: ["background-color: #dcfce7; font-weight: 600;" if row["选中"] == "是" else "" for _ in row],
            axis=1,
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(table_data, use_container_width=True)


def build_agent_context(
    *,
    db_path: str,
    level_label: str,
    level: str,
    filters: Dict[str, str],
    forecast_horizon: int,
    backtest_windows: int,
    min_train_periods: int,
    selected_detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fields = {}
    if selected_detail:
        fields = selected_detail.get("item", {}).get("fields", {}) or {}

    channel = fields.get("渠道大类") or filters.get("channel") or "全部"
    channel_subtype = fields.get("渠道细分类") or filters.get("channel_detail") or "全部"
    product_model = fields.get("型号") or filters.get("product_model") or "全部"

    return {
        "db_path": db_path,
        "database_path": db_path,
        "level": level_label,
        "level_key": level,
        "channel": channel,
        "channel_detail": channel_subtype,
        "channel_subtype": channel_subtype,
        "product_model": product_model,
        "horizon": int(forecast_horizon),
        "forecast_horizon": int(forecast_horizon),
        "backtest_windows": int(backtest_windows),
        "min_train_periods": int(min_train_periods),
        "limit": 1,
    }


def call_agent_analysis(
    question: str,
    context: Dict[str, Any],
    *,
    allow_channel_override: bool = False,
    response_mode: str = "rule",
    llm_config: Optional[LLMConfig] = None,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    clean_question = (question or "").strip()
    if not clean_question:
        raise ValueError("请输入业务问题后再开始智能分析。")
    final_context, validation = prepare_agent_context_for_question(
        clean_question,
        context,
        allow_channel_override=allow_channel_override,
    )
    if validation["blocked"]:
        return build_context_conflict_response(validation)

    if response_mode == "tool_calling":
        config = llm_config or LLMConfig.from_env()
        if config.mode != "llm":
            return build_tool_calling_error_response(
                "LLM Tool Calling Agent 未启用，请将 LLM_MODE 设置为 llm。",
                final_context,
            )
        if not config.is_complete():
            missing = "、".join(config.missing_fields())
            return build_tool_calling_error_response(
                f"LLM 配置不完整，缺少 {missing}。",
                final_context,
            )

        client = llm_client or OpenAICompatibleChatClient()
        try:
            agent_response = run_tool_calling_agent(
                clean_question,
                final_context,
                client,
                config,
            )
        except Exception as exc:  # pragma: no cover - 第三方客户端或测试替身兜底
            message = str(exc) or exc.__class__.__name__
            if config.api_key:
                message = message.replace(config.api_key, "[redacted]")
            return build_tool_calling_error_response(
                f"LLM Tool Calling Agent 执行失败：{message}",
                final_context,
            )

        result = dict(agent_response or {})
        result.setdefault("ok", False)
        result.setdefault("answer", "")
        result.setdefault("tool_calls", [])
        result.setdefault("iterations", 0)
        result.setdefault("error", None)
        result.setdefault("analysis_context", {})
        result["intent"] = "llm_tool_calling"
        result["evidence"] = []
        result["disclaimer"] = AGENT_SCOPE_NOTE
        result["answer_mode_requested"] = "tool_calling"
        result["answer_mode_effective"] = "tool_calling" if result.get("ok") else "unavailable"
        if result.get("answer"):
            result = with_actual_object_answer(
                result,
                final_context,
                validation.get("override_channel"),
            )
        return result

    if response_mode == "llm":
        grounded_result = run_grounded_llm_analysis(
            clean_question,
            final_context,
            config=llm_config,
            client=llm_client,
            rule_analyzer=analyze_sales_question,
        )
        rule_response = with_actual_object_answer(
            grounded_result["rule_response"],
            final_context,
            validation.get("override_channel"),
        )
        response = dict(rule_response)
        response["answer_mode_requested"] = "llm"
        response["answer_mode_effective"] = "llm" if grounded_result["llm_status"].get("used") else "rule"
        response["llm_status"] = grounded_result["llm_status"]
        response["llm_explanation"] = grounded_result.get("llm_explanation")
        response["fact_pack"] = grounded_result.get("fact_pack")
        return response

    response = analyze_sales_question(clean_question, final_context)
    result = with_actual_object_answer(response, final_context, validation.get("override_channel"))
    result["answer_mode_requested"] = "rule"
    result["answer_mode_effective"] = "rule"
    result["llm_status"] = {
        "requested": False,
        "used": False,
        "mode": "rule",
        "message": "规则驱动分析（V2.1）。",
        "fallback_reason": "",
    }
    return result


def build_tool_calling_error_response(message: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": False,
        "answer": "",
        "tool_calls": [],
        "iterations": 0,
        "error": message,
        "intent": "llm_tool_calling",
        "evidence": [],
        "disclaimer": AGENT_SCOPE_NOTE,
        "analysis_context": {
            "level": context.get("level"),
            "channel": context.get("channel"),
            "channel_subtype": context.get("channel_subtype") or context.get("channel_detail"),
            "product_model": context.get("product_model"),
            "horizon": context.get("horizon"),
        },
        "answer_mode_requested": "tool_calling",
        "answer_mode_effective": "unavailable",
    }


def detect_question_object_keywords(question: str) -> Dict[str, Any]:
    text = question or ""
    channels = [keyword for keyword in CHANNEL_KEYWORDS if keyword in text]
    return {
        "channels": channels,
        "channel": channels[0] if len(channels) == 1 else None,
        "mentions_channel_level": "渠道大类" in text,
        "mentions_channel_detail": "渠道细分类" in text or "渠道细分" in text,
        "mentions_model": "型号" in text,
    }


def _clean_context_object_value(value: Any) -> str:
    text = str(value or "").strip()
    return text or "全部"


def prepare_agent_context_for_question(
    question: str,
    context: Dict[str, Any],
    *,
    allow_channel_override: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    final_context = dict(context or {})
    detected = detect_question_object_keywords(question)
    requested_channel = detected.get("channel")
    current_channel = _clean_context_object_value(final_context.get("channel"))

    validation = {
        "blocked": False,
        "question_object": requested_channel,
        "current_channel": current_channel,
        "override_channel": None,
        "detected": detected,
    }

    if requested_channel and current_channel not in {"全部", requested_channel}:
        if not allow_channel_override:
            validation["blocked"] = True
            return final_context, validation
        final_context["channel"] = requested_channel
        validation["override_channel"] = requested_channel
    elif requested_channel and current_channel == "全部":
        final_context["channel"] = requested_channel

    return final_context, validation


def build_context_conflict_response(validation: Dict[str, Any]) -> Dict[str, Any]:
    requested = validation.get("question_object") or "未识别"
    current = validation.get("current_channel") or "全部"
    warning = (
        f"问题中请求分析：{requested}；\n"
        f"当前左侧筛选对象：{current}；\n"
        "为避免错误分析，系统未执行预测。请调整左侧筛选条件后重新分析。"
    )
    return {
        "blocked": True,
        "intent": "context_conflict",
        "tool_calls": [],
        "evidence": [],
        "answer": warning,
        "disclaimer": "当前问题对象与筛选对象不一致，未调用预测工具。",
    }


def actual_object_text(context: Dict[str, Any]) -> str:
    return (
        f"预测层级={_clean_context_object_value(context.get('level'))}；"
        f"渠道大类={_clean_context_object_value(context.get('channel'))}；"
        f"渠道细分类={_clean_context_object_value(context.get('channel_subtype') or context.get('channel_detail'))}；"
        f"型号={_clean_context_object_value(context.get('product_model'))}；"
        f"预测步长={_clean_context_object_value(context.get('horizon'))}个月；"
        f"回测窗口={_clean_context_object_value(context.get('backtest_windows'))}个月"
    )


def with_actual_object_answer(
    response: Dict[str, Any],
    context: Dict[str, Any],
    override_channel: Optional[str] = None,
) -> Dict[str, Any]:
    result = dict(response or {})
    answer = str(result.get("answer") or "本次智能分析未返回结论。")
    actual_line = f"实际分析对象：{actual_object_text(context)}"

    if answer.startswith("实际分析对象："):
        lines = answer.splitlines()
        rest = "\n".join(lines[1:]).lstrip()
        answer = lines[0]
        if override_channel:
            answer += f"\n\n本次分析对象已由问题文本覆盖为：{override_channel}。"
        if rest:
            answer += f"\n\n{rest}"
    else:
        prefix = actual_line
        if override_channel:
            prefix += f"\n\n本次分析对象已由问题文本覆盖为：{override_channel}。"
        answer = f"{prefix}\n\n{answer}"

    result["answer"] = answer
    result["analysis_context"] = {
        "level": context.get("level"),
        "channel": context.get("channel"),
        "channel_subtype": context.get("channel_subtype") or context.get("channel_detail"),
        "product_model": context.get("product_model"),
        "horizon": context.get("horizon"),
        "backtest_windows": context.get("backtest_windows"),
    }
    return result


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _sanitize_agent_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_agent_trace_value(item)
            for key, item in value.items()
            if str(key).strip().lower() not in SENSITIVE_AGENT_TRACE_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_agent_trace_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_agent_trace_value(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if any(key in lowered for key in SENSITIVE_AGENT_TRACE_KEYS):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return "[已隐藏受保护内容]"
            return _sanitize_agent_trace_value(parsed)
    return value


def build_tool_call_trace_rows(tool_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    rows = []
    for index, record in enumerate(tool_calls, start=1):
        if not isinstance(record, dict):
            continue
        arguments = record.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = _sanitize_agent_trace_value(arguments)
        rows.append(
            {
                "index": index,
                "tool_name": str(record.get("tool_name") or "未知工具"),
                "arguments": _sanitize_agent_trace_value(arguments),
                "ok": bool(record.get("ok")),
                "result": _sanitize_agent_trace_value(record.get("result", {})),
            }
        )
    return rows


def render_agent_context_card(
    context: Dict[str, Any],
    mode_label: str,
    response_mode: str,
    strategy_label: str = "自动选择",
):
    values = (
        ("预测层级", context.get("level", "未选择")),
        ("渠道", context.get("channel", "全部")),
        ("渠道细分类", context.get("channel_subtype", "全部")),
        ("型号", context.get("product_model", "全部")),
        ("预测步长", f"{context.get('horizon', '-')} 个月"),
        ("策略", strategy_label),
    )
    cells = "".join(
        f'<div class="stat-cell"><span class="stat-label">{escape(label)}</span>'
        f'<span class="stat-value" title="{escape(str(value))}">{escape(str(value))}</span></div>'
        for label, value in values
    )
    render_section_heading("Current Context", "当前分析上下文", ANSWER_MODE_DESCRIPTIONS[response_mode])
    st.markdown(f'<div class="context-strip">{cells}</div>', unsafe_allow_html=True)


def render_agent_metric_snapshot(selected_detail: Optional[Dict[str, Any]]):
    if not selected_detail:
        return

    selection = selected_detail["selection"]
    report = selected_detail["report"]
    best = selection.best_backtest

    render_section_heading("Forecast Signal", "当前预测对象", "已运行预测后显示关键模型与误差指标")
    with st.container(border=True):
        metric_cols = st.columns(5)
        metric_cols[0].metric("预测对象", selected_detail["item"]["label"])
        metric_cols[1].metric("最优模型", selection.selected_algorithm_label)
        metric_cols[2].metric("WAPE", percent_text(best.wape if best else report.best_wape), help=METRIC_HELP["WAPE"])
        metric_cols[3].metric("sMAPE", percent_text(best.smape if best else report.best_smape), help=METRIC_HELP["sMAPE"])
        metric_cols[4].metric("Bias", signed_percent_text(best.bias if best else None), help=METRIC_HELP["Bias"])


def _summary_strip(values: Sequence[Tuple[str, Any]]) -> None:
    cells = "".join(
        f'<div class="stat-cell"><span class="stat-label">{escape(label)}</span>'
        f'<span class="stat-value">{escape(str(value))}</span></div>'
        for label, value in values
    )
    st.markdown(f'<div class="summary-strip">{cells}</div>', unsafe_allow_html=True)


def _rule_trace_rows(tool_calls: Sequence[str]) -> List[Dict[str, Any]]:
    return [
        {
            "index": index,
            "tool_name": tool_name,
            "arguments": {},
            "result": {},
            "ok": True,
        }
        for index, tool_name in enumerate(tool_calls, start=1)
    ]


def render_tool_activity(trace_rows: Sequence[Dict[str, Any]], final_ok: bool) -> None:
    render_section_heading("Tool Activity", "工具活动", "仅展示结构化调用轨迹，不展示内部思维过程")
    if not trace_rows:
        st.info("本次回答未调用业务工具。")
    for row in trace_rows:
        tool_name = str(row.get("tool_name") or "未知工具")
        title, purpose, icon = TOOL_UI_META.get(tool_name, (tool_name, "执行已注册的业务工具", "build"))
        ok = bool(row.get("ok"))
        state_class = "" if ok else " failed"
        state_text = "成功" if ok else "失败"
        st.markdown(
            f"""
            <div class="activity-item">
              <div class="activity-icon"><span class="material-symbols-rounded">{escape(icon)}</span></div>
              <div><div class="activity-name">{escape(title)}</div><div class="activity-purpose">{escape(purpose)}</div></div>
              <div class="activity-state{state_class}">{state_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if row.get("arguments") or row.get("result"):
            with st.expander(f"{row.get('index', '-')}. {title} · 参数与 Observation", expanded=False):
                detail_cols = st.columns(2)
                with detail_cols[0]:
                    st.caption("Arguments")
                    st.json(row.get("arguments", {}))
                with detail_cols[1]:
                    st.caption("Observation")
                    st.json(row.get("result", {}))

    final_class = "" if final_ok else " failed"
    final_text = "已生成" if final_ok else "未完成"
    st.markdown(
        f"""
        <div class="activity-item">
          <div class="activity-icon"><span class="material-symbols-rounded">done_all</span></div>
          <div><div class="activity-name">Final Answer</div><div class="activity-purpose">基于可用 Tool Observation 生成最终回答</div></div>
          <div class="activity-state{final_class}">{final_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_analysis() -> None:
    with st.container(border=True):
        empty_cols = st.columns([1, 2, 1])
        with empty_cols[1]:
            if EMPTY_ANALYSIS_ASSET.is_file():
                st.image(str(EMPTY_ANALYSIS_ASSET), width=180)
            st.markdown(
                '<div style="text-align:center"><strong>等待第一次分析</strong><br>'
                '<span style="color:#8B9BB0;font-size:0.8rem">输入业务问题、确认回答模式，然后运行智能分析。</span></div>',
                unsafe_allow_html=True,
            )


def render_agent_result(response: Dict[str, Any], question: str):
    answer = response.get("answer") or "本次智能分析未返回结论。"
    intent = response.get("intent") or "未识别"
    evidence = _as_text_list(response.get("evidence"))
    disclaimer = response.get("disclaimer") or "预测结果仅供业务分析参考。"
    llm_explanation = response.get("llm_explanation")
    llm_status = response.get("llm_status")
    is_tool_calling = response.get("answer_mode_requested") == "tool_calling"
    tool_calls = [] if is_tool_calling else _as_text_list(response.get("tool_calls"))
    trace_rows = (
        build_tool_call_trace_rows(response.get("tool_calls"))
        if is_tool_calling
        else _rule_trace_rows(tool_calls)
    )
    blocked = bool(response.get("blocked"))
    succeeded = bool(response.get("ok", not blocked)) and bool(response.get("answer"))
    iterations = response.get("iterations", "-") if is_tool_calling else "-"

    render_section_heading("Analysis Summary", "分析摘要", "本次 Agent 执行状态与证据范围")
    _summary_strip(
        (
            ("Agent 状态", "成功" if succeeded else "受限" if blocked else "失败"),
            ("Tool 调用", len(trace_rows)),
            ("LLM 迭代", iterations),
            ("风险边界", "已声明" if disclaimer else "未提供"),
        )
    )

    if response.get("error"):
        st.error(response["error"])

    render_section_heading("Final Answer", "最终回答", f"Intent · {intent}")
    with st.container(border=True):
        st.caption(f"问题 · {question}")
        if response.get("answer"):
            st.markdown(answer)
        else:
            st.warning("Agent 未返回最终回答。")

    if llm_explanation:
        render_llm_explanation(llm_explanation)

    if evidence:
        with st.expander("Supporting Evidence · 分析证据", expanded=False):
            for item in evidence:
                st.markdown(f"- {item}")

    render_tool_activity(trace_rows, succeeded)
    st.warning(f"**预测与数据风险边界**\n\n{disclaimer}")
    render_llm_status(llm_status)


def render_llm_explanation(explanation: Dict[str, Any]):
    st.markdown("### LLM 业务解读")
    with st.container(border=True):
        st.markdown("**管理层摘要**")
        st.markdown(explanation.get("executive_summary", ""))
        st.markdown("**业务解读**")
        st.markdown(explanation.get("business_interpretation", ""))

        rec_cols = st.columns(2)
        with rec_cols[0]:
            st.markdown("**风险建议**")
            for item in explanation.get("risk_recommendations", []):
                st.markdown(f"- {item}")
        with rec_cols[1]:
            st.markdown("**建议继续追问**")
            for item in explanation.get("follow_up_questions", []):
                st.markdown(f"- {item}")

        refs = explanation.get("evidence_references", [])
        if refs:
            st.caption("引用事实字段：" + "；".join(str(item) for item in refs))


def render_llm_status(status: Optional[Dict[str, Any]]):
    if not status:
        return
    st.markdown("### LLM 调用状态")
    if status.get("used"):
        st.success(status.get("message", "LLM 增强解释已启用。"))
    elif status.get("requested"):
        message = status.get("message", "LLM 未使用，已回退到规则驱动分析。")
        reason = status.get("fallback_reason")
        if reason:
            st.warning(f"{message}\n\n原因：{reason}")
        else:
            st.warning(message)
    else:
        st.info(status.get("message", "当前使用规则驱动分析。"))


def render_agent_assistant(
    *,
    context: Dict[str, Any],
    db_ready: bool,
    selected_detail: Optional[Dict[str, Any]] = None,
    strategy_label: str = "自动选择",
):
    if "agent_question" not in st.session_state:
        st.session_state["agent_question"] = "分析线下渠道未来 3 个月动销趋势，并说明模型选择原因。"

    render_section_heading("Ask OMS Intelligence", "智能分析", "从业务问题开始，Agent 将按所选模式调用本地工具")
    with st.container(border=True):
        st.caption("常用工作流")
        button_cols = st.columns(3)
        for index, (label, preset_question) in enumerate(AGENT_PRESET_QUESTIONS.items()):
            if button_cols[index % 3].button(label, use_container_width=True):
                st.session_state["agent_question"] = preset_question

        question = st.text_area(
            "请输入业务问题",
            key="agent_question",
            placeholder="例如：分析线下渠道未来 3 个月动销趋势，并说明模型选择原因。",
            height=76,
            label_visibility="collapsed",
        )

        control_cols = st.columns([4, 1])
        with control_cols[0]:
            allow_channel_override = st.checkbox(
                "允许问题中的渠道条件覆盖当前筛选",
                value=False,
                help="默认关闭。开启后，问题中出现线上/线下时可覆盖当前 context 的渠道大类。",
            )
        with control_cols[1]:
            submit_analysis = st.button(
                "运行智能分析",
                type="primary",
                disabled=not db_ready,
                use_container_width=True,
            )

        if not db_ready:
            st.info("数据库可读后即可使用智能分析助手。")
        st.markdown(
            f'<div class="assistant-boundary">{AGENT_SCOPE_NOTE}</div>',
            unsafe_allow_html=True,
        )

    render_section_heading("Reasoning Policy", "回答模式", "选择 Agent 如何获取事实并组织回答")
    mode_label = st.radio(
        "回答模式选择",
        list(ANSWER_MODE_OPTIONS.keys()),
        captions=[ANSWER_MODE_DESCRIPTIONS[value] for value in ANSWER_MODE_OPTIONS.values()],
        horizontal=True,
        label_visibility="collapsed",
        key="agent_response_mode",
        width="stretch",
    )
    response_mode = ANSWER_MODE_OPTIONS[mode_label]

    llm_config = LLMConfig.from_env()
    if response_mode in {"llm", "tool_calling"}:
        if llm_config.mode != "llm":
            if response_mode == "llm":
                st.warning("当前 LLM_MODE 未设置为 llm，将自动使用规则驱动分析。")
            else:
                st.warning("当前 LLM_MODE 未设置为 llm，Tool Calling Agent 暂不可用。")
        elif not llm_config.is_complete():
            missing = "、".join(llm_config.missing_fields())
            if response_mode == "llm":
                st.warning(f"LLM 配置不完整，缺少 {missing}，将自动使用规则驱动分析。")
            else:
                st.warning(f"LLM 配置不完整，缺少 {missing}，Tool Calling Agent 暂不可用。")

    render_agent_context_card(context, mode_label, response_mode, strategy_label)
    render_agent_metric_snapshot(selected_detail)

    if submit_analysis:
        try:
            spinner_text = {
                "rule": "正在调用本地规则驱动 Agent...",
                "llm": "正在执行受控 LLM 增强解释...",
                "tool_calling": "正在运行 LLM Tool Calling Agent...",
            }[response_mode]
            with st.spinner(spinner_text):
                response = call_agent_analysis(
                    question,
                    context,
                    allow_channel_override=allow_channel_override,
                    response_mode=response_mode,
                    llm_config=llm_config,
                )
            st.session_state["agent_last_response"] = response
            st.session_state["agent_last_question"] = question
        except Exception as exc:  # pragma: no cover - Streamlit 页面兜底提示
            st.error(f"智能分析失败：{exc} 请检查数据库路径、筛选条件和历史数据是否可用。")

    last_response = st.session_state.get("agent_last_response")
    last_question = st.session_state.get("agent_last_question", question)
    if last_response:
        render_agent_result(last_response, last_question)
    else:
        render_empty_analysis()


def main():
    st.set_page_config(
        page_title="OMS Sales Forecasting Workspace",
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_ui_theme()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <strong>OMS Intelligence</strong>
              <span>Workspace controls</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_sidebar_label("Data")
        db_path = st.text_input(
            "数据库路径",
            value=get_default_db_path(),
            help="可通过环境变量 OMS_SQLITE_PATH 设置默认路径。",
        )

        filters: Dict[str, str] = {}
        db_ready = False
        if db_path:
            try:
                row_info = ensure_database_readable(db_path)
                db_ready = True
                render_sidebar_status(f"Database connected · {row_info['row_count']:,}", True)
            except DataAccessError as exc:
                st.error(str(exc))
        else:
            st.info("请设置 OMS_SQLITE_PATH 或输入数据库路径。")
        if not db_ready:
            render_sidebar_status("Database waiting", False)
        render_sidebar_status("Agent ready" if db_ready else "Agent waiting for data", db_ready)

        render_sidebar_label("Scope")
        level_label = st.selectbox("预测层级", list(LEVEL_OPTIONS.keys()), index=0)
        level = LEVEL_OPTIONS[level_label]

        if db_ready:
            try:
                channel_options = ["全部"] + cached_distinct_values(db_path, "channel", tuple())
                filters["channel"] = st.selectbox("渠道大类筛选", channel_options)

                detail_filter_base = _filters_to_tuple({"channel": filters["channel"]})
                channel_detail_options = ["全部"] + cached_distinct_values(
                    db_path, "channel_detail", detail_filter_base
                )
                filters["channel_detail"] = st.selectbox("渠道细分类筛选", channel_detail_options)

                model_filter_base = _filters_to_tuple(
                    {"channel": filters["channel"], "channel_detail": filters["channel_detail"]}
                )
                if level == "channel_model":
                    model_options = ["全部"] + cached_distinct_values(
                        db_path, "product_model", model_filter_base, limit=300
                    )
                    filters["product_model"] = st.selectbox("型号筛选", model_options)
                else:
                    filters["product_model"] = "全部"
                    st.selectbox("型号筛选", ["全部"], help="渠道大类/细分类层级默认不按型号过滤。")
            except DataAccessError as exc:
                st.error(str(exc))
                db_ready = False
        else:
            st.selectbox("渠道大类筛选", ["全部"])
            st.selectbox("渠道细分类筛选", ["全部"])
            st.selectbox("型号筛选", ["全部"])

        render_sidebar_label("Forecast")
        forecast_horizon = st.selectbox("预测步长选择", [1, 3, 6], index=2)
        strategy_label = st.selectbox("算法策略选择", ["自动选择", "手动指定"])
        strategy = "auto" if strategy_label == "自动选择" else "manual"
        manual_algorithm = None
        if strategy == "manual":
            manual_label = st.selectbox(
                "手动指定算法",
                [label for key, label in ALGORITHM_LABELS.items() if key not in ("zero", "mean")],
            )
            manual_algorithm = {label: key for key, label in ALGORITHM_LABELS.items()}[manual_label]

        min_train_periods = st.number_input(
            "最少训练期（月）",
            min_value=6,
            max_value=36,
            value=MIN_TRAIN_PERIODS,
            step=1,
        )
        backtest_windows = st.slider(
            "回测窗口（月）",
            min_value=3,
            max_value=12,
            value=DEFAULT_BACKTEST_WINDOWS,
        )
        max_objects = st.slider(
            "最多分析对象数",
            min_value=1,
            max_value=MAX_OBJECTS_DEFAULT,
            value=MAX_OBJECTS_DEFAULT,
            help="渠道型号层级默认只分析筛选条件下前 50 个对象，建议继续筛选后再计算。",
        )
        start = st.button(
            "开始分析与预测",
            type="primary",
            disabled=not db_ready,
            use_container_width=True,
        )

    knowledge_ready = (Path(__file__).resolve().parent / "knowledge").is_dir()
    llm_config = LLMConfig.from_env()
    llm_ready = llm_config.mode == "llm" and llm_config.is_complete()
    render_page_header(db_ready=db_ready, knowledge_ready=knowledge_ready, agent_ready=db_ready or llm_ready)

    selected_detail: Optional[Dict[str, Any]] = None
    agent_context = build_agent_context(
        db_path=db_path,
        level_label=level_label,
        level=level,
        filters=filters,
        forecast_horizon=int(forecast_horizon),
        backtest_windows=int(backtest_windows),
        min_train_periods=int(min_train_periods),
    )
    agent_context_signature = (
        db_path,
        level_label,
        filters.get("channel", "全部"),
        filters.get("channel_detail", "全部"),
        filters.get("product_model", "全部"),
        int(forecast_horizon),
        int(backtest_windows),
        int(min_train_periods),
    )
    if st.session_state.get("agent_context_signature") == agent_context_signature:
        agent_context = st.session_state.get("agent_context", agent_context)

    agent_tab, prediction_tab = st.tabs(
        [":material/auto_awesome: AI 分析工作台", ":material/monitoring: 预测分析"]
    )

    with prediction_tab:
        if not start:
            st.caption("设置筛选条件后点击“开始分析与预测”。")
        else:
            try:
                monthly_summary = cached_global_monthly_summary(db_path)
                latest_info = detect_global_latest_incomplete(monthly_summary)
                if latest_info.get("is_incomplete"):
                    for reason in latest_info.get("reasons", []):
                        st.warning(reason)

                series_list = cached_aggregated_series(db_path, level, _filters_to_tuple(filters), max_objects)
            except DataAccessError as exc:
                st.error(str(exc))
                series_list = []
                latest_info = None

            if not series_list:
                st.warning("当前筛选条件下没有可分析的聚合序列。")
            else:
                if len(series_list) >= max_objects:
                    st.info(
                        f"当前仅展示符合筛选条件的前 {max_objects} 个对象。"
                        "若处于渠道型号层级，建议继续选择渠道细分类或型号以减少计算范围。"
                    )

                details = []
                progress = st.progress(0, text="正在执行滚动回测与预测...")
                for index, item in enumerate(series_list, start=1):
                    completed = normalize_points(item["points"])
                    selection = select_model_and_forecast(
                        completed.months,
                        completed.values,
                        horizon=max(6, forecast_horizon),
                        strategy=strategy,
                        manual_algorithm=manual_algorithm,
                        min_train_periods=int(min_train_periods),
                        backtest_windows=int(backtest_windows),
                        global_latest_info=latest_info,
                    )
                    report = assess_predictability(completed, selection, latest_info)
                    details.append({"item": item, "completed": completed, "selection": selection, "report": report})
                    progress.progress(index / len(series_list), text=f"已完成 {index}/{len(series_list)} 个对象")
                progress.empty()

                summary_rows = [
                    build_summary_row(detail["item"], detail["selection"], detail["report"]) for detail in details
                ]
                st.markdown("### 预测结果")
                st.dataframe(dataframe(summary_rows), use_container_width=True)

                labels = [detail["item"]["label"] for detail in details]
                selected_label = st.selectbox("选择对象查看详情", labels)
                selected_detail = details[labels.index(selected_label)]
                agent_context = build_agent_context(
                    db_path=db_path,
                    level_label=level_label,
                    level=level,
                    filters=filters,
                    forecast_horizon=int(forecast_horizon),
                    backtest_windows=int(backtest_windows),
                    min_train_periods=int(min_train_periods),
                    selected_detail=selected_detail,
                )
                st.session_state["agent_context"] = agent_context
                st.session_state["agent_context_signature"] = agent_context_signature

                render_model_selection_card(selected_detail)

                st.markdown("### 历史动销、点预测与经验预测区间")
                render_chart(selected_detail)

                render_predictability_evidence(selected_detail, latest_info)

                st.markdown("### 候选算法回测误差对比")
                render_backtest_table(selected_detail)

    with agent_tab:
        render_agent_assistant(
            context=agent_context,
            db_ready=db_ready,
            selected_detail=selected_detail,
            strategy_label=strategy_label,
        )

    render_intro()


if __name__ == "__main__":
    main()
