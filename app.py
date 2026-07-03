from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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
    st.markdown("### 数据口径说明")
    st.markdown(
        """
        <div style="line-height:1.85; padding:1rem 1.1rem; border:1px solid #dbeafe;
                    border-radius:8px; background:#eff6ff;">
        <p><strong>数据来源：</strong>SQLite 表 <code>adb_model_sales_summary</code>。</p>
        <p><strong>字段映射：</strong><code>fin_year_month</code> 为时间字段，
        <code>channel</code> 为渠道大类，<code>channel_</code> 为渠道细分类，
        <code>product_model</code> 为型号，<code>product_category</code> 为品类，
        <code>product_class</code> 为细分类，<code>qty_total</code> 为预测目标。</p>
        <p><strong>口径提醒：</strong>当前以 <code>qty_total</code> 作为动销数量口径，
        最终项目交付前需要与业务方确认该字段是否与动销定义完全一致。</p>
        <p><strong>可测性提醒：</strong>可测性表示当前数据对预测的支撑程度，
        不等同于保证准确率；实际预测可信度应结合滚动回测误差判断。</p>
        </div>
        """,
        unsafe_allow_html=True,
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
      <span><span style="display:inline-block;width:24px;height:3px;background:#2563eb;margin-right:6px;"></span>历史动销</span>
      <span><span style="display:inline-block;width:24px;border-top:3px dashed #f97316;margin-right:6px;"></span>点预测</span>
      <span><span style="display:inline-block;width:24px;height:10px;background:rgba(245,158,11,.25);margin-right:6px;"></span>经验预测区间</span>
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
                    "mark": {"type": "area", "opacity": 0.22, "color": "#f59e0b"},
                    "encoding": {
                        "x": {"field": "月份", "type": "ordinal", "sort": sort_order, "title": "账期"},
                        "y": {"field": "下界", "type": "quantitative", "title": "动销数量"},
                        "y2": {"field": "上界"},
                    },
                },
                {
                    "transform": [{"filter": "datum['系列'] == '历史动销'"}],
                    "mark": {"type": "line", "strokeWidth": 3, "color": "#2563eb"},
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
                        "color": "#f97316",
                    },
                    "encoding": {
                        "x": {"field": "月份", "type": "ordinal", "sort": sort_order, "title": "账期"},
                        "y": {"field": "数量", "type": "quantitative", "title": "动销数量"},
                    },
                },
                {
                    "transform": [{"filter": "datum['系列'] == '点预测'"}],
                    "mark": {"type": "point", "filled": True, "size": 60, "color": "#f97316"},
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


def main():
    st.set_page_config(page_title="OMS 动销预测与可测性分析", layout="wide")
    st.title("OMS 动销预测与可测性分析")

    with st.sidebar:
        st.header("分析设置")
        db_path = st.text_input(
            "数据库路径",
            value=get_default_db_path(),
            help="可通过环境变量 OMS_SQLITE_PATH 设置默认路径。",
        )
        level_label = st.selectbox("预测层级", list(LEVEL_OPTIONS.keys()), index=0)
        level = LEVEL_OPTIONS[level_label]

        filters: Dict[str, str] = {}
        db_ready = False
        if db_path:
            try:
                row_info = ensure_database_readable(db_path)
                st.caption(f"数据库可读，源表记录数：{row_info['row_count']:,}")
                db_ready = True
            except DataAccessError as exc:
                st.error(str(exc))
        else:
            st.info("请设置 OMS_SQLITE_PATH 或输入数据库路径。")

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
        start = st.button("开始分析与预测", type="primary", disabled=not db_ready)

    render_intro()

    if not start:
        st.caption("设置筛选条件后点击“开始分析与预测”。")
        return

    try:
        monthly_summary = cached_global_monthly_summary(db_path)
        latest_info = detect_global_latest_incomplete(monthly_summary)
        if latest_info.get("is_incomplete"):
            for reason in latest_info.get("reasons", []):
                st.warning(reason)

        series_list = cached_aggregated_series(db_path, level, _filters_to_tuple(filters), max_objects)
    except DataAccessError as exc:
        st.error(str(exc))
        return

    if not series_list:
        st.warning("当前筛选条件下没有可分析的聚合序列。")
        return

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

    summary_rows = [build_summary_row(detail["item"], detail["selection"], detail["report"]) for detail in details]
    st.markdown("### 预测结果")
    st.dataframe(dataframe(summary_rows), use_container_width=True)

    labels = [detail["item"]["label"] for detail in details]
    selected_label = st.selectbox("选择对象查看详情", labels)
    selected_detail = details[labels.index(selected_label)]

    render_model_selection_card(selected_detail)

    st.markdown("### 历史动销、点预测与经验预测区间")
    render_chart(selected_detail)

    render_predictability_evidence(selected_detail, latest_info)

    st.markdown("### 候选算法回测误差对比")
    render_backtest_table(selected_detail)


if __name__ == "__main__":
    main()
