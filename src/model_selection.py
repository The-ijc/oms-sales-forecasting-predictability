from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .backtest import BacktestResult, walk_forward_backtest
from .forecasting import ForecastError, forecast, forecast_with_fallback
from .utils import (
    ADI_INTERMITTENT,
    ALGORITHM_LABELS,
    DEFAULT_BACKTEST_WINDOWS,
    MIN_TRAIN_PERIODS,
    MODEL_COMPLEXITY_RANK,
    RANDOM_FOREST_MIN_MONTHS,
    RANDOM_FOREST_MIN_NONZERO,
    SEASONAL_MIN_MONTHS,
    SEASONAL_STRENGTH_THRESHOLD,
    SHORT_HISTORY_MONTHS,
    SMAPE_TIE_TOLERANCE,
    WAPE_TIE_TOLERANCE,
    ZERO_RATIO_HIGH,
    add_months,
    adi,
    detect_latest_incomplete,
    nonzero_values,
    seasonality_strength,
    zero_ratio,
)


CONSERVATIVE_ALGORITHMS = ["naive", "mean", "zero"]
STANDARD_ALGORITHMS = [
    "naive",
    "moving_average_3",
    "ses",
    "holt_linear",
    "linear_trend",
]

DISPLAY_ALGORITHMS = [
    "naive",
    "moving_average_3",
    "ses",
    "holt_linear",
    "linear_trend",
    "seasonal_naive",
    "random_forest_lag",
    "croston_sba",
]


@dataclass
class ModelSelectionResult:
    selected_algorithm: str
    selected_algorithm_label: str
    candidate_algorithms: List[str]
    backtest_results: List[BacktestResult]
    forecast_months: List[str]
    forecast_values: List[float]
    training_months: List[str]
    training_values: List[float]
    excluded_latest_month: Optional[str] = None
    logs: List[str] = field(default_factory=list)

    @property
    def best_backtest(self) -> Optional[BacktestResult]:
        for result in self.backtest_results:
            if result.algorithm == self.selected_algorithm and result.fold_count > 0:
                return result
        return None


def candidate_algorithms_for_series(values: Sequence[float]) -> List[str]:
    """根据历史长度、零值比例、ADI、季节性选择候选算法。"""
    values = list(values)
    if len(values) < SHORT_HISTORY_MONTHS:
        return [algorithm for algorithm in CONSERVATIVE_ALGORITHMS if algorithm != "naive" or values]

    candidates = list(STANDARD_ALGORITHMS)
    zr = zero_ratio(values)
    series_adi = adi(values)
    if zr >= ZERO_RATIO_HIGH or series_adi >= ADI_INTERMITTENT:
        candidates.append("croston_sba")

    if len(values) >= SEASONAL_MIN_MONTHS and seasonality_strength(values) >= SEASONAL_STRENGTH_THRESHOLD:
        candidates.append("seasonal_naive")

    if len(values) >= RANDOM_FOREST_MIN_MONTHS and len(nonzero_values(values)) >= RANDOM_FOREST_MIN_NONZERO:
        candidates.append("random_forest_lag")

    return candidates


def algorithm_availability_for_series(values: Sequence[float]) -> dict:
    """返回所有展示算法的启用状态和中文原因。"""
    values = list(values)
    candidates = set(candidate_algorithms_for_series(values))
    zr = zero_ratio(values)
    series_adi = adi(values)
    seasonal = seasonality_strength(values)
    nonzero_count = len(nonzero_values(values))
    availability = {}

    for algorithm in DISPLAY_ALGORITHMS:
        enabled = algorithm in candidates
        reason = "已纳入候选模型，将参与滚动回测。"
        status = "已启用"

        if not enabled:
            status = "未启用"
            if len(values) < SHORT_HISTORY_MONTHS and algorithm not in CONSERVATIVE_ALGORITHMS:
                reason = f"历史不足 {SHORT_HISTORY_MONTHS} 个月，只启用保守策略。"
            elif algorithm == "seasonal_naive":
                if len(values) < SEASONAL_MIN_MONTHS:
                    reason = f"历史不足 {SEASONAL_MIN_MONTHS} 个月，不适合季节性模型。"
                else:
                    reason = (
                        f"季节性强度 {seasonal:.2f} 低于阈值 "
                        f"{SEASONAL_STRENGTH_THRESHOLD:.2f}。"
                    )
            elif algorithm == "random_forest_lag":
                if len(values) < RANDOM_FOREST_MIN_MONTHS:
                    reason = f"历史不足 {RANDOM_FOREST_MIN_MONTHS} 个月，滞后特征样本不足。"
                elif nonzero_count < RANDOM_FOREST_MIN_NONZERO:
                    reason = (
                        f"非零样本数 {nonzero_count} 少于 "
                        f"{RANDOM_FOREST_MIN_NONZERO}，不启用 Random Forest。"
                    )
                else:
                    reason = "样本条件未达到 Random Forest 启用要求。"
            elif algorithm == "croston_sba":
                reason = (
                    f"零值比例 {zr:.2%}、ADI {series_adi:.2f} 未达到间歇性需求阈值。"
                )
            else:
                reason = "当前序列条件不适用该算法。"

        availability[algorithm] = {
            "enabled": enabled,
            "status": status,
            "reason": reason,
        }

    return availability


def _within_tolerance(value: float, best: float, tolerance: float) -> bool:
    if best == 0:
        return value <= tolerance
    return value <= best * (1 + tolerance)


def _select_from_backtests(results: Sequence[BacktestResult]) -> Optional[BacktestResult]:
    valid = [
        result
        for result in results
        if result.fold_count > 0 and result.primary_score != float("inf")
    ]
    if not valid:
        return None

    best_score = min(result.primary_score for result in valid)
    shortlist = [
        result
        for result in valid
        if _within_tolerance(result.primary_score, best_score, WAPE_TIE_TOLERANCE)
    ]
    best_smape = min(result.smape for result in shortlist)
    smape_shortlist = [
        result
        for result in shortlist
        if _within_tolerance(result.smape, best_smape, SMAPE_TIE_TOLERANCE)
    ]
    return sorted(
        smape_shortlist,
        key=lambda result: (
            MODEL_COMPLEXITY_RANK.get(result.algorithm, 999),
            result.primary_score,
            result.smape,
        ),
    )[0]


def _future_months(last_month: Optional[str], horizon: int) -> List[str]:
    if not last_month:
        return []
    return [add_months(last_month, step) for step in range(1, horizon + 1)]


def select_model_and_forecast(
    months: Sequence[str],
    values: Sequence[float],
    horizon: int = 6,
    strategy: str = "auto",
    manual_algorithm: Optional[str] = None,
    min_train_periods: int = MIN_TRAIN_PERIODS,
    backtest_windows: int = DEFAULT_BACKTEST_WINDOWS,
    global_latest_info: Optional[dict] = None,
) -> ModelSelectionResult:
    """自动回测选择模型并输出未来预测。"""
    months = list(months)
    values = list(values)
    logs: List[str] = []

    if not months:
        return ModelSelectionResult(
            selected_algorithm="zero",
            selected_algorithm_label=ALGORITHM_LABELS["zero"],
            candidate_algorithms=["zero"],
            backtest_results=[],
            forecast_months=[],
            forecast_values=[],
            training_months=[],
            training_values=[],
            logs=["序列为空，使用零预测。"],
        )

    excluded_latest_month = None
    local_incomplete, local_reasons = detect_latest_incomplete(months, values)
    global_incomplete = bool(global_latest_info and global_latest_info.get("is_incomplete"))
    global_latest_month = global_latest_info.get("latest_month") if global_latest_info else None
    should_exclude_latest = local_incomplete or (
        global_incomplete and global_latest_month and months[-1] == global_latest_month
    )

    training_months = months[:]
    training_values = values[:]
    if should_exclude_latest and len(months) > 1:
        excluded_latest_month = months[-1]
        training_months = months[:-1]
        training_values = values[:-1]
        logs.extend(local_reasons)
        if global_incomplete and global_latest_month == excluded_latest_month:
            logs.extend(global_latest_info.get("reasons", []))
        logs.append(f"已从回测和训练中排除最新疑似不完整月份：{excluded_latest_month}。")

    if strategy == "manual" and manual_algorithm:
        if len(training_values) < SHORT_HISTORY_MONTHS and manual_algorithm not in CONSERVATIVE_ALGORITHMS:
            candidates = candidate_algorithms_for_series(training_values)
            logs.append("历史不足 6 个月，手动算法被限制为保守策略。")
        else:
            candidates = [manual_algorithm]
    else:
        candidates = candidate_algorithms_for_series(training_values)

    if not candidates:
        candidates = ["zero"]

    backtest_results = walk_forward_backtest(
        training_values,
        candidates,
        min_train_periods=min_train_periods,
        backtest_windows=backtest_windows,
    )
    selected_backtest = _select_from_backtests(backtest_results)
    selected_algorithm = selected_backtest.algorithm if selected_backtest else candidates[0]

    try:
        forecast_values = forecast(training_values, horizon, selected_algorithm)
    except ForecastError as exc:
        logs.append(f"{ALGORITHM_LABELS.get(selected_algorithm, selected_algorithm)} 预测失败：{exc}")
        selected_algorithm = "naive" if training_values else "zero"
        logs.append(f"已降级为 {ALGORITHM_LABELS[selected_algorithm]}。")
        forecast_values = forecast_with_fallback(training_values, horizon, selected_algorithm)

    forecast_months = _future_months(training_months[-1] if training_months else None, horizon)
    return ModelSelectionResult(
        selected_algorithm=selected_algorithm,
        selected_algorithm_label=ALGORITHM_LABELS.get(selected_algorithm, selected_algorithm),
        candidate_algorithms=list(candidates),
        backtest_results=backtest_results,
        forecast_months=forecast_months,
        forecast_values=forecast_values,
        training_months=training_months,
        training_values=training_values,
        excluded_latest_month=excluded_latest_month,
        logs=logs,
    )
