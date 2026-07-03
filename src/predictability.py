from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .model_selection import ModelSelectionResult
from .utils import (
    ADI_INTERMITTENT,
    BACKTEST_WEIGHT,
    HIGH_PREDICTABILITY_SCORE,
    MEDIUM_PREDICTABILITY_SCORE,
    STRUCTURE_WEIGHT,
    ZERO_RATIO_HIGH,
    coefficient_of_variation,
    grade_from_score,
    mean,
    nonzero_cv2,
    outlier_ratio,
    round_float,
    seasonality_strength,
    stddev,
    trend_strength,
    zero_ratio,
    adi,
)


@dataclass
class PredictabilityReport:
    history_months: int
    valid_months: int
    missing_months: int
    missing_rate: float
    zero_ratio: float
    mean_qty: float
    std_qty: float
    cv: float
    adi: float
    nonzero_cv2: float
    outlier_ratio: float
    seasonality_strength: float
    trend_strength: float
    recent_complete_month: Optional[str]
    best_wape: Optional[float]
    best_smape: Optional[float]
    best_mae: Optional[float]
    best_bias: Optional[float]
    backtest_windows: int
    error_stability: Optional[float]
    structure_score: float
    backtest_score: float
    overall_score: float
    level: str
    evidence: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict:
        return {
            "历史月份数": self.history_months,
            "有效月份数": self.valid_months,
            "缺失月份数": self.missing_months,
            "缺失比例": round_float(self.missing_rate),
            "零值比例": round_float(self.zero_ratio),
            "动销均值": round_float(self.mean_qty, 2),
            "动销标准差": round_float(self.std_qty, 2),
            "CV": round_float(self.cv),
            "ADI": round_float(self.adi),
            "非零需求CV²": round_float(self.nonzero_cv2),
            "异常值比例": round_float(self.outlier_ratio),
            "季节性强度": round_float(self.seasonality_strength),
            "趋势强度": round_float(self.trend_strength),
            "最近完整月份": self.recent_complete_month,
            "最佳WAPE": round_float(self.best_wape),
            "最佳sMAPE": round_float(self.best_smape),
            "最佳MAE": round_float(self.best_mae, 2),
            "最佳Bias": round_float(self.best_bias),
            "回测窗口数量": self.backtest_windows,
            "误差稳定性": round_float(self.error_stability),
            "结构质量得分": round_float(self.structure_score, 1),
            "回测表现得分": round_float(self.backtest_score, 1),
            "综合可测性得分": round_float(self.overall_score, 1),
            "可测性等级": self.level,
        }


def _volatility_score(cv_value: float) -> float:
    if cv_value == float("inf"):
        return 2.0
    if cv_value <= 0.5:
        return 20.0
    if cv_value <= 1.0:
        return 15.0
    if cv_value <= 2.0:
        return 8.0
    return 3.0


def _error_score(metric: Optional[float]) -> float:
    if metric is None:
        return 35.0
    if metric <= 0.20:
        return 100.0
    if metric <= 0.35:
        return 85.0
    if metric <= 0.50:
        return 70.0
    if metric <= 0.80:
        return 50.0
    if metric <= 1.20:
        return 30.0
    return 15.0


def _stability_score(stability: Optional[float]) -> float:
    if stability is None:
        return 35.0
    if stability <= 0.20:
        return 100.0
    if stability <= 0.50:
        return 80.0
    if stability <= 1.00:
        return 55.0
    if stability <= 1.50:
        return 35.0
    return 15.0


def _structure_score(
    history_months: int,
    missing_rate: float,
    zero_rate: float,
    cv_value: float,
    season_strength: float,
    trend_strength_value: float,
) -> float:
    history_score = min(history_months / 24, 1.0) * 25
    completeness_score = max(0.0, 1 - missing_rate) * 20
    activity_score = max(0.0, 1 - zero_rate) * 20
    volatility = _volatility_score(cv_value)
    pattern_base = max(season_strength, trend_strength_value, 0.35 if history_months >= 12 else 0.10)
    pattern_score = min(pattern_base, 1.0) * 15
    return max(0.0, min(100.0, history_score + completeness_score + activity_score + volatility + pattern_score))


def _backtest_score(best_wape, best_mae, avg_qty, stability, has_backtest: bool) -> float:
    if not has_backtest:
        return 30.0
    metric = best_wape
    if metric is None:
        metric = best_mae / avg_qty if avg_qty else None
    return 0.8 * _error_score(metric) + 0.2 * _stability_score(stability)


def assess_predictability(
    completed_series,
    selection: ModelSelectionResult,
    latest_info: Optional[dict] = None,
) -> PredictabilityReport:
    values = list(completed_series.values)
    best = selection.best_backtest

    history_months = len(completed_series.months)
    zero_rate = zero_ratio(values)
    avg_qty = mean(values)
    std_qty = stddev(values)
    cv_value = coefficient_of_variation(values)
    adi_value = adi(values)
    nz_cv2 = nonzero_cv2(values)
    outlier_rate = outlier_ratio(values)
    seasonal = seasonality_strength(values)
    trend = trend_strength(values)
    recent_complete_month = selection.training_months[-1] if selection.training_months else None

    structure = _structure_score(
        history_months,
        completed_series.missing_rate,
        zero_rate,
        cv_value,
        seasonal,
        trend,
    )

    has_backtest = best is not None and best.fold_count > 0
    best_wape = best.wape if best else None
    best_smape = best.smape if best else None
    best_mae = best.mae if best else None
    best_bias = best.bias if best else None
    stability = best.error_stability if best else None
    backtest = _backtest_score(best_wape, best_mae, avg_qty, stability, has_backtest)
    overall = STRUCTURE_WEIGHT * structure + BACKTEST_WEIGHT * backtest
    level = grade_from_score(overall)

    evidence: List[str] = []
    if history_months < 12:
        evidence.append("历史月份少于 12 个月，滚动回测支撑不足。")
    elif history_months >= 24:
        evidence.append("历史月份不少于 24 个月，可支持更完整的趋势或季节性检查。")
    if completed_series.missing_rate > 0:
        evidence.append(f"序列存在缺失月份，缺失比例为 {completed_series.missing_rate:.2%}。")
    if zero_rate >= ZERO_RATIO_HIGH:
        evidence.append(f"零动销比例为 {zero_rate:.2%}，属于高零值序列。")
    if adi_value >= ADI_INTERMITTENT:
        evidence.append(f"ADI 为 {adi_value:.2f}，需求呈现间歇性特征。")
    if cv_value > 1.0:
        evidence.append(f"CV 为 {cv_value:.2f}，波动较大。")
    if outlier_rate > 0.05:
        evidence.append(f"异常值比例为 {outlier_rate:.2%}，需要关注促销或异常波动。")
    if seasonal >= 0.30:
        evidence.append(f"季节性强度为 {seasonal:.2f}，可考虑季节性模型。")
    if trend >= 0.35:
        evidence.append(f"趋势强度为 {trend:.2f}，趋势模型可能有帮助。")
    if best:
        if best.wape is None:
            evidence.append("回测真实值总和为 0，WAPE 不可计算，已参考 MAE。")
        elif best.wape > 0.8:
            evidence.append(f"最佳模型 WAPE 为 {best.wape:.2%}，回测误差偏高。")
        else:
            evidence.append(f"最佳模型 WAPE 为 {best.wape:.2%}，可作为可信度判断依据。")
    else:
        evidence.append("没有足够回测窗口，可测性更多依赖结构质量判断。")
    if selection.excluded_latest_month:
        evidence.append(f"最新月份 {selection.excluded_latest_month} 已按不完整风险排除。")
    if latest_info and latest_info.get("is_incomplete"):
        evidence.extend(latest_info.get("reasons", []))
    if not evidence:
        evidence.append("未发现明显结构性风险，但仍需结合业务节奏解释预测结果。")

    return PredictabilityReport(
        history_months=history_months,
        valid_months=completed_series.valid_months,
        missing_months=completed_series.missing_months,
        missing_rate=completed_series.missing_rate,
        zero_ratio=zero_rate,
        mean_qty=avg_qty,
        std_qty=std_qty,
        cv=cv_value,
        adi=adi_value,
        nonzero_cv2=nz_cv2,
        outlier_ratio=outlier_rate,
        seasonality_strength=seasonal,
        trend_strength=trend,
        recent_complete_month=recent_complete_month,
        best_wape=best_wape,
        best_smape=best_smape,
        best_mae=best_mae,
        best_bias=best_bias,
        backtest_windows=best.fold_count if best else 0,
        error_stability=stability,
        structure_score=structure,
        backtest_score=backtest,
        overall_score=overall,
        level=level,
        evidence=evidence,
    )
