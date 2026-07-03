from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


MIN_TRAIN_PERIODS = 12
DEFAULT_BACKTEST_WINDOWS = 6
MAX_OBJECTS_DEFAULT = 50

SHORT_HISTORY_MONTHS = 6
SEASONAL_MIN_MONTHS = 24
RANDOM_FOREST_MIN_MONTHS = 24
RANDOM_FOREST_MIN_NONZERO = 12

ZERO_RATIO_HIGH = 0.5
ADI_INTERMITTENT = 1.32
SEASONAL_STRENGTH_THRESHOLD = 0.30
TREND_STRENGTH_THRESHOLD = 0.35

INCOMPLETE_LATEST_RATIO = 0.35
WAPE_TIE_TOLERANCE = 0.03
SMAPE_TIE_TOLERANCE = 0.03

STRUCTURE_WEIGHT = 0.45
BACKTEST_WEIGHT = 0.55
HIGH_PREDICTABILITY_SCORE = 75
MEDIUM_PREDICTABILITY_SCORE = 50


ALGORITHM_LABELS = {
    "zero": "零预测",
    "mean": "历史均值",
    "naive": "Naive 最近值",
    "moving_average_3": "三期移动平均",
    "ses": "简单指数平滑 SES",
    "holt_linear": "Holt 线性趋势",
    "linear_trend": "线性回归趋势",
    "seasonal_naive": "Seasonal Naive",
    "random_forest_lag": "Random Forest 滞后特征",
    "croston_sba": "Croston SBA",
}

MODEL_COMPLEXITY_RANK = {
    "zero": 0,
    "mean": 1,
    "naive": 2,
    "moving_average_3": 3,
    "ses": 4,
    "croston_sba": 5,
    "seasonal_naive": 6,
    "linear_trend": 7,
    "holt_linear": 8,
    "random_forest_lag": 9,
}


@dataclass
class CompletedSeries:
    months: List[str]
    values: List[float]
    valid_months: int
    missing_months: int
    missing_rate: float


def safe_div(numerator: float, denominator: float, default: Optional[float] = None):
    if denominator == 0:
        return default
    return numerator / denominator


def clean_qty(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_month(month: str) -> Tuple[int, int]:
    text = str(month).strip()
    if len(text) == 6 and text.isdigit():
        return int(text[:4]), int(text[4:6])
    if len(text) == 7 and text[4] == "-" and text[:4].isdigit() and text[5:].isdigit():
        return int(text[:4]), int(text[5:7])
    raise ValueError(f"不支持的账期格式：{month}")


def format_month(year: int, month: int) -> str:
    return f"{year:04d}{month:02d}"


def add_months(month: str, offset: int) -> str:
    year, mon = parse_month(month)
    total = year * 12 + (mon - 1) + offset
    new_year = total // 12
    new_month = total % 12 + 1
    return format_month(new_year, new_month)


def month_range(start_month: str, end_month: str) -> List[str]:
    start_year, start_mon = parse_month(start_month)
    end_year, end_mon = parse_month(end_month)
    start_index = start_year * 12 + start_mon
    end_index = end_year * 12 + end_mon
    if end_index < start_index:
        return []
    return [add_months(start_month, offset) for offset in range(end_index - start_index + 1)]


def normalize_points(points: Sequence[Tuple[str, float]]) -> CompletedSeries:
    cleaned = {}
    for month, value in points:
        try:
            normalized_month = format_month(*parse_month(month))
        except ValueError:
            continue
        cleaned[normalized_month] = cleaned.get(normalized_month, 0.0) + clean_qty(value)

    if not cleaned:
        return CompletedSeries([], [], 0, 0, 0.0)

    months = month_range(min(cleaned), max(cleaned))
    values = [cleaned.get(month, 0.0) for month in months]
    valid_months = len(cleaned)
    missing_months = len(months) - valid_months
    missing_rate = missing_months / len(months) if months else 0.0
    return CompletedSeries(months, values, valid_months, missing_months, missing_rate)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def coefficient_of_variation(values: Sequence[float]) -> float:
    avg = mean(values)
    if avg == 0:
        return 0.0 if stddev(values) == 0 else float("inf")
    return stddev(values) / abs(avg)


def zero_ratio(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return sum(1 for value in values if value == 0) / len(values)


def nonzero_values(values: Sequence[float]) -> List[float]:
    return [value for value in values if value != 0]


def adi(values: Sequence[float]) -> float:
    nonzero_count = len(nonzero_values(values))
    if nonzero_count == 0:
        return float("inf")
    return len(values) / nonzero_count


def nonzero_cv2(values: Sequence[float]) -> float:
    nz = nonzero_values(values)
    if len(nz) < 2:
        return 0.0
    cv = coefficient_of_variation(nz)
    return cv * cv


def outlier_ratio(values: Sequence[float]) -> float:
    if len(values) < 6:
        return 0.0
    median_value = statistics.median(values)
    deviations = [abs(value - median_value) for value in values]
    mad = statistics.median(deviations)
    if mad == 0:
        return 0.0
    threshold = 3.5 * 1.4826 * mad
    return sum(1 for value in values if abs(value - median_value) > threshold) / len(values)


def seasonality_strength(values: Sequence[float], season_length: int = 12) -> float:
    if len(values) < season_length * 2:
        return 0.0
    total_std = stddev(values)
    if total_std == 0:
        return 0.0

    seasonal_means = []
    for offset in range(season_length):
        bucket = [values[index] for index in range(offset, len(values), season_length)]
        seasonal_means.append(mean(bucket))
    strength = stddev(seasonal_means) / total_std
    return max(0.0, min(1.0, strength))


def linear_regression_fit(values: Sequence[float]) -> Tuple[float, float, float]:
    if len(values) < 2:
        return 0.0, values[0] if values else 0.0, 0.0
    xs = list(range(len(values)))
    x_bar = mean(xs)
    y_bar = mean(values)
    denominator = sum((x - x_bar) ** 2 for x in xs)
    slope = 0.0 if denominator == 0 else sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, values)) / denominator
    intercept = y_bar - slope * x_bar
    fitted = [intercept + slope * x for x in xs]
    ss_total = sum((y - y_bar) ** 2 for y in values)
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(values, fitted))
    r2 = 0.0 if ss_total == 0 else max(0.0, 1 - ss_res / ss_total)
    return slope, intercept, min(1.0, r2)


def trend_strength(values: Sequence[float]) -> float:
    return linear_regression_fit(values)[2]


def wape(actual: Sequence[float], predicted: Sequence[float]) -> Optional[float]:
    denominator = sum(abs(value) for value in actual)
    if denominator == 0:
        return None
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / denominator


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    total = 0.0
    for a, p in zip(actual, predicted):
        denominator = abs(a) + abs(p)
        total += 0.0 if denominator == 0 else 2 * abs(a - p) / denominator
    return total / len(actual)


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def bias(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    denominator = sum(abs(value) for value in actual)
    raw_bias = sum(p - a for a, p in zip(actual, predicted))
    if denominator == 0:
        return raw_bias / len(actual)
    return raw_bias / denominator


def clip_forecast(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, value)


def detect_latest_incomplete(months: Sequence[str], values: Sequence[float]) -> Tuple[bool, List[str]]:
    reasons = []
    if len(months) < 4 or len(values) < 4:
        return False, reasons

    latest_value = values[-1]
    previous_values = values[-4:-1]
    previous_median = statistics.median(previous_values)
    if previous_median > 0 and latest_value <= previous_median * INCOMPLETE_LATEST_RATIO:
        reasons.append(
            f"最新月份 {months[-1]} 的动销量显著低于前三个月中位数，可能尚未完整。"
        )
    if previous_median > 0 and latest_value == 0:
        reasons.append(f"最新月份 {months[-1]} 动销量为 0，存在未完整入库风险。")
    return bool(reasons), reasons


def error_stability(errors: Sequence[float]) -> float:
    if len(errors) < 2:
        return 0.0
    avg_error = mean([abs(value) for value in errors])
    if avg_error == 0:
        return 0.0
    return stddev([abs(value) for value in errors]) / avg_error


def grade_from_score(score: float) -> str:
    if score >= HIGH_PREDICTABILITY_SCORE:
        return "高可测"
    if score >= MEDIUM_PREDICTABILITY_SCORE:
        return "中可测"
    return "低可测"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "不可计算"
    return f"{value:.2%}"


def round_float(value: Optional[float], digits: int = 4):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)
