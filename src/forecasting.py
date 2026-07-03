from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .utils import (
    ALGORITHM_LABELS,
    RANDOM_FOREST_MIN_MONTHS,
    RANDOM_FOREST_MIN_NONZERO,
    clip_forecast,
    linear_regression_fit,
    mean,
    nonzero_values,
)


class ForecastError(RuntimeError):
    """预测算法失败。"""


def _ensure_history(values: Sequence[float]):
    if not values:
        raise ForecastError("历史序列为空，无法预测。")


def zero_forecast(values: Sequence[float], horizon: int) -> List[float]:
    return [0.0 for _ in range(horizon)]


def mean_forecast(values: Sequence[float], horizon: int) -> List[float]:
    _ensure_history(values)
    avg = mean(values)
    return [clip_forecast(avg) for _ in range(horizon)]


def naive_forecast(values: Sequence[float], horizon: int) -> List[float]:
    _ensure_history(values)
    return [clip_forecast(values[-1]) for _ in range(horizon)]


def moving_average_forecast(values: Sequence[float], horizon: int, window: int = 3) -> List[float]:
    _ensure_history(values)
    avg = mean(values[-window:])
    return [clip_forecast(avg) for _ in range(horizon)]


def ses_forecast(values: Sequence[float], horizon: int, alpha: float = 0.35) -> List[float]:
    _ensure_history(values)
    level = values[0]
    for value in values[1:]:
        level = alpha * value + (1 - alpha) * level
    return [clip_forecast(level) for _ in range(horizon)]


def holt_linear_forecast(
    values: Sequence[float], horizon: int, alpha: float = 0.45, beta: float = 0.20
) -> List[float]:
    _ensure_history(values)
    if len(values) < 3:
        return moving_average_forecast(values, horizon)
    level = values[0]
    trend = values[1] - values[0]
    for value in values[1:]:
        previous_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
    return [clip_forecast(level + step * trend) for step in range(1, horizon + 1)]


def linear_trend_forecast(values: Sequence[float], horizon: int) -> List[float]:
    _ensure_history(values)
    if len(values) < 2:
        return naive_forecast(values, horizon)
    slope, intercept, _ = linear_regression_fit(values)
    start = len(values)
    return [clip_forecast(intercept + slope * (start + step)) for step in range(horizon)]


def seasonal_naive_forecast(values: Sequence[float], horizon: int, season_length: int = 12) -> List[float]:
    _ensure_history(values)
    if len(values) < season_length:
        return naive_forecast(values, horizon)
    forecasts = []
    for step in range(horizon):
        source_index = len(values) - season_length + (step % season_length)
        forecasts.append(clip_forecast(values[source_index]))
    return forecasts


def croston_sba_forecast(values: Sequence[float], horizon: int, alpha: float = 0.10) -> List[float]:
    _ensure_history(values)
    nonzero = [(index, value) for index, value in enumerate(values) if value > 0]
    if not nonzero:
        return zero_forecast(values, horizon)

    first_index, first_value = nonzero[0]
    demand_level = first_value
    interval_level = max(1.0, first_index + 1.0)
    last_demand_index = first_index

    for index, value in enumerate(values[first_index + 1 :], start=first_index + 1):
        if value > 0:
            interval = index - last_demand_index
            demand_level = alpha * value + (1 - alpha) * demand_level
            interval_level = alpha * interval + (1 - alpha) * interval_level
            last_demand_index = index

    croston = demand_level / interval_level if interval_level else 0.0
    sba = (1 - alpha / 2) * croston
    return [clip_forecast(sba) for _ in range(horizon)]


@dataclass
class _TreeNode:
    prediction: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


def _sse(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values)


def _build_tree(
    x_rows: Sequence[Sequence[float]],
    y_values: Sequence[float],
    rng: random.Random,
    depth: int = 0,
    max_depth: int = 4,
    min_leaf: int = 3,
) -> _TreeNode:
    prediction = mean(y_values)
    if depth >= max_depth or len(y_values) <= min_leaf * 2 or len(set(y_values)) <= 1:
        return _TreeNode(prediction=prediction)

    feature_count = len(x_rows[0])
    sampled_features = rng.sample(range(feature_count), k=max(1, int(math.sqrt(feature_count))))
    best_feature = None
    best_threshold = None
    best_score = float("inf")
    best_left = []
    best_right = []

    for feature_index in sampled_features:
        thresholds = sorted({row[feature_index] for row in x_rows})
        if len(thresholds) > 8:
            step = max(1, len(thresholds) // 8)
            thresholds = thresholds[::step]
        for threshold in thresholds:
            left_indices = [i for i, row in enumerate(x_rows) if row[feature_index] <= threshold]
            right_indices = [i for i, row in enumerate(x_rows) if row[feature_index] > threshold]
            if len(left_indices) < min_leaf or len(right_indices) < min_leaf:
                continue
            left_y = [y_values[i] for i in left_indices]
            right_y = [y_values[i] for i in right_indices]
            score = _sse(left_y) + _sse(right_y)
            if score < best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold
                best_left = left_indices
                best_right = right_indices

    if best_feature is None:
        return _TreeNode(prediction=prediction)

    left_x = [x_rows[i] for i in best_left]
    left_y = [y_values[i] for i in best_left]
    right_x = [x_rows[i] for i in best_right]
    right_y = [y_values[i] for i in best_right]
    return _TreeNode(
        prediction=prediction,
        feature_index=best_feature,
        threshold=best_threshold,
        left=_build_tree(left_x, left_y, rng, depth + 1, max_depth, min_leaf),
        right=_build_tree(right_x, right_y, rng, depth + 1, max_depth, min_leaf),
    )


def _predict_tree(node: _TreeNode, features: Sequence[float]) -> float:
    current = node
    while current.feature_index is not None and current.threshold is not None:
        if features[current.feature_index] <= current.threshold:
            if current.left is None:
                break
            current = current.left
        else:
            if current.right is None:
                break
            current = current.right
    return current.prediction


def _lag_features(history: Sequence[float], next_index: int) -> List[float]:
    lag1 = history[-1] if len(history) >= 1 else 0.0
    lag2 = history[-2] if len(history) >= 2 else lag1
    lag3 = history[-3] if len(history) >= 3 else lag2
    trailing_mean = mean(history[-3:]) if history else 0.0
    month_position = next_index % 12
    return [lag1, lag2, lag3, trailing_mean, float(month_position)]


def random_forest_lag_forecast(values: Sequence[float], horizon: int) -> List[float]:
    _ensure_history(values)
    if len(values) < RANDOM_FOREST_MIN_MONTHS or len(nonzero_values(values)) < RANDOM_FOREST_MIN_NONZERO:
        raise ForecastError("Random Forest 需要至少 24 个月历史且非零样本不少于 12。")

    x_rows = []
    y_values = []
    for index in range(3, len(values)):
        x_rows.append(_lag_features(values[:index], index))
        y_values.append(values[index])
    if len(y_values) < 12:
        raise ForecastError("Random Forest 可训练样本不足。")

    rng = random.Random(2026)
    forest = []
    tree_count = 25
    for _ in range(tree_count):
        sample_indices = [rng.randrange(len(y_values)) for _ in range(len(y_values))]
        sample_x = [x_rows[i] for i in sample_indices]
        sample_y = [y_values[i] for i in sample_indices]
        forest.append(_build_tree(sample_x, sample_y, rng))

    history = list(values)
    forecasts = []
    for _ in range(horizon):
        features = _lag_features(history, len(history))
        prediction = mean([_predict_tree(tree, features) for tree in forest])
        prediction = clip_forecast(prediction)
        forecasts.append(prediction)
        history.append(prediction)
    return forecasts


FORECAST_FUNCTIONS: Dict[str, Callable[[Sequence[float], int], List[float]]] = {
    "zero": zero_forecast,
    "mean": mean_forecast,
    "naive": naive_forecast,
    "moving_average_3": moving_average_forecast,
    "ses": ses_forecast,
    "holt_linear": holt_linear_forecast,
    "linear_trend": linear_trend_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "random_forest_lag": random_forest_lag_forecast,
    "croston_sba": croston_sba_forecast,
}


def forecast(values: Sequence[float], horizon: int, algorithm: str) -> List[float]:
    if algorithm not in FORECAST_FUNCTIONS:
        raise ForecastError(f"未知预测算法：{algorithm}")
    if horizon <= 0:
        return []
    try:
        return FORECAST_FUNCTIONS[algorithm](list(values), horizon)
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError(f"{ALGORITHM_LABELS.get(algorithm, algorithm)} 训练或预测失败：{exc}") from exc


def forecast_with_fallback(values: Sequence[float], horizon: int, algorithm: str) -> List[float]:
    """模型失败时退回到最近值，最近值也不可用时退回零预测。"""
    try:
        return forecast(values, horizon, algorithm)
    except ForecastError:
        if values:
            return naive_forecast(values, horizon)
        return zero_forecast(values, horizon)
