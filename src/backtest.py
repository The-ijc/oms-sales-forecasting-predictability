from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .forecasting import ForecastError, forecast
from .utils import ALGORITHM_LABELS, bias, error_stability, mae, smape, wape


@dataclass
class BacktestResult:
    algorithm: str
    algorithm_label: str
    wape: Optional[float]
    smape: float
    mae: float
    bias: float
    primary_score: float
    fold_count: int
    actuals: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    fold_errors: List[float] = field(default_factory=list)
    error_stability: float = 0.0
    status: str = "ok"
    message: str = ""

    def to_dict(self):
        return {
            "算法": self.algorithm_label,
            "WAPE": self.wape,
            "sMAPE": self.smape,
            "MAE": self.mae,
            "Bias": self.bias,
            "回测窗口数": self.fold_count,
            "误差稳定性": self.error_stability,
            "状态": self.status,
            "说明": self.message,
        }


def walk_forward_backtest(
    values: Sequence[float],
    algorithms: Sequence[str],
    min_train_periods: int = 12,
    backtest_windows: int = 6,
) -> List[BacktestResult]:
    """Walk-Forward 回测：每次只用过去数据预测下一个月。"""
    results: List[BacktestResult] = []
    values = list(values)

    for algorithm in algorithms:
        actuals: List[float] = []
        predictions: List[float] = []
        messages: List[str] = []

        if len(values) <= min_train_periods:
            results.append(
                BacktestResult(
                    algorithm=algorithm,
                    algorithm_label=ALGORITHM_LABELS.get(algorithm, algorithm),
                    wape=None,
                    smape=0.0,
                    mae=0.0,
                    bias=0.0,
                    primary_score=float("inf"),
                    fold_count=0,
                    status="skipped",
                    message="历史月份不足，无法形成回测窗口。",
                )
            )
            continue

        first_cutoff = max(min_train_periods, len(values) - backtest_windows)
        for cutoff in range(first_cutoff, len(values)):
            train_values = values[:cutoff]
            actual = values[cutoff]
            try:
                predicted = forecast(train_values, 1, algorithm)[0]
            except ForecastError as exc:
                messages.append(str(exc))
                continue
            actuals.append(actual)
            predictions.append(predicted)

        if not actuals:
            results.append(
                BacktestResult(
                    algorithm=algorithm,
                    algorithm_label=ALGORITHM_LABELS.get(algorithm, algorithm),
                    wape=None,
                    smape=0.0,
                    mae=0.0,
                    bias=0.0,
                    primary_score=float("inf"),
                    fold_count=0,
                    status="failed",
                    message="; ".join(sorted(set(messages))) or "模型回测失败。",
                )
            )
            continue

        wape_value = wape(actuals, predictions)
        mae_value = mae(actuals, predictions)
        fold_errors = [prediction - actual for actual, prediction in zip(actuals, predictions)]
        primary_score = wape_value if wape_value is not None else mae_value
        status = "partial" if messages else "ok"
        message = "; ".join(sorted(set(messages)))
        results.append(
            BacktestResult(
                algorithm=algorithm,
                algorithm_label=ALGORITHM_LABELS.get(algorithm, algorithm),
                wape=wape_value,
                smape=smape(actuals, predictions),
                mae=mae_value,
                bias=bias(actuals, predictions),
                primary_score=primary_score,
                fold_count=len(actuals),
                actuals=actuals,
                predictions=predictions,
                fold_errors=fold_errors,
                error_stability=error_stability(fold_errors),
                status=status,
                message=message,
            )
        )

    return results
