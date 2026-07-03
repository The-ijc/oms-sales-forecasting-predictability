from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_access import (  # noqa: E402
    DataAccessError,
    detect_global_latest_incomplete,
    get_aggregated_series,
    validate_db_path,
)
from src.model_selection import candidate_algorithms_for_series, select_model_and_forecast  # noqa: E402
from src.predictability import assess_predictability  # noqa: E402
from src.utils import normalize_points  # noqa: E402


class PredictabilityAppTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "sample.sqlite"
        self._create_sample_db()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_sample_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE adb_model_sales_summary (
                fin_year_month TEXT,
                channel TEXT,
                channel_ TEXT,
                product_model TEXT,
                product_category TEXT,
                product_class TEXT,
                qty_total INTEGER
            )
            """
        )
        rows = []
        months = [f"2023{month:02d}" for month in range(1, 13)]
        for index, month in enumerate(months, start=1):
            rows.append((month, "线下", "直营", "M1", "品类A", "细分A", 10 + index))
            rows.append((month, "线下", "直营", "M2", "品类A", "细分B", 5 + index))
            rows.append((month, "线上", "电商", "M3", "品类B", "细分C", index % 2))
        conn.executemany(
            """
            INSERT INTO adb_model_sales_summary
            (fin_year_month, channel, channel_, product_model, product_category, product_class, qty_total)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        conn.close()

    def test_database_path_not_exists(self):
        with self.assertRaises(DataAccessError):
            validate_db_path(str(Path(self.temp_dir.name) / "missing.sqlite"))

    def test_channel_aggregation(self):
        series = get_aggregated_series(str(self.db_path), "channel", limit=10)
        self.assertEqual(len(series), 2)
        self.assertTrue(all(len(item["key"]) == 1 for item in series))
        self.assertTrue(any(item["label"] == "线下" for item in series))

    def test_channel_detail_aggregation(self):
        series = get_aggregated_series(str(self.db_path), "channel_detail", limit=10)
        self.assertEqual(len(series), 2)
        self.assertTrue(all(len(item["key"]) == 2 for item in series))

    def test_channel_model_aggregation(self):
        series = get_aggregated_series(
            str(self.db_path),
            "channel_model",
            filters={"channel": "线下"},
            limit=10,
        )
        self.assertEqual(len(series), 2)
        self.assertTrue(all(len(item["key"]) == 3 for item in series))

    def test_short_history_uses_conservative_candidates(self):
        candidates = candidate_algorithms_for_series([2, 3, 4])
        self.assertEqual(candidates, ["naive", "mean", "zero"])

    def test_high_zero_series_includes_croston(self):
        values = [0, 0, 10, 0, 0, 0, 8, 0, 0, 0, 12, 0, 0, 0]
        candidates = candidate_algorithms_for_series(values)
        self.assertIn("croston_sba", candidates)

    def test_model_selection_outputs_forecast(self):
        months = [f"2023{month:02d}" for month in range(1, 13)] + [
            f"2024{month:02d}" for month in range(1, 7)
        ]
        values = [10 + index for index in range(len(months))]
        selection = select_model_and_forecast(months, values, horizon=3, backtest_windows=3)
        self.assertTrue(selection.selected_algorithm)
        self.assertEqual(len(selection.forecast_values), 3)
        self.assertTrue(selection.candidate_algorithms)

    def test_predictability_level_output(self):
        points = [(f"2023{month:02d}", 10 + month) for month in range(1, 13)]
        completed = normalize_points(points)
        selection = select_model_and_forecast(completed.months, completed.values, horizon=3, backtest_windows=3)
        report = assess_predictability(completed, selection)
        self.assertIn(report.level, {"高可测", "中可测", "低可测"})
        self.assertGreaterEqual(report.overall_score, 0)

    def test_latest_incomplete_warning(self):
        monthly_summary = [
            {"month": "202401", "record_count": 100, "qty_total": 1000},
            {"month": "202402", "record_count": 110, "qty_total": 900},
            {"month": "202403", "record_count": 105, "qty_total": 950},
            {"month": "202404", "record_count": 20, "qty_total": 100},
        ]
        latest = detect_global_latest_incomplete(monthly_summary)
        self.assertTrue(latest["is_incomplete"])
        self.assertEqual(latest["latest_complete_month"], "202403")


if __name__ == "__main__":
    unittest.main()
