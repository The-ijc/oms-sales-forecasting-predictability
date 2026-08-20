from __future__ import annotations

import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
AGENT_DIR = PROJECT_DIR / "动销预测"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent.orchestrator import analyze_sales_question  # noqa: E402
from agent.tools import (  # noqa: E402
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    execute_tool,
    generate_business_summary,
    get_forecast_result,
    get_model_comparison,
    get_predictability_evidence,
    search_knowledge_base,
)


class AgentToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "sample.sqlite"
        self._create_sample_db(month_count=18)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_sample_db(self, month_count: int):
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
        for index in range(month_count):
            year = 2023 + index // 12
            month = index % 12 + 1
            period = f"{year}{month:02d}"
            rows.append((period, "线下", "直营", "M1", "品类A", "细分A", 20 + index))
            rows.append((period, "线下", "直营", "M2", "品类A", "细分B", 10 + index % 3))
            rows.append((period, "线上", "电商", "M3", "品类B", "细分C", index % 2))
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

    def _context(self):
        return {
            "db_path": str(self.db_path),
            "level": "渠道型号",
            "channel": "线下",
            "channel_subtype": "直营",
            "product_model": "M1",
            "horizon": 3,
            "backtest_windows": 3,
        }

    def _walk(self, value: Any):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from self._walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from self._walk(item)
        else:
            yield value

    def test_forecast_tool_returns_complete_structure(self):
        result = get_forecast_result(context=self._context())
        self.assertTrue(result["ok"])
        for key in ["预测对象名称", "历史月份数", "最优模型", "未来预测值", "预测区间", "最近完整账期"]:
            self.assertIn(key, result)
        self.assertEqual(len(result["未来预测值"]), 3)

    def test_tools_do_not_return_customer_detail_fields(self):
        result = get_forecast_result(context=self._context())
        forbidden = {"客户", "客户明细", "原始明细", "交易明细", "points", "raw_rows", "fin_year_month"}
        text_values = {str(item) for item in self._walk(result)}
        self.assertTrue(forbidden.isdisjoint(text_values))

    def test_model_choice_intent_is_detected(self):
        result = analyze_sales_question("为什么这个对象选择 Naive 模型？", self._context())
        self.assertEqual(result["intent"], "model_choice")
        self.assertIn("get_model_comparison", result["tool_calls"])

    def test_data_quality_intent_is_detected(self):
        result = analyze_sales_question("这个型号的数据质量怎么样？", self._context())
        self.assertEqual(result["intent"], "data_quality")
        self.assertIn("get_predictability_evidence", result["tool_calls"])

    def test_risk_intent_is_detected(self):
        result = analyze_sales_question("有哪些预测风险？", self._context())
        self.assertEqual(result["intent"], "forecast_risk")
        self.assertIn("风险", result["answer"])

    def test_business_summary_uses_input_numbers_only(self):
        forecast = {
            "ok": True,
            "预测对象名称": "测试对象",
            "历史月份数": 18,
            "最优模型": "Naive 最近值",
            "未来预测值": [{"月份": "202407", "预测值": 12.0}],
            "预测区间": [{"月份": "202407", "下界": 10.0, "上界": 14.0}],
            "最近完整账期": "202406",
        }
        comparison = {"ok": True, "最优模型": "Naive 最近值", "所有候选模型": []}
        evidence = {
            "ok": True,
            "综合可测性得分": 66.6,
            "可测性等级": "中可测",
            "回测 WAPE": 0.12,
            "风险提示": ["历史波动需要关注"],
        }
        summary = generate_business_summary(forecast, comparison, evidence)
        self.assertTrue(summary["ok"])
        text = summary["业务摘要"]
        self.assertIn("12.00", text)
        self.assertIn("66.6", text)
        self.assertIn("预测区间不是准确率保证", text)
        self.assertNotIn("999", text)

    def test_short_history_answer_contains_risk(self):
        short_db = Path(self.temp_dir.name) / "short.sqlite"
        self.db_path = short_db
        self._create_sample_db(month_count=5)
        result = analyze_sales_question("为什么这个对象不建议强行预测？", self._context())
        self.assertEqual(result["intent"], "not_recommend_forecast")
        self.assertRegex(result["answer"], re.compile("历史月份|回测|风险"))

    def test_tool_registry_exposes_business_and_knowledge_tools(self):
        self.assertEqual(
            set(TOOL_REGISTRY),
            {
                "get_forecast_result",
                "get_model_comparison",
                "get_predictability_evidence",
                "search_knowledge_base",
                "search_product_knowledge",
            },
        )
        self.assertNotIn("generate_business_summary", TOOL_REGISTRY)

    def test_tool_schemas_match_registry_and_do_not_expose_db_path(self):
        schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
        self.assertEqual(schema_names, set(TOOL_REGISTRY))
        for schema in TOOL_SCHEMAS:
            properties = schema["function"]["parameters"]["properties"]
            self.assertNotIn("db_path", properties)
            expected_required = (
                ["query"]
                if schema["function"]["name"] in {"search_knowledge_base", "search_product_knowledge"}
                else []
            )
            self.assertEqual(schema["function"]["parameters"]["required"], expected_required)

        knowledge_schema = next(
            schema for schema in TOOL_SCHEMAS if schema["function"]["name"] == "search_knowledge_base"
        )
        top_k = knowledge_schema["function"]["parameters"]["properties"]["top_k"]
        self.assertEqual(top_k["minimum"], 1)
        self.assertEqual(top_k["maximum"], 5)

        product_schema = next(
            schema for schema in TOOL_SCHEMAS if schema["function"]["name"] == "search_product_knowledge"
        )
        product_properties = product_schema["function"]["parameters"]["properties"]
        self.assertEqual(set(product_properties), {"query", "product_model", "top_k"})
        self.assertEqual(product_properties["top_k"]["minimum"], 1)
        self.assertEqual(product_properties["top_k"]["maximum"], 5)

    def test_execute_tool_unknown_tool_returns_structured_error(self):
        result = execute_tool("unknown_tool", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "unknown_tool")
        self.assertIn("未知工具", result["错误"])

    def test_execute_tool_invalid_json_returns_structured_error(self):
        result = execute_tool("get_forecast_result", "{not json")
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "get_forecast_result")
        self.assertIn("JSON", result["错误"])

    def test_execute_tool_protected_context_overrides_llm_arguments(self):
        def fake_tool(*, context=None, **_kwargs):
            return {"ok": True, "tool": "get_forecast_result", "context": context}

        with patch.dict(TOOL_REGISTRY, {"get_forecast_result": fake_tool}):
            result = execute_tool(
                "get_forecast_result",
                {"db_path": "llm.sqlite", "channel": "线上", "horizon": 12},
                protected_context={"db_path": "protected.sqlite", "channel": "线下", "horizon": 3},
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["context"]["db_path"], "protected.sqlite")
        self.assertEqual(result["context"]["channel"], "线下")
        self.assertEqual(result["context"]["horizon"], 3)
        self.assertNotIn("llm.sqlite", set(str(value) for value in result["context"].values()))


if __name__ == "__main__":
    unittest.main()
