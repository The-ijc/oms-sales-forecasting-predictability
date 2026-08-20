from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

try:
    import streamlit  # noqa: F401
except ImportError:
    class _CacheData:
        def __call__(self, **_kwargs):
            return lambda function: function

    sys.modules["streamlit"] = types.SimpleNamespace(cache_data=_CacheData())

import app


class AppAgentIntegrationTest(unittest.TestCase):
    def _context(self, channel: str = "线下"):
        return app.build_agent_context(
            db_path="sample.sqlite",
            level_label="渠道型号",
            level="channel_model",
            filters={"channel": channel, "channel_detail": "直营", "product_model": "M1"},
            forecast_horizon=3,
            backtest_windows=3,
            min_train_periods=12,
        )

    def _llm_config(self):
        return app.LLMConfig(
            mode="llm",
            api_key="secret",
            base_url="https://example.test/v1",
            model="mock-model",
        )

    def test_conflicting_question_channel_blocks_tool_call_by_default(self):
        context = self._context(channel="线上")

        with patch.object(app, "analyze_sales_question") as mocked:
            result = app.call_agent_analysis("分析线下渠道未来 3 个月动销趋势", context)

        mocked.assert_not_called()
        self.assertTrue(result["blocked"])
        self.assertEqual(result["intent"], "context_conflict")
        self.assertEqual(result["tool_calls"], [])
        self.assertIn("问题中请求分析：线下", result["answer"])
        self.assertIn("当前左侧筛选对象：线上", result["answer"])

    def test_conflicting_question_channel_blocks_llm_mode_before_service_call(self):
        context = self._context(channel="线上")

        with patch.object(app, "run_grounded_llm_analysis") as mocked:
            result = app.call_agent_analysis(
                "分析线下渠道未来 3 个月动销趋势",
                context,
                response_mode="llm",
                llm_config=app.LLMConfig(mode="llm", api_key="secret", base_url="https://example.test/v1", model="mock"),
            )

        mocked.assert_not_called()
        self.assertTrue(result["blocked"])
        self.assertEqual(result["tool_calls"], [])

    def test_override_switch_changes_context_channel_to_question_channel(self):
        context = self._context(channel="线上")

        with patch.object(
            app,
            "analyze_sales_question",
            return_value={
                "answer": "已完成分析。",
                "intent": "forecast_trend",
                "tool_calls": ["get_forecast_result"],
                "evidence": ["预测对象：线下 / 直营 / M1"],
                "disclaimer": "预测结果来自本地工具。",
            },
        ) as mocked:
            result = app.call_agent_analysis(
                "分析线下渠道未来 3 个月动销趋势",
                context,
                allow_channel_override=True,
            )

        called_question, called_context = mocked.call_args.args
        self.assertEqual(called_question, "分析线下渠道未来 3 个月动销趋势")
        self.assertEqual(called_context["db_path"], "sample.sqlite")
        self.assertEqual(called_context["level"], "渠道型号")
        self.assertEqual(called_context["channel"], "线下")
        self.assertEqual(called_context["channel_subtype"], "直营")
        self.assertEqual(called_context["product_model"], "M1")
        self.assertEqual(called_context["horizon"], 3)
        self.assertEqual(called_context["backtest_windows"], 3)
        self.assertTrue(result["answer"].startswith("实际分析对象："))
        self.assertIn("本次分析对象已由问题文本覆盖为：线下。", result["answer"])

    def test_matching_question_channel_runs_analysis_normally(self):
        context = self._context(channel="线下")

        with patch.object(
            app,
            "analyze_sales_question",
            return_value={
                "answer": "已完成分析。",
                "intent": "forecast_trend",
                "tool_calls": ["get_forecast_result"],
                "evidence": ["预测对象：线下 / 直营 / M1"],
                "disclaimer": "预测结果来自本地工具。",
            },
        ) as mocked:
            result = app.call_agent_analysis("分析线下渠道未来 3 个月动销趋势", context)

        mocked.assert_called_once()
        self.assertFalse(result.get("blocked", False))
        self.assertIn("已完成分析。", result["answer"])

    def test_answer_actual_object_matches_tool_context(self):
        context = self._context(channel="线下")

        def fake_agent(_question, tool_context):
            return {
                "answer": f"工具收到渠道：{tool_context['channel']}",
                "intent": "forecast_trend",
                "tool_calls": ["get_forecast_result"],
                "evidence": [],
                "disclaimer": "预测结果来自本地工具。",
            }

        with patch.object(app, "analyze_sales_question", side_effect=fake_agent) as mocked:
            result = app.call_agent_analysis("分析线下渠道未来 3 个月动销趋势", context)

        _called_question, called_context = mocked.call_args.args
        first_line = result["answer"].splitlines()[0]
        self.assertEqual(called_context["channel"], "线下")
        self.assertIn("实际分析对象：", first_line)
        self.assertIn("渠道大类=线下", first_line)
        self.assertIn("工具收到渠道：线下", result["answer"])
        self.assertEqual(result["analysis_context"]["channel"], called_context["channel"])

    def test_tool_calling_mode_invokes_new_agent(self):
        context = self._context()
        fake_client = object()
        config = self._llm_config()
        agent_response = {
            "ok": True,
            "answer": "未来三个月销量预计保持稳定。",
            "tool_calls": [],
            "iterations": 1,
            "error": None,
            "analysis_context": {},
        }

        with patch.object(app, "run_tool_calling_agent", return_value=agent_response) as mocked:
            result = app.call_agent_analysis(
                "分析线下渠道未来三个月销量",
                context,
                response_mode="tool_calling",
                llm_config=config,
                llm_client=fake_client,
            )

        called_question, called_context, called_client, called_config = mocked.call_args.args
        self.assertEqual(called_question, "分析线下渠道未来三个月销量")
        self.assertEqual(called_context["db_path"], "sample.sqlite")
        self.assertEqual(called_context["level"], "渠道型号")
        self.assertEqual(called_context["channel"], "线下")
        self.assertEqual(called_context["channel_subtype"], "直营")
        self.assertEqual(called_context["product_model"], "M1")
        self.assertEqual(called_context["horizon"], 3)
        self.assertEqual(called_context["min_train_periods"], 12)
        self.assertEqual(called_context["backtest_windows"], 3)
        self.assertEqual(called_context["limit"], 1)
        self.assertIs(called_client, fake_client)
        self.assertIs(called_config, config)
        self.assertTrue(result["ok"])
        self.assertEqual(result["answer_mode_requested"], "tool_calling")
        self.assertTrue(result["answer"].startswith("实际分析对象："))

    def test_grounded_llm_mode_keeps_existing_service_flow(self):
        context = self._context()
        grounded_result = {
            "rule_response": {
                "answer": "规则事实结论。",
                "intent": "forecast_trend",
                "tool_calls": ["get_forecast_result"],
                "evidence": [],
                "disclaimer": "预测仅供参考。",
            },
            "llm_status": {
                "requested": True,
                "used": True,
                "mode": "llm",
                "message": "LLM 增强解释已启用。",
                "fallback_reason": "",
            },
            "llm_explanation": {"executive_summary": "LLM 解释。"},
            "fact_pack": {"事实": "可用"},
        }

        with patch.object(app, "run_grounded_llm_analysis", return_value=grounded_result) as mocked:
            result = app.call_agent_analysis(
                "分析线下渠道趋势",
                context,
                response_mode="llm",
                llm_config=self._llm_config(),
            )

        mocked.assert_called_once()
        self.assertEqual(result["answer_mode_requested"], "llm")
        self.assertEqual(result["answer_mode_effective"], "llm")
        self.assertEqual(result["llm_explanation"], grounded_result["llm_explanation"])
        self.assertIn("规则事实结论。", result["answer"])

    def test_tool_calling_success_response_can_build_ui_trace(self):
        context = self._context()
        agent_response = {
            "ok": True,
            "answer": "预测属于估计值。",
            "tool_calls": [
                {
                    "tool_call_id": "call_1",
                    "tool_name": "get_forecast_result",
                    "arguments": '{"horizon": 3}',
                    "result": {"ok": True, "tool": "get_forecast_result", "未来预测值": [100]},
                    "ok": True,
                }
            ],
            "iterations": 2,
            "error": None,
            "analysis_context": {},
        }

        with patch.object(app, "run_tool_calling_agent", return_value=agent_response):
            result = app.call_agent_analysis(
                "预测线下渠道未来三个月销量",
                context,
                response_mode="tool_calling",
                llm_config=self._llm_config(),
                llm_client=object(),
            )

        rows = app.build_tool_call_trace_rows(result["tool_calls"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool_name"], "get_forecast_result")
        self.assertEqual(rows[0]["arguments"], {"horizon": 3})
        self.assertTrue(rows[0]["ok"])

    def test_tool_calling_exception_returns_error_without_propagating(self):
        context = self._context()

        with patch.object(
            app,
            "run_tool_calling_agent",
            side_effect=RuntimeError("request failed with secret"),
        ):
            result = app.call_agent_analysis(
                "预测线下渠道销量",
                context,
                response_mode="tool_calling",
                llm_config=self._llm_config(),
                llm_client=object(),
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["answer_mode_effective"], "unavailable")
        self.assertIn("request failed", result["error"])
        self.assertNotIn("secret", result["error"])

    def test_tool_calling_incomplete_config_does_not_invoke_agent(self):
        context = self._context()

        with patch.object(app, "run_tool_calling_agent") as mocked:
            result = app.call_agent_analysis(
                "预测线下渠道销量",
                context,
                response_mode="tool_calling",
                llm_config=app.LLMConfig(mode="llm"),
            )

        mocked.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("配置不完整", result["error"])

    def test_tool_call_trace_hides_protected_context(self):
        rows = app.build_tool_call_trace_rows(
            [
                {
                    "tool_name": "get_forecast_result",
                    "arguments": '{"horizon":3,"db_path":"private.sqlite","limit":50}',
                    "result": {
                        "ok": False,
                        "错误": "数据库暂不可用",
                        "database_path": "private.sqlite",
                        "backtest_windows": 6,
                    },
                    "ok": False,
                }
            ]
        )

        rendered = str(rows)
        self.assertNotIn("db_path", rendered)
        self.assertNotIn("database_path", rendered)
        self.assertNotIn("private.sqlite", rendered)
        self.assertNotIn("backtest_windows", rendered)
        self.assertEqual(rows[0]["arguments"], {"horizon": 3})


if __name__ == "__main__":
    unittest.main()
