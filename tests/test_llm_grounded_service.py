from __future__ import annotations

import json
import unittest

from 动销预测.llm.client import LLMClientError
from 动销预测.llm.config import LLMConfig
from 动销预测.llm.grounded_service import run_grounded_llm_analysis


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def complete(self, messages, config):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def fake_rule_analyzer(_question, context):
    forecast = {
        "ok": True,
        "tool": "get_forecast_result",
        "预测对象名称": "线下 / 直营 / M1",
        "最优模型": "Naive 最近值",
        "未来预测值": [{"月份": "202407", "预测值": 30.0}],
        "预测区间": [{"月份": "202407", "下界": 24.0, "上界": 36.0}],
        "最近完整账期": "202406",
    }
    comparison = {
        "ok": True,
        "tool": "get_model_comparison",
        "最优模型": "Naive 最近值",
        "所有候选模型": [
            {
                "模型": "Naive 最近值",
                "是否最优": True,
                "WAPE": 0.12,
                "sMAPE": 0.2,
                "MAE": 5.0,
                "Bias": -0.05,
            }
        ],
    }
    evidence = {
        "ok": True,
        "tool": "get_predictability_evidence",
        "综合可测性得分": 72.5,
        "可测性等级": "中可测",
        "回测 WAPE": 0.12,
        "风险提示": ["历史波动需要关注"],
        "可测性证据列表": ["回测 WAPE 为 12.00%"],
    }
    return {
        "intent": "forecast_trend",
        "tool_calls": ["get_forecast_result", "get_model_comparison", "get_predictability_evidence"],
        "evidence": ["预测对象：线下 / 直营 / M1", "最优模型：Naive 最近值"],
        "answer": "实际分析对象：预测层级=渠道型号；渠道大类=线下；渠道细分类=直营；型号=M1；预测步长=3个月；回测窗口=3个月\n\n本地规则结论。",
        "disclaimer": "预测区间不代表准确率保证。",
        "analysis_context": {
            "level": context["level"],
            "channel": context["channel"],
            "channel_subtype": context["channel_subtype"],
            "product_model": context["product_model"],
            "horizon": context["horizon"],
            "backtest_windows": context["backtest_windows"],
        },
        "tool_outputs": {
            "get_forecast_result": forecast,
            "get_model_comparison": comparison,
            "get_predictability_evidence": evidence,
        },
    }


def context():
    return {
        "level": "渠道型号",
        "channel": "线下",
        "channel_subtype": "直营",
        "product_model": "M1",
        "horizon": 3,
        "backtest_windows": 3,
    }


def valid_llm_json():
    return json.dumps(
        {
            "executive_summary": "线下 / 直营 / M1 使用 Naive 最近值，202407 预测值 30.0，WAPE 12.00%，预测区间不代表准确率保证。",
            "business_interpretation": "当前可测性等级为中可测，综合得分 72.5，需关注历史波动。",
            "risk_recommendations": ["结合库存和促销节奏复核预测结论"],
            "follow_up_questions": ["查看模型选择依据"],
            "evidence_references": ["best_model", "forecast_values", "model_metrics", "predictability"],
        },
        ensure_ascii=False,
    )


class GroundedLLMServiceTest(unittest.TestCase):
    def _llm_config(self, api_key: str = "secret-key"):
        return LLMConfig(
            mode="llm",
            api_key=api_key,
            base_url="https://example.test/v1",
            model="mock-model",
            timeout_seconds=3,
        )

    def test_rule_mode_does_not_call_client(self):
        client = FakeClient(error=AssertionError("should not call"))
        result = run_grounded_llm_analysis(
            "分析线下渠道未来 3 个月动销趋势",
            context(),
            config=LLMConfig(mode="rule"),
            client=client,
            rule_analyzer=fake_rule_analyzer,
        )

        self.assertEqual(client.calls, 0)
        self.assertFalse(result["llm_status"]["used"])

    def test_missing_config_falls_back_without_client_call(self):
        client = FakeClient(error=AssertionError("should not call"))
        result = run_grounded_llm_analysis(
            "分析线下渠道未来 3 个月动销趋势",
            context(),
            config=LLMConfig(mode="llm", api_key="", base_url="https://example.test/v1", model="mock-model"),
            client=client,
            rule_analyzer=fake_rule_analyzer,
        )

        self.assertEqual(client.calls, 0)
        self.assertFalse(result["llm_status"]["used"])
        self.assertIn("配置不完整", result["llm_status"]["message"])

    def test_network_error_falls_back_and_redacts_api_key(self):
        client = FakeClient(error=LLMClientError("network failed for secret-key"))
        result = run_grounded_llm_analysis(
            "分析线下渠道未来 3 个月动销趋势",
            context(),
            config=self._llm_config(api_key="secret-key"),
            client=client,
            rule_analyzer=fake_rule_analyzer,
        )

        self.assertFalse(result["llm_status"]["used"])
        self.assertNotIn("secret-key", result["llm_status"]["fallback_reason"])
        self.assertIn("[redacted]", result["llm_status"]["fallback_reason"])

    def test_valid_mock_json_is_accepted(self):
        client = FakeClient(response=valid_llm_json())
        result = run_grounded_llm_analysis(
            "分析线下渠道未来 3 个月动销趋势",
            context(),
            config=self._llm_config(),
            client=client,
            rule_analyzer=fake_rule_analyzer,
        )

        self.assertTrue(result["llm_status"]["used"])
        self.assertIn("executive_summary", result["llm_explanation"])
        self.assertEqual(result["llm_explanation"]["follow_up_questions"], ["查看模型选择依据"])

    def test_invalid_json_is_rejected(self):
        client = FakeClient(response="not json")
        result = run_grounded_llm_analysis(
            "分析线下渠道未来 3 个月动销趋势",
            context(),
            config=self._llm_config(),
            client=client,
            rule_analyzer=fake_rule_analyzer,
        )

        self.assertFalse(result["llm_status"]["used"])
        self.assertIn("事实一致性校验", result["llm_status"]["message"])

    def test_missing_field_is_rejected(self):
        payload = json.loads(valid_llm_json())
        payload.pop("evidence_references")
        client = FakeClient(response=json.dumps(payload, ensure_ascii=False))

        result = run_grounded_llm_analysis(
            "分析线下渠道未来 3 个月动销趋势",
            context(),
            config=self._llm_config(),
            client=client,
            rule_analyzer=fake_rule_analyzer,
        )

        self.assertFalse(result["llm_status"]["used"])

    def test_unknown_model_name_is_rejected(self):
        payload = json.loads(valid_llm_json())
        payload["business_interpretation"] = "建议改用 ARIMA，因为它更适合当前对象。"
        client = FakeClient(response=json.dumps(payload, ensure_ascii=False))

        result = run_grounded_llm_analysis(
            "分析线下渠道未来 3 个月动销趋势",
            context(),
            config=self._llm_config(),
            client=client,
            rule_analyzer=fake_rule_analyzer,
        )

        self.assertFalse(result["llm_status"]["used"])
        self.assertIn("模型名", result["llm_status"]["fallback_reason"])

    def test_inconsistent_number_is_rejected(self):
        payload = json.loads(valid_llm_json())
        payload["executive_summary"] = "202407 预测值 999，WAPE 12.00%，预测区间不代表准确率保证。"
        client = FakeClient(response=json.dumps(payload, ensure_ascii=False))

        result = run_grounded_llm_analysis(
            "分析线下渠道未来 3 个月动销趋势",
            context(),
            config=self._llm_config(),
            client=client,
            rule_analyzer=fake_rule_analyzer,
        )

        self.assertFalse(result["llm_status"]["used"])
        self.assertIn("数值", result["llm_status"]["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
