from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from 动销预测.agent.tool_calling_orchestrator import run_tool_calling_agent  # noqa: E402
from 动销预测.agent.prompts import TOOL_CALLING_AGENT_SYSTEM_PROMPT  # noqa: E402
from 动销预测.llm.client import LLMClientError  # noqa: E402
from 动销预测.llm.config import LLMConfig  # noqa: E402


def tool_call(call_id, name, arguments="{}"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_message(self, messages, config, tools=None, tool_choice=None):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "config": config,
                "tools": copy.deepcopy(tools),
                "tool_choice": tool_choice,
            }
        )
        return copy.deepcopy(self.responses.pop(0))


class RaisingChatClient:
    def __init__(self, error):
        self.error = error

    def complete_message(self, _messages, _config, tools=None, tool_choice=None):
        raise self.error


class ToolCallingOrchestratorTest(unittest.TestCase):
    def setUp(self):
        self.context = {
            "db_path": "protected.sqlite",
            "level": "渠道型号",
            "channel": "线下",
            "channel_subtype": "直营",
            "product_model": "M1",
            "horizon": 3,
        }
        self.config = LLMConfig(
            mode="llm",
            api_key="secret",
            base_url="https://example.test/v1",
            model="mock-model",
        )

    def test_single_tool_call_completes_full_loop(self):
        client = FakeChatClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_1", "get_forecast_result", '{"horizon": 3}')],
                },
                {"role": "assistant", "content": "未来三个月销量为预测值，并非保证值。"},
            ]
        )
        tool_result = {"ok": True, "tool": "get_forecast_result", "未来预测值": [100, 110, 120]}

        with patch(
            "动销预测.agent.tool_calling_orchestrator.execute_tool",
            return_value=tool_result,
        ) as execute:
            result = run_tool_calling_agent("预测未来三个月销量", self.context, client, self.config)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "未来三个月销量为预测值，并非保证值。")
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["tool_calls"][0]["tool_call_id"], "call_1")
        execute.assert_called_once_with(
            "get_forecast_result",
            '{"horizon": 3}',
            protected_context=unittest.mock.ANY,
        )
        second_round_messages = client.calls[1]["messages"]
        self.assertEqual(second_round_messages[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(second_round_messages[3]["role"], "tool")
        self.assertEqual(json.loads(second_round_messages[3]["content"]), tool_result)
        self.assertEqual(client.calls[0]["tool_choice"], "auto")

    def test_system_prompt_contains_strict_observation_grounding_rules(self):
        client = FakeChatClient([{"role": "assistant", "content": "当前问题不需要工具。"}])

        result = run_tool_calling_agent("你好", self.context, client, self.config)

        self.assertTrue(result["ok"])
        system_message = client.calls[0]["messages"][0]
        self.assertEqual(system_message["role"], "system")
        self.assertEqual(system_message["content"], TOOL_CALLING_AGENT_SYSTEM_PROMPT)
        for required_rule in [
            "不得用知识工具替代 SQLite 业务工具",
            "优先调用 search_product_knowledge",
            "商品资料不得用于生成销量或预测值",
            "优先调用 search_knowledge_base",
            "组合调用业务工具和 Product Knowledge Tool",
            "知识库依据不足",
            "必须逐项可追溯到 Tool Observation",
            "行业阈值",
            "因果关系",
            "不得把 WAPE 直接解释为单点预测的 ±误差区间",
            "不根据系统当前时间判断",
            "基于最近完整账期，向后预测若干账期",
            "不引入新事实的格式化",
            "不得扩展为新的风险事实",
            "不包含实时库存、补货或进销存信息",
            "chain-of-thought",
        ]:
            self.assertIn(required_rule, system_message["content"])

    def test_forecast_and_product_knowledge_can_be_combined(self):
        client = FakeChatClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("call_forecast", "get_forecast_result", '{"horizon":3}'),
                        tool_call(
                            "call_product",
                            "search_product_knowledge",
                            '{"query":"EWH-10B2 是什么产品","product_model":"EWH-10B2"}',
                        ),
                    ],
                },
                {"role": "assistant", "content": "已根据预测和商品资料完成回答。"},
            ]
        )

        def fake_execute(name, _arguments, protected_context=None):
            if name == "get_forecast_result":
                return {"ok": True, "tool": name, "未来预测值": [100, 110, 120]}
            return {
                "ok": True,
                "tool": name,
                "results": [{"model": "EWH-10B2", "content": "当前商品资料字段为空。"}],
            }

        context = dict(self.context, product_model="EWH-10B2")
        with patch(
            "动销预测.agent.tool_calling_orchestrator.execute_tool",
            side_effect=fake_execute,
        ) as execute:
            result = run_tool_calling_agent(
                "EWH-10B2 三个账期销量怎么样？它是什么产品？",
                context,
                client,
                self.config,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(
            [record["tool_name"] for record in result["tool_calls"]],
            ["get_forecast_result", "search_product_knowledge"],
        )
        observations = client.calls[1]["messages"][-2:]
        self.assertEqual(
            [message["tool_call_id"] for message in observations],
            ["call_forecast", "call_product"],
        )

    def test_knowledge_tool_observation_can_produce_final_answer(self):
        client = FakeChatClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("call_knowledge", "search_knowledge_base", '{"query":"WAPE 定义"}')
                    ],
                },
                {"role": "assistant", "content": "根据知识库，WAPE 是聚合回测误差指标。"},
            ]
        )
        knowledge_result = {
            "ok": True,
            "tool": "search_knowledge_base",
            "query": "WAPE 定义",
            "results": [{"source": "forecasting_glossary.md", "content": "WAPE 是聚合回测误差指标。"}],
        }

        with patch(
            "动销预测.agent.tool_calling_orchestrator.execute_tool",
            return_value=knowledge_result,
        ) as execute:
            result = run_tool_calling_agent("WAPE 是什么意思？", self.context, client, self.config)

        self.assertTrue(result["ok"])
        self.assertEqual(result["iterations"], 2)
        execute.assert_called_once_with(
            "search_knowledge_base",
            '{"query":"WAPE 定义"}',
            protected_context=unittest.mock.ANY,
        )
        observation = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertEqual(observation, knowledge_result)

    def test_forecast_then_knowledge_tool_can_produce_final_answer(self):
        client = FakeChatClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_forecast", "get_forecast_result")],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("call_knowledge", "search_knowledge_base", '{"query":"WAPE 定义"}')
                    ],
                },
                {"role": "assistant", "content": "已根据预测结果和知识定义完成回答。"},
            ]
        )

        def fake_execute(name, _arguments, protected_context=None):
            if name == "get_forecast_result":
                return {"ok": True, "tool": name, "未来预测值": [100, 110, 120]}
            return {
                "ok": True,
                "tool": name,
                "results": [{"source": "forecasting_glossary.md", "content": "WAPE 定义"}],
            }

        with patch(
            "动销预测.agent.tool_calling_orchestrator.execute_tool",
            side_effect=fake_execute,
        ) as execute:
            result = run_tool_calling_agent(
                "线上三个账期预测是多少，WAPE 是什么意思？",
                self.context,
                client,
                self.config,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(
            [record["tool_name"] for record in result["tool_calls"]],
            ["get_forecast_result", "search_knowledge_base"],
        )
        self.assertEqual(execute.call_count, 2)
        final_messages = client.calls[2]["messages"]
        self.assertEqual(final_messages[3]["tool_call_id"], "call_forecast")
        self.assertEqual(final_messages[5]["tool_call_id"], "call_knowledge")

    def test_multiple_tool_calls_add_matching_observations(self):
        client = FakeChatClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("call_forecast", "get_forecast_result"),
                        tool_call("call_evidence", "get_predictability_evidence"),
                    ],
                },
                {"role": "assistant", "content": "预测已完成，但需关注可测性风险。"},
            ]
        )

        def fake_execute(name, _arguments, protected_context=None):
            return {"ok": True, "tool": name, "中文事实": "可用"}

        with patch(
            "动销预测.agent.tool_calling_orchestrator.execute_tool",
            side_effect=fake_execute,
        ) as execute:
            result = run_tool_calling_agent("预测并判断是否靠谱", self.context, client, self.config)

        self.assertTrue(result["ok"])
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(
            [record["tool_name"] for record in result["tool_calls"]],
            ["get_forecast_result", "get_predictability_evidence"],
        )
        observations = client.calls[1]["messages"][-2:]
        self.assertEqual(
            [message["tool_call_id"] for message in observations],
            ["call_forecast", "call_evidence"],
        )
        self.assertIn("中文事实", observations[0]["content"])

    def test_sequential_tool_calls_accumulate_conversation(self):
        client = FakeChatClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_forecast", "get_forecast_result")],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_evidence", "get_predictability_evidence")],
                },
                {"role": "assistant", "content": "预测和风险证据均已获取。"},
            ]
        )

        with patch(
            "动销预测.agent.tool_calling_orchestrator.execute_tool",
            side_effect=lambda name, _arguments, protected_context=None: {
                "ok": True,
                "tool": name,
            },
        ) as execute:
            result = run_tool_calling_agent("预测并判断风险", self.context, client, self.config)

        self.assertTrue(result["ok"])
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(
            [record["tool_name"] for record in result["tool_calls"]],
            ["get_forecast_result", "get_predictability_evidence"],
        )
        final_messages = client.calls[2]["messages"]
        self.assertEqual(
            [message["role"] for message in final_messages],
            ["system", "user", "assistant", "tool", "assistant", "tool"],
        )
        self.assertEqual(final_messages[3]["tool_call_id"], "call_forecast")
        self.assertEqual(final_messages[5]["tool_call_id"], "call_evidence")

    def test_llm_can_answer_directly(self):
        client = FakeChatClient([{"role": "assistant", "content": "这是最终回答。"}])

        result = run_tool_calling_agent("你好", self.context, client, self.config)

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "这是最终回答。")
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["tool_calls"], [])

    def test_tool_error_is_returned_as_observation(self):
        client = FakeChatClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_error", "get_forecast_result")],
                },
                {"role": "assistant", "content": "当前无法完成预测，因为数据不可用。"},
            ]
        )
        tool_error = {"ok": False, "tool": "get_forecast_result", "错误": "数据不可用"}

        with patch(
            "动销预测.agent.tool_calling_orchestrator.execute_tool",
            return_value=tool_error,
        ):
            result = run_tool_calling_agent("预测销量", self.context, client, self.config)

        self.assertTrue(result["ok"])
        self.assertFalse(result["tool_calls"][0]["ok"])
        observation = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertEqual(observation, tool_error)

    def test_max_iterations_stops_repeated_tool_calls(self):
        client = FakeChatClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call(f"call_{index}", "get_forecast_result")],
                }
                for index in range(3)
            ]
        )

        with patch(
            "动销预测.agent.tool_calling_orchestrator.execute_tool",
            return_value={"ok": True, "tool": "get_forecast_result"},
        ) as execute:
            result = run_tool_calling_agent(
                "持续预测",
                self.context,
                client,
                self.config,
                max_iterations=3,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["iterations"], 3)
        self.assertIn("reached_max_iterations", result["error"])
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(len(client.calls), 3)

    def test_empty_assistant_message_returns_structured_error(self):
        client = FakeChatClient([{"role": "assistant", "content": None}])

        result = run_tool_calling_agent("预测销量", self.context, client, self.config)

        self.assertFalse(result["ok"])
        self.assertEqual(result["iterations"], 1)
        self.assertIn("没有非空 content", result["error"])

    def test_llm_api_errors_return_safe_structured_response(self):
        errors = [
            TimeoutError("timeout secret"),
            LLMClientError("LLM API HTTP 401：invalid secret"),
            LLMClientError("LLM API 返回不是合法 JSON：secret"),
        ]

        for error in errors:
            with self.subTest(error=error):
                result = run_tool_calling_agent(
                    "预测销量",
                    self.context,
                    RaisingChatClient(error),
                    self.config,
                )

                self.assertFalse(result["ok"])
                self.assertEqual(result["answer"], "")
                self.assertEqual(result["tool_calls"], [])
                self.assertEqual(result["iterations"], 1)
                self.assertIn("LLM 请求失败", result["error"])
                self.assertNotIn("secret", result["error"])
                self.assertIn("analysis_context", result)


if __name__ == "__main__":
    unittest.main()
