from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from 动销预测.llm.client import LLMClientError, OpenAICompatibleChatClient  # noqa: E402
from 动销预测.llm.config import LLMConfig  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class RawResponse(FakeResponse):
    def read(self):
        return self.payload


class CapturingOpener:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.requests = []
        self.timeouts = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return FakeResponse(self.response_payload)

    @property
    def last_payload(self):
        return json.loads(self.requests[-1].data.decode("utf-8"))


def llm_config():
    return LLMConfig(
        mode="llm",
        api_key="secret",
        base_url="https://example.test/v1",
        model="mock-model",
        timeout_seconds=3,
        max_tokens=123,
    )


class OpenAICompatibleChatClientTest(unittest.TestCase):
    def test_complete_keeps_content_return_and_json_response_format(self):
        opener = CapturingOpener({"choices": [{"message": {"role": "assistant", "content": "{\"ok\":true}"}}]})
        client = OpenAICompatibleChatClient(opener=opener)

        result = client.complete([{"role": "user", "content": "hello"}], llm_config())

        self.assertEqual(result, "{\"ok\":true}")
        self.assertEqual(opener.last_payload["response_format"], {"type": "json_object"})

    def test_complete_message_sends_tools_and_tool_choice(self):
        tool_schema = {
            "type": "function",
            "function": {
                "name": "get_forecast_result",
                "description": "查询预测",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        opener = CapturingOpener({"choices": [{"message": {"role": "assistant", "content": "final"}}]})
        client = OpenAICompatibleChatClient(opener=opener)

        message = client.complete_message(
            [{"role": "user", "content": "预测未来三个月销量"}],
            llm_config(),
            tools=[tool_schema],
            tool_choice="auto",
        )

        self.assertEqual(message["content"], "final")
        self.assertEqual(opener.last_payload["tools"], [tool_schema])
        self.assertEqual(opener.last_payload["tool_choice"], "auto")
        self.assertNotIn("response_format", opener.last_payload)

    def test_complete_message_preserves_tool_calls_with_null_content(self):
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_forecast_result",
                "arguments": "{\"horizon\":3}",
            },
        }
        opener = CapturingOpener(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    }
                ]
            }
        )
        client = OpenAICompatibleChatClient(opener=opener)

        message = client.complete_message([{"role": "user", "content": "预测"}], llm_config())

        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["id"], "call_123")
        self.assertEqual(message["tool_calls"][0]["type"], "function")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_forecast_result")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], "{\"horizon\":3}")

    def test_tool_role_message_is_serialized_for_next_round(self):
        opener = CapturingOpener({"choices": [{"message": {"role": "assistant", "content": "final answer"}}]})
        client = OpenAICompatibleChatClient(opener=opener)
        messages = [
            {"role": "user", "content": "预测"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_forecast_result", "arguments": "{\"horizon\":3}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "{\"ok\":true}",
            },
        ]

        client.complete_message(messages, llm_config())

        self.assertEqual(opener.last_payload["messages"], messages)

    def test_timeout_is_converted_to_client_error(self):
        def timeout_opener(_request, timeout):
            raise TimeoutError("socket timeout")

        client = OpenAICompatibleChatClient(opener=timeout_opener)

        with self.assertRaisesRegex(LLMClientError, "调用超时"):
            client.complete_message([{"role": "user", "content": "预测"}], llm_config())

    def test_http_auth_error_redacts_api_key(self):
        def auth_error_opener(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                None,
                io.BytesIO(b'{"error":"invalid secret"}'),
            )

        client = OpenAICompatibleChatClient(opener=auth_error_opener)

        with self.assertRaises(LLMClientError) as caught:
            client.complete_message([{"role": "user", "content": "预测"}], llm_config())

        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))

    def test_malformed_json_response_is_converted_to_client_error(self):
        def malformed_opener(_request, timeout):
            return RawResponse(b"{not-json")

        client = OpenAICompatibleChatClient(opener=malformed_opener)

        with self.assertRaisesRegex(LLMClientError, "不是合法 JSON"):
            client.complete_message([{"role": "user", "content": "预测"}], llm_config())


if __name__ == "__main__":
    unittest.main()
