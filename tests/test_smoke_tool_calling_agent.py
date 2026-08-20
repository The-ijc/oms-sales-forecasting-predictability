from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import smoke_test_tool_calling_agent as smoke  # noqa: E402


class SmokeTestToolCallingAgentTest(unittest.TestCase):
    def test_missing_configuration_exits_before_agent_call(self):
        output = io.StringIO()

        with patch.object(smoke, "run_tool_calling_agent") as agent, redirect_stdout(output):
            exit_code = smoke.main([], {"LLM_MODE": "llm"})

        agent.assert_not_called()
        self.assertEqual(exit_code, 2)
        self.assertIn("LLM_API_KEY", output.getvalue())
        self.assertIn("OMS_SQLITE_PATH", output.getvalue())

    def test_smoke_output_redacts_secrets_and_protected_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "private.sqlite"
            db_path.touch()
            environ = {
                "LLM_MODE": "llm",
                "LLM_API_KEY": "top-secret-key",
                "LLM_BASE_URL": "https://example.test/v1",
                "LLM_MODEL": "mock-model",
                "OMS_SQLITE_PATH": str(db_path),
            }
            response = {
                "ok": True,
                "answer": "预测属于估计值。",
                "iterations": 2,
                "error": None,
                "tool_calls": [
                    {
                        "tool_name": "get_forecast_result",
                        "arguments": '{"horizon":3,"db_path":"' + str(db_path) + '"}',
                        "ok": True,
                        "result": {
                            "ok": True,
                            "db_path": str(db_path),
                            "说明": "source=" + str(db_path),
                            "未来预测值": [100],
                        },
                    }
                ],
            }
            output = io.StringIO()

            with (
                patch.object(smoke, "run_tool_calling_agent", return_value=response) as agent,
                patch.object(smoke, "OpenAICompatibleChatClient", return_value=object()),
                redirect_stdout(output),
            ):
                exit_code = smoke.main([], environ)

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        agent.assert_called_once()
        self.assertIn("get_forecast_result", rendered)
        self.assertIn("未来预测值", rendered)
        self.assertNotIn("top-secret-key", rendered)
        self.assertNotIn(str(db_path), rendered)
        self.assertNotIn("db_path", rendered)
        self.assertNotIn("system_prompt", rendered)


if __name__ == "__main__":
    unittest.main()
