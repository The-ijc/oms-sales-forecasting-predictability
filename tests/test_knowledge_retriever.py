from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from 动销预测.agent.knowledge_retriever import (  # noqa: E402
    DEFAULT_KNOWLEDGE_DIR,
    KnowledgeRetriever,
)
from 动销预测.agent.tools import execute_tool  # noqa: E402


class KnowledgeRetrieverTest(unittest.TestCase):
    def test_related_query_returns_relevant_chunk(self):
        results = KnowledgeRetriever().search("WAPE 是什么意思", top_k=3)

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "forecasting_glossary.md")
        self.assertIn("WAPE", results[0]["title"] + results[0]["content"])
        self.assertIn("sum(abs(actual - predicted))", results[0]["content"])

    def test_unmatched_query_returns_empty_results(self):
        results = KnowledgeRetriever().search("火星轨道量子天气预报", top_k=3)

        self.assertEqual(results, [])

    def test_top_k_limits_result_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge_dir = Path(temp_dir)
            for index in range(6):
                (knowledge_dir / f"doc_{index}.md").write_text(
                    f"# 文档 {index}\n\n## 共同检索词\n\n共同检索词的说明 {index}。",
                    encoding="utf-8",
                )

            results = KnowledgeRetriever(knowledge_dir).search("共同检索词", top_k=2)

        self.assertEqual(len(results), 2)

    def test_results_do_not_expose_absolute_paths(self):
        results = KnowledgeRetriever().search("最近完整账期", top_k=3)

        self.assertTrue(results)
        rendered = str(results)
        self.assertNotIn(str(DEFAULT_KNOWLEDGE_DIR.resolve()), rendered)
        for result in results:
            self.assertEqual(result["source"], Path(result["source"]).name)
            self.assertEqual(result["document"], Path(result["document"]).name)

    def test_knowledge_tool_is_available_through_execute_tool(self):
        result = execute_tool(
            "search_knowledge_base",
            {"query": "经验预测区间是什么意思", "top_k": 2, "db_path": "untrusted.sqlite"},
            protected_context={"db_path": "protected.sqlite"},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "search_knowledge_base")
        self.assertEqual(result["query"], "经验预测区间是什么意思")
        self.assertLessEqual(len(result["results"]), 2)
        rendered = str(result)
        self.assertNotIn("untrusted.sqlite", rendered)
        self.assertNotIn("protected.sqlite", rendered)


if __name__ == "__main__":
    unittest.main()
