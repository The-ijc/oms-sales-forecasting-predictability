from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from 动销预测.agent.product_knowledge_retriever import (  # noqa: E402
    ProductKnowledgeRetriever,
    normalize_product_model,
)
from 动销预测.agent.tools import execute_tool  # noqa: E402


def product_record(model: str, **overrides):
    record = {
        "model": model,
        "normalized_model": normalize_product_model(model),
        "brand": "",
        "category": "",
        "subcategory": "",
        "product_name": "",
        "launch_date": "",
        "specifications": {},
        "features": [],
        "description": "",
        "sources": [],
    }
    record.update(overrides)
    return record


class ProductKnowledgeRetrieverTest(unittest.TestCase):
    def _retriever_with_records(self, records):
        temp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(temp_dir.name)
        (data_dir / "products.json").write_text(
            json.dumps(records, ensure_ascii=False),
            encoding="utf-8",
        )
        return temp_dir, ProductKnowledgeRetriever(data_dir)

    def test_model_normalization_preserves_letters_and_digits(self):
        self.assertEqual(normalize_product_model("EWH-10B2"), "EWH10B2")
        self.assertEqual(normalize_product_model("EWH 10B2"), "EWH10B2")
        self.assertEqual(normalize_product_model("EWH10B2"), "EWH10B2")
        self.assertEqual(normalize_product_model("CXW-350-Q2CWi"), "CXW350Q2CWI")

    def test_exact_raw_model_match(self):
        results = ProductKnowledgeRetriever().search("EWH-10B2 是什么产品？")

        self.assertTrue(results)
        self.assertEqual(results[0]["model"], "EWH-10B2")
        self.assertEqual(results[0]["match_type"], "exact_model")

    def test_model_without_hyphen_matches_normalized_model(self):
        results = ProductKnowledgeRetriever().search("EWH10B2 是什么产品？")

        self.assertTrue(results)
        self.assertEqual(results[0]["model"], "EWH-10B2")
        self.assertEqual(results[0]["match_type"], "normalized_model_exact")

    def test_short_model_is_not_treated_as_exact_substring_of_longer_model(self):
        temp_dir, retriever = self._retriever_with_records(
            [product_record("EWH-10B2"), product_record("CEWH-10B2")]
        )
        try:
            results = retriever.search("CEWH-10B2 是什么产品？")
        finally:
            temp_dir.cleanup()

        self.assertTrue(results)
        self.assertEqual(results[0]["model"], "CEWH-10B2")
        self.assertEqual(results[0]["match_type"], "exact_model")

    def test_exact_model_precedes_keyword_fallback(self):
        temp_dir, retriever = self._retriever_with_records(
            [
                product_record("EWH-10B2"),
                product_record(
                    "OTHER-1",
                    category="测试分类",
                    description="目标关键词 目标关键词 目标关键词 商品介绍",
                ),
            ]
        )
        try:
            results = retriever.search("EWH-10B2 目标关键词 商品介绍", top_k=2)
        finally:
            temp_dir.cleanup()

        self.assertEqual(results[0]["model"], "EWH-10B2")
        self.assertEqual(results[0]["match_type"], "exact_model")

    def test_top_k_limits_product_results(self):
        records = [
            product_record(f"SKU-{index}", description=f"共同功能 商品资料 {index}")
            for index in range(6)
        ]
        temp_dir, retriever = self._retriever_with_records(records)
        try:
            results = retriever.search("共同功能 商品资料", top_k=2)
        finally:
            temp_dir.cleanup()

        self.assertEqual(len(results), 2)

    def test_unmatched_query_returns_empty_results(self):
        results = ProductKnowledgeRetriever().search("火星轨道量子天气")

        self.assertEqual(results, [])

    def test_results_do_not_expose_local_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = str((Path(temp_dir) / "cached_page.html").resolve())
            records = [
                product_record(
                    "SAFE-1",
                    sources=[
                        {
                            "title": "资料页",
                            "url": local_path,
                            "source_type": "local_cache",
                            "retrieved_at": "2026-01-01T00:00:00Z",
                        }
                    ],
                )
            ]
            (Path(temp_dir) / "products.json").write_text(
                json.dumps(records, ensure_ascii=False),
                encoding="utf-8",
            )
            results = ProductKnowledgeRetriever(Path(temp_dir)).search("SAFE-1")

        self.assertTrue(results)
        self.assertNotIn(local_path, str(results))
        self.assertEqual(results[0]["sources"][0]["url"], "")

    def test_product_tool_uses_protected_product_model(self):
        result = execute_tool(
            "search_product_knowledge",
            {"query": "这个商品是什么？", "product_model": "EWH-6B2", "top_k": 3},
            protected_context={"product_model": "EWH-10B2", "db_path": "protected.sqlite"},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "search_product_knowledge")
        self.assertEqual(result["product_model"], "EWH-10B2")
        self.assertTrue(result["results"])
        self.assertEqual(result["results"][0]["model"], "EWH-10B2")
        self.assertNotIn("protected.sqlite", str(result))

    def test_records_without_sources_keep_unknown_product_fields_empty(self):
        records = ProductKnowledgeRetriever().load_records()

        self.assertEqual(len(records), 12)
        for record in records:
            if record["sources"]:
                continue
            self.assertEqual(record["brand"], "")
            self.assertEqual(record["product_name"], "")
            self.assertEqual(record["specifications"], {})
            self.assertEqual(record["features"], [])


if __name__ == "__main__":
    unittest.main()
