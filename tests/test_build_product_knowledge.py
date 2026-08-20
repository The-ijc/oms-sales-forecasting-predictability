from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.build_product_knowledge import (  # noqa: E402
    SkuRecord,
    SourceCandidate,
    choose_source,
    collect_product_records,
    extract_skus,
    merge_product_records,
    parse_product_html,
    write_products_json,
)
from 动销预测.agent.product_knowledge_retriever import ProductKnowledgeRetriever  # noqa: E402


PRODUCT_HTML = """
<html>
  <head>
    <title>官方 EWH-10B2 商品页</title>
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "EWH-10B2 测试商品",
        "brand": {"@type": "Brand", "name": "测试品牌"},
        "category": "测试品类",
        "releaseDate": "2025-03-01",
        "description": "来自公开商品页的产品介绍。",
        "featureList": ["功能 A", "功能 B"],
        "additionalProperty": [
          {"@type": "PropertyValue", "name": "容量", "value": "10L"}
        ]
      }
    </script>
  </head>
</html>
"""


def product_record(model: str, **overrides):
    record = SkuRecord(model, "").to_product_record()
    record.update(overrides)
    return record


class BuildProductKnowledgeTest(unittest.TestCase):
    def _create_sku_db(self, directory: str) -> Path:
        path = Path(directory) / "sku.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                'CREATE TABLE source_skus ("型号" TEXT, "品类" TEXT, "细分类" TEXT, "上市时间" TEXT)'
            )
            connection.executemany(
                "INSERT INTO source_skus VALUES (?, ?, ?, ?)",
                [
                    ("EWH-10B2", "热水器", "小厨宝", "202503"),
                    ("EWH 10B2", "", "", ""),
                    ("PF25C1", "净水", "前置过滤器", None),
                    (None, "无效", "无效", None),
                ],
            )
            connection.execute(
                'CREATE VIEW "vw_product_sales_customer_3" AS SELECT * FROM source_skus'
            )
            connection.commit()
        finally:
            connection.close()
        return path

    def test_extracts_sqlite_skus_with_business_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skus = extract_skus(str(self._create_sku_db(temp_dir)))

        self.assertEqual([sku.normalized_model for sku in skus], ["EWH10B2", "PF25C1"])
        self.assertEqual(skus[0].category, "热水器")
        self.assertEqual(skus[0].subcategory, "小厨宝")
        self.assertEqual(skus[0].launch_date, "202503")

    def test_sku_extraction_deduplicates_equivalent_model_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skus = extract_skus(str(self._create_sku_db(temp_dir)))

        self.assertEqual(sum(sku.normalized_model == "EWH10B2" for sku in skus), 1)

    def test_parses_one_product_from_json_ld(self):
        record = parse_product_html(PRODUCT_HTML)

        self.assertEqual(record["brand"], "测试品牌")
        self.assertEqual(record["product_name"], "EWH-10B2 测试商品")
        self.assertEqual(record["specifications"], {"容量": "10L"})
        self.assertEqual(record["features"], ["功能 A", "功能 B"])

    def test_two_column_specifications_table_is_used_as_fallback(self):
        page_html = """
        <html><body>
          <h2>Specifications</h2>
          <table>
            <tr><th>Model</th><td>EWH-10B2</td></tr>
            <tr><th>Storage Capacity</th><td>2.6 gal (9.84L)</td></tr>
            <tr><th>Heating Element</th><td>2kW @ 220V</td></tr>
            <tr><th>Empty Value</th><td> </td></tr>
          </table>
        </body></html>
        """

        record = parse_product_html(page_html)

        self.assertEqual(
            record["specifications"],
            {
                "Model": "EWH-10B2",
                "Storage Capacity": "2.6 gal (9.84L)",
                "Heating Element": "2kW @ 220V",
            },
        )

    def test_features_list_is_used_as_fallback(self):
        page_html = """
        <html><body>
          <h2>Key Features</h2>
          <ul>
            <li> Compact installation </li>
            <li>Fast heating</li>
            <li>Compact installation</li>
          </ul>
        </body></html>
        """

        record = parse_product_html(page_html)

        self.assertEqual(record["features"], ["Compact installation", "Fast heating"])

    def test_json_ld_values_take_priority_over_html_fallback(self):
        page_html = """
        <html><head>
          <script type="application/ld+json">
            {
              "@type": "Product",
              "brand": {"name": "Structured Brand"},
              "featureList": ["Structured feature"],
              "additionalProperty": [
                {"name": "Model", "value": "STRUCTURED-1"}
              ]
            }
          </script>
        </head><body>
          <h2>Features</h2><ul><li>Fallback feature</li></ul>
          <h2>Specifications</h2>
          <table><tr><td>Model</td><td>FALLBACK-1</td></tr></table>
        </body></html>
        """

        record = parse_product_html(page_html, "https://www.aosmith.com.ph/product")

        self.assertEqual(record["brand"], "Structured Brand")
        self.assertEqual(record["features"], ["Structured feature"])
        self.assertEqual(record["specifications"], {"Model": "STRUCTURED-1"})

    def test_fallback_fills_only_empty_structured_fields(self):
        page_html = """
        <html><head>
          <meta name="product:brand" content="Metadata Brand">
          <script type="application/ld+json">
            {"@type": "Product", "name": "Structured Product"}
          </script>
        </head><body>
          <h2>Product Features</h2><ul><li>Fallback feature</li></ul>
          <h2>Specifications</h2>
          <table><tr><td>Model</td><td>EWH-10B2</td></tr></table>
        </body></html>
        """

        record = parse_product_html(page_html, "https://www.aosmith.com.ph/product")

        self.assertEqual(record["brand"], "Metadata Brand")
        self.assertEqual(record["product_name"], "Structured Product")
        self.assertEqual(record["features"], ["Fallback feature"])
        self.assertEqual(record["specifications"], {"Model": "EWH-10B2"})

    def test_explicit_domain_mapping_supplies_brand(self):
        record = parse_product_html(
            "<html><body><h1>EWH-10B2</h1></body></html>",
            "https://www.aosmith.com.ph/products/EWH-10B2",
        )

        self.assertEqual(record["brand"], "A.O. Smith")

    def test_navigation_list_is_not_treated_as_features(self):
        page_html = """
        <html><body>
          <nav>
            <h2>Features</h2>
            <ul><li>Products</li><li>Support</li><li>Contact</li></ul>
          </nav>
          <h2>Overview</h2>
          <ul><li>General page item</li></ul>
        </body></html>
        """

        record = parse_product_html(page_html)

        self.assertEqual(record["features"], [])

    def test_empty_html_keeps_unknown_fields_empty(self):
        record = parse_product_html("")

        self.assertEqual(record["brand"], "")
        self.assertEqual(record["product_name"], "")
        self.assertEqual(record["specifications"], {})
        self.assertEqual(record["features"], [])

    def test_network_failure_does_not_interrupt_batch(self):
        skus = [
            SkuRecord("FAIL-1", "FAIL1"),
            SkuRecord("EWH-10B2", "EWH10B2"),
        ]

        def resolver(sku):
            return SourceCandidate(f"https://example.com/{sku.normalized_model}", "brand_official")

        def fetcher(url):
            if url.endswith("FAIL1"):
                raise TimeoutError("request timed out")
            return PRODUCT_HTML

        records, failures = collect_product_records(skus, resolver, fetcher)

        self.assertEqual([record["model"] for record in records], ["EWH-10B2"])
        self.assertEqual(failures[0]["model"], "FAIL-1")

    def test_products_json_merge_preserves_existing_valid_data(self):
        existing = product_record(
            "EWH-10B2",
            brand="已有品牌",
            sources=[{
                "title": "已有官方页",
                "url": "https://official.example/EWH-10B2",
                "source_type": "brand_official",
                "retrieved_at": "2026-01-01T00:00:00Z",
            }],
        )
        incoming = product_record(
            "EWH 10B2",
            description="新补充的可靠介绍",
            sources=[{
                "title": "零售页",
                "url": "https://retailer.example/EWH10B2",
                "source_type": "retailer",
                "retrieved_at": "2026-02-01T00:00:00Z",
            }],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "products.json"
            write_products_json(output, merge_product_records([existing], [incoming]))
            records = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["brand"], "已有品牌")
        self.assertEqual(records[0]["description"], "新补充的可靠介绍")
        self.assertEqual(records[0]["sources"][0]["url"], "https://official.example/EWH-10B2")

    def test_normalized_model_deduplicates_during_merge(self):
        merged = merge_product_records(
            [product_record("EWH-10B2", brand="品牌")],
            [product_record("EWH10B2", description="介绍")],
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["normalized_model"], "EWH10B2")

    def test_source_url_is_saved_and_only_one_source_is_selected(self):
        sku = SkuRecord("EWH-10B2", "EWH10B2")
        candidates = [
            SourceCandidate("https://retailer.example/item", "retailer"),
            SourceCandidate("https://brand.example/item", "brand_official", "品牌官网"),
        ]
        selected = choose_source(candidates)
        records, failures = collect_product_records(
            [sku],
            lambda _sku: selected,
            lambda _url: PRODUCT_HTML,
            lambda: "2026-08-20T00:00:00+00:00",
        )

        self.assertEqual(failures, [])
        self.assertEqual(len(records[0]["sources"]), 1)
        self.assertEqual(records[0]["sources"][0]["url"], "https://brand.example/item")
        self.assertEqual(records[0]["sources"][0]["source_type"], "brand_official")

    def test_product_retriever_reads_generated_record(self):
        sku = SkuRecord("EWH-10B2", "EWH10B2")
        records, _ = collect_product_records(
            [sku],
            lambda _sku: SourceCandidate("https://brand.example/item", "brand_official"),
            lambda _url: PRODUCT_HTML,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            write_products_json(data_dir / "products.json", records)
            results = ProductKnowledgeRetriever(data_dir).search("EWH10B2 是什么商品")

        self.assertTrue(results)
        self.assertEqual(results[0]["model"], "EWH-10B2")
        self.assertIn("测试品牌", results[0]["content"])
        self.assertNotIn(str(data_dir.resolve()), str(results))


if __name__ == "__main__":
    unittest.main()
