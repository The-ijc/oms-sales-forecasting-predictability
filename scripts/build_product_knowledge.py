from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access import connect_read_only, get_default_db_path, quote_identifier  # noqa: E402
from 动销预测.agent.product_knowledge_retriever import normalize_product_model  # noqa: E402


SKU_VIEW = "vw_product_sales_customer_3"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "product_knowledge" / "products.json"
SOURCE_FIELDS = ("title", "url", "source_type", "retrieved_at")
SOURCE_PRIORITY = {
    "brand_official": 1,
    "official_flagship": 2,
    "retailer": 3,
    "public_parameters": 4,
    "unknown": 99,
}
DOMAIN_BRAND_MAP = {
    "aosmith.com.ph": "A.O. Smith",
}
MAX_SPECIFICATION_FIELDS = 40
MAX_FEATURES = 20
MAX_FEATURE_LENGTH = 300
_FEATURE_HEADINGS = {"features", "product features", "key features"}
_SPECIFICATION_HEADINGS = {"specification", "specifications", "product specifications"}


@dataclass(frozen=True)
class SkuRecord:
    model: str
    normalized_model: str
    category: str = ""
    subcategory: str = ""
    launch_date: str = ""

    def to_product_record(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "normalized_model": self.normalized_model,
            "brand": "",
            "category": self.category,
            "subcategory": self.subcategory,
            "product_name": "",
            "launch_date": self.launch_date,
            "specifications": {},
            "features": [],
            "description": "",
            "sources": [],
        }


@dataclass(frozen=True)
class SourceCandidate:
    url: str
    source_type: str = "unknown"
    title: str = ""


class _ProductHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta: Dict[str, str] = {}
        self.title_parts: List[str] = []
        self.json_ld_parts: List[List[str]] = []
        self.tables: List[Dict[str, Any]] = []
        self.fallback_features: List[str] = []
        self._in_title = False
        self._in_json_ld = False
        self._heading_tag = ""
        self._heading_parts: List[str] = []
        self._last_heading = ""
        self._table: Optional[Dict[str, Any]] = None
        self._row: Optional[List[str]] = None
        self._cell_parts: Optional[List[str]] = None
        self._features_section = False
        self._list_depth = 0
        self._list_item_parts: Optional[List[str]] = None
        self._blocked_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if tag in {"nav", "footer", "header"}:
            self._blocked_depth += 1
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            if key and attributes.get("content"):
                self.meta[key] = attributes["content"]
        elif tag == "script" and "ld+json" in attributes.get("type", "").lower():
            self._in_json_ld = True
            self.json_ld_parts.append([])
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and not self._blocked_depth:
            self._heading_tag = tag
            self._heading_parts = []
        elif tag == "table" and self._table is None and not self._blocked_depth:
            self._table = {"heading": self._last_heading, "rows": []}
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell_parts = []
        elif tag in {"ul", "ol"} and self._features_section and not self._blocked_depth:
            self._list_depth += 1
        elif tag == "li" and self._list_depth and not self._blocked_depth:
            self._list_item_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_json_ld = False
        elif tag == self._heading_tag:
            heading = _clean_text("".join(self._heading_parts))
            self._last_heading = heading
            normalized_heading = _normalize_section_heading(heading)
            self._features_section = normalized_heading in _FEATURE_HEADINGS
            self._heading_tag = ""
            self._heading_parts = []
        elif tag in {"th", "td"} and self._cell_parts is not None:
            self._row.append(_clean_text("".join(self._cell_parts)))
            self._cell_parts = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            self._table["rows"].append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None
        elif tag == "li" and self._list_item_parts is not None:
            feature = _clean_text("".join(self._list_item_parts))
            if feature:
                self.fallback_features.append(feature)
            self._list_item_parts = None
        elif tag in {"ul", "ol"} and self._list_depth:
            self._list_depth -= 1
        if tag in {"nav", "footer", "header"} and self._blocked_depth:
            self._blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._in_json_ld and self.json_ld_parts:
            self.json_ld_parts[-1].append(data)
        if self._heading_tag:
            self._heading_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        if self._list_item_parts is not None:
            self._list_item_parts.append(data)


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _find_product_object(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            found = _find_product_object(item)
            if found:
                return found
        return None
    if not isinstance(value, dict):
        return None
    object_type = value.get("@type")
    types = object_type if isinstance(object_type, list) else [object_type]
    if any(str(item).lower() == "product" for item in types):
        return value
    for key in ("@graph", "mainEntity", "itemListElement"):
        found = _find_product_object(value.get(key))
        if found:
            return found
    return None


def _brand_name(value: Any) -> str:
    if isinstance(value, dict):
        return _clean_text(value.get("name"))
    return _clean_text(value)


def _specifications(value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        return {}
    result: Dict[str, Any] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("name") or item.get("propertyID"))
        item_value = item.get("value")
        if name and item_value not in (None, ""):
            result[name] = _clean_text(item_value)
    return result


def _features(value: Any) -> List[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(r"[;；,，\n]", value)
    else:
        items = []
    return list(dict.fromkeys(text for item in items if (text := _clean_text(item))))


def _normalize_section_heading(value: Any) -> str:
    return re.sub(r"[^a-z]+", " ", _clean_text(value).lower()).strip()


def _fallback_specifications(tables: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    for table in tables:
        heading = _normalize_section_heading(table.get("heading"))
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        if not heading and rows and len(rows[0]) == 1:
            heading = _normalize_section_heading(rows[0][0])
            rows = rows[1:]
        if heading not in _SPECIFICATION_HEADINGS:
            continue

        result: Dict[str, str] = {}
        invalid_table = False
        for row in rows:
            if not isinstance(row, list) or not any(_clean_text(cell) for cell in row):
                continue
            if len(row) != 2:
                invalid_table = True
                break
            key, value = (_clean_text(cell) for cell in row)
            if not key or not value:
                continue
            if len(key) > 120 or len(value) > 500:
                continue
            if key not in result:
                result[key] = value
            if len(result) >= MAX_SPECIFICATION_FIELDS:
                break
        if result and not invalid_table:
            return result
    return {}


def _fallback_features(values: Iterable[Any]) -> List[str]:
    features: List[str] = []
    for value in values:
        feature = _clean_text(value)
        if not feature or len(feature) > MAX_FEATURE_LENGTH or feature in features:
            continue
        features.append(feature)
        if len(features) >= MAX_FEATURES:
            break
    return features


def _brand_from_domain(source_url: str) -> str:
    hostname = (urllib.parse.urlparse(str(source_url or "")).hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    for domain, brand in DOMAIN_BRAND_MAP.items():
        if hostname == domain or hostname.endswith("." + domain):
            return brand
    return ""


def parse_product_html(page_html: str, source_url: str = "") -> Dict[str, Any]:
    parser = _ProductHTMLParser()
    parser.feed(str(page_html or ""))
    product: Dict[str, Any] = {}
    for parts in parser.json_ld_parts:
        try:
            value = json.loads("".join(parts))
        except json.JSONDecodeError:
            continue
        product = _find_product_object(value) or product
        if product:
            break

    page_title = _clean_text("".join(parser.title_parts))
    structured_specifications = _specifications(product.get("additionalProperty"))
    structured_features = _features(product.get("featureList"))
    return {
        "brand": _first_nonempty(
            _brand_name(product.get("brand")),
            parser.meta.get("product:brand"),
            parser.meta.get("og:brand"),
            parser.meta.get("brand"),
            _brand_from_domain(source_url),
        ),
        "category": _first_nonempty(product.get("category"), parser.meta.get("product:category")),
        "subcategory": "",
        "product_name": _first_nonempty(
            product.get("name"),
            parser.meta.get("og:title"),
            page_title,
        ),
        "launch_date": _first_nonempty(product.get("releaseDate"), product.get("datePublished")),
        "specifications": structured_specifications or _fallback_specifications(parser.tables),
        "features": structured_features or _fallback_features(parser.fallback_features),
        "description": _first_nonempty(
            product.get("description"),
            parser.meta.get("og:description"),
            parser.meta.get("description"),
        ),
        "page_title": page_title,
    }


def extract_skus(db_path: str) -> List[SkuRecord]:
    columns = {
        "model": quote_identifier("型号"),
        "category": quote_identifier("品类"),
        "subcategory": quote_identifier("细分类"),
        "launch_date": quote_identifier("上市时间"),
    }
    sql = f"""
        SELECT
            TRIM(CAST({columns['model']} AS TEXT)) AS model,
            TRIM(CAST({columns['category']} AS TEXT)) AS category,
            TRIM(CAST({columns['subcategory']} AS TEXT)) AS subcategory,
            TRIM(CAST({columns['launch_date']} AS TEXT)) AS launch_date
        FROM {quote_identifier(SKU_VIEW)}
        WHERE {columns['model']} IS NOT NULL
          AND TRIM(CAST({columns['model']} AS TEXT)) <> ''
        ORDER BY model
    """
    by_model: Dict[str, SkuRecord] = {}
    with closing(connect_read_only(db_path)) as connection:
        for row in connection.execute(sql):
            model = str(row["model"] or "").strip()
            normalized = normalize_product_model(model)
            if not normalized:
                continue
            incoming = SkuRecord(
                model=model,
                normalized_model=normalized,
                category=str(row["category"] or "").strip(),
                subcategory=str(row["subcategory"] or "").strip(),
                launch_date=str(row["launch_date"] or "").strip(),
            )
            existing = by_model.get(normalized)
            if existing is None:
                by_model[normalized] = incoming
            else:
                by_model[normalized] = SkuRecord(
                    model=existing.model,
                    normalized_model=normalized,
                    category=existing.category or incoming.category,
                    subcategory=existing.subcategory or incoming.subcategory,
                    launch_date=existing.launch_date or incoming.launch_date,
                )
    return sorted(by_model.values(), key=lambda item: item.normalized_model)


def _valid_web_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def choose_source(candidates: Sequence[SourceCandidate]) -> Optional[SourceCandidate]:
    valid = [candidate for candidate in candidates if _valid_web_url(candidate.url)]
    if not valid:
        return None
    return min(valid, key=lambda item: (SOURCE_PRIORITY.get(item.source_type, 99), item.url))


def fetch_public_page(url: str, timeout: float = 15.0, max_bytes: int = 2_000_000) -> str:
    if not _valid_web_url(url):
        raise ValueError("只允许公开 HTTP/HTTPS 商品页。")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OMS-ProductKnowledge-Collector/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("商品页面超过允许的大小。")
        charset = response.headers.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _canonical_source(value: Any) -> Optional[Dict[str, str]]:
    if not isinstance(value, dict):
        return None
    url = str(value.get("url") or "").strip()
    if not _valid_web_url(url):
        return None
    return {field: _clean_text(value.get(field)) for field in SOURCE_FIELDS}


def canonical_product_record(value: Mapping[str, Any]) -> Dict[str, Any]:
    model = str(value.get("model") or "").strip()
    normalized = normalize_product_model(model)
    if not model or not normalized:
        raise ValueError("商品记录缺少有效 model。")
    specifications = value.get("specifications") if isinstance(value.get("specifications"), dict) else {}
    features = value.get("features") if isinstance(value.get("features"), list) else []
    sources = value.get("sources") if isinstance(value.get("sources"), list) else []
    source = next((item for item in (_canonical_source(item) for item in sources) if item), None)
    return {
        "model": model,
        "normalized_model": normalized,
        "brand": _clean_text(value.get("brand")),
        "category": _clean_text(value.get("category")),
        "subcategory": _clean_text(value.get("subcategory")),
        "product_name": _clean_text(value.get("product_name")),
        "launch_date": _clean_text(value.get("launch_date")),
        "specifications": {str(key): item for key, item in specifications.items()},
        "features": list(dict.fromkeys(text for item in features if (text := _clean_text(item)))),
        "description": _clean_text(value.get("description")),
        "sources": [source] if source else [],
    }


def merge_product_record(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> Dict[str, Any]:
    left = canonical_product_record(existing)
    right = canonical_product_record(incoming)
    if left["normalized_model"] != right["normalized_model"]:
        raise ValueError("不能合并不同 normalized_model 的商品记录。")
    merged = dict(left)
    for field in ("brand", "category", "subcategory", "product_name", "launch_date", "description"):
        merged[field] = left[field] or right[field]
    merged["specifications"] = {**right["specifications"], **left["specifications"]}
    merged["features"] = list(dict.fromkeys([*left["features"], *right["features"]]))
    merged["sources"] = left["sources"][:1] or right["sources"][:1]
    return canonical_product_record(merged)


def merge_product_records(
    existing_records: Iterable[Mapping[str, Any]],
    incoming_records: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for value in [*existing_records, *incoming_records]:
        record = canonical_product_record(value)
        key = record["normalized_model"]
        merged[key] = merge_product_record(merged[key], record) if key in merged else record
    return sorted(merged.values(), key=lambda item: item["normalized_model"])


def load_products_json(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("products.json 顶层必须是数组。")
    return [canonical_product_record(item) for item in value if isinstance(item, dict)]


def write_products_json(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    canonical_records = merge_product_records([], records)
    payload = json.dumps(canonical_records, ensure_ascii=False, indent=2)
    json.loads(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(payload + "\n", encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def collect_product_records(
    skus: Sequence[SkuRecord],
    source_resolver: Callable[[SkuRecord], Optional[SourceCandidate]],
    fetcher: Callable[[str], str] = fetch_public_page,
    retrieved_at_factory: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat(),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for sku in skus:
        base_record = sku.to_product_record()
        try:
            candidate = source_resolver(sku)
            if candidate is None:
                records.append(base_record)
                continue
            page_html = fetcher(candidate.url)
            parsed = parse_product_html(page_html, candidate.url)
            collected = {
                **base_record,
                **{field: parsed.get(field) for field in (
                    "brand",
                    "category",
                    "subcategory",
                    "product_name",
                    "launch_date",
                    "specifications",
                    "features",
                    "description",
                )},
                "sources": [
                    {
                        "title": candidate.title or parsed.get("page_title") or "",
                        "url": candidate.url,
                        "source_type": candidate.source_type,
                        "retrieved_at": retrieved_at_factory(),
                    }
                ],
            }
            records.append(merge_product_record(base_record, collected))
        except Exception as exc:
            failures.append({"model": sku.model, "error": str(exc) or exc.__class__.__name__})
            continue
    return records, failures


def load_source_map(path: Optional[Path]) -> Dict[str, List[SourceCandidate]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source map 必须是以型号为 key 的 JSON object。")
    result: Dict[str, List[SourceCandidate]] = {}
    for model, raw_candidates in value.items():
        candidates = raw_candidates if isinstance(raw_candidates, list) else [raw_candidates]
        parsed_candidates = []
        for item in candidates:
            if isinstance(item, str):
                parsed_candidates.append(SourceCandidate(url=item))
            elif isinstance(item, dict):
                parsed_candidates.append(
                    SourceCandidate(
                        url=str(item.get("url") or ""),
                        source_type=str(item.get("source_type") or "unknown"),
                        title=str(item.get("title") or ""),
                    )
                )
        result[normalize_product_model(model)] = parsed_candidates
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 SQLite SKU 清单离线构建 Product Knowledge")
    parser.add_argument("--db-path", default="", help="SQLite 路径；默认读取 OMS_SQLITE_PATH")
    parser.add_argument("--list", action="store_true", help="仅列出待采集 SKU，不访问网络、不写文件")
    parser.add_argument("--model", default="", help="只处理一个型号，支持有无横杠的等价写法")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个 SKU")
    parser.add_argument("--source-url", default="", help="单型号最终可信网页 URL，必须配合 --model")
    parser.add_argument("--source-type", default="unknown", choices=list(SOURCE_PRIORITY))
    parser.add_argument("--source-map", type=Path, default=None, help="型号到候选 URL 的离线 JSON 映射")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    db_path = str(args.db_path or get_default_db_path() or "").strip()
    if not db_path:
        print("缺少数据库路径：请使用 --db-path 或设置 OMS_SQLITE_PATH。")
        return 2

    try:
        skus = extract_skus(db_path)
    except Exception as exc:
        print(f"SKU 提取失败：{exc}")
        return 2

    if args.model:
        requested = normalize_product_model(args.model)
        skus = [sku for sku in skus if sku.normalized_model == requested]
        if not skus:
            print(f"SQLite 中未找到型号：{args.model}")
            return 2
    if args.limit:
        if args.limit < 1:
            print("--limit 必须大于 0。")
            return 2
        skus = skus[: args.limit]

    if args.list:
        for sku in skus:
            print(
                f"{sku.model}\t品类={sku.category or '-'}\t"
                f"细分类={sku.subcategory or '-'}\t上市时间={sku.launch_date or '-'}"
            )
        print(f"SKU 数量：{len(skus)}")
        return 0

    if args.source_url and not args.model:
        print("--source-url 必须配合 --model 使用。")
        return 2

    try:
        source_map = load_source_map(args.source_map)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"source map 读取失败：{exc}")
        return 2
    if args.source_url and skus:
        source_map[skus[0].normalized_model] = [
            SourceCandidate(url=args.source_url, source_type=args.source_type)
        ]

    def resolve_source(sku: SkuRecord) -> Optional[SourceCandidate]:
        return choose_source(source_map.get(sku.normalized_model, []))

    records, failures = collect_product_records(skus, resolve_source)
    try:
        existing = load_products_json(args.output)
        merged = merge_product_records(existing, records)
        write_products_json(args.output, merged)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"products.json 更新失败：{exc}")
        return 2

    for failure in failures:
        print(f"采集失败，已跳过 {failure['model']}：{failure['error']}")
    print(f"SKU={len(skus)}，写入/合并={len(records)}，失败={len(failures)}，总记录={len(merged)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
