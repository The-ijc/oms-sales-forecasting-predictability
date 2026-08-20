from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PRODUCT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "product_knowledge"
MAX_TOP_K = 5
_ASCII_TOKEN_PATTERN = re.compile(r"[A-Z]+|\d+|[\u4e00-\u9fff]+")
_SOURCE_FIELDS = ("title", "url", "source_type", "retrieved_at")


def normalize_product_model(model: Any) -> str:
    """Normalize separators and case while preserving every ASCII letter and digit."""
    return re.sub(r"[^A-Z0-9]", "", str(model or "").upper())


def _tokens(text: Any) -> List[str]:
    tokens: List[str] = []
    for token in _ASCII_TOKEN_PATTERN.findall(str(text or "").upper()):
        tokens.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 1:
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


def _model_tokens(model: str) -> List[str]:
    return re.findall(r"[A-Z]+|\d+", normalize_product_model(model))


def _is_local_absolute_path(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        text.lower().startswith("file://")
        or re.match(r"^[A-Za-z]:[\\/]", text)
        or text.startswith("/")
    )


def _raw_model_in_query(model: str, query: str) -> bool:
    pattern = rf"(?<![A-Z0-9]){re.escape(model.upper())}(?![A-Z0-9])"
    return re.search(pattern, query.upper()) is not None


def _normalized_model_candidates(query: str) -> set[str]:
    components = re.findall(r"[A-Za-z0-9]+", str(query or ""))
    candidates = {normalize_product_model(component) for component in components}
    for start in range(len(components)):
        for size in range(2, min(4, len(components) - start) + 1):
            candidates.add(normalize_product_model("".join(components[start : start + size])))
    return {candidate for candidate in candidates if candidate}


def _safe_sources(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    sources = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = {field: str(item.get(field) or "") for field in _SOURCE_FIELDS}
        for field in _SOURCE_FIELDS:
            if _is_local_absolute_path(source[field]):
                source[field] = ""
        sources.append(source)
    return sources


def _record_from_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    model = str(value.get("model") or "").strip()
    if not model:
        return None
    normalized_model = normalize_product_model(value.get("normalized_model") or model)
    if not normalized_model:
        return None
    specifications = value.get("specifications") if isinstance(value.get("specifications"), dict) else {}
    features = value.get("features") if isinstance(value.get("features"), list) else []
    return {
        "model": model,
        "normalized_model": normalized_model,
        "brand": str(value.get("brand") or "").strip(),
        "category": str(value.get("category") or "").strip(),
        "subcategory": str(value.get("subcategory") or "").strip(),
        "product_name": str(value.get("product_name") or "").strip(),
        "launch_date": str(value.get("launch_date") or "").strip(),
        "specifications": specifications,
        "features": [str(item) for item in features if str(item).strip()],
        "description": str(value.get("description") or "").strip(),
        "sources": _safe_sources(value.get("sources")),
    }


def _json_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("products"), list):
        value = value["products"]
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        yield value


def _content(record: Dict[str, Any]) -> str:
    parts = [f"商品型号：{record['model']}"]
    field_labels = (
        ("brand", "品牌"),
        ("product_name", "商品名称"),
        ("category", "分类"),
        ("subcategory", "子分类"),
        ("launch_date", "上市日期"),
        ("description", "产品介绍"),
    )
    for field, label in field_labels:
        if record.get(field):
            parts.append(f"{label}：{record[field]}")
    if record.get("specifications"):
        parts.append("主要规格：" + json.dumps(record["specifications"], ensure_ascii=False, sort_keys=True))
    if record.get("features"):
        parts.append("功能特点：" + "；".join(record["features"]))
    if len(parts) == 1:
        parts.append("当前结构化资料中尚无已验证的品牌、名称、规格、功能或产品介绍。")
    return "\n".join(parts)


class ProductKnowledgeRetriever:
    """Structured product retriever with model-first ranking and lexical fallback."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir or DEFAULT_PRODUCT_DATA_DIR)

    def load_records(self) -> List[Dict[str, Any]]:
        if not self.data_dir.is_dir():
            return []
        records = []
        paths = sorted(self.data_dir.glob("*.json")) + sorted(self.data_dir.glob("*.jsonl"))
        for path in paths:
            try:
                for value in _json_records(path):
                    record = _record_from_mapping(value)
                    if record:
                        records.append(record)
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
        return records

    def search(
        self,
        query: str,
        product_model: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        clean_query = str(query or "").strip()
        clean_model = str(product_model or "").strip()
        if clean_model == "全部":
            clean_model = ""
        if not clean_query and not clean_model:
            return []
        try:
            bounded_top_k = max(1, min(int(top_k), MAX_TOP_K))
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k 必须是整数。") from exc

        records = self.load_records()
        if not records:
            return []

        searchable_texts = [_content(record) for record in records]
        token_counts = [Counter(_tokens(text)) for text in searchable_texts]
        document_frequency = Counter()
        for counts in token_counts:
            document_frequency.update(counts.keys())
        average_length = sum(max(1, sum(counts.values())) for counts in token_counts) / len(token_counts)
        query_counts = Counter(_tokens(clean_query))
        query_upper = clean_query.upper()
        normalized_query_candidates = _normalized_model_candidates(clean_query)
        normalized_requested_model = normalize_product_model(clean_model)

        ranked = []
        for record, text, counts in zip(records, searchable_texts, token_counts):
            model = record["model"]
            normalized_model = record["normalized_model"]
            lexical_score = _bm25_score(
                query_counts,
                counts,
                document_frequency,
                document_count=len(records),
                document_length=max(1, sum(counts.values())),
                average_length=max(1.0, average_length),
            )

            if (clean_model and clean_model.upper() == model.upper()) or _raw_model_in_query(model, query_upper):
                priority, match_type = 4, "exact_model"
            elif (
                normalized_requested_model
                and normalized_requested_model == normalized_model
            ) or normalized_model in normalized_query_candidates:
                priority, match_type = 3, "normalized_model_exact"
            else:
                model_tokens = _model_tokens(model)
                query_model_tokens = set(_tokens(clean_query))
                if model_tokens and all(token in query_model_tokens for token in model_tokens):
                    priority, match_type = 2, "model_token_strong"
                elif lexical_score > 0:
                    priority, match_type = 1, "keyword_fallback"
                else:
                    continue

            score = priority * 100.0 + lexical_score
            ranked.append((priority, score, model, record, text, match_type))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [
            {
                "model": record["model"],
                "normalized_model": record["normalized_model"],
                "brand": record["brand"],
                "product_name": record["product_name"],
                "category": record["category"],
                "content": text,
                "sources": record["sources"],
                "score": round(score, 6),
                "match_type": match_type,
            }
            for _priority, score, _model, record, text, match_type in ranked[:bounded_top_k]
        ]


def _bm25_score(
    query_counts: Counter,
    document_counts: Counter,
    document_frequency: Counter,
    *,
    document_count: int,
    document_length: int,
    average_length: float,
) -> float:
    score = 0.0
    for token, query_frequency in query_counts.items():
        term_frequency = document_counts.get(token, 0)
        if not term_frequency:
            continue
        frequency = document_frequency.get(token, 0)
        inverse_document_frequency = math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
        denominator = term_frequency + 1.5 * (0.25 + 0.75 * document_length / average_length)
        score += inverse_document_frequency * term_frequency * 2.5 / denominator * query_frequency
    return score


def retrieve_product_knowledge(
    query: str,
    product_model: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    return ProductKnowledgeRetriever().search(query, product_model=product_model, top_k=top_k)
