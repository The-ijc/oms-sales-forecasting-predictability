from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Set


REQUIRED_FIELDS = {
    "executive_summary",
    "business_interpretation",
    "risk_recommendations",
    "follow_up_questions",
    "evidence_references",
}

BANNED_PHRASES = [
    "保证准确率",
    "上传客户明细",
    "上传数据库",
    "上传数据库文件",
    "上传原始交易",
    "客户明细",
    "原始交易数据",
]

MODEL_TERMS = [
    "Naive",
    "最近值",
    "三期移动平均",
    "SES",
    "Holt",
    "线性回归",
    "Seasonal Naive",
    "Random Forest",
    "Croston",
    "ARIMA",
    "Prophet",
    "XGBoost",
    "LSTM",
]


class SafetyValidationError(ValueError):
    """LLM output failed grounding checks."""


def parse_and_validate_llm_json(raw_text: str, fact_pack: Dict[str, Any], max_output_chars: int = 2400) -> Dict[str, Any]:
    if len(raw_text or "") > max_output_chars:
        raise SafetyValidationError("LLM 输出过长。")
    parsed = _parse_json_object(raw_text)
    _validate_required_fields(parsed)
    _validate_text_lengths(parsed, max_output_chars)
    _validate_banned_phrases(parsed)
    _validate_model_names(parsed, fact_pack)
    _validate_numbers_and_months(parsed, fact_pack)
    return parsed


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SafetyValidationError("LLM 输出不是合法 JSON。") from exc
    if not isinstance(parsed, dict):
        raise SafetyValidationError("LLM 输出必须是 JSON 对象。")
    return parsed


def _validate_required_fields(parsed: Dict[str, Any]):
    missing = REQUIRED_FIELDS - set(parsed)
    if missing:
        raise SafetyValidationError(f"LLM 输出缺少字段：{', '.join(sorted(missing))}")
    if not isinstance(parsed["executive_summary"], str):
        raise SafetyValidationError("executive_summary 必须是字符串。")
    if not isinstance(parsed["business_interpretation"], str):
        raise SafetyValidationError("business_interpretation 必须是字符串。")
    for field in ("risk_recommendations", "follow_up_questions", "evidence_references"):
        if not isinstance(parsed[field], list) or not all(isinstance(item, str) for item in parsed[field]):
            raise SafetyValidationError(f"{field} 必须是字符串列表。")


def _validate_text_lengths(parsed: Dict[str, Any], max_output_chars: int):
    if len(parsed["executive_summary"]) > 180:
        raise SafetyValidationError("executive_summary 超过 180 字。")
    if len(_combined_text(parsed)) > max_output_chars:
        raise SafetyValidationError("LLM 输出正文过长。")
    for field in ("risk_recommendations", "follow_up_questions", "evidence_references"):
        for item in parsed[field]:
            if len(item) > 220:
                raise SafetyValidationError(f"{field} 中存在过长条目。")


def _validate_banned_phrases(parsed: Dict[str, Any]):
    text = _combined_text(parsed)
    for phrase in BANNED_PHRASES:
        if phrase in text:
            raise SafetyValidationError(f"LLM 输出包含禁止表达：{phrase}")


def _validate_model_names(parsed: Dict[str, Any], fact_pack: Dict[str, Any]):
    text = _combined_text(parsed)
    allowed_models = set(str(item) for item in fact_pack.get("allowed_models", []) if item)
    allowed_text = " ".join(allowed_models)
    for term in MODEL_TERMS:
        if term in text and term not in allowed_text:
            raise SafetyValidationError(f"LLM 输出包含 fact pack 中不存在的模型名：{term}")


def _validate_numbers_and_months(parsed: Dict[str, Any], fact_pack: Dict[str, Any]):
    text = _combined_text(parsed)
    allowed_tokens = _allowed_numeric_tokens(fact_pack)

    month_tokens = set(re.findall(r"\b20\d{4}\b", text))
    allowed_months = {token for token in allowed_tokens if re.fullmatch(r"20\d{4}", token)}
    unknown_months = month_tokens - allowed_months
    if unknown_months:
        raise SafetyValidationError(f"LLM 输出包含 fact pack 中不存在的月份：{', '.join(sorted(unknown_months))}")

    numeric_tokens = re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?", text)
    for token in numeric_tokens:
        normalized = _normalize_number_token(token)
        if normalized and normalized not in allowed_tokens:
            raise SafetyValidationError(f"LLM 输出包含 fact pack 中不存在或不一致的数值：{token}")


def _combined_text(parsed: Dict[str, Any]) -> str:
    parts = [parsed.get("executive_summary", ""), parsed.get("business_interpretation", "")]
    for field in ("risk_recommendations", "follow_up_questions", "evidence_references"):
        parts.extend(parsed.get(field, []))
    return "\n".join(str(part) for part in parts if part is not None)


def _allowed_numeric_tokens(fact_pack: Dict[str, Any]) -> Set[str]:
    tokens: Set[str] = set()
    for value in _walk_values(fact_pack):
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            tokens.update(_number_variants(value))
        elif isinstance(value, str):
            for token in re.findall(r"[-+]?\d+(?:\.\d+)?%?", value):
                tokens.add(_normalize_number_token(token))
    return {token for token in tokens if token}


def _number_variants(value: float) -> Set[str]:
    variants = {
        _normalize_number_token(str(value)),
        _normalize_number_token(f"{value:.0f}"),
        _normalize_number_token(f"{value:.1f}"),
        _normalize_number_token(f"{value:.2f}"),
    }
    if -1 <= float(value) <= 1:
        percent = float(value) * 100
        variants.update(
            {
                _normalize_number_token(f"{percent:.0f}%"),
                _normalize_number_token(f"{percent:.1f}%"),
                _normalize_number_token(f"{percent:.2f}%"),
            }
        )
    return variants


def _normalize_number_token(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return ""
    if text.endswith("%"):
        number = text[:-1]
        try:
            return f"{float(number):.2f}%"
        except ValueError:
            return text
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value
