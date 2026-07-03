from __future__ import annotations

import os
import sqlite3
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import INCOMPLETE_LATEST_RATIO, MAX_OBJECTS_DEFAULT


TABLE_NAME = "adb_model_sales_summary"

FIELD_MAPPING = {
    "month": "fin_year_month",
    "channel": "channel",
    "channel_detail": "channel_",
    "product_model": "product_model",
    "product_category": "product_category",
    "product_class": "product_class",
    "target": "qty_total",
}

LEVEL_CONFIG = {
    "channel": {
        "label": "渠道大类",
        "fields": [("channel", "channel", "渠道大类")],
    },
    "channel_detail": {
        "label": "渠道细分类",
        "fields": [
            ("channel", "channel", "渠道大类"),
            ("channel_", "channel_detail", "渠道细分类"),
        ],
    },
    "channel_model": {
        "label": "渠道型号",
        "fields": [
            ("channel", "channel", "渠道大类"),
            ("channel_", "channel_detail", "渠道细分类"),
            ("product_model", "product_model", "型号"),
        ],
    },
}


class DataAccessError(RuntimeError):
    """数据库读取错误。"""


def get_default_db_path() -> str:
    """从环境变量读取默认数据库路径。"""
    return os.environ.get("OMS_SQLITE_PATH", "")


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def validate_db_path(db_path_text: str) -> Path:
    """检查数据库路径是否存在。"""
    if not db_path_text:
        raise DataAccessError("数据库路径为空，请设置 OMS_SQLITE_PATH 或在侧边栏输入路径。")
    db_path = Path(db_path_text).expanduser()
    if not db_path.exists():
        raise DataAccessError(f"数据库文件不存在：{db_path}")
    if not db_path.is_file():
        raise DataAccessError(f"数据库路径不是文件：{db_path}")
    return db_path


def connect_read_only(db_path_text: str) -> sqlite3.Connection:
    """以只读方式连接 SQLite，并开启 query_only 防止写入。"""
    db_path = validate_db_path(db_path_text)
    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _build_filter_clause(filters: Optional[Dict[str, str]]) -> Tuple[str, List[str]]:
    filters = filters or {}
    conditions = []
    params: List[str] = []

    for field_key in ("channel", "channel_detail", "product_model"):
        value = filters.get(field_key)
        if not value or value == "全部":
            continue
        source_field = FIELD_MAPPING[field_key]
        conditions.append(f"TRIM({quote_identifier(source_field)}) = ?")
        params.append(value.strip())

    month_field = quote_identifier(FIELD_MAPPING["month"])
    conditions.append(f"{month_field} IS NOT NULL")
    conditions.append(f"TRIM({month_field}) <> ''")

    if not conditions:
        return "", params
    return "WHERE " + " AND ".join(conditions), params


def _display_expr(source_field: str, alias: str) -> str:
    qfield = quote_identifier(source_field)
    return f"COALESCE(NULLIF(TRIM({qfield}), ''), '未填写') AS {quote_identifier(alias)}"


def _level_fields(level: str):
    if level not in LEVEL_CONFIG:
        raise DataAccessError(f"不支持的预测层级：{level}")
    return LEVEL_CONFIG[level]["fields"]


def get_distinct_values(
    db_path_text: str,
    field_key: str,
    filters: Optional[Dict[str, str]] = None,
    limit: int = 200,
) -> List[str]:
    """读取筛选项，只返回维度取值，不读取明细。"""
    if field_key not in FIELD_MAPPING:
        raise DataAccessError(f"不支持的筛选字段：{field_key}")
    source_field = FIELD_MAPPING[field_key]
    qfield = quote_identifier(source_field)
    where_clause, params = _build_filter_clause(filters)
    if where_clause:
        where_clause += f" AND {qfield} IS NOT NULL AND TRIM({qfield}) <> ''"
    else:
        where_clause = f"WHERE {qfield} IS NOT NULL AND TRIM({qfield}) <> ''"

    sql = f"""
        SELECT TRIM({qfield}) AS value,
               SUM(COALESCE({quote_identifier(FIELD_MAPPING['target'])}, 0)) AS total_qty
        FROM {quote_identifier(TABLE_NAME)}
        {where_clause}
        GROUP BY value
        ORDER BY total_qty DESC, value
        LIMIT ?
    """
    try:
        with connect_read_only(db_path_text) as conn:
            rows = conn.execute(sql, [*params, limit]).fetchall()
            return [row["value"] for row in rows]
    except sqlite3.Error as exc:
        raise DataAccessError(f"读取筛选项失败：{exc}") from exc


def get_global_monthly_summary(db_path_text: str) -> List[Dict[str, float]]:
    """读取全库月度聚合，用于判断最新月份是否可能不完整。"""
    month_field = quote_identifier(FIELD_MAPPING["month"])
    target_field = quote_identifier(FIELD_MAPPING["target"])
    sql = f"""
        SELECT TRIM({month_field}) AS month,
               COUNT(*) AS record_count,
               SUM(COALESCE({target_field}, 0)) AS qty_total
        FROM {quote_identifier(TABLE_NAME)}
        WHERE {month_field} IS NOT NULL AND TRIM({month_field}) <> ''
        GROUP BY month
        ORDER BY month
    """
    try:
        with connect_read_only(db_path_text) as conn:
            return [
                {
                    "month": row["month"],
                    "record_count": int(row["record_count"] or 0),
                    "qty_total": float(row["qty_total"] or 0),
                }
                for row in conn.execute(sql)
            ]
    except sqlite3.Error as exc:
        raise DataAccessError(f"读取月度汇总失败：{exc}") from exc


def detect_global_latest_incomplete(monthly_summary: Sequence[Dict[str, float]]):
    """根据全库最近月份记录数和动销量判断是否可能未完整入库。"""
    if len(monthly_summary) < 4:
        latest = monthly_summary[-1]["month"] if monthly_summary else None
        return {
            "latest_month": latest,
            "latest_complete_month": latest,
            "is_incomplete": False,
            "reasons": [],
        }

    latest = monthly_summary[-1]
    previous = monthly_summary[-4:-1]
    median_records = statistics.median(row["record_count"] for row in previous)
    median_qty = statistics.median(row["qty_total"] for row in previous)
    reasons = []

    if median_records > 0 and latest["record_count"] < median_records * 0.5:
        reasons.append(
            f"最新月份 {latest['month']} 的全库记录数低于前三个月中位数的 50%，可能尚未完整。"
        )
    if median_qty > 0 and latest["qty_total"] <= median_qty * INCOMPLETE_LATEST_RATIO:
        reasons.append(
            f"最新月份 {latest['month']} 的全库动销量低于前三个月中位数的 {INCOMPLETE_LATEST_RATIO:.0%}，可能尚未完整。"
        )

    is_incomplete = bool(reasons)
    latest_complete_month = monthly_summary[-2]["month"] if is_incomplete else latest["month"]
    return {
        "latest_month": latest["month"],
        "latest_complete_month": latest_complete_month,
        "is_incomplete": is_incomplete,
        "reasons": reasons,
    }


def get_aggregated_series(
    db_path_text: str,
    level: str,
    filters: Optional[Dict[str, str]] = None,
    limit: int = MAX_OBJECTS_DEFAULT,
) -> List[Dict]:
    """按预测层级读取月度聚合序列，默认最多返回前 50 个对象。"""
    fields = _level_fields(level)
    select_fields = ", ".join(_display_expr(source, alias) for source, alias, _ in fields)
    alias_names = [alias for _, alias, _ in fields]
    group_fields = ", ".join(quote_identifier(alias) for alias in alias_names)
    join_conditions = " AND ".join(
        f"monthly.{quote_identifier(alias)} = top_objects.{quote_identifier(alias)}"
        for alias in alias_names
    )
    order_fields = ", ".join(f"monthly.{quote_identifier(alias)}" for alias in alias_names)
    where_clause, params = _build_filter_clause(filters)

    month_field = quote_identifier(FIELD_MAPPING["month"])
    target_field = quote_identifier(FIELD_MAPPING["target"])
    sql = f"""
        WITH monthly AS (
            SELECT
                {select_fields},
                TRIM({month_field}) AS month,
                SUM(COALESCE({target_field}, 0)) AS qty
            FROM {quote_identifier(TABLE_NAME)}
            {where_clause}
            GROUP BY {group_fields}, month
        ),
        top_objects AS (
            SELECT {group_fields}, SUM(qty) AS total_qty
            FROM monthly
            GROUP BY {group_fields}
            ORDER BY total_qty DESC, {group_fields}
            LIMIT ?
        )
        SELECT monthly.*, top_objects.total_qty
        FROM monthly
        JOIN top_objects ON {join_conditions}
        ORDER BY top_objects.total_qty DESC, {order_fields}, monthly.month
    """

    try:
        with connect_read_only(db_path_text) as conn:
            rows = conn.execute(sql, [*params, limit]).fetchall()
    except sqlite3.Error as exc:
        raise DataAccessError(f"读取聚合序列失败：{exc}") from exc

    grouped: Dict[Tuple[str, ...], Dict] = {}
    for row in rows:
        key = tuple(row[alias] for alias in alias_names)
        if key not in grouped:
            grouped[key] = {
                "key": key,
                "label": " / ".join(key),
                "level": level,
                "fields": {
                    label: row[alias]
                    for (_, alias, label) in fields
                },
                "total_qty": float(row["total_qty"] or 0),
                "points": [],
            }
        grouped[key]["points"].append((row["month"], float(row["qty"] or 0)))

    return list(grouped.values())


def ensure_database_readable(db_path_text: str) -> Dict[str, int]:
    """验证数据库可读并返回基础记录数。"""
    sql = f"SELECT COUNT(*) AS row_count FROM {quote_identifier(TABLE_NAME)}"
    try:
        with connect_read_only(db_path_text) as conn:
            row = conn.execute(sql).fetchone()
            return {"row_count": int(row["row_count"])}
    except sqlite3.Error as exc:
        raise DataAccessError(f"数据库读取失败：{exc}") from exc
