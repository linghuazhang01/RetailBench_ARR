import json
from typing import Iterable, Sequence


def format_sql_rows(
    columns: Sequence[str],
    rows: Iterable[Sequence],
    *,
    max_rows: int = 50
) -> str:
    """
    将 SQL 查询结果格式化为便于 LLM 消化的字符串。

    - 自动把 JSON 字段（record_data/order_data 等）尝试反序列化。
    - 控制预览行数，避免长输出。
    """
    rows = list(rows)
    shown = rows[:max_rows]

    lines = [
        f"rows_total={len(rows)}, rows_shown={len(shown)}, columns={list(columns)}"
    ]

    for idx, row in enumerate(shown):
        row_dict = {}
        for col, val in zip(columns, row):
            parsed = _maybe_parse_json(val)
            row_dict[col] = parsed
        lines.append(f"[{idx}] {json.dumps(row_dict, ensure_ascii=False)}")

    if len(rows) > max_rows:
        lines.append(f"... truncated {len(rows) - max_rows} rows")

    return "\n".join(lines)


def _maybe_parse_json(val):
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except Exception:
        return val
