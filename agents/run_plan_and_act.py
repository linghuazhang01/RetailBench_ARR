#!/usr/bin/env python3
"""
Plan-and-Act loop for RetailEnvironment using OpenAI tools.

Plan phase: generate a concise plan for the day (no tool calls).
Act phase: execute the plan with tools and end the day with end_today.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if sys.version_info < (3, 10):
    raise SystemExit("RetailBench agents require Python 3.10 or newer.")

from openai import OpenAI

from retail_environment import RetailEnvironment
from util.config_registry import CONFIG_TYPE_CHOICES, load_config_by_type
from util.run_logging import initialize_run_artifacts
from util.tool_call_parser import parse_tool_args, parse_tool_calls

from module.stream_chat import get_cached_prompt_tokens, stream_chat
from agents.run_reflection import (
    DAILY_OPERATION_CLOSEOUT_INSTRUCTION,
    USER_ENVIRONMENT_WARNINGS,
    apply_negative_funds_update,
    begin_quality_first_action_round,
    build_action_reconsideration_tool_response_messages,
    end_quality_first_action_round,
    is_action_reconsideration_status,
    make_tool_response_message,
    render_system_prompt,
    sanitize_messages_for_api,
    tool_result_action_executed,
    tool_result_has_error,
    tool_result_status,
)


DEFAULT_MODEL = "qwen3-235b-a22b-thinking-2507"

PLAN_SYSTEM_PROMPT = """You are a retail planning agent.

Environment Characteristics

- The store operates with a large number of SKUs, where products within the same category interact and may substitute or cannibalize each other's demand.
- Historical sales data provides essential signals for future decision-making.
- Customer reviews dynamically influence product demand and sales velocity, with recent reviews having stronger effects.
- Supply chains involve delivery lead times, requiring forward-looking inventory planning.
- Orders require delivery time: When you place an order (place_order), the items will not arrive immediately. The delivery time varies and can take up to 7 days (within 7 days). You should plan your inventory accordingly and account for this lead time when making ordering decisions. Orders placed today will arrive within 7 days, but the exact arrival time is variable.
- Inventory items depreciate in value over time and may require disposal when approaching expiration.
- Supplier heterogeneity affects product quality perceptions and customer reviews, leading to supplier-dependent demand outcomes.
- Product demand is influenced by multiple factors including pricing, customer reviews, and external market conditions.
- Return rates are affected by customer reviews and product quality perceptions.
- Store shelf space is limited. A SKU can sell only after it is placed on shelf. Newly arrived inventory is not put on shelf automatically; use set_shelf_skus as a normal top-level action to choose the active shelf assortment.
- Daily rent: The store incurs a fixed daily rent cost of {daily_rent} that must be paid each day. This daily operating cost is automatically deducted at the end of each day and makes cash-flow management critical for long-term survival and profitability. You must ensure sufficient funds are available to cover the daily rent. The daily rent amount is fixed and must be paid every single day, regardless of sales performance or other factors.

Action Policy

When using execute_code for data analysis:
- Available variables: funds, current_date, rent, sku_ids. All execute_code tool helpers return clean data directly (result auto-extracted).
- Always available helpers: view_funds_and_date(), view_inventory(), view_shelf_status(), view_sales_profit_history(sku_ids, start_date, end_date, supplier_id=None), view_current_date_supplier_prices(sku_ids), view_supplier_price_history(sku_id, start_date, end_date, supplier_id=None), view_sku_prices(sku_ids), view_sku_inventory_cost(sku_ids), view_notes().
- When reviews are enabled, also available: view_sku_avg_ratings(sku_ids, start_date, end_date), view_supplier_returns_avg_rate(supplier_id, start_date, end_date, sku_ids).
- When news is enabled, also available: view_today_news(), view_news_detail(news_id), view_news_history(start_date, end_date).
- Libraries/objects: json, math, datetime, date, timedelta, defaultdict. Safe imports are allowed only for datetime, collections, math, statistics, itertools, functools, decimal, operator, and json.
- execute_code is read-only. Do not call place_order, modify_sku_price, or set_shelf_skus inside execute_code; use execute_code only to inspect/analyze data, then call mutation tools as normal top-level tool calls if needed.
- current_date is a "YYYY-MM-DD" string. Convert it before date arithmetic, e.g. `today = date.fromisoformat(current_date); start = str(today - timedelta(days=14))`.
- Each execute_code call is isolated. Recompute local variables such as inv, orders, sales, velocity, and result inside the same snippet; do not rely on variables from earlier code calls.
- For open supplier orders, inspect `pending_orders` from view_inventory(); use `supplier_id`, `expected_delivery_date`, `days_until_arrival`, `ordered_sku`, `items`, and item `count`; do not use arrival_date, lines, or item quantity.
- Avoid lambda/key-function constructs inside execute_code examples and generated code. Prefer explicit loops when choosing a low-return supplier or building ranked lists.
- Do not define helper functions or classes inside execute_code unless absolutely necessary; keep snippets linear and short.
- Call view_* functions with the same parameters as the tool versions (e.g., view_sales_profit_history(sku_ids, start_date, end_date, supplier_id=None), view_current_date_supplier_prices(sku_ids)). Dates as "YYYY-MM-DD" strings. view_sales_profit_history returns {{sku_id: {{"daily": [{{"date": ..., "selling_price": ..., "units_sold": ..., "net_profit": ...}}], "total_units": ..., "days": ..., "net_profit": ..., "avg_selling_price": ...}}}}.
- For tools that take sku_ids, pass a non-empty list of legal SKU ids; use `sku_ids` or a non-empty slice such as `sku_ids[:10]`, never `[]`.
- First, map each plan step to the data needed (e.g., "check reorder" → inventory + supplier prices + sales velocity).
- Then, write code that directly produces actionable outputs matching the plan's intent (e.g., a sorted reorder list with supplier and quantity).
- Assign structured results to `result` so subsequent actions (place_order, modify_sku_price) can use the analysis output directly.
- Example — generate a prioritized reorder list with supplier candidates using available return-rate evidence for a later normal place_order tool call:
  <tool_call>
  {{"name": "execute_code", "arguments": {{"code": "today = date.fromisoformat(current_date)\\nstart = str(today - timedelta(days=30))\\ninv = view_inventory()\\nsp = view_current_date_supplier_prices(sku_ids)\\nsales = view_sales_profit_history(sku_ids, start, current_date)\\nreorder = []\\nfor s in sku_ids:\\n    item = inv['inventory'].get(s, {{}})\\n    qty = item.get('quantity', 0)\\n    wait = item.get('waiting', 0)\\n    sold = sales.get(s, {{}}).get('total_units', 0)\\n    if qty + wait < 7 and sold > 0:\\n        best = None\\n        for q in sp.get(s, []):\\n            sid = q.get('supplier_id')\\n            ret = view_supplier_returns_avg_rate(sid, start, current_date, [s])\\n            rate = ret.get('average_return_rate')\\n            denom = ret.get('denominator_count', 0)\\n            if rate is None or denom <= 0:\\n                continue\\n            return_quality = 1.0 - float(rate)\\n            price = float(q.get('price', 0))\\n            candidate = {{'supplier_id': sid, 'price': price, 'return_rate': float(rate), 'return_quality': return_quality, 'denominator': denom}}\\n            if best is None or candidate['return_quality'] > best['return_quality'] or (candidate['return_quality'] == best['return_quality'] and candidate['price'] < best['price']):\\n                best = candidate\\n        if best:\\n            need = 7 - qty - wait\\n            reorder.append({{'sku_id': s, 'need': need, 'supplier': best['supplier_id'], 'unit_cost': best['price'], 'return_rate': best['return_rate'], 'return_quality': best['return_quality'], 'return_denominator': best['denominator']}})\\nfor r in reorder[:10]:\\n    print(r['sku_id'] + ': order ' + str(r['need']) + ' from ' + r['supplier'] + ' cost=' + str(round(r['unit_cost'], 2)) + ' return_rate=' + str(round(r['return_rate'], 3)))\\nresult = reorder"}}}}
  </tool_call>

When placing an order (place_order):
- First, carefully consider current inventory levels and whether reordering is truly necessary.
- Then, evaluate the order quantity against the product's average daily sales volume. Avoid ordering too little, which could lead to future stockouts or understocking.
- When referencing daily sales volume, use a sufficiently long time window (e.g., 14-30 days) rather than a short period, to ensure the data reflects stable demand patterns.
- Carefully analyze return rates and product review information before ordering; supplier choice can directly affect product quality, customer reviews, demand, and return rates.
- Supplier price quotes expose price and delivery fields, not internal supplier quality labels. Use review and return-rate tools when comparing supplier quality.

When managing shelf assortment (set_shelf_skus):
- Use view_shelf_status to inspect current on-shelf assortment and capacity; use view_inventory separately when selecting inventory-backed SKUs to place on shelf.
- Keep at most the shelf capacity number of SKUs shown by view_shelf_status.
- Put SKUs on shelf deliberately after inventory arrives or when assortment should change; arrivals are not automatically shelved.
- The provided sku_ids list replaces the entire shelf assortment, so preserve existing useful shelf SKUs when making incremental changes.

When modifying a price (modify_sku_price):
- First, consider whether the new price can generate sufficient profit margin.
- Then, evaluate whether the price change might cause significant fluctuations in the product's sales volume.
- When analyzing pricing trends, reference data over a sufficiently long time window rather than a short interval, to avoid making decisions based on noisy or unrepresentative data.
- Use supplier purchase prices and inventory costs as references when modifying prices, avoiding prices that are too low to preserve margin or too high to sustain demand.

Your job is to produce a concise, actionable plan for today's operations. The plan should be 3-7 steps.
Use the provided current status and yesterday's actions/results to identify priorities, risks, and quick wins.
Do NOT call tools in the plan phase. Output only the plan in plain text.
"""

EXECUTION_SYSTEM_PROMPT = """You are a retail operations agent executing the plan for today.

Environment Characteristics

- The store operates with a large number of SKUs, where products within the same category interact and may substitute or cannibalize each other's demand.
- Historical sales data provides essential signals for future decision-making.
- Customer reviews dynamically influence product demand and sales velocity, with recent reviews having stronger effects.
- Supply chains involve delivery lead times, requiring forward-looking inventory planning.
- Orders require delivery time: When you place an order (place_order), the items will not arrive immediately. The delivery time varies and can take up to 7 days (within 7 days). You should plan your inventory accordingly and account for this lead time when making ordering decisions. Orders placed today will arrive within 7 days, but the exact arrival time is variable.
- Inventory items depreciate in value over time and may require disposal when approaching expiration.
- Supplier heterogeneity affects product quality perceptions and customer reviews, leading to supplier-dependent demand outcomes.
- Product demand is influenced by multiple factors including pricing, customer reviews, and external market conditions.
- Return rates are affected by customer reviews and product quality perceptions.
- Store shelf space is limited. A SKU can sell only after it is placed on shelf. Newly arrived inventory is not put on shelf automatically; use set_shelf_skus as a normal top-level action to choose the active shelf assortment.
- Daily rent: The store incurs a fixed daily rent cost of {daily_rent} that must be paid each day. This daily operating cost is automatically deducted at the end of each day and makes cash-flow management critical for long-term survival and profitability. You must ensure sufficient funds are available to cover the daily rent. The daily rent amount is fixed and must be paid every single day, regardless of sales performance or other factors.

Action Policy

When using execute_code for data analysis:
- Available variables: funds, current_date, rent, sku_ids. All execute_code tool helpers return clean data directly (result auto-extracted).
- Always available helpers: view_funds_and_date(), view_inventory(), view_shelf_status(), view_sales_profit_history(sku_ids, start_date, end_date, supplier_id=None), view_current_date_supplier_prices(sku_ids), view_supplier_price_history(sku_id, start_date, end_date, supplier_id=None), view_sku_prices(sku_ids), view_sku_inventory_cost(sku_ids), view_notes().
- When reviews are enabled, also available: view_sku_avg_ratings(sku_ids, start_date, end_date), view_supplier_returns_avg_rate(supplier_id, start_date, end_date, sku_ids).
- When news is enabled, also available: view_today_news(), view_news_detail(news_id), view_news_history(start_date, end_date).
- Libraries/objects: json, math, datetime, date, timedelta, defaultdict. Safe imports are allowed only for datetime, collections, math, statistics, itertools, functools, decimal, operator, and json.
- execute_code is read-only. Do not call place_order, modify_sku_price, or set_shelf_skus inside execute_code; use execute_code only to inspect/analyze data, then call mutation tools as normal top-level tool calls if needed.
- current_date is a "YYYY-MM-DD" string. Convert it before date arithmetic, e.g. `today = date.fromisoformat(current_date); start = str(today - timedelta(days=14))`.
- Each execute_code call is isolated. Recompute local variables such as inv, orders, sales, velocity, and result inside the same snippet; do not rely on variables from earlier code calls.
- For open supplier orders, inspect `pending_orders` from view_inventory(); use `supplier_id`, `expected_delivery_date`, `days_until_arrival`, `ordered_sku`, `items`, and item `count`; do not use arrival_date, lines, or item quantity.
- Avoid lambda/key-function constructs inside execute_code examples and generated code. Prefer explicit loops when choosing a low-return supplier or building ranked lists.
- Do not define helper functions or classes inside execute_code unless absolutely necessary; keep snippets linear and short.
- Call view_* functions with the same parameters as the tool versions (e.g., view_sales_profit_history(sku_ids, start_date, end_date, supplier_id=None), view_current_date_supplier_prices(sku_ids)). Dates as "YYYY-MM-DD" strings. view_sales_profit_history returns {{sku_id: {{"daily": [{{"date": ..., "selling_price": ..., "units_sold": ..., "net_profit": ...}}], "total_units": ..., "days": ..., "net_profit": ..., "avg_selling_price": ...}}}}.
- For tools that take sku_ids, pass a non-empty list of legal SKU ids; use `sku_ids` or a non-empty slice such as `sku_ids[:10]`, never `[]`.
- First, map each plan step to the data needed (e.g., "check reorder" → inventory + supplier prices + sales velocity).
- Then, write code that directly produces actionable outputs matching the plan's intent (e.g., a sorted reorder list with supplier and quantity).
- Assign structured results to `result` so subsequent actions (place_order, modify_sku_price) can use the analysis output directly.
- Example — generate a prioritized reorder list with supplier candidates using available return-rate evidence for a later normal place_order tool call:
  <tool_call>
  {{"name": "execute_code", "arguments": {{"code": "today = date.fromisoformat(current_date)\\nstart = str(today - timedelta(days=30))\\ninv = view_inventory()\\nsp = view_current_date_supplier_prices(sku_ids)\\nsales = view_sales_profit_history(sku_ids, start, current_date)\\nreorder = []\\nfor s in sku_ids:\\n    item = inv['inventory'].get(s, {{}})\\n    qty = item.get('quantity', 0)\\n    wait = item.get('waiting', 0)\\n    sold = sales.get(s, {{}}).get('total_units', 0)\\n    if qty + wait < 7 and sold > 0:\\n        best = None\\n        for q in sp.get(s, []):\\n            sid = q.get('supplier_id')\\n            ret = view_supplier_returns_avg_rate(sid, start, current_date, [s])\\n            rate = ret.get('average_return_rate')\\n            denom = ret.get('denominator_count', 0)\\n            if rate is None or denom <= 0:\\n                continue\\n            return_quality = 1.0 - float(rate)\\n            price = float(q.get('price', 0))\\n            candidate = {{'supplier_id': sid, 'price': price, 'return_rate': float(rate), 'return_quality': return_quality, 'denominator': denom}}\\n            if best is None or candidate['return_quality'] > best['return_quality'] or (candidate['return_quality'] == best['return_quality'] and candidate['price'] < best['price']):\\n                best = candidate\\n        if best:\\n            need = 7 - qty - wait\\n            reorder.append({{'sku_id': s, 'need': need, 'supplier': best['supplier_id'], 'unit_cost': best['price'], 'return_rate': best['return_rate'], 'return_quality': best['return_quality'], 'return_denominator': best['denominator']}})\\nfor r in reorder[:10]:\\n    print(r['sku_id'] + ': order ' + str(r['need']) + ' from ' + r['supplier'] + ' cost=' + str(round(r['unit_cost'], 2)) + ' return_rate=' + str(round(r['return_rate'], 3)))\\nresult = reorder"}}}}
  </tool_call>

When placing an order (place_order):
- First, carefully consider current inventory levels and whether reordering is truly necessary.
- Then, evaluate the order quantity against the product's average daily sales volume. Avoid ordering too little, which could lead to future stockouts or understocking.
- When referencing daily sales volume, use a sufficiently long time window (e.g., 14-30 days) rather than a short period, to ensure the data reflects stable demand patterns.
- Carefully analyze return rates and product review information before ordering; supplier choice can directly affect product quality, customer reviews, demand, and return rates.
- Supplier price quotes expose price and delivery fields, not internal supplier quality labels. Use review and return-rate tools when comparing supplier quality.

When managing shelf assortment (set_shelf_skus):
- Use view_shelf_status to inspect current on-shelf assortment and capacity; use view_inventory separately when selecting inventory-backed SKUs to place on shelf.
- Keep at most the shelf capacity number of SKUs shown by view_shelf_status.
- Put SKUs on shelf deliberately after inventory arrives or when assortment should change; arrivals are not automatically shelved.
- The provided sku_ids list replaces the entire shelf assortment, so preserve existing useful shelf SKUs when making incremental changes.

When modifying a price (modify_sku_price):
- First, consider whether the new price can generate sufficient profit margin.
- Then, evaluate whether the price change might cause significant fluctuations in the product's sales volume.
- When analyzing pricing trends, reference data over a sufficiently long time window rather than a short interval, to avoid making decisions based on noisy or unrepresentative data.
- Use supplier purchase prices and inventory costs as references when modifying prices, avoiding prices that are too low to preserve margin or too high to sustain demand.

# Your Role

You should:
1. Analyze the current business situation using data tools (inventory, sales, suppliers, news, funds, etc.)
2. Make data-driven decisions about inventory management, pricing, and ordering
3. Execute actions such as placing orders, adjusting prices, and managing shelf assortment based on your analysis
4. End the day by calling end_today when all operations are complete

Follow the provided plan, but adapt if live data suggests a better action.

# Available Tools

The available function signatures are provided within <tools></tools> XML tags:
<tools>
{tool_definitions}
</tools>

For each function call, return a JSON object with function name and arguments inside <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Ending the Day

When you have completed all reasonable operations for the day, you MUST call end_today to advance to the next day.
"""

DEFAULT_GOAL = (
    "Please optimize inventory assortment and turnover for long-term store viability: "
    "minimize stockouts, shrink, and cash risk while covering rent/operating costs "
    "and growing gross margin via data-driven, proactive decisions."
)


def build_openai_tools(env: RetailEnvironment, exclude_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Convert RetailEnvironment.get_tools() to OpenAI tool schema."""
    if exclude_tools is None:
        exclude_tools = []
    tools = []
    for name, meta in env.get_tools().items():
        if name in exclude_tools:
            continue
        params = meta.get("input_schema") or meta.get(
            "parameters",
            {"type": "object", "properties": {}, "required": []},
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return tools


def render_sku_descriptions(env: RetailEnvironment) -> str:
    lines = []
    lines.append(
        "SKU catalog (grouped by category). SKU_ID is the unique product identifier. "
        "Promotion_Days indicates when the item will be discounted/cleared and should be sold before that window expires."
    )
    for category, sku_list in env.skus_category_map.items():
        lines.append(f"## Category: {category}")
        for sku in sku_list:
            desc = sku.attributes or {}
            detail = desc.get("description") or desc.get("DESCRIP") or ""
            brand = sku.brand
            promotion_days = getattr(sku, "promotion_day", None) or desc.get("PROMOTION_TIME")
            lines.append(
                f"- SKU_id={sku.sku_id}, Expiration_Days={promotion_days}, "
                f"Brand={brand}, Desc={detail}, Category={category}"
            )
        lines.append("")

    return "\n".join(lines).strip()


def render_tool_definitions(tools: List[Dict[str, Any]]) -> str:
    """Render tool definitions as newline-delimited JSON objects for the prompt."""
    return "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)


def safe_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text if it exceeds max_length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def log_message(records: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    """Append a timestamped record to the in-memory run log."""
    record = {"ts": datetime.utcnow().isoformat() + "Z", **payload}
    records.append(record)


def write_log_json_array(log_path: Path, records: List[Dict[str, Any]]) -> None:
    """Persist the run log as a JSON array."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)


def save_turn_calls_to_json(
    log_dir: Path,
    phase: str,
    turn_index: int,
    turn_data: Dict[str, Any],
    day: int,
    round_index: Optional[int] = None,
) -> None:
    """Save turn calls to a separate JSON file."""
    if round_index is None:
        filename = f"{phase}_{day}_{turn_index}.json"
    else:
        filename = f"{phase}_{day}_r{round_index}_t{turn_index}.json"
    filepath = log_dir / str(day) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(turn_data, f, ensure_ascii=False, indent=2, default=str)


def build_goal(base_goal: str, config: Dict[str, Any]) -> str:
    """Blend the base goal with readable operational context from config."""
    context = (
        f"You are operating store {config.get('store_id', '')} with initial funds of {config.get('initial_funds', '')}. "
        f"Your available operational data ranges from {config.get('data_begin_time', '')} to {config.get('data_end_time', '')}. "
        f"Today is {config.get('store_begin_time', '')}. Daily Rent is {config.get('everyday_rent', 0)}."
    )
    return f"{context}\n\n{base_goal}"


DEFAULT_API_KEY = ""
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI client with configurable API key and base URL.
    """
    if base_url is None:
        base_url = (
            os.environ.get("BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or DEFAULT_BASE_URL
        )
    if api_key is None:
        if "openrouter.ai" in base_url:
            api_key = (
                os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or DEFAULT_API_KEY
            )
        else:
            api_key = (
                os.environ.get("API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("DASHSCOPE_API_KEY")
                or DEFAULT_API_KEY
            )

    return OpenAI(api_key=api_key, base_url=base_url)


def save_checkpoint(checkpoint_dir: Path, turn: int, messages: List[Dict[str, Any]], env: RetailEnvironment) -> None:
    """Save checkpoint: messages + environment state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / f"messages_turn_{turn}.json").open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=str)
    env.save_checkpoint(checkpoint_dir / f"env_checkpoint_turn_{turn}.json")
    print(f"[Checkpoint] Saved checkpoint at turn {turn}")


def recover_from_checkpoint(checkpoint_dir: Path, turn: int) -> tuple[RetailEnvironment, List[Dict[str, Any]]]:
    """Recover environment state and messages from checkpoint."""
    messages_path = checkpoint_dir / f"messages_turn_{turn}.json"
    env_checkpoint_path = checkpoint_dir / f"env_checkpoint_turn_{turn}.json"
    if not messages_path.exists() or not env_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for turn {turn}")

    with messages_path.open("r", encoding="utf-8") as f:
        messages = json.load(f)
    env = RetailEnvironment.recover_from_checkpoint(env_checkpoint_path)
    print(f"[Checkpoint] Recovered from turn {turn}")
    return env, messages


def recover_from_day_checkpoint(
    checkpoint_dir: Path,
    day: int,
) -> tuple[RetailEnvironment, List[str], Optional[str], int, int]:
    """Recover environment state and previous-day info from day checkpoint."""
    day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
    if not day_checkpoint_path.exists():
        raise FileNotFoundError(f"Day checkpoint not found: {day_checkpoint_path}")

    with day_checkpoint_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    env = RetailEnvironment.recover_from_checkpoint(checkpoint_dir / metadata["env_checkpoint_path"])
    previous_day_actions = metadata.get("previous_day_actions", [])
    previous_day_end_today_formatted = metadata.get("previous_day_end_today_formatted")
    start_day = metadata.get("day", day) + 1
    start_turn = metadata.get("global_turn", 0)

    print(
        f"[Checkpoint] Recovered from day {day}, will start from day {start_day}, "
        f"previous actions: {len(previous_day_actions)} entries"
    )
    return env, previous_day_actions, previous_day_end_today_formatted, start_day, start_turn


def generate_plan(
    client: OpenAI,
    model: str,
    day: int,
    goal: str,
    status_text: str,
    sku_desc: str,
    previous_day_actions: Optional[List[str]] = None,
    previous_day_end_today_formatted: Optional[str] = None,
    daily_rent: float = 0,
    quality_first_mode: str = "off",
) -> tuple[str, Dict[str, Any]]:
    """
    Generate a concise plan for the day.
    Returns (plan_text, usage).
    """
    sku_desc = truncate_text(sku_desc, max_length=8000)

    if previous_day_actions:
        previous_actions_text = "\n".join(f"- {action}" for action in previous_day_actions)
        previous_actions_text = truncate_text(previous_actions_text, max_length=2000)
    else:
        previous_actions_text = "None"

    if previous_day_end_today_formatted:
        previous_end_today_text = truncate_text(previous_day_end_today_formatted, max_length=2000)
    else:
        previous_end_today_text = "None"

    plan_prompt = f"""# Task Goal
{goal}

# Day {day} Status
{status_text}

# Yesterday's Actions
{previous_actions_text}

# Yesterday's end_today Result
{previous_end_today_text}

# SKU Catalog (truncated if long)
{sku_desc}

{USER_ENVIRONMENT_WARNINGS}

# Instructions
Produce a short plan (3-7 steps). Do NOT call tools.
"""

    messages = [
        {
            "role": "system",
            "content": render_system_prompt(
                PLAN_SYSTEM_PROMPT,
                quality_first_mode=quality_first_mode,
                daily_rent=daily_rent,
            ),
        },
        {"role": "user", "content": plan_prompt},
    ]

    try:
        full_content, final_content, reasoning_content, usage = stream_chat(client, model, messages)
        plan_text = (final_content or "").strip() or (full_content or "").strip()
    except Exception as exc:
        print(f"[WARN] Plan generation failed: {type(exc).__name__}: {exc}")
        plan_text = ""
        usage = {}

    if not plan_text:
        plan_text = "Plan: Review status, check inventory and sales, adjust prices/orders, and end the day."

    return plan_text, usage or {}


def run_plan_and_act_loop(
    goal: str,
    env: RetailEnvironment,
    model: str = DEFAULT_MODEL,
    max_turns: int = 50,
    log_path: Path = Path("logs/run_env_history.json"),
    max_input_tokens: int = 60000,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 0,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    start_turn: int = 0,
    start_day: int = 1,
    previous_day_actions: Optional[List[str]] = None,
    previous_day_end_today_formatted: Optional[str] = None,
    log_dir: Optional[Path] = None,
    max_days: int = 30,
    max_execution_turns_per_day: int = 20,
    client: Optional[OpenAI] = None,
) -> None:
    all_tools = build_openai_tools(env)

    if not env.config.get("enable_new", False):
        news_tools = ["view_news_history", "view_today_news", "view_news_detail"]
        all_tools = [t for t in all_tools if t["function"]["name"] not in news_tools]
    allowed_tool_names = {tool["function"]["name"] for tool in all_tools}

    run_log: List[Dict[str, Any]] = []

    execution_system_prompt = render_system_prompt(
        EXECUTION_SYSTEM_PROMPT,
        quality_first_mode=env.config.get("quality_first_mode", "off"),
        tool_definitions=render_tool_definitions(all_tools),
        daily_rent=env.config.get('everyday_rent', 0),
    )

    if client is None:
        client = create_openai_client()

    input_tokens = 0
    global_turn = start_turn

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cached_prompt_tokens = 0

    consecutive_negative_days = 0
    previous_day_actions = previous_day_actions or []
    daily_execution_turn_limit = max_turns if max_turns and max_turns > 0 else max_execution_turns_per_day
    if daily_execution_turn_limit <= 0:
        daily_execution_turn_limit = 1

    for day in range(start_day, max_days + 1):
        print(f"\n{'='*80}\nDay {day} - {env.current_date}\n{'='*80}\n")
        log_message(run_log, {"role": "system", "day": day, "current_date": str(env.current_date), "message": f"Day {day} started"})

        day_prompt_tokens = 0
        day_completion_tokens = 0
        day_tokens = 0
        day_cached_prompt_tokens = 0

        day_actions: List[str] = []
        day_round = 0
        day_execution_turns_total = 0
        day_complete = False

        while not day_complete and day_execution_turns_total < daily_execution_turn_limit:
            day_round += 1

            # --- Plan Phase ---
            print(f"\n[Day {day}] === PLAN PHASE (Round {day_round}) ===")

            funds_result = env.exec_tools("view_funds_and_date")
            inventory_result = env.exec_tools("view_inventory")
            shelf_result = env.exec_tools("view_shelf_status")

            funds_formatted = funds_result.get("formatted", "")
            inventory_formatted = truncate_text(inventory_result.get("formatted", ""), max_length=3000)
            shelf_formatted = shelf_result.get("formatted", "")

            status_text = (
                f"{funds_formatted}\n\n"
                f"Shelf Status:\n{shelf_formatted}\n\n"
                f"Inventory and Open Orders:\n{inventory_formatted}"
            )
            sku_desc = render_sku_descriptions(env)

            plan_text, plan_usage = generate_plan(
                client=client,
                model=model,
                day=day,
                goal=goal,
                daily_rent=env.config.get('everyday_rent', 0),
                quality_first_mode=env.config.get("quality_first_mode", "off"),
                status_text=status_text,
                sku_desc=sku_desc,
                previous_day_actions=previous_day_actions,
                previous_day_end_today_formatted=previous_day_end_today_formatted,
            )

            print(f"[Day {day}] Plan:\n{plan_text}\n")
            log_message(run_log, {"role": "plan", "day": day, "round": day_round, "content": plan_text})
            log_message(
                run_log,
                {
                    "role": "usage",
                    "day": day,
                    "round": day_round,
                    "phase": "plan",
                    "prompt_tokens": plan_usage.get("prompt_tokens"),
                    "completion_tokens": plan_usage.get("completion_tokens"),
                    "total_tokens": plan_usage.get("total_tokens"),
                    "cached_prompt_tokens": get_cached_prompt_tokens(plan_usage),
                },
            )

            if plan_usage:
                prompt_tokens = plan_usage.get("prompt_tokens", 0)
                completion_tokens = plan_usage.get("completion_tokens", 0)
                tokens = plan_usage.get("total_tokens", 0)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += tokens
                cached_prompt_tokens = get_cached_prompt_tokens(plan_usage)
                total_cached_prompt_tokens += cached_prompt_tokens
                day_prompt_tokens += prompt_tokens
                day_completion_tokens += completion_tokens
                day_tokens += tokens
                day_cached_prompt_tokens += cached_prompt_tokens

            if log_dir is not None:
                plan_turn_data = {
                    "day": day,
                    "round": day_round,
                    "phase": "plan",
                    "turn": 1,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "plan_text": plan_text,
                    "usage": plan_usage,
                    "status": status_text,
                }
                save_turn_calls_to_json(log_dir, "plan", 1, plan_turn_data, day, round_index=day_round)

            # --- Execution Phase ---
            print(f"\n[Day {day}] === ACT PHASE (Round {day_round}) ===")
            execution_phase_complete = False
            execution_turns = 0
            consecutive_no_valid_tool_calls = 0
            remaining_execution_turns = daily_execution_turn_limit - day_execution_turns_total
            round_turn_limit = min(max_execution_turns_per_day, remaining_execution_turns)
            if round_turn_limit <= 0:
                break

            execution_system_msg = {
                "role": "system",
                "content": execution_system_prompt,
            }

            execution_user_msg = {
                "role": "user",
                "content": (
                    f"# Day {day} - Execution Phase\n\n"
                    f"## Plan\n{plan_text}\n\n"
                    f"## Current Status\n{status_text}\n\n"
                    f"## Shelf Status\n{shelf_formatted}\n\n"
                    f"## SKU Catalog\n{sku_desc}\n\n"
                    "## Instructions\n\n"
                    f"{USER_ENVIRONMENT_WARNINGS}\n\n"
                    "Follow the plan. "
                    f"{DAILY_OPERATION_CLOSEOUT_INSTRUCTION}"
                ),
            }

            log_message(run_log, {**execution_system_msg, "day": day, "round": day_round, "phase": "execution"})
            log_message(run_log, {**execution_user_msg, "day": day, "round": day_round, "phase": "execution"})

            execution_messages: List[Dict[str, Any]] = []

            while not execution_phase_complete and execution_turns < round_turn_limit:
                execution_turns += 1
                global_turn += 1

                if input_tokens > max_input_tokens:
                    assistant_idxs = [i for i, m in enumerate(execution_messages) if m.get("role") == "assistant"]
                    if len(assistant_idxs) >= 3:
                        execution_messages = execution_messages[assistant_idxs[2]:]
                        assert execution_messages[0]["role"] == "assistant"
                    elif len(assistant_idxs) >= 2:
                        execution_messages = execution_messages[assistant_idxs[1]:]
                        assert execution_messages[0]["role"] == "assistant"

                request_messages = (
                    [execution_system_msg]
                    + execution_messages
                    + [execution_user_msg]
                )
                api_messages = sanitize_messages_for_api(request_messages)

                try:
                    full_content, final_content, reasoning_content, usage = stream_chat(
                        client=client,
                        model=model,
                        messages=api_messages,
                    )
                except Exception as stream_exc:
                    err_msg = (
                        f"LLM stream_chat failed on day {day} execution turn {execution_turns}: "
                        f"{type(stream_exc).__name__}: {stream_exc}"
                    )
                    print(f"[ERROR] {err_msg}")
                    log_message(
                        run_log,
                        {
                            "role": "error",
                            "day": day,
                            "round": day_round,
                            "phase": "execution",
                            "turn": execution_turns,
                            "message": err_msg,
                            "error_type": type(stream_exc).__name__,
                        },
                    )
                    write_log_json_array(log_path, run_log)
                    execution_messages.append({"role": "user", "content": f"System error: {err_msg}. Continue."})
                    continue

                parse_method_tag = "none"
                try:
                    parse_source = final_content or full_content or ""
                    tool_calls_list, parse_method_tag = parse_tool_calls(
                        parse_source,
                        allowed_tool_names=allowed_tool_names,
                    )
                except Exception as parse_exc:
                    err_msg = (
                        f"Failed to parse tool calls on day {day} execution turn {execution_turns}: "
                        f"{type(parse_exc).__name__}: {parse_exc}"
                    )
                    print(f"[ERROR] {err_msg}")
                    log_message(
                        run_log,
                        {
                            "role": "error",
                            "day": day,
                            "round": day_round,
                            "phase": "execution",
                            "turn": execution_turns,
                            "message": err_msg,
                            "error_type": type(parse_exc).__name__,
                        },
                    )
                    tool_calls_list, parse_method_tag = [], "none"

                turn_data: Optional[Dict[str, Any]] = None
                if log_dir is not None:
                    turn_data = {
                        "day": day,
                        "round": day_round,
                        "phase": "execution",
                        "turn": execution_turns,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "think_content": full_content,
                        "full_content": full_content,
                        "final_content": final_content,
                        "reasoning_content": reasoning_content,
                        "tool_calls": tool_calls_list,
                        "usage": usage,
                        "messages": request_messages,
                        "api_messages": api_messages,
                    }
                    save_turn_calls_to_json(log_dir, "execute", execution_turns, turn_data, day, round_index=day_round)

                log_message(
                    run_log,
                    {
                        "role": "usage",
                        "day": day,
                        "round": day_round,
                        "phase": "execution",
                        "turn": execution_turns,
                        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                        "completion_tokens": usage.get("completion_tokens") if usage else None,
                        "total_tokens": usage.get("total_tokens") if usage else None,
                        "cached_prompt_tokens": get_cached_prompt_tokens(usage),
                    },
                )
                input_tokens = usage.get("prompt_tokens") if usage else input_tokens

                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    tokens = usage.get("total_tokens", 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_tokens += tokens
                    cached_prompt_tokens = get_cached_prompt_tokens(usage)
                    total_cached_prompt_tokens += cached_prompt_tokens
                    day_prompt_tokens += prompt_tokens
                    day_completion_tokens += completion_tokens
                    day_tokens += tokens
                    day_cached_prompt_tokens += cached_prompt_tokens

                print(
                    f"[Day {day} Execution Turn {execution_turns}] tokens: "
                    f"prompt={usage.get('prompt_tokens') if usage else None}, "
                    f"completion={usage.get('completion_tokens') if usage else None}"
                )

                log_message(
                    run_log,
                    {
                        "role": "assistant",
                        "day": day,
                        "round": day_round,
                        "phase": "execution",
                        "turn": execution_turns,
                        "content": full_content,
                        "full_content": full_content,
                        "final_content": final_content,
                        "reasoning": reasoning_content,
                        "tool_calls": tool_calls_list,
                    },
                )

                assistant_message = {"role": "assistant", "content": full_content}
                tool_response_messages: List[Dict[str, Any]] = []
                tool_response_meta: List[Dict[str, Any]] = []
                suppress_assistant_message = False
                valid_tool_calls_count = 0
                skipped_tool_calls_after_reconsideration: List[Dict[str, Any]] = []

                begin_quality_first_action_round(env)
                for call_index, call in enumerate(tool_calls_list):
                    if not isinstance(call, dict) or not call.get("name"):
                        err_msg = f"Invalid tool call: {call}"
                        print(f"[ERROR] {err_msg}")
                        log_message(
                            run_log,
                            {
                                "role": "error",
                                "day": day,
                                "round": day_round,
                                "phase": "execution",
                                "turn": execution_turns,
                                "message": err_msg,
                                "call": str(call),
                            },
                        )
                        tool_response_messages.append(make_tool_response_message(f"Error: {err_msg}"))
                        continue

                    name, args = call.get("name"), parse_tool_args(call.get("arguments"))
                    try:
                        result = env.exec_tools(name, **args)
                        if not isinstance(result, dict) or "formatted" not in result:
                            result = {"formatted": str(result), "result": result}
                    except Exception as exc:
                        err_msg = f"Error executing tool {name}: {type(exc).__name__}: {exc}"
                        print(f"[ERROR] {err_msg}")
                        if hasattr(env, "logger"):
                            env.logger.error(err_msg)
                        result = {"formatted": err_msg, "result": {"error": str(exc), "error_type": type(exc).__name__}}

                    tool_executed_successfully = not tool_result_has_error(result)
                    action_executed = tool_result_action_executed(result)
                    result_status = tool_result_status(result)
                    if is_action_reconsideration_status(result_status):
                        suppress_assistant_message = True

                    log_message(
                        run_log,
                        {
                            "role": "tool",
                            "day": day,
                            "round": day_round,
                            "phase": "execution",
                            "turn": execution_turns,
                            "name": name,
                            "args": args,
                            "content": result.get("formatted", safe_dump(result)),
                            "raw": result.get("result", safe_dump(result)),
                            "success": tool_executed_successfully,
                            "action_executed": action_executed,
                            "result_status": result_status,
                        },
                    )

                    if tool_executed_successfully:
                        valid_tool_calls_count += 1
                        if action_executed:
                            day_actions.append(f"{name}({safe_dump(args)})")
                        elif is_action_reconsideration_status(result_status):
                            day_actions.append(
                                f"{name}({safe_dump(args)}) [action reconsideration required; action not executed]"
                            )

                    if is_action_reconsideration_status(result_status):
                        reconsideration_messages = build_action_reconsideration_tool_response_messages(
                            name,
                            args,
                            result,
                        )
                        tool_response_messages.extend(reconsideration_messages)
                        skipped_tool_calls_after_reconsideration = tool_calls_list[call_index + 1 :]
                        if skipped_tool_calls_after_reconsideration:
                            log_message(
                                run_log,
                                {
                                    "role": "system",
                                    "day": day,
                                    "round": day_round,
                                    "phase": "execution",
                                    "turn": execution_turns,
                                    "message": "Skipped remaining parsed tool calls after action reconsideration.",
                                    "skipped_tool_calls": skipped_tool_calls_after_reconsideration,
                                },
                            )
                        # Stop this tool batch so the next model round can reconsider first.
                        break
                    else:
                        tool_response_messages.append(
                            make_tool_response_message(
                                result.get("formatted", safe_dump(result)),
                            )
                        )

                    if name == "end_today" and tool_executed_successfully:
                        execution_phase_complete = True
                        day_complete = True
                        print(f"[Day {day}] Execution phase complete - end_today called")
                        previous_day_end_today_formatted = result.get("formatted") or safe_dump(result)
                        consecutive_negative_days, funds = apply_negative_funds_update(
                            result,
                            env.funds,
                            consecutive_negative_days,
                        )
                        if funds < 0:
                            if consecutive_negative_days >= 5:
                                print(f"[FAILURE] {consecutive_negative_days} consecutive days with negative funds")
                                log_message(
                                    run_log,
                                    {
                                        "role": "system",
                                        "day": day,
                                        "round": day_round,
                                        "message": f"Failure: {consecutive_negative_days} consecutive days with negative funds",
                                        "funds": funds,
                                    },
                                )
                                end_quality_first_action_round(env)
                                write_log_json_array(log_path, run_log)
                                return
                        break

                end_quality_first_action_round(env)
                if turn_data is not None and skipped_tool_calls_after_reconsideration:
                    turn_data["skipped_tool_calls_after_reconsideration"] = skipped_tool_calls_after_reconsideration
                    turn_data["executed_tool_call_count_before_reconsideration"] = (
                        len(tool_calls_list) - len(skipped_tool_calls_after_reconsideration)
                    )
                    save_turn_calls_to_json(
                        log_dir,
                        "execute",
                        execution_turns,
                        turn_data,
                        day,
                        round_index=day_round,
                    )

                if valid_tool_calls_count > 0:
                    consecutive_no_valid_tool_calls = 0
                else:
                    consecutive_no_valid_tool_calls += 1
                    print(
                        f"[Day {day} Execution Turn {execution_turns}] "
                        f"No valid tool calls (consecutive: {consecutive_no_valid_tool_calls}/5)"
                    )

                if consecutive_no_valid_tool_calls >= 5 and not execution_phase_complete:
                    print(
                        f"[Day {day}] Execution round {day_round} ended: "
                        "5 consecutive turns with no valid tool calls"
                    )
                    print(f"[Day {day}] Executing end_today due to consecutive invalid tool calls")
                    log_message(
                        run_log,
                        {
                            "role": "system",
                            "day": day,
                            "round": day_round,
                            "message": "Execution round ended: 5 consecutive turns with no valid tool calls",
                        },
                    )
                    try:
                        end_today_result = env.exec_tools("end_today")
                        if tool_result_has_error(end_today_result):
                            err_msg = (
                                "Forced end_today failed after consecutive invalid tool calls: "
                                f"{end_today_result.get('formatted', safe_dump(end_today_result))}"
                            )
                            print(f"[ERROR] {err_msg}")
                            log_message(
                                run_log,
                                {
                                    "role": "error",
                                    "day": day,
                                    "round": day_round,
                                    "phase": "execution",
                                    "turn": execution_turns,
                                    "message": err_msg,
                                    "error_type": "end_today_failed",
                                },
                            )
                            write_log_json_array(log_path, run_log)
                            return
                        execution_phase_complete = True
                        day_complete = True
                        previous_day_end_today_formatted = end_today_result.get("formatted") or safe_dump(end_today_result)
                        consecutive_negative_days, funds = apply_negative_funds_update(
                            end_today_result,
                            env.funds,
                            consecutive_negative_days,
                        )
                        if funds < 0 and consecutive_negative_days >= 5:
                            print(f"[FAILURE] {consecutive_negative_days} consecutive days with negative funds")
                            log_message(
                                run_log,
                                {
                                    "role": "system",
                                    "day": day,
                                    "round": day_round,
                                    "message": f"Failure: {consecutive_negative_days} consecutive days with negative funds",
                                    "funds": funds,
                                },
                            )
                            write_log_json_array(log_path, run_log)
                            return
                        log_message(
                            run_log,
                            {
                                "role": "tool",
                                "day": day,
                                "round": day_round,
                                "phase": "execution",
                                "turn": execution_turns,
                                "name": "end_today",
                                "content": end_today_result.get("formatted", safe_dump(end_today_result)),
                                "raw": end_today_result.get("result", safe_dump(end_today_result)),
                                "forced": True,
                                "reason": "5 consecutive invalid tool-call turns",
                            },
                        )
                    except Exception as end_today_exc:
                        err_msg = (
                            "Failed to execute end_today after consecutive invalid tool calls: "
                            f"{type(end_today_exc).__name__}: {end_today_exc}"
                        )
                        print(f"[ERROR] {err_msg}")
                        log_message(
                            run_log,
                            {
                                "role": "error",
                                "day": day,
                                "round": day_round,
                                "phase": "execution",
                                "turn": execution_turns,
                                "message": err_msg,
                                "error_type": type(end_today_exc).__name__,
                            },
                        )
                    break

                if tool_calls_list and tool_response_messages:
                    if not suppress_assistant_message:
                        execution_messages.append(assistant_message)
                    execution_messages.extend(tool_response_messages)
                    logged_user_content = "\n".join(
                        message.get("content", "") for message in tool_response_messages
                    )
                    log_message(
                        run_log,
                        {
                            "role": "user",
                            "day": day,
                            "round": day_round,
                            "phase": "execution",
                            "turn": execution_turns,
                            "content": logged_user_content,
                            "messages": tool_response_messages,
                            "parse_method": parse_method_tag,
                            "meta": {"tool_responses": tool_response_meta},
                        },
                    )
                else:
                    execution_messages.append(assistant_message)
                    execution_messages.append({"role": "user", "content": "No valid tool call detected. Continue or call end_today."})

                write_log_json_array(log_path, run_log)

            day_execution_turns_total += execution_turns

            if not execution_phase_complete:
                projected_day_turns_total = day_execution_turns_total
                if projected_day_turns_total >= daily_execution_turn_limit:
                    print(
                        f"[Day {day}] Execution stopped: reached daily max turns "
                        f"({daily_execution_turn_limit})"
                    )
                    log_message(
                        run_log,
                        {
                            "role": "system",
                            "day": day,
                            "round": day_round,
                            "message": f"Execution stopped: reached daily max turns ({daily_execution_turn_limit})",
                        },
                    )
                elif execution_turns >= round_turn_limit:
                    print(
                        f"[Day {day}] Execution round {day_round} ended: "
                        f"reached round max turns ({round_turn_limit})"
                    )
                    log_message(
                        run_log,
                        {
                            "role": "system",
                            "day": day,
                            "round": day_round,
                            "message": f"Execution round ended: reached round max turns ({round_turn_limit})",
                        },
                    )
                if projected_day_turns_total < daily_execution_turn_limit:
                    print(f"[Day {day}] Round {day_round} ended without end_today; re-planning")
                    log_message(
                        run_log,
                        {
                            "role": "system",
                            "day": day,
                            "round": day_round,
                            "message": "Round ended without end_today; re-planning",
                        },
                    )

        if not day_complete:
            print(f"[Day {day}] Forcing end_today after {daily_execution_turn_limit} execution turns")
            try:
                end_today_result = env.exec_tools("end_today")
                if tool_result_has_error(end_today_result):
                    err_msg = (
                        "Forced end_today failed after daily execution turn limit: "
                        f"{end_today_result.get('formatted', safe_dump(end_today_result))}"
                    )
                    print(f"[ERROR] {err_msg}")
                    log_message(run_log, {"role": "error", "day": day, "message": err_msg, "error_type": "end_today_failed"})
                    write_log_json_array(log_path, run_log)
                    return
                day_complete = True
                previous_day_end_today_formatted = end_today_result.get("formatted") or safe_dump(end_today_result)
                consecutive_negative_days, funds = apply_negative_funds_update(
                    end_today_result,
                    env.funds,
                    consecutive_negative_days,
                )
                if funds < 0 and consecutive_negative_days >= 5:
                    print(f"[FAILURE] {consecutive_negative_days} consecutive days with negative funds")
                    log_message(
                        run_log,
                        {
                            "role": "system",
                            "day": day,
                            "message": f"Failure: {consecutive_negative_days} consecutive days with negative funds",
                            "funds": funds,
                        },
                    )
                    write_log_json_array(log_path, run_log)
                    return
                log_message(run_log, {"role": "system", "day": day, "message": "Forced end_today"})
            except Exception as e:
                print(f"[ERROR] Failed to force end_today after daily execution turn limit: {e}")

        if log_dir is not None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                day_token_path = log_dir / f"day_{day}_token_usage.json"
                day_token_data = {
                    "day": day,
                    "current_date": str(env.current_date),
                    "prompt_tokens": day_prompt_tokens,
                    "completion_tokens": day_completion_tokens,
                    "total_tokens": day_tokens,
                    "cached_prompt_tokens": day_cached_prompt_tokens,
                    "uncached_prompt_tokens": max(day_prompt_tokens - day_cached_prompt_tokens, 0),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                with day_token_path.open("w", encoding="utf-8") as f:
                    json.dump(day_token_data, f, ensure_ascii=False, indent=2, default=str)
                print(
                    f"[Day {day}] Token usage: "
                    f"prompt={day_prompt_tokens:,}, completion={day_completion_tokens:,}, total={day_tokens:,}"
                )
            except Exception as exc:
                print(f"[WARN] Failed to write day {day} token usage to json: {exc}")

        write_log_json_array(log_path, run_log)

        # Store previous day summary for next day's plan
        previous_day_actions = day_actions

        # Save day checkpoint
        should_save_checkpoint = (
            checkpoint_dir is not None
            and checkpoint_interval > 0
            and (day == max_days or day % checkpoint_interval == 0)
        )
        if should_save_checkpoint:
            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(checkpoint_dir, global_turn, execution_messages, env)
                day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
                day_messages_path = checkpoint_dir / f"day_{day}_messages.json"
                day_env_checkpoint_path = checkpoint_dir / f"day_{day}_env_checkpoint.json"

                with day_messages_path.open("w", encoding="utf-8") as f:
                    json.dump(execution_messages, f, ensure_ascii=False, indent=2, default=str)

                env.save_checkpoint(day_env_checkpoint_path)

                checkpoint_metadata = {
                    "day": day,
                    "current_date": str(env.current_date),
                    "global_turn": global_turn,
                    "execution_turns": day_execution_turns_total,
                    "messages_path": str(day_messages_path.relative_to(checkpoint_dir)),
                    "env_checkpoint_path": str(day_env_checkpoint_path.relative_to(checkpoint_dir)),
                    "previous_day_actions": previous_day_actions,
                    "previous_day_end_today_formatted": previous_day_end_today_formatted,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                with day_checkpoint_path.open("w", encoding="utf-8") as f:
                    json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2, default=str)

                print(f"[Checkpoint] Saved day {day} checkpoint to {checkpoint_dir}")
            except Exception as exc:
                print(f"[WARN] Failed to save day {day} checkpoint: {exc}")

    print("\nSimulation completed.")

    print("\n" + "=" * 80)
    print("Token Usage Statistics:")
    print("=" * 80)
    print(f"Total Prompt Tokens: {total_prompt_tokens:,}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Cached Prompt Tokens: {total_cached_prompt_tokens:,}")
    print("=" * 80 + "\n")

    if log_dir is not None:
        try:
            token_stats_path = log_dir / "token_statistics.json"
            token_stats = {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cached_prompt_tokens": total_cached_prompt_tokens,
                "total_uncached_prompt_tokens": max(total_prompt_tokens - total_cached_prompt_tokens, 0),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            with token_stats_path.open("w", encoding="utf-8") as f:
                json.dump(token_stats, f, ensure_ascii=False, indent=2, default=str)
            print(f"[INFO] Token statistics saved to {token_stats_path}")
        except Exception as exc:
            print(f"[WARN] Failed to save token statistics: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retail environment with Plan-and-Act loop")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save/load checkpoints")
    parser.add_argument("--recover_turn", type=int, help="Turn number to recover from (if recovering)")
    parser.add_argument("--recover_day", type=int, help="Day number to recover from (if recovering from day checkpoint)")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Save checkpoint every N days and on the final day; 0 disables checkpoint saving")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--max_turns",
        type=int,
        default=50,
        help="Maximum total execution turns per day across all plan-act rounds",
    )
    parser.add_argument("--db_path", type=str, default=None, help="Database path for order records (default: 'model_run_time')")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for agent/environment logs and run artifacts")
    parser.add_argument(
        "--config_type",
        type=str,
        choices=CONFIG_TYPE_CHOICES,
        default="hard",
        help="Configuration type, supports easy/middle/hard",
    )
    parser.add_argument("--max_input_tokens", type=int, default=50000, help="Maximum input tokens for context window")
    parser.add_argument("--max_days", type=int, default=30, help="Maximum number of days to simulate")
    parser.add_argument("--max_execution_turns", type=int, default=20, help="Maximum turns per execution round")
    parser.add_argument("--quality_first_mode", type=str, choices=["off", "gate"], default="off", help="Quality-first supplier gate mode")
    parser.add_argument("--caec_mode", type=str, choices=["off", "gate"], default="off", help="Legacy no-op argument")
    parser.add_argument("--caec_config", type=str, default=None, help="Legacy no-op argument")
    parser.add_argument("--fcr_mode", type=str, choices=["off", "gate"], default="off", help="Legacy no-op argument")
    parser.add_argument("--fcr_window_days", type=int, default=14, help="Legacy no-op argument")
    parser.add_argument("--api_key", type=str, default=None, help=f"OpenAI API key (default: {DEFAULT_API_KEY})")
    parser.add_argument("--base_url", type=str, default=None, help=f"OpenAI API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    config = load_config_by_type(args.config_type)
    config["quality_first_mode"] = args.quality_first_mode
    artifacts = initialize_run_artifacts(
        prefix="run_plan_and_act",
        config=config,
        args_payload={
            "recover_turn": args.recover_turn,
            "recover_day": args.recover_day,
            "checkpoint_interval": args.checkpoint_interval,
            "model": args.model,
            "max_turns": args.max_turns,
            "config_type": args.config_type,
            "max_input_tokens": args.max_input_tokens,
            "max_days": args.max_days,
            "max_execution_turns": args.max_execution_turns,
            "quality_first_mode": args.quality_first_mode,
            "legacy_caec_mode": args.caec_mode,
            "legacy_caec_config": args.caec_config,
            "legacy_fcr_mode": args.fcr_mode,
            "legacy_fcr_window_days": args.fcr_window_days,
            "api_key": "***" if args.api_key else None,
            "base_url": args.base_url,
        },
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        db_path=args.db_path,
    )

    goal = build_goal(DEFAULT_GOAL, config)

    client = create_openai_client(api_key=args.api_key, base_url=args.base_url)

    checkpoint_dir = artifacts["checkpoint_dir"]

    initial_messages = None
    start_turn = 0
    start_day = 1
    previous_day_actions: List[str] = []
    previous_day_end_today_formatted: Optional[str] = None

    if args.recover_day is not None:
        print(f"[Checkpoint] Recovering from day {args.recover_day} checkpoint...")
        env, previous_day_actions, previous_day_end_today_formatted, start_day, start_turn = recover_from_day_checkpoint(
            checkpoint_dir, args.recover_day
        )
    elif args.recover_turn is not None:
        print(f"[Checkpoint] Recovering from turn {args.recover_turn}...")
        env, recovered_messages = recover_from_checkpoint(checkpoint_dir, args.recover_turn)
        initial_messages = recovered_messages
        start_turn = args.recover_turn
    else:
        env = RetailEnvironment(config)

    run_plan_and_act_loop(
        goal=goal,
        env=env,
        model=args.model,
        max_turns=args.max_turns,
        log_path=artifacts["log_path"],
        max_input_tokens=args.max_input_tokens,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        initial_messages=initial_messages,
        start_turn=start_turn,
        start_day=start_day,
        previous_day_actions=previous_day_actions,
        previous_day_end_today_formatted=previous_day_end_today_formatted,
        log_dir=artifacts["run_dir"],
        max_days=args.max_days,
        max_execution_turns_per_day=args.max_execution_turns,
        client=client,
    )


if __name__ == "__main__":
    main()
