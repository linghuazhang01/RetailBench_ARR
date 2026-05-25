"""Run 365-day oracle (best-parameter) simulations for all config types.

Uses optimal parameters from 105-experiment grid search:
  easy:   bulk=4, sample=6
  middle: bulk=3, sample=4
  hard:   bulk=3, sample=5

Results saved to logs/oracle/
"""
from __future__ import annotations

import sys
import os
import csv
import json
import time
import io
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/retailbench-matplotlib")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module"))

from agents.run_non_llm_simulation import simulate_quality_based_environment
from util.default_config import get_config_by_name

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "oracle")
DAYS = 365

CONFIGS = [
    {"config_type": "easy",   "bulk": 4, "sample": 6},
    {"config_type": "middle", "bulk": 3, "sample": 4},
    {"config_type": "hard",   "bulk": 3, "sample": 5},
]


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not math.isnan(float(v))
    ]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _round_optional(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _fmt_number(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.{digits}f}"


def _successful_order_lines(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = record.get("result", {})
    if not isinstance(result, dict):
        return []
    result_data = result.get("result", {})
    if not isinstance(result_data, dict):
        return []
    lines = result_data.get("lines", [])
    if not isinstance(lines, list):
        return []
    return [line for line in lines if isinstance(line, dict)]


def _extract_sku_ids(raw_items: Any) -> set[str]:
    skus: set[str] = set()
    if not isinstance(raw_items, list):
        return skus
    for item in raw_items:
        if isinstance(item, str):
            skus.add(item)
        elif isinstance(item, dict):
            for key in ("sku_id", "sku", "upc"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    skus.add(value)
                    break
    return skus


def _load_daily_price_ratios(db_path: Path) -> Dict[str, List[float]]:
    ratios_by_date: Dict[str, List[float]] = {}
    if not db_path.exists():
        return ratios_by_date
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT sold_date, selling_price, buy_price FROM product_lifecycle "
            "WHERE status='sold' AND buy_price > 0"
        )
        for sold_date, selling_price, buy_price in cursor.fetchall():
            if sold_date and selling_price and buy_price:
                ratios_by_date.setdefault(str(sold_date), []).append(
                    float(selling_price) / float(buy_price)
                )
        conn.close()
    except Exception as exc:
        print(f"Daily price-ratio extraction skipped for {db_path}: {exc}")
    return ratios_by_date


def _plot_daily_metrics(daily_rows: List[Dict[str, Any]], figures_dir: Path,
                        title: str) -> List[str]:
    if not daily_rows:
        return []
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(figures_dir / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Daily metric plot generation skipped: matplotlib unavailable ({exc})")
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    days = [row["day"] for row in daily_rows]
    specs = [
        ("price_ratio", "Price Ratio"),
        ("order_ratio", "Order Ratio"),
        ("net_worth", "Net Worth"),
        ("funds", "Funds"),
    ]

    def values_for(key: str) -> List[float]:
        values = []
        for row in daily_rows:
            value = row.get(key)
            values.append(float(value) if isinstance(value, (int, float)) else math.nan)
        return values

    paths: List[str] = []
    fig, axes = plt.subplots(len(specs), 1, figsize=(10, 12), sharex=True)
    for ax, (key, label) in zip(axes, specs):
        ax.plot(days, values_for(key), linewidth=1.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Day")
    fig.tight_layout()
    combined_path = figures_dir / "metrics_over_time.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(combined_path))

    for key, label in specs:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(days, values_for(key), linewidth=1.5)
        ax.set_title(f"{title}: {label}")
        ax.set_xlabel("Day")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out_path = figures_dir / f"{key}_over_time.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out_path))

    return paths


def _write_daily_metrics(log_dir: Path, config_type: str) -> Dict[str, Any]:
    tool_calls_path = log_dir / "tool_calls.jsonl"
    db_path = log_dir / "db" / "records.db"
    config = get_config_by_name(config_type)
    daily_rent = float(config.get("everyday_rent", 0.0) or 0.0)
    price_ratios_by_date = _load_daily_price_ratios(db_path)

    daily_rows: List[Dict[str, Any]] = []
    daily_sales_by_sku: List[Dict[str, float]] = []
    daily_stockout_skus: List[set[str]] = []
    current_orders_by_sku: Dict[str, int] = {}

    if not tool_calls_path.exists():
        return {}

    with tool_calls_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            tool = record.get("tool")
            if tool == "place_order":
                for item in _successful_order_lines(record):
                    sku_id = item.get("sku_id")
                    quantity = item.get("quantity")
                    if isinstance(sku_id, str) and isinstance(quantity, (int, float)):
                        current_orders_by_sku[sku_id] = (
                            current_orders_by_sku.get(sku_id, 0) + int(quantity)
                        )
                continue

            if tool != "end_today":
                continue

            result = record.get("result", {})
            result_data = result.get("result", {}) if isinstance(result, dict) else {}
            if not isinstance(result_data, dict):
                current_orders_by_sku = {}
                continue

            day = len(daily_rows) + 1
            current_date = str(result_data.get("current_date", ""))
            sales_by_sku = result_data.get("sales_by_sku", {})
            expired_by_sku = result_data.get("expired_discount_by_sku", {})
            returns_by_sku = result_data.get("returns_by_sku", {})
            insufficient_skus = _extract_sku_ids(result_data.get("insufficient_skus", []))

            sales_today = (
                {str(k): float(v) for k, v in sales_by_sku.items() if isinstance(v, (int, float))}
                if isinstance(sales_by_sku, dict)
                else {}
            )
            daily_sales_by_sku.append(sales_today)
            daily_stockout_skus.append(insufficient_skus)

            ratios = price_ratios_by_date.get(current_date, [])
            price_ratio = _mean(ratios)

            order_ratio = None
            if current_orders_by_sku:
                order_ratios: List[float] = []
                day_idx = len(daily_rows)
                lookback_start = max(0, day_idx - 6)
                for sku_id, order_qty in current_orders_by_sku.items():
                    non_stockout_sales = []
                    for idx in range(lookback_start, day_idx + 1):
                        if sku_id not in daily_stockout_skus[idx]:
                            non_stockout_sales.append(daily_sales_by_sku[idx].get(sku_id, 0.0))
                    avg_sales = _mean(non_stockout_sales)
                    if avg_sales and avg_sales > 0:
                        order_ratios.append(float(order_qty) / avg_sales)
                order_ratio = _mean(order_ratios)

            funds = float(result_data.get("funds", record.get("funds", 0.0)) or 0.0)
            net_worth = float(result_data.get("net_worth", record.get("net_worth", 0.0)) or 0.0)
            daily_rows.append(
                {
                    "day": day,
                    "date": current_date,
                    "funds": round(funds, 6),
                    "net_worth": round(net_worth, 6),
                    "net_worth_without_rent": round(net_worth + daily_rent * day, 6),
                    "price_ratio": _round_optional(price_ratio),
                    "order_ratio": _round_optional(order_ratio),
                    "daily_sold": round(sum(sales_today.values()), 6),
                    "daily_profit": round(float(result_data.get("money_earned", 0.0) or 0.0), 6),
                    "daily_expired": (
                        sum(v for v in expired_by_sku.values() if isinstance(v, (int, float)))
                        if isinstance(expired_by_sku, dict)
                        else 0
                    ),
                    "daily_returns": (
                        sum(v for v in returns_by_sku.values() if isinstance(v, (int, float)))
                        if isinstance(returns_by_sku, dict)
                        else 0
                    ),
                }
            )
            current_orders_by_sku = {}

    json_path = log_dir / "daily_metrics.json"
    csv_path = log_dir / "daily_metrics.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(daily_rows, handle, indent=2, ensure_ascii=False)
    if daily_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(daily_rows[0].keys()))
            writer.writeheader()
            writer.writerows(daily_rows)

    plot_paths = _plot_daily_metrics(
        daily_rows,
        log_dir / "figures",
        f"Oracle {config_type}",
    )

    avg_price_ratio = _mean([row["price_ratio"] for row in daily_rows])
    avg_order_ratio = _mean([row["order_ratio"] for row in daily_rows])
    final_without_rent = (
        daily_rows[-1]["net_worth_without_rent"] if daily_rows else None
    )
    return {
        "daily_rent": daily_rent,
        "final_net_worth_without_rent": _round_optional(final_without_rent, 2),
        "avg_price_ratio": _round_optional(avg_price_ratio),
        "avg_order_ratio": _round_optional(avg_order_ratio),
        "daily_metrics_json": str(json_path),
        "daily_metrics_csv": str(csv_path),
        "metric_plot_paths": plot_paths,
    }


def _jsonl_to_json_array(jsonl_path: Path, json_path: Path) -> int:
    """Convert tool_calls.jsonl to a JSON array for easier inspection."""
    rows = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return len(rows)


def run_oracle(cfg: dict, run_root: Path) -> dict:
    """Run a single oracle simulation with suppressed output."""
    config_type = cfg["config_type"]
    log_dir = run_root / config_type
    db_dir = log_dir / "db"
    log_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    _logger = logging.getLogger("RetailBench")
    _logger.setLevel(logging.CRITICAL)
    for h in _logger.handlers:
        h.setLevel(logging.CRITICAL)

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        result = simulate_quality_based_environment(
            days=DAYS,
            sample_size=cfg["sample"],
            config_type=config_type,
            bulk_qty_multiplier=cfg["bulk"],
            db_path=str(db_dir),
            log_dir=str(log_dir),
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        _logger.setLevel(logging.INFO)
        for h in _logger.handlers:
            h.setLevel(logging.INFO)

    tool_jsonl = log_dir / "tool_calls.jsonl"
    tool_json = log_dir / "tool_calls.json"
    tool_call_count = _jsonl_to_json_array(tool_jsonl, tool_json)
    daily_metric_summary = _write_daily_metrics(log_dir, config_type)
    result["oracle_log_dir"] = str(log_dir)
    result["tool_calls_jsonl"] = str(tool_jsonl)
    result["tool_calls_json"] = str(tool_json)
    result["tool_call_count"] = tool_call_count
    result.update(daily_metric_summary)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(OUTPUT_DIR) / f"oracle_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Oracle Simulation (365 days, best parameters)")
    print("=" * 60)
    print(f"  Log root: {run_root}")

    for cfg in CONFIGS:
        ct = cfg["config_type"]
        print(f"\n[{ct}] bulk={cfg['bulk']}, sample={cfg['sample']}, days={DAYS}")
        t0 = time.time()
        result = run_oracle(cfg, run_root)
        elapsed = time.time() - t0
        all_results.append(result)
        print(f"  net_worth={result['final_net_worth']:,.2f}  "
              f"profit/day={result['avg_daily_profit']:,.2f}  "
              f"expired={result['expired_ratio']:.4f}  "
              f"returns={result['return_ratio']:.4f}  "
              f"tool_calls={result['tool_call_count']:,}  "
              f"({elapsed:.0f}s)")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, f"oracle_{timestamp}.json")

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "days": DAYS,
            "parameter_source": "105-experiment grid search (para-search/)",
            "log_root": str(run_root),
        },
        "best_params": {
            c["config_type"]: {"bulk_qty_multiplier": c["bulk"], "sample_size": c["sample"]}
            for c in CONFIGS
        },
        "results": all_results,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with (run_root / "oracle_results.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Also save as latest
    latest_path = os.path.join(OUTPUT_DIR, "oracle_latest.json")
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary report
    lines = [
        f"# Oracle Simulation Results ({DAYS} days)",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Best Parameters (from 105-experiment grid search)",
        f"",
        f"| Config | bulk | sample |",
        f"|--------|-----:|-------:|",
    ]
    for c in CONFIGS:
        lines.append(f"| {c['config_type']} | {c['bulk']} | {c['sample']} |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Config | Days | Net Worth | Net Worth w/o Rent | Funds | Avg Price Ratio | Avg Order Ratio | Avg Daily Sold | Avg Daily Profit | Total Sold | Total Ordered | Expired | Returns | Expired Ratio | Return Ratio | Tool Calls | Log Dir |")
    lines.append("|--------|-----:|----------:|-------------------:|------:|----------------:|----------------:|---------------:|-----------------:|-----------:|--------------:|--------:|--------:|--------------:|-------------:|-----------:|:--------|")
    for r in all_results:
        lines.append(
            f"| {r['config_type']} "
            f"| {r['run_days']} "
            f"| {r['final_net_worth']:,.2f} "
            f"| {_fmt_number(r.get('final_net_worth_without_rent'), 2)} "
            f"| {r['final_funds']:,.2f} "
            f"| {_fmt_number(r.get('avg_price_ratio'), 4)} "
            f"| {_fmt_number(r.get('avg_order_ratio'), 4)} "
            f"| {r['avg_daily_sold']:,.1f} "
            f"| {r['avg_daily_profit']:,.2f} "
            f"| {r['total_sold']:,} "
            f"| {r['total_ordered']:,} "
            f"| {r['total_expired']:,} "
            f"| {r['total_returns']:,} "
            f"| {r['expired_ratio']:.4f} "
            f"| {r['return_ratio']:.4f} "
            f"| {r['tool_call_count']:,} "
            f"| `{r['oracle_log_dir']}` |"
        )
    lines.append("")

    report_path = os.path.join(OUTPUT_DIR, f"oracle_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    with (run_root / "oracle_results.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUTPUT_DIR, "oracle_latest.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  oracle_{timestamp}.json")
    print(f"  oracle_{timestamp}.md")


if __name__ == "__main__":
    main()
