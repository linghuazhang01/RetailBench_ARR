"""
Non-LLM Simulation Runners for RetailBench

This module contains simulation functions that run retail environment operations
without any LLM-based decision making. These serve as baselines and test harnesses.

Environment Types and Their Mapping:
=====================================

| Function Name                      | Environment Type      | Key Features                                      | Config Types Supported          |
|-------------------------------------|----------------------|---------------------------------------------------|--------------------------------|
| simulate_ultra_easy_environment     | Ultra Easy           | Minimal logic, simple reorder                    | ultra_easy                     |
| simulate_simple_logic_environment   | Simple Logic         | Basic pricing/ordering, cheapest supplier         | dynamic_hard/middle, still_hard/middle |
| simulate_simple_review_environment  | Review-Aware         | Supplier selection based on customer reviews      | dynamic_hard/middle, still_hard/middle |
| simulate_news_aware_environment     | News-Aware           | Uses news to adjust demand forecasting            | dynamic_hard/middle, still_hard/middle |
| simulate_quality_based_environment  | Quality-Based        | Considers product quality in decisions            | dynamic_hard/middle, still_hard/middle |
| simulate_quality_based_environment  | Quality-Based        | Adds explicit shelf assortment for hard_v2        | hard_v2                              |

Config Type Mapping:
- dynamic_hard:   High demand variability, hard difficulty
- hard_v2:        Hard setting with power category attraction and shelf capacity
- dynamic_middle: Medium demand variability, medium difficulty
- still_hard:    Low demand variability, hard difficulty
- still_middle:  Low demand variability, medium difficulty
- ultra_easy:    Minimal complexity for testing
"""
from __future__ import annotations
import sys
import os


# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module"))

from collections import defaultdict
from datetime import date, datetime, timedelta
import random
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from util.config_registry import CONFIG_TYPE_CHOICES, load_config_by_type
from retail_environment import RetailEnvironment
from util.logger import set_logger_level
from util.default_config import get_config_by_name


def _shelf_constraint_enabled(env: RetailEnvironment) -> bool:
    return bool(getattr(env, "enable_shelf_constraint", False))


def _shelf_capacity(env: RetailEnvironment, fallback: int) -> int:
    capacity = getattr(env, "shelf_sku_capacity", None)
    if capacity is None:
        return fallback
    return max(int(capacity), 0)


def _select_diverse_shelf_skus(env: RetailEnvironment, sku_ids: List[str], capacity: int) -> List[str]:
    if capacity <= 0:
        return []

    seen = set()
    by_category: Dict[str, List[str]] = defaultdict(list)
    for sku_id in sku_ids:
        if sku_id in seen or sku_id not in env.skus_id_map:
            continue
        seen.add(sku_id)
        category = getattr(env.skus_id_map[sku_id], "category", "")
        by_category[category].append(sku_id)

    selected: List[str] = []
    max_len = max((len(items) for items in by_category.values()), default=0)
    for rank in range(max_len):
        for category in sorted(by_category):
            items = by_category[category]
            if rank >= len(items):
                continue
            selected.append(items[rank])
            if len(selected) >= capacity:
                return selected
    return selected


def _apply_baseline_shelf_assortment(
    env: RetailEnvironment,
    sampled_skus: List[str],
    label: str,
) -> List[str]:
    """Choose the non-LLM baseline's active SKU set when shelf capacity is enabled."""
    if not _shelf_constraint_enabled(env):
        return sampled_skus

    capacity = _shelf_capacity(env, len(sampled_skus))
    shelf_skus = _select_diverse_shelf_skus(env, sampled_skus, capacity)
    env.exec_tools("set_shelf_skus", sku_ids=shelf_skus)
    env.log_info(
        f"[Shelf-{label}] Enabled shelf constraint: using {len(shelf_skus)}/"
        f"{capacity} sampled SKUs as the active shelf assortment."
    )
    return shelf_skus



def _snapshot_env_state_for_test(env: RetailEnvironment) -> Dict[str, Any]:
    """
    Snapshot environment state for checkpoint consistency testing.
    """
    state: Dict[str, Any] = {}
    state["funds"] = round(float(env.funds), 6)
    state["current_date"] = env.current_date.isoformat()
    state["shelf_sku_ids"] = sorted(getattr(env, "shelf_sku_ids", set()))

    sku_prices: Dict[str, float] = {}
    for sku_id, sku in env.skus_id_map.items():
        sku_prices[sku_id] = round(float(sku.price), 6)
    state["sku_prices"] = sku_prices

    inv_snapshot: Dict[str, Any] = {"capacity": env.inventory.capacity, "items_by_sku": {}, "waiting_items": []}
    for sku_id, items in env.inventory.items_by_sku.items():
        items_data = []
        for m in items:
            items_data.append({
                "sku_id": m.sku.sku_id,
                "begin_time": m.begin_time.isoformat(),
                "expired_time": m.expired_time.isoformat(),
                "buy_price": round(float(m.buy_price), 6),
                "merch_id": m.merch_id,
                "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
                "supplier_id": getattr(m, "supplier_id", None),
            })
        items_data.sort(key=lambda x: x["merch_id"])
        inv_snapshot["items_by_sku"][sku_id] = items_data

    waiting_data = []
    for m in env.inventory.waiting_items:
        waiting_data.append({
            "sku_id": m.sku.sku_id,
            "begin_time": m.begin_time.isoformat(),
            "expired_time": m.expired_time.isoformat(),
            "buy_price": round(float(m.buy_price), 6),
            "merch_id": m.merch_id,
            "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
            "supplier_id": getattr(m, "supplier_id", None),
        })
    waiting_data.sort(key=lambda x: x["merch_id"])
    inv_snapshot["waiting_items"] = waiting_data
    state["inventory"] = inv_snapshot

    orders_snapshot: Dict[str, Any] = {}
    for order_id, order in env.order_manager.orders.items():
        items_data = []
        for m in order.items:
            items_data.append({
                "sku_id": m.sku.sku_id,
                "begin_time": m.begin_time.isoformat(),
                "expired_time": m.expired_time.isoformat(),
                "buy_price": round(float(m.buy_price), 6),
                "merch_id": m.merch_id,
                "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
                "supplier_id": getattr(m, "supplier_id", None),
            })
        items_data.sort(key=lambda x: x["merch_id"])
        ordered_sku_dict = {sku.sku_id: qty for sku, qty in order.ordered_sku.items()}
        orders_snapshot[order_id] = {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "delivery_time": order.delivery_time,
            "cost": round(float(order.cost), 6),
            "items": items_data,
            "ordered_sku": ordered_sku_dict,
        }
    state["orders"] = orders_snapshot

    if env.skus_list:
        sample_sku_id = env.skus_list[0].sku_id
        cfg_begin = env.config.get("data_begin_time") or env.config.get("begin_time")
        try:
            start_dt = datetime.strptime(cfg_begin, "%m/%d/%y").date()
        except Exception:
            start_dt = env.current_date
        sales = env.record_manager.read_sku(sample_sku_id, start_dt, env.current_date)
        sales_snapshot: Dict[str, Any] = {}
        for day, recs in sales.items():
            move_sum = sum(r.move for r in recs)
            last = recs[-1]
            sales_snapshot[day] = {
                "total_move": int(move_sum),
                "last_price": round(float(last.price), 6),
                "last_customer_count": int(last.customer_count) if last.customer_count is not None else None,
                "records": len(recs),
            }
        state["sample_sales"] = {"sku_id": sample_sku_id, "by_day": sales_snapshot}

    if getattr(env, "news_manager", None) and env.config.get("enable_new"):
        today_news = env.news_manager.get_today_news()
        rolling = getattr(env.news_manager, "_rolling", [])
        state["news_manager"] = {
            "today_news_count": len(today_news),
            "rolling_count": len(rolling),
        }

    return state


def _run_checkpoint_consistency_test(
    env: RetailEnvironment,
    label: str,
    checkpoint_prefix: str,
    checkpoint_filename: str,
) -> None:
    """
    Save and restore an environment checkpoint, then compare stable state fields.
    """
    import shutil
    import tempfile

    checkpoint_dir: Optional[Path] = None
    try:
        env.log_info(f"[CheckpointTest-{label}] 开始保存与恢复环境，用于一致性验证...")
        original_state = _snapshot_env_state_for_test(env)

        checkpoint_dir = Path(tempfile.mkdtemp(prefix=checkpoint_prefix))
        checkpoint_path = checkpoint_dir / checkpoint_filename
        env.save_checkpoint(checkpoint_path)

        recovered_env = RetailEnvironment.recover_from_checkpoint(checkpoint_path)
        recovered_state = _snapshot_env_state_for_test(recovered_env)

        assert original_state["funds"] == recovered_state["funds"]
        assert original_state["current_date"] == recovered_state["current_date"]
        assert original_state["sku_prices"] == recovered_state["sku_prices"]
        assert original_state["shelf_sku_ids"] == recovered_state["shelf_sku_ids"]
        assert original_state["inventory"] == recovered_state["inventory"]
        assert original_state["orders"] == recovered_state["orders"]

        if "sample_sales" in original_state:
            assert original_state["sample_sales"] == recovered_state.get("sample_sales")
        if "news_manager" in original_state:
            assert original_state["news_manager"] == recovered_state.get("news_manager")

        env.log_info(f"[CheckpointTest-{label}] 环境恢复后一致性验证通过。")
    except AssertionError as exc:
        env.log_info(f"[CheckpointTest-{label}] 一致性验证失败: {exc}")
    except Exception as exc:
        env.log_info(f"[CheckpointTest-{label}] 执行 checkpoint 测试时发生异常: {exc}")
    finally:
        if checkpoint_dir is not None:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def simulate_simple_logic_environment(
    days: int = 7,
    sample_size: int = 3,
    config_type: str = "dynamic_hard",
    db_path: str = None,
    enable_checkpoint_test: bool = False,
):
    """
    初始化环境后，模拟正常经营若干天：
    - 每天调用 step() 推进一天
    - 如果某个测试 SKU 库存太低，就自动下单补货
    - 打印每天的资金变化、销量情况和订单数
    - 每日展示新闻、供给商报价，并在日终查看记录管理器写入的数据
    
    Args:
        days: 模拟天数
        sample_size: 每个品类选择的 SKU 数量
        config_type: 配置类型 ("dynamic_hard", "dynamic_middle", "still_hard", "still_middle")
        db_path: 数据库路径（可选）
        enable_checkpoint_test: 是否在模拟结束后执行 checkpoint 保存/恢复一致性测试
    """
    # 根据 config_type 选择配置
    config = get_config_by_name(config_type)
    
    env = RetailEnvironment(config)

    # 为了避免模拟过程中输出大量 debug 日志（尤其是 Attraction 相关），
    # 这里将日志级别降到 INFO，只保留关键信息与耗时统计。
    set_logger_level(True)
    supplier_choice_by_sku: Dict[str, Optional[str]] = {}

    def parse_date_safe(raw: str) -> date:
        try:
            return datetime.strptime(raw, "%m/%d/%y").date()
        except Exception:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except Exception:
                return date.today()

    def avg_profit_for_sku(sku_id: str, start_dt: date, end_dt: date) -> float:
        sales = env.record_manager.read_sku(sku_id, start_dt, end_dt)
        all_records = [rec for recs in sales.values() for rec in recs]
        if not all_records:
            return float("-inf")
        price_rows = env.record_manager.read_supplier_prices(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
        )
        min_price_by_date: Dict[str, float] = {}
        for row in price_rows:
            d = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
            min_price_by_date[d] = min(min_price_by_date.get(d, row.price), row.price)

        total_profit = 0.0
        total_units = 0
        for rec in all_records:
            d = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            cost = min_price_by_date.get(d, rec.price)
            profit_per_unit = rec.price - cost
            total_profit += profit_per_unit * rec.move
            total_units += rec.move

        return total_profit

    def select_skus_per_category(n: int) -> List[str]:
        selected: List[str] = []
        begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
        start_dt = parse_date_safe(begin_raw)
        # 使用当前日期作为窗口上限
        end_dt = env.current_date

        for category, sku_objs in env.skus_category_map.items():
            scores = []
            for sku_obj in sku_objs:
                score = avg_profit_for_sku(sku_obj.sku_id, start_dt, end_dt)
                scores.append((score, sku_obj.sku_id))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [sku_id for _, sku_id in scores[:n] if _ != float("-inf")]
            if len(top) < n:  # 补足
                remaining = [s.sku_id for s in sku_objs if s.sku_id not in top]
                top.extend(remaining[: max(0, n - len(top))])
            selected.extend(top)
        return selected

    def log_record_manager_snapshot(dt: date) -> None:
        try:
            today_str = dt.isoformat()
            supplier_rows = []
            for sku_id in env.skus_id_map.keys():
                supplier_rows.extend(
                    env.record_manager.read_supplier_prices(
                        sku_id=sku_id,
                        start_date=dt,
                        end_date=dt,
                    )
                )
            env.log_debug(f"记录管理器-当日供给商价格 ({len(supplier_rows)}): {[r.to_dict() for r in supplier_rows]}")

            news_rows = env.exec_tools(
                "view_news_history",
                start_date=today_str,
                end_date=today_str,
            ).get("result", [])
            env.log_debug(f"记录管理器-当日新闻 ({len(news_rows)}): {news_rows[:5]}")

            order_rows = env.record_manager.read_supplier_orders(
                start_order_date=dt,
                end_order_date=dt,
            )
            preview_orders = [
                {"supplier_id": o.supplier_id, "order_date": o.order_date, "arrival_date": o.arrival_date, "items": o.items}
                for o in order_rows[:5]
            ]
            env.log_debug(f"记录管理器-当日订单 ({len(order_rows)}): {preview_orders}")
        except Exception as e:
            env.log_debug(f"读取记录管理器数据失败: {e}")

    try:
        # 按品类选择 SKU：每个品类挑选 sample_size 个平均收益最高的 SKU
        if env.skus_category_map:
            sampled_skus = select_skus_per_category(sample_size)
        else:
            sampled_skus = []

        env.log_debug(f"开始模拟 {days} 天经营")
        env.log_debug(f"初始资金 & 日期：{env.view_funds_and_date()['formatted']}")
        env.log_debug(f"测试用 SKU: {sampled_skus}\n")

        consecutive_negative_days = 0  # 跟踪连续负资金天数
        
        for day in range(1, days + 1):
            # 整天耗时统计
            day_start_time = time.time()
            
            # 检查资金状态
            if env.funds < 0:
                consecutive_negative_days += 1
                if consecutive_negative_days >= 10:
                    env.log_debug(f"连续 {consecutive_negative_days} 天资金为负数，模拟结束。")
                    break
            else:
                consecutive_negative_days = 0  # 重置计数器
            
            env.log_debug(f"\n====== 第 {day} 天 ======")

            # 0. 查看当前日期与资金
            funds_snapshot = env.view_funds_and_date()
            env.log_debug(f"资金与日期：{funds_snapshot['formatted']}")

            # 0.0 生成并展示今日新闻
            if env.config.get("enable_new") and env.news_manager:
                try:
                    today_news = env.exec_tools("view_today_news")
                    env.log_debug("今日新闻：")
                    env.log_debug(today_news.get("formatted", ""))
                except Exception as e:
                    env.log_debug(f"今日新闻获取失败: {e}")

            try:
                for sku_id_for_test in sampled_skus:
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test]
                    )
                    env.log_debug(f"供应商报价（{sku_id_for_test}）: {quotes_resp['formatted']}")
            except Exception as e:
                env.log_debug(f"今日供应商价格获取失败: {e}")

            # 1. 查看当前库存
            inv_info = env.view_inventory()
            inv_result = inv_info.get("result", {})
            env.log_debug(f"当前库存: {inv_info['formatted']}")
            inventory_map = inv_result.get("inventory", {})
            if not sampled_skus:
                env.log_debug("没有找到可用的测试 SKU，跳过补货逻辑。")
            else:
                for sku_id_for_test in sampled_skus:
                    sku_start_time = time.time()
                    current_quantity = inventory_map.get(sku_id_for_test, {}).get("quantity", 0)
                    env.log_debug(f"SKU {sku_id_for_test} 当前库存：{current_quantity}")

                    # 1. 查看历史销售数据
                    t0 = time.time()
                    sales_resp = env.exec_tools(
                        "view_sales_profit_history",
                        sku_ids=[sku_id_for_test],
                        start_date=(env.current_date - timedelta(days=30)).isoformat(),
                        end_date=env.current_date.isoformat(),
                    )
                    t_sales = time.time() - t0

                    env.log_debug(
                        "历史销售数据:" + sales_resp['formatted']
                    )
                    sales_records = sales_resp.get("result", {}).get(sku_id_for_test, {}).get("records", {})

                    # 2. 查看当前供货商价格，选择最低
                    t0 = time.time()
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test]
                    )
                    t_quotes = time.time() - t0

                    env.log_debug("供应商报价:" + quotes_resp['formatted'])
                    supplier_quotes = quotes_resp.get("result", {}).get(sku_id_for_test, [])
                    current_supplier = supplier_choice_by_sku.get(sku_id_for_test)
                    chosen_supplier = None
                    chosen_supplier_price = None

                    def pick_supplier(quotes: List[Dict[str, Any]], prefer: Optional[str] = None):
                        if prefer:
                            preferred = next((q for q in quotes if q.get("supplier_id") == prefer), None)
                            if preferred:
                                return preferred.get("supplier_id"), preferred.get("price")
                        if not quotes:
                            return None, None
                        cheapest = min(quotes, key=lambda x: x.get("price", float("inf")))
                        return cheapest.get("supplier_id"), cheapest.get("price")

                    chosen_supplier, chosen_supplier_price = pick_supplier(
                        supplier_quotes,
                        prefer=current_supplier,
                    )

                    if chosen_supplier_price is None:
                        import pdb; pdb.set_trace()
                    if chosen_supplier:
                        supplier_choice_by_sku[sku_id_for_test] = chosen_supplier
                        env.log_debug(f"选择供应商 {chosen_supplier}，报价 {chosen_supplier_price}")
                    else:
                        env.log_debug("未找到可用供应商报价。")

                    # 3. 基于历史销售选择最优价格（总收入最大）
                    t0 = time.time()
                    revenue_by_price: Dict[float, List[int]] = {}
                    moves_by_price: Dict[float, List[int]] = {}
                    # 若暂无供给商报价，回退使用销售价作为成本避免 None
                    cost_price = chosen_supplier_price if chosen_supplier_price is not None else None

                    for day_records in sales_records.values():
                        for rec in day_records:
                            price = float(rec.get("price", 0))
                            move = int(rec.get("move", 0))
                            effective_cost = cost_price if cost_price is not None else price
                            revenue_by_price.setdefault(price, []).append((price - effective_cost) * move)
                            moves_by_price.setdefault(price, []).append(move)

                    best_price = None
                    expected_move = 0
                    if revenue_by_price:
                        best_price = max(revenue_by_price.items(), key=lambda x: sum(x[1]) / len(x[1]))[0]
                        moves = moves_by_price.get(best_price, [])
                        expected_move = sum(moves) / len(moves) if moves else None
                        env.log_debug(f"历史最优价格 {best_price}，预期销量 {expected_move}")

                    # 4. 修改价格
                    t_price_block = time.time() - t0
                    t0 = time.time()
                    if best_price is not None:
                        res = env.exec_tools("modify_sku_price", sku_id=sku_id_for_test, new_price=best_price)
                        env.log_debug(f"修改价格结果: {res['formatted']}")
                    t_price_change = time.time() - t0

                    # 5. 库存为空则下单，数量为预测销量的 4 倍
                    t0 = time.time()
                    if current_quantity <= expected_move * 4:
                        open_orders_info = env.exec_tools("view_inventory")
                        open_orders = open_orders_info.get("result", {}).get("pending_orders", []) or []
                        env.log_debug("当前未完成订单：" + open_orders_info['formatted'])
                        has_pending = any(
                            any(item.get("sku_id") == sku_id_for_test for item in (order or {}).get("items", []))
                            for order in open_orders
                        )
                        if has_pending:
                            env.log_debug(f"已有未完成订单包含 {sku_id_for_test}，跳过下单。")
                            continue

                        predicted_sales = expected_move if expected_move is not None else 10
                        order_qty = max(1, int(predicted_sales * 4))
                        place_kwargs = {
                            "items": [{"sku_id": sku_id_for_test, "quantity": order_qty}],
                        }
                        if not chosen_supplier:
                            env.log_debug("未选择到供给商，无法下单，跳过。")
                        else:
                            place_kwargs["supplier_id"] = chosen_supplier
                            order_res = env.exec_tools("place_order", **place_kwargs)
                            env.log_debug("触发补货，下单结果：" + order_res['formatted'])
                    t_order_block = time.time() - t0

                    # 单个 SKU 的耗时拆分日志
                    sku_total = time.time() - sku_start_time
                    # 使用 markdown 表格结构化输出单个 SKU 的性能数据，便于在日志中阅读
                    perf_md = (
                        f"### [PERF] Day {day} - SKU `{sku_id_for_test}`\n"
                        f"| 阶段 | 耗时 |\n"
                        f"|---|---|\n"
                        f"| total | `{sku_total:.4f}s` |\n"
                        f"| sales_hist (view_sales_profit_history) | `{t_sales:.4f}s` |\n"
                        f"| quotes (view_current_date_supplier_prices) | `{t_quotes:.4f}s` |\n"
                        f"| pricing (Python 计算最优价格) | `{t_price_block:.4f}s` |\n"
                        f"| price_change (modify_sku_price) | `{t_price_change:.4f}s` |\n"
                        f"| order_logic (open_orders + place_order) | `{t_order_block:.4f}s` |"
                    )
                    env.log_info(perf_md)

                # 3. 推进一天
                end_time = time.time()
                env.log_info(
                    f"第 {day} 天决策逻辑耗时：{end_time - day_start_time:.4f} 秒（不含 end_today）"
                )
                step_res = env.exec_tools("end_today")
                step_payload = step_res.get("result", {})
                env.log_info("step 结果：")
                env.log_info(step_res.get("formatted", ""))
                if step_payload:
                    if step_payload.get("insufficient_skus"):
                        env.log_debug(f"  当日缺货 SKU：{step_payload['insufficient_skus']}")

                ratings = env.get_sku_rating_report()

                if ratings:
                    for rating, rating_item in ratings.items():
                        env.log_debug(
                            f"SKU: {rating}, InitialRating: {rating_item['initial_rating']}, AvgRating: {rating_item['avg_rating']}"
                        )
                
        # 4. 查看当前订单数量
        orders_info = env.exec_tools("view_inventory")
        open_orders = orders_info.get("result", {}).get("pending_order_count")
        env.log_debug(f"当前未完成订单数：{open_orders}")

        # 5. 日终：查看 record_manager 当日写入的供给商价格、新闻、订单
        # log_record_manager_snapshot(env.current_date - timedelta(days=1))

        env.log_debug("\n====== 模拟结束 ======")
        env.log_debug(f"最终资金 & 日期：{env.exec_tools('view_funds_and_date')}")
        ratings = env.get_sku_rating_report()
        
        final_inv = env.exec_tools("view_inventory")
        final_total_skus = final_inv.get("result", {}).get("total_skus")
        env.log_debug(f"最终总 SKU 数：{final_total_skus}")

        if enable_checkpoint_test:
            _run_checkpoint_consistency_test(
                env,
                label="Logic",
                checkpoint_prefix="simulate_logic_ckpt_",
                checkpoint_filename="logic_env.json",
            )
    finally:
        # 结束时清理容器
        pass


def simulate_simple_review_environment(
    days: int = 7,
    sample_size: int = 3,
    db_path: str = 'data/simulate_data/15/records_no_review/',
    config_type: str = "dynamic_hard",
    enable_checkpoint_test: bool = False,
):
    """
    初始化环境后，模拟正常经营若干天：
    目标逻辑：
    1. 第一次：查询历史数据中客户评价最好的供给商，作为每个 SKU 的"首选供给商"；
    2. 每隔 7 天：对当前有报价的每个供给商做一次小量试单，用于更新该供给商的评价认知；
    3. 常规/大量订货：基于当前对各供给商评价水平（平均评分）的认知，选择最优供给商进行大单补货。
    
    Args:
        days: 模拟天数
        sample_size: 每个品类选择的 SKU 数量
        db_path: 数据库路径
        config_type: 配置类型 ("dynamic_hard", "dynamic_middle", "still_hard", "still_middle")
        enable_checkpoint_test: 是否在模拟结束后执行 checkpoint 保存/恢复一致性测试
    """

    print(
        "========================simulate_simple_review_environment =========================="
    )

    config = get_config_by_name(config_type)
    
    env = RetailEnvironment(config)
    env = RetailEnvironment(config)

    # 记录每个 SKU 当前“认为最优”的供给商
    best_supplier_by_sku: Dict[str, Optional[str]] = {}

    # 探索相关参数
    exploration_interval = 1000        # 每隔多少天做一次小量试单
    bulk_qty_multiplier = 3         # 大量订货：预计销量的倍数
    
    def calculate_exploration_qty(sku_id: str) -> int:
        """
        根据历史30天的平均销售量计算探索订单数量。
        返回：历史30天平均销售量的1/5，至少为1。
        """
        start_date = env.current_date - timedelta(days=30)
        sales = env.record_manager.read_sku(sku_id, start_date, env.current_date)
        
        # 计算总销量
        total_sales = 0
        for day_records in sales.values():
            for rec in day_records:
                total_sales += rec.move
        
        # 计算平均日销量（30天）
        avg_daily_sales = total_sales / 30.0 if total_sales > 0 else 0
        
        # 探索数量 = 平均日销量的1/5，至少为1
        exploration_qty = max(1, int(avg_daily_sales / 5.0))
        
        return exploration_qty

    def parse_date_safe(raw: str) -> date:
        try:
            return datetime.strptime(raw, "%m/%d/%y").date()
        except Exception:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except Exception:
                return date.today()

    def avg_profit_for_sku(sku_id: str, start_dt: date, end_dt: date) -> float:
        sales = env.record_manager.read_sku(sku_id, start_dt, end_dt)
        all_records = [rec for recs in sales.values() for rec in recs]
        if not all_records:
            return float("-inf")
        price_rows = env.record_manager.read_supplier_prices(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
        )
        min_price_by_date: Dict[str, float] = {}
        for row in price_rows:
            d = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
            min_price_by_date[d] = min(min_price_by_date.get(d, row.price), row.price)

        total_profit = 0.0
        total_units = 0
        for rec in all_records:
            d = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            cost = min_price_by_date.get(d, rec.price)
            profit_per_unit = rec.price - cost
            total_profit += profit_per_unit * rec.move
            total_units += rec.move

        return total_profit

    def select_skus_per_category(n: int) -> List[str]:
        selected: List[str] = []
        begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
        start_dt = parse_date_safe(begin_raw)
        # 使用当前日期作为窗口上限
        end_dt = env.current_date

        for category, sku_objs in env.skus_category_map.items():
            scores = []
            for sku_obj in sku_objs:
                score = avg_profit_for_sku(sku_obj.sku_id, start_dt, end_dt)
                scores.append((score, sku_obj.sku_id))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [sku_id for _, sku_id in scores[:n] if _ != float("-inf")]
            if len(top) < n:  # 补足
                remaining = [s.sku_id for s in sku_objs if s.sku_id not in top]
                top.extend(remaining[: max(0, n - len(top))])
            selected.extend(top)
        return selected

    def compute_supplier_ratings(
        sku_id: str,
        start_dt: date,
        end_dt: date,
    ) -> Dict[str, float]:
        """
        计算某个 SKU 在一段时间内，不同供给商的加权平均评分。
        使用贝叶斯平均方法，根据评论数量进行加权，避免偶发评论造成的影响。
        公式：weighted_rating = (prior_mean * prior_count + actual_sum) / (prior_count + actual_count)
        返回：{supplier_id: weighted_avg_rating}
        """
        reviews = env.review_manager.get_reviews_in_range(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
            ratings=[1, 2, 3, 4, 5],
        )

        rating_sum: Dict[str, float] = {}
        rating_cnt: Dict[str, int] = {}
        for r in reviews:
            sid = getattr(r, "supplier_id", None)
            if not sid or str(sid).upper() == "UNKNOWN":
                continue
            score = float(getattr(r, "rating", 0) or 0)
            rating_sum[sid] = rating_sum.get(sid, 0.0) + score
            rating_cnt[sid] = rating_cnt.get(sid, 0) + 1

        # 计算所有供应商的平均评分作为先验平均值
        all_ratings = [float(getattr(r, "rating", 0) or 0) for r in reviews if getattr(r, "supplier_id", None) and str(getattr(r, "supplier_id", None)).upper() != "UNKNOWN"]
        prior_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 3.0  # 默认中位数
        prior_count = 5  # 先验评论数量，用于平滑
        
        # 使用贝叶斯平均方法计算加权评分
        # weighted_rating = (prior_mean * prior_count + actual_sum) / (prior_count + actual_count)
        # 评论数量越多，实际评分权重越大，避免偶发评论造成的影响
        avg_by_supplier: Dict[str, float] = {}
        for sid, s in rating_sum.items():
            actual_count = rating_cnt.get(sid, 0)
            if actual_count > 0:
                # 贝叶斯加权：评论数量越多，实际评分权重越大
                weighted_avg = (prior_mean * prior_count + s) / (prior_count + actual_count)
                avg_by_supplier[sid] = weighted_avg
        
        # 打印 rating_map (包含原始平均和加权平均)
        raw_avg = {sid: rating_sum[sid] / rating_cnt[sid] for sid in rating_sum.keys() if rating_cnt.get(sid, 0) > 0}
        env.log_debug(f"[compute_supplier_ratings] SKU {sku_id} ({start_dt} ~ {end_dt}):")
        env.log_debug(f"  - Raw avg ratings: {raw_avg}")
        env.log_debug(f"  - Weighted avg ratings (Bayesian, prior_count={prior_count}): {avg_by_supplier}")
        env.log_debug(f"  - Rating counts: {rating_cnt}")
        
        return avg_by_supplier

    try:
        # 按品类选择 SKU：每个品类挑选 sample_size 个平均收益最高的 SKU
        if env.skus_category_map:
            sampled_skus = select_skus_per_category(sample_size)
        else:
            sampled_skus = []

        env.log_debug(f"开始模拟 {days} 天经营")
        env.log_debug("初始资金 & 日期：" + env.view_funds_and_date()['formatted'])
        env.log_debug(f"测试用 SKU: {sampled_skus}\n")

        # ---------- 第一次：用历史评价为每个 SKU 选出"首选供给商" ----------
        if sampled_skus:
            begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
            history_start = parse_date_safe(begin_raw)
            history_end = env.current_date
            for sku_id in sampled_skus:
                rating_map = compute_supplier_ratings(sku_id, history_start, history_end)
                if rating_map:
                    best_sid = max(rating_map.items(), key=lambda x: x[1])[0]
                    best_supplier_by_sku[sku_id] = best_sid
                    env.log_debug(
                        f"[初始化首选供给商] SKU {sku_id} 首选 {best_sid}，历史平均评分 {rating_map[best_sid]:.2f}"
                    )

        consecutive_negative_days = 0  # 跟踪连续负资金天数
        
        for day in range(1, days + 1):
            start_time = time.time()
            
            # 检查资金状态
            if env.funds < 0:
                consecutive_negative_days += 1
                if consecutive_negative_days >= 10:
                    env.log_debug(f"连续 {consecutive_negative_days} 天资金为负数，模拟结束。")
                    break
            else:
                consecutive_negative_days = 0  # 重置计数器
            
            env.log_debug(f"\n====== 第 {day} 天 ======")

            # 0. 查看当前日期与资金
            funds_snapshot = env.view_funds_and_date()
            env.log_debug(f"资金与日期：{funds_snapshot['formatted']}")

            # 1. 查看当前库存
            inv_info = env.view_inventory()
            inv_result = inv_info.get("result", {})
            env.log_debug(f"当前库存: {inv_info['formatted']}")
            inventory_map = inv_result.get("inventory", {})
            if not sampled_skus:
                env.log_debug("没有找到可用的测试 SKU，跳过补货逻辑。")
            else:
                for sku_id_for_test in sampled_skus:
                    # 计算当前库存：包括已入库的商品和等待入库的商品
                    sku_inv_info = inventory_map.get(sku_id_for_test, {})
                    base_quantity = sku_inv_info.get("quantity", 0)
                    waiting_count = sku_inv_info.get("waiting", 0)
                    current_quantity = base_quantity + waiting_count
                    env.log_debug(f"SKU {sku_id_for_test} 当前库存：{current_quantity} (已入库: {base_quantity}, 等待入库: {waiting_count})")

                    # 1.1 查看历史销售数据（用于定价 & 需求估计）
                    sales_resp = env.exec_tools(
                        "view_sales_profit_history",
                        sku_ids=[sku_id_for_test],
                        start_date=(env.current_date - timedelta(days=60)).isoformat(),
                        end_date=env.current_date.isoformat(),
                    )

                    env.log_debug(f"历史销售数据: {sales_resp['formatted']}")
                    sales_records = sales_resp.get("result", {}).get(sku_id_for_test, {}).get("records", {})

                    # 1.2 查看当前所有供给商报价
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test],
                    )
                    env.log_debug(f"供应商报价: {quotes_resp['formatted']}")
                    supplier_quotes: List[Dict[str, Any]] = quotes_resp.get("result", {}).get(sku_id_for_test, []) or []

                    if not supplier_quotes:
                        env.log_debug("今日该 SKU 无供应商报价，跳过。")
                        continue

                    # 1.3 根据最近一段时间的评论，为各供给商打分
                    review_window_start = env.current_date - timedelta(days=60)
                    rating_map_recent = compute_supplier_ratings(
                        sku_id_for_test,
                        # review_window_start,
                        parse_date_safe(begin_raw),
                        parse_date_safe(begin_raw) + timedelta(days=90),
                        # env.current_date,
                    )

                    # 如果之前没有首选供给商，这里再尝试用“最近 60 天评论”初始化一次
                    if sku_id_for_test not in best_supplier_by_sku or not best_supplier_by_sku[sku_id_for_test]:
                        if rating_map_recent:
                            best_sid = max(rating_map_recent.items(), key=lambda x: x[1])[0]
                            best_supplier_by_sku[sku_id_for_test] = best_sid
                            env.log_debug(
                                f"[补充首选供给商] SKU {sku_id_for_test} 选择 {best_sid} 作为首选（最近 60 天平均评分 {rating_map_recent[best_sid]:.2f}）"
                            )

                    # 2. 每隔 exploration_interval 天，对所有供给商小量试单
                    if day > 1 and day % exploration_interval == 1:
                        # 根据历史30天平均销售量计算探索数量
                        exploration_qty = calculate_exploration_qty(sku_id_for_test)
                        env.log_debug(f"[探索] 第 {day} 天，对 SKU {sku_id_for_test} 所有供给商做小量试单（数量：{exploration_qty}，基于历史30天平均销售量的1/5）")
                        for quote in supplier_quotes:
                            sid = quote.get("supplier_id")
                            if not sid:
                                continue
                            try:
                                order_res = env.exec_tools(
                                    "place_order",
                                    items=[{"sku_id": sku_id_for_test, "quantity": exploration_qty}],
                                    supplier_id=sid,
                                )
                                env.log_debug(f"  探索下单：供给商 {sid}，数量 {exploration_qty}，结果：{order_res['formatted']}")
                            except Exception as e:
                                env.log_debug(f"  探索下单失败（{sid}）：{e}")

                    # 3. 基于“当前认知的供给商评价”决定大单的供给商
                    chosen_supplier: Optional[str] = None
                    chosen_supplier_price: Optional[float] = None

                    def cheapest_supplier(quotes: List[Dict[str, Any]]) -> tuple:
                        q = min(quotes, key=lambda x: x.get("price", float("inf")))
                        return q.get("supplier_id"), q.get("price")

                    if rating_map_recent:
                        # 有评论数据：按平均评分排序，评分高者优先，若没有价格信息则再按价格兜底
                        def sort_key(entry: Dict[str, Any]) -> tuple:
                            sid = entry.get("supplier_id")
                            rating = rating_map_recent.get(sid, 0.0)
                            return (-rating, entry.get("price", float("inf")))

                        best_entry = sorted(supplier_quotes, key=sort_key)[0]
                        chosen_supplier = best_entry.get("supplier_id")
                        chosen_supplier_price = best_entry.get("price")
                    else:
                        # 没有任何评论数据，则退化为历史初始化的首选供给商 / 最便宜供给商
                        sid_pref = best_supplier_by_sku.get(sku_id_for_test)
                        if sid_pref:
                            preferred = next(
                                (q for q in supplier_quotes if q.get("supplier_id") == sid_pref),
                                None,
                            )
                            if preferred:
                                chosen_supplier = preferred.get("supplier_id")
                                chosen_supplier_price = preferred.get("price")
                        if not chosen_supplier:
                            chosen_supplier, chosen_supplier_price = cheapest_supplier(supplier_quotes)

                    if chosen_supplier:
                        best_supplier_by_sku[sku_id_for_test] = chosen_supplier
                        env.log_debug(
                            f"[大单供给商选择] SKU {sku_id_for_test} 选 {chosen_supplier} 供货，单价 {chosen_supplier_price}"
                        )
                    else:
                        env.log_debug("未能选出大单供给商，跳过后续补货逻辑。")
                        continue

                    # 4. 基于历史销售选择最优售价（总利润最大）
                    revenue_by_price: Dict[float, List[float]] = {}
                    moves_by_price: Dict[float, List[int]] = {}
                    cost_price = chosen_supplier_price if chosen_supplier_price is not None else 0.0

                    for day_records in sales_records.values():
                        for rec in day_records:
                            price = float(rec.get("price", 0))
                            move = int(rec.get("move", 0))
                            profit = (price - cost_price) * move
                            revenue_by_price.setdefault(price, []).append(profit)
                            moves_by_price.setdefault(price, []).append(move)

                    best_price: Optional[float] = None
                    expected_move = 0.0
                    if revenue_by_price:
                        def avg_profit(item):
                            price, profits = item
                            if not profits:
                                return float("-inf")
                            return sum(profits) / len(profits)

                        best_price = max(revenue_by_price.items(), key=avg_profit)[0]
                        moves = moves_by_price.get(best_price, [])
                        expected_move = (sum(moves) / len(moves)) if moves else 0.0
                        env.log_debug(f"历史最优售价 {best_price}，基于历史预期日销量 {expected_move}")

                    # 5. 修改零售价
                    if best_price is not None and best_price > 0:
                        try:
                            res = env.exec_tools(
                                "modify_sku_price",
                                sku_id=sku_id_for_test,
                                new_price=best_price,
                            )
                            env.log_debug(f"修改价格结果: {res['formatted']}")
                        except Exception as e:
                            env.log_debug(f"修改价格失败: {e}")

                    # 6. 库存不足时触发大单补货：数量为预期销量的 bulk_qty_multiplier 倍
                    target_move = expected_move if expected_move > 0 else 10
                    reorder_threshold = target_move * bulk_qty_multiplier
                    if current_quantity <= reorder_threshold:
                        open_orders_info = env.exec_tools("view_inventory")
                        open_orders = open_orders_info.get("result", {}).get("pending_orders", []) or []
                        env.log_debug(f"当前未完成订单：{open_orders_info['formatted']}")
                        
                        # 计算未完成订单中该 SKU 的总数量（而不是简单检查是否存在）
                        pending_qty = 0
                        for order in open_orders:
                            items = order.get("items", []) or []
                            for item in items:
                                if item.get("sku_id") == sku_id_for_test:
                                    pending_qty += item.get("count", item.get("quantity", 0))
                        
                        # 计算需要补货的数量
                        order_qty = max(1, int(target_move * bulk_qty_multiplier))
                        
                        # Check available capacity before placing the order
                        current_total_inventory = sum(len(items) for items in env.inventory.items_by_sku.values()) + len(env.inventory.waiting_items)
                        available_capacity = env.inventory.capacity - current_total_inventory if env.inventory.capacity is not None else float('inf')
                        
                        # Leave a buffer of 100 units
                        max_order_qty_considering_capacity = max(0, available_capacity - 100)
                        
                        if order_qty - pending_qty > max_order_qty_considering_capacity:
                            env.log_debug(
                                f"[库存容量限制] SKU {sku_id_for_test} 计划订货 {order_qty - pending_qty}，但可用容量仅 {max_order_qty_considering_capacity}，跳过或减少订货。"
                            )
                            order_qty = max(0, max_order_qty_considering_capacity + pending_qty)  # Adjust order_qty to fit capacity
                            if order_qty <= pending_qty:  # If adjusted order_qty is less than or equal to pending, no new order needed
                                env.log_debug(f"[库存容量限制] SKU {sku_id_for_test} 调整后无需新订货。")
                                continue
                        
                        # 如果未完成订单的数量已经足够（大于等于需要补货的数量），则跳过
                        # 否则，即使有试单等小量订单，仍然可以下大单补货
                        if pending_qty >= order_qty:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，数量 {pending_qty} >= 需要补货数量 {order_qty}，跳过大单补货。"
                            )
                            continue
                        elif pending_qty > 0:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，但数量 {pending_qty} < 需要补货数量 {order_qty}，继续补货。"
                            )
                        place_kwargs = {
                            "items": [{"sku_id": sku_id_for_test, "quantity": order_qty - pending_qty}],
                            "supplier_id": chosen_supplier,
                        }

                        try:
                            order_res = env.exec_tools("place_order", **place_kwargs)
                            env.log_debug(f"[大单补货] 触发补货，下单结果：{order_res['formatted']}")
                        except Exception as e:
                            env.log_debug(f"[大单补货] 下单失败：{e}")

            # 3. 推进一天
            step_res = env.exec_tools("end_today")
            step_payload = step_res.get("result", {})
            env.log_debug("step 结果：")
            env.log_debug(step_res.get("formatted", ""))
            if step_payload:
                if step_payload.get("insufficient_skus"):
                    env.log_debug(f"  当日缺货 SKU：{step_payload['insufficient_skus']}")
                
                # 统计 waiting_items（直接从 inventory 获取，避免序列化问题）
                # 注意：step_payload 中的 waiting_items 已被 _json_safe 序列化为字符串，无法访问 sku 属性
                # 因此直接使用 env.inventory.waiting_items，与 _format_step_result 方法保持一致
                waiting_items = env.inventory.waiting_items
                if waiting_items:
                    waiting_by_sku = defaultdict(int)
                    for merch in waiting_items:
                        waiting_by_sku[merch.sku.sku_id] += 1
                    total_waiting = len(waiting_items)
                    env.log_debug(f"  [库存统计] 等待入库商品总数: {total_waiting}")
                    if waiting_by_sku:
                        waiting_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(waiting_by_sku.items()))
                        env.log_debug(f"  [库存统计] 等待入库商品按SKU分布: {waiting_lines}")
                else:
                    env.log_debug(f"  [库存统计] 等待入库商品总数: 0")
                
                # 统计 expired_discount_by_sku
                expired_by_sku = step_payload.get("expired_discount_by_sku", {})
                if expired_by_sku:
                    total_expired = sum(expired_by_sku.values())
                    env.log_debug(f"  [库存统计] 过期清仓商品总数: {total_expired}")
                    expired_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(expired_by_sku.items()))
                    env.log_debug(f"  [库存统计] 过期清仓商品按SKU分布: {expired_lines}")
                else:
                    env.log_debug(f"  [库存统计] 过期清仓商品总数: 0")
                
                # 统计 returns_by_sku
                returns_by_sku = step_payload.get("returns_by_sku", {})
                if returns_by_sku:
                    total_returns = sum(returns_by_sku.values())
                    env.log_debug(f"  [库存统计] 退货商品总数: {total_returns}")
                    return_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(returns_by_sku.items()))
                    env.log_debug(f"  [库存统计] 退货商品按SKU分布: {return_lines}")
                else:
                    env.log_debug(f"  [库存统计] 退货商品总数: 0")

            ratings = env.get_sku_rating_report()

            if ratings:
                for rating, rating_item in ratings.items():
                    env.log_debug(
                        f"SKU: {rating}, InitialRating: {rating_item['initial_rating']}, AvgRating: {rating_item['avg_rating']}"
                    )

            # 4. 查看当前订单数量
            orders_info = env.exec_tools("view_inventory")
            open_orders = orders_info.get("result", {}).get("pending_order_count")
            env.log_debug(f"当前未完成订单数：{open_orders}")

            end_time = time.time()
            env.log_info(
                f"第 {day} 天模拟结束，耗时 {end_time - start_time:.4f} 秒"
            )

        env.log_debug("\n====== 模拟结束 ======")
        env.log_debug("最终资金 & 日期：" + env.exec_tools("view_funds_and_date")['formatted'])
        ratings = env.get_sku_rating_report()
        
        final_inv = env.exec_tools("view_inventory")
        final_total_skus = final_inv.get("result", {}).get("total_skus")
        env.log_debug(f"最终总 SKU 数：{final_total_skus}")

        if enable_checkpoint_test:
            _run_checkpoint_consistency_test(
                env,
                label="Review",
                checkpoint_prefix="simulate_review_ckpt_",
                checkpoint_filename="review_env.json",
            )
    finally:
        # 结束时清理容器
        pass


def simulate_news_aware_environment(
    days: int = 7,
    sample_size: int = 3,
    db_path: str = 'data/simulate_data/15/records_no_review/',
    config_type: str = "dynamic_hard",
    enable_checkpoint_test: bool = False,
):
    """
    基于 simulate_simple_review_environment，但加入新闻影响的决策优化：
    1. 启用新闻模块，从新闻数据中获取对各 SKU 的影响程度
    2. 在计算预期销量时，结合新闻影响调整需求预测
    3. 在补货决策时，使用新闻调整后的预期销量来优化订货量
    4. 保持原有的供给商选择和评价逻辑
    
    Args:
        days: 模拟天数
        sample_size: 每个品类选择的 SKU 数量
        db_path: 数据库路径
        config_type: 配置类型 ("dynamic_hard", "dynamic_middle", "still_hard", "still_middle")
        enable_checkpoint_test: 是否在模拟结束后执行 checkpoint 保存/恢复一致性测试
    """

    print(
        "========================simulate_news_aware_environment =========================="
    )
    
    config = get_config_by_name(config_type)
    
    # 设置 order_record_dir
    config["order_record_dir"] = db_path if db_path else 'model_run_time'
    env = RetailEnvironment(config)

    # 记录每个 SKU 当前"认为最优"的供给商
    best_supplier_by_sku: Dict[str, Optional[str]] = {}

    # 探索相关参数
    exploration_interval = 7        # 每隔多少天做一次小量试单
    bulk_qty_multiplier = 4       # 大量订货：预计销量的倍数
    
    # 追踪新闻影响倍数：用于统计整个模拟过程中的比例
    # 结构：{day: {sku_id: {"need_mult": float, "supply_mult": float, "ratio": float}}}
    multiplier_tracking: Dict[int, Dict[str, Dict[str, float]]] = {}
    
    def calculate_exploration_qty(sku_id: str) -> int:
        """
        根据历史30天的平均销售量计算探索订单数量。
        返回：历史30天平均销售量的1/5，至少为1。
        """
        start_date = env.current_date - timedelta(days=30)
        sales = env.record_manager.read_sku(sku_id, start_date, env.current_date)
        
        # 计算总销量
        total_sales = 0
        for day_records in sales.values():
            for rec in day_records:
                total_sales += rec.move
        
        # 计算平均日销量（30天）
        avg_daily_sales = total_sales / 30.0 if total_sales > 0 else 0
        
        # 探索数量 = 平均日销量的1/5，至少为1
        exploration_qty = max(1, int(avg_daily_sales / 5.0))
        
        return exploration_qty

    def parse_date_safe(raw: str) -> date:
        try:
            return datetime.strptime(raw, "%m/%d/%y").date()
        except Exception:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except Exception:
                return date.today()

    def avg_profit_for_sku(sku_id: str, start_dt: date, end_dt: date) -> float:
        sales = env.record_manager.read_sku(sku_id, start_dt, end_dt)
        all_records = [rec for recs in sales.values() for rec in recs]
        if not all_records:
            return float("-inf")
        price_rows = env.record_manager.read_supplier_prices(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
        )
        min_price_by_date: Dict[str, float] = {}
        for row in price_rows:
            d = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
            min_price_by_date[d] = min(min_price_by_date.get(d, row.price), row.price)

        total_profit = 0.0
        total_units = 0
        for rec in all_records:
            d = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            cost = min_price_by_date.get(d, rec.price)
            profit_per_unit = rec.price - cost
            total_profit += profit_per_unit * rec.move
            total_units += rec.move

        return total_profit

    def select_skus_per_category(n: int) -> List[str]:
        selected: List[str] = []
        begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
        start_dt = parse_date_safe(begin_raw)
        # 使用当前日期作为窗口上限
        end_dt = env.current_date

        for category, sku_objs in env.skus_category_map.items():
            scores = []
            for sku_obj in sku_objs:
                score = avg_profit_for_sku(sku_obj.sku_id, start_dt, end_dt)
                scores.append((score, sku_obj.sku_id))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [sku_id for _, sku_id in scores[:n] if _ != float("-inf")]
            if len(top) < n:  # 补足
                remaining = [s.sku_id for s in sku_objs if s.sku_id not in top]
                top.extend(remaining[: max(0, n - len(top))])
            selected.extend(top)
        return selected

    def compute_supplier_ratings(
        sku_id: str,
        start_dt: date,
        end_dt: date,
    ) -> Dict[str, float]:
        """
        计算某个 SKU 在一段时间内，不同供给商的加权平均评分。
        使用贝叶斯平均方法，根据评论数量进行加权，避免偶发评论造成的影响。
        公式：weighted_rating = (prior_mean * prior_count + actual_sum) / (prior_count + actual_count)
        返回：{supplier_id: weighted_avg_rating}
        """
        reviews = env.review_manager.get_reviews_in_range(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
            ratings=[1, 2, 3, 4, 5],
        )

        rating_sum: Dict[str, float] = {}
        rating_cnt: Dict[str, int] = {}
        for r in reviews:
            sid = getattr(r, "supplier_id", None)
            if not sid or str(sid).upper() == "UNKNOWN":
                continue
            score = float(getattr(r, "rating", 0) or 0)
            rating_sum[sid] = rating_sum.get(sid, 0.0) + score
            rating_cnt[sid] = rating_cnt.get(sid, 0) + 1

        # 计算所有供应商的平均评分作为先验平均值
        all_ratings = [float(getattr(r, "rating", 0) or 0) for r in reviews if getattr(r, "supplier_id", None) and str(getattr(r, "supplier_id", None)).upper() != "UNKNOWN"]
        prior_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 3.0  # 默认中位数
        prior_count = 5  # 先验评论数量，用于平滑
        
        # 使用贝叶斯平均方法计算加权评分
        # weighted_rating = (prior_mean * prior_count + actual_sum) / (prior_count + actual_count)
        # 评论数量越多，实际评分权重越大，避免偶发评论造成的影响
        avg_by_supplier: Dict[str, float] = {}
        for sid, s in rating_sum.items():
            actual_count = rating_cnt.get(sid, 0)
            if actual_count > 0:
                # 贝叶斯加权：评论数量越多，实际评分权重越大
                weighted_avg = (prior_mean * prior_count + s) / (prior_count + actual_count)
                avg_by_supplier[sid] = weighted_avg
        
        # 打印 rating_map (包含原始平均和加权平均)
        raw_avg = {sid: rating_sum[sid] / rating_cnt[sid] for sid in rating_sum.keys() if rating_cnt.get(sid, 0) > 0}
        env.log_debug(f"[compute_supplier_ratings] SKU {sku_id} ({start_dt} ~ {end_dt}):")
        env.log_debug(f"  - Raw avg ratings: {raw_avg}")
        env.log_debug(f"  - Weighted avg ratings (Bayesian, prior_count={prior_count}): {avg_by_supplier}")
        env.log_debug(f"  - Rating counts: {rating_cnt}")
        
        return avg_by_supplier

    try:
        # 按品类选择 SKU：每个品类挑选 sample_size 个平均收益最高的 SKU
        if env.skus_category_map:
            sampled_skus = select_skus_per_category(sample_size)
        else:
            sampled_skus = []

        env.log_debug(f"开始模拟 {days} 天经营（新闻感知版本）")
        env.log_debug("初始资金 & 日期：" + env.view_funds_and_date()['formatted'])
        env.log_debug(f"测试用 SKU: {sampled_skus}\n")

        # ---------- 第一次：用历史评价为每个 SKU 选出"首选供给商" ----------
        if sampled_skus:
            begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
            history_start = parse_date_safe(begin_raw)
            history_end = env.current_date
            for sku_id in sampled_skus:
                rating_map = compute_supplier_ratings(sku_id, history_start, history_end)
                if rating_map:
                    best_sid = max(rating_map.items(), key=lambda x: x[1])[0]
                    best_supplier_by_sku[sku_id] = best_sid
                    env.log_debug(
                        f"[初始化首选供给商] SKU {sku_id} 首选 {best_sid}，历史平均评分 {rating_map[best_sid]:.2f}"
                    )

        consecutive_negative_days = 0  # 跟踪连续负资金天数
        
        for day in range(1, days + 1):
            start_time = time.time()
            
            # 检查资金状态
            if env.funds < 0:
                consecutive_negative_days += 1
                if consecutive_negative_days >= 10:
                    env.log_debug(f"连续 {consecutive_negative_days} 天资金为负数，模拟结束。")
                    break
            else:
                consecutive_negative_days = 0  # 重置计数器
            
            env.log_debug(f"\n====== 第 {day} 天 ======")

            # 0. 查看当前日期与资金
            funds_snapshot = env.view_funds_and_date()
            env.log_debug(f"资金与日期：{funds_snapshot['formatted']}")

            # 0.0 生成并展示今日新闻（如果启用）
            if env.config.get("enable_new") and env.news_manager:
                try:
                    today_news = env.exec_tools("view_today_news")
                    env.log_debug("今日新闻：")
                    env.log_debug(today_news.get("formatted", ""))
                except Exception as e:
                    env.log_debug(f"今日新闻获取失败: {e}")

            # 1. 查看当前库存
            inv_info = env.view_inventory()
            inv_result = inv_info.get("result", {})
            env.log_debug(f"当前库存: {inv_info['formatted']}")
            inventory_map = inv_result.get("inventory", {})
            if not sampled_skus:
                env.log_debug("没有找到可用的测试 SKU，跳过补货逻辑。")
            else:
                for sku_id_for_test in sampled_skus:
                    # 计算当前库存：包括已入库的商品和等待入库的商品
                    sku_inv_info = inventory_map.get(sku_id_for_test, {})
                    base_quantity = sku_inv_info.get("quantity", 0)
                    waiting_count = sku_inv_info.get("waiting", 0)
                    current_quantity = base_quantity + waiting_count
                    env.log_debug(f"SKU {sku_id_for_test} 当前库存：{current_quantity} (已入库: {base_quantity}, 等待入库: {waiting_count})")

                    # 1.1 查看历史销售数据（用于定价 & 需求估计）
                    sales_resp = env.exec_tools(
                        "view_sales_profit_history",
                        sku_ids=[sku_id_for_test],
                        start_date=(env.current_date - timedelta(days=30)).isoformat(),
                        end_date=env.current_date.isoformat(),
                    )

                    env.log_debug(f"历史销售数据: {sales_resp['formatted']}")
                    sales_records = sales_resp.get("result", {}).get(sku_id_for_test, {}).get("records", {})

                    # 1.2 查看当前所有供给商报价
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test],
                    )
                    env.log_debug(f"供应商报价: {quotes_resp['formatted']}")
                    supplier_quotes: List[Dict[str, Any]] = quotes_resp.get("result", {}).get(sku_id_for_test, []) or []

                    if not supplier_quotes:
                        env.log_debug("今日该 SKU 无供应商报价，跳过。")
                        continue

                    # 1.3 根据最近一段时间的评论，为各供给商打分
                    review_window_start = env.current_date - timedelta(days=60)
                    rating_map_recent = compute_supplier_ratings(
                        sku_id_for_test,
                        review_window_start,
                        env.current_date,
                    )

                    # 如果之前没有首选供给商，这里再尝试用"最近 60 天评论"初始化一次
                    if sku_id_for_test not in best_supplier_by_sku or not best_supplier_by_sku[sku_id_for_test]:
                        if rating_map_recent:
                            best_sid = max(rating_map_recent.items(), key=lambda x: x[1])[0]
                            best_supplier_by_sku[sku_id_for_test] = best_sid
                            env.log_debug(
                                f"[补充首选供给商] SKU {sku_id_for_test} 选择 {best_sid} 作为首选（最近 60 天平均评分 {rating_map_recent[best_sid]:.2f}）"
                            )

                    # 2. 每隔 exploration_interval 天，对所有供给商小量试单
                    if day % exploration_interval == 1:
                        # 根据历史30天平均销售量计算探索数量
                        exploration_qty = calculate_exploration_qty(sku_id_for_test)
                        env.log_debug(f"[探索] 第 {day} 天，对 SKU {sku_id_for_test} 所有供给商做小量试单（数量：{exploration_qty}，基于历史30天平均销售量的1/5）")
                        for quote in supplier_quotes:
                            sid = quote.get("supplier_id")
                            if not sid:
                                continue
                            try:
                                order_res = env.exec_tools(
                                    "place_order",
                                    items=[{"sku_id": sku_id_for_test, "quantity": exploration_qty}],
                                    supplier_id=sid,
                                )
                                env.log_debug(f"  探索下单：供给商 {sid}，数量 {exploration_qty}，结果：{order_res['formatted']}")
                            except Exception as e:
                                env.log_debug(f"  探索下单失败（{sid}）：{e}")

                    # 3. 基于"当前认知的供给商评价"决定大单的供给商
                    chosen_supplier: Optional[str] = None
                    chosen_supplier_price: Optional[float] = None

                    def cheapest_supplier(quotes: List[Dict[str, Any]]) -> tuple:
                        q = min(quotes, key=lambda x: x.get("price", float("inf")))
                        return q.get("supplier_id"), q.get("price")

                    if rating_map_recent:
                        # 有评论数据：按平均评分排序，评分高者优先，若没有价格信息则再按价格兜底
                        def sort_key(entry: Dict[str, Any]) -> tuple:
                            sid = entry.get("supplier_id")
                            rating = rating_map_recent.get(sid, 0.0)
                            return (-rating, entry.get("price", float("inf")))

                        best_entry = sorted(supplier_quotes, key=sort_key)[0]
                        chosen_supplier = best_entry.get("supplier_id")
                        chosen_supplier_price = best_entry.get("price")
                    else:
                        # 没有任何评论数据，则退化为历史初始化的首选供给商 / 最便宜供给商
                        sid_pref = best_supplier_by_sku.get(sku_id_for_test)
                        if sid_pref:
                            preferred = next(
                                (q for q in supplier_quotes if q.get("supplier_id") == sid_pref),
                                None,
                            )
                            if preferred:
                                chosen_supplier = preferred.get("supplier_id")
                                chosen_supplier_price = preferred.get("price")
                        if not chosen_supplier:
                            chosen_supplier, chosen_supplier_price = cheapest_supplier(supplier_quotes)

                    if chosen_supplier:
                        best_supplier_by_sku[sku_id_for_test] = chosen_supplier
                        env.log_debug(
                            f"[大单供给商选择] SKU {sku_id_for_test} 选 {chosen_supplier} 供货，单价 {chosen_supplier_price}"
                        )
                    else:
                        env.log_debug("未能选出大单供给商，跳过后续补货逻辑。")
                        continue

                    # 4. 基于历史销售选择最优售价（总利润最大）
                    revenue_by_price: Dict[float, List[float]] = {}
                    moves_by_price: Dict[float, List[int]] = {}
                    cost_price = chosen_supplier_price if chosen_supplier_price is not None else 0.0

                    for day_records in sales_records.values():
                        for rec in day_records:
                            price = float(rec.get("price", 0))
                            move = int(rec.get("move", 0))
                            profit = (price - cost_price) * move
                            revenue_by_price.setdefault(price, []).append(profit)
                            moves_by_price.setdefault(price, []).append(move)

                    best_price: Optional[float] = None
                    expected_move = 0.0
                    if revenue_by_price:
                        def avg_profit(item):
                            price, profits = item
                            if not profits:
                                return float("-inf")
                            return sum(profits) / len(profits)

                        best_price = max(revenue_by_price.items(), key=avg_profit)[0]
                        moves = moves_by_price.get(best_price, [])
                        expected_move = (sum(moves) / len(moves)) if moves else 0.0
                        env.log_debug(f"历史最优售价 {best_price}，基于历史预期日销量 {expected_move}")

                    # 5. 结合新闻影响 & 供给变化，调整对未来需求和订货策略的预估
                    need_effect = 0.0
                    supply_effect = 0.0
                    if env.config.get("enable_new") and getattr(env, "news_manager", None):
                        try:
                            sku_obj = env.skus_id_map.get(sku_id_for_test)
                            sku_category = getattr(sku_obj, "category", None) if sku_obj else None

                            # 5.1 需求侧：impact_factor = "need"
                            need_info = env.news_manager.evaluate_impact_for_sku(
                                sku_id=sku_id_for_test,
                                sku_category=sku_category,
                                impact_factors=["need"],
                            )
                            need_effect = float(need_info.get("total_effect", 0.0) or 0.0)

                            # 5.2 供给侧：impact_factor = "supply"
                            supply_info = env.news_manager.evaluate_impact_for_sku(
                                sku_id=sku_id_for_test,
                                sku_category=sku_category,
                                impact_factors=["supply"],
                            )
                            supply_effect = float(supply_info.get("total_effect", 0.0) or 0.0)

                            matched_need = need_info.get("matched_news", []) or []
                            matched_supply = supply_info.get("matched_news", []) or []
                            if matched_need or matched_supply:
                                env.log_debug(
                                    f"[NewsImpact] SKU {sku_id_for_test} need_news={len(matched_need)}, "
                                    f"supply_news={len(matched_supply)}, "
                                    f"need_effect={need_effect:.4f}, supply_effect={supply_effect:.4f}"
                                )
                        except Exception as e:
                            env.log_debug(f"[NewsImpact] 计算 SKU {sku_id_for_test} 新闻影响失败: {e}")

                    # 基于历史销量 + 新闻影响得到"有效预期销量"：
                    # - expected_move 为历史均值
                    # - need_effect 为需求侧的相对提升/降低系数（已按 config 加权）
                    # - supply_effect 为供给侧的影响：> 0 表示供给条件变好（更容易/更便宜拿货），应增加订货量
                    base_expected = expected_move if expected_move > 0 else 10.0
                    # 需求侧倍数：news_effect 可能是正负值，例如 0.1 表示需求提升 10%
                    # 限制在合理范围内：0.5 ~ 2.0，避免过度放大或缩小
                    need_multiplier = 1.0 + need_effect
                    # 供给侧倍数：supply_effect > 0 代表供给条件变好（更容易/更便宜拿货），应增加订货量
                    # 例如 supply_effect = 0.2 -> multiplier ≈ 1.2，表示可以多进 20% 的货
                    # 限制在合理范围内：0.5 ~ 2.0
                    supply_multiplier = 1.0 + supply_effect
                    # 总倍数 = 需求侧倍数 * 供给侧倍数，但限制在合理范围内：0.5 ~ 3.0
                    # 避免过度放大（移除之前硬编码的最小值 2）
                    total_multiplier = max(0.5, min(2.0, need_multiplier * supply_multiplier))
                    effective_expected = base_expected * total_multiplier
                    
                    # 计算比例：need_mult / supply_mult（如果 supply_mult > 0）
                    ratio = need_multiplier / supply_multiplier if supply_multiplier > 0 else float('inf')
                    
                    # 记录到追踪数据结构中
                    if day not in multiplier_tracking:
                        multiplier_tracking[day] = {}
                    multiplier_tracking[day][sku_id_for_test] = {
                        "need_mult": need_multiplier,
                        "supply_mult": supply_multiplier,
                        "ratio": ratio,
                        "need_effect": need_effect,
                        "supply_effect": supply_effect,
                    }
                    
                    env.log_debug(
                        f"[NewsImpact] SKU {sku_id_for_test} base_expected={base_expected:.2f}, "
                        f"need_effect={need_effect:.4f}, need_mult={need_multiplier:.3f}, "
                        f"supply_effect={supply_effect:.4f}, supply_mult={supply_multiplier:.3f}, "
                        f"ratio(need/supply)={ratio:.3f}, total_mult={total_multiplier:.3f}, effective_expected={effective_expected:.2f}"
                    )

                    # 6. 修改零售价（可选：也可以根据新闻影响微调价格）
                    if best_price is not None and best_price > 0:
                        try:
                            res = env.exec_tools(
                                "modify_sku_price",
                                sku_id=sku_id_for_test,
                                new_price=best_price,
                            )
                            env.log_debug(f"修改价格结果: {res['formatted']}")
                        except Exception as e:
                            env.log_debug(f"修改价格失败: {e}")

                    # 7. 库存不足时触发大单补货：数量为"新闻修正后预期销量"的 bulk_qty_multiplier 倍
                    target_move = effective_expected
                    reorder_threshold = target_move * bulk_qty_multiplier
                    if current_quantity <= reorder_threshold:
                        open_orders_info = env.exec_tools("view_inventory")
                        open_orders = open_orders_info.get("result", {}).get("pending_orders", []) or []
                        env.log_debug(f"当前未完成订单：{open_orders_info['formatted']}")
                        
                        # 计算未完成订单中该 SKU 的总数量（而不是简单检查是否存在）
                        pending_qty = 0
                        for order in open_orders:
                            items = order.get("items", []) or []
                            for item in items:
                                if item.get("sku_id") == sku_id_for_test:
                                    pending_qty += item.get("count", item.get("quantity", 0))
                        
                        # 计算需要补货的数量（基于新闻修正后的预期销量）
                        order_qty = max(1, int(target_move * bulk_qty_multiplier))
                        
                        # Check available capacity before placing the order
                        current_total_inventory = sum(len(items) for items in env.inventory.items_by_sku.values()) + len(env.inventory.waiting_items)
                        available_capacity = env.inventory.capacity - current_total_inventory if env.inventory.capacity is not None else float('inf')
                        
                        # Leave a buffer of 100 units
                        max_order_qty_considering_capacity = max(0, available_capacity - 100)
                        
                        if order_qty - pending_qty > max_order_qty_considering_capacity:
                            env.log_debug(
                                f"[库存容量限制] SKU {sku_id_for_test} 计划订货 {order_qty - pending_qty}，但可用容量仅 {max_order_qty_considering_capacity}，跳过或减少订货。"
                            )
                            order_qty = max(0, max_order_qty_considering_capacity + pending_qty)  # Adjust order_qty to fit capacity
                            if order_qty <= pending_qty:  # If adjusted order_qty is less than or equal to pending, no new order needed
                                env.log_debug(f"[库存容量限制] SKU {sku_id_for_test} 调整后无需新订货。")
                                continue
                        
                        # 如果未完成订单的数量已经足够（大于等于需要补货的数量），则跳过
                        # 否则，即使有试单等小量订单，仍然可以下大单补货
                        if pending_qty >= order_qty:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，数量 {pending_qty} >= 需要补货数量 {order_qty}，跳过大单补货。"
                            )
                            continue
                        elif pending_qty > 0:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，但数量 {pending_qty} < 需要补货数量 {order_qty}，继续补货。"
                            )
                        place_kwargs = {
                            "items": [{"sku_id": sku_id_for_test, "quantity": order_qty - pending_qty}],
                            "supplier_id": chosen_supplier,
                        }

                        try:
                            order_res = env.exec_tools("place_order", **place_kwargs)
                            env.log_debug(
                                f"[大单补货-新闻感知] 触发补货，下单结果：{order_res['formatted']} "
                                f"(基于新闻修正后的预期销量: {effective_expected:.2f})"
                            )
                        except Exception as e:
                            env.log_debug(f"[大单补货] 下单失败：{e}")

            # 3. 推进一天
            step_res = env.exec_tools("end_today")
            step_payload = step_res.get("result", {})
            env.log_debug("step 结果：")
            env.log_debug(step_res.get("formatted", ""))
            if step_payload:
                if step_payload.get("insufficient_skus"):
                    env.log_debug(f"  当日缺货 SKU：{step_payload['insufficient_skus']}")
                
                # 统计 waiting_items（直接从 inventory 获取，避免序列化问题）
                # 注意：step_payload 中的 waiting_items 已被 _json_safe 序列化为字符串，无法访问 sku 属性
                # 因此直接使用 env.inventory.waiting_items，与 _format_step_result 方法保持一致
                waiting_items = env.inventory.waiting_items
                if waiting_items:
                    waiting_by_sku = defaultdict(int)
                    for merch in waiting_items:
                        waiting_by_sku[merch.sku.sku_id] += 1
                    total_waiting = len(waiting_items)
                    env.log_debug(f"  [库存统计] 等待入库商品总数: {total_waiting}")
                    if waiting_by_sku:
                        waiting_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(waiting_by_sku.items()))
                        env.log_debug(f"  [库存统计] 等待入库商品按SKU分布: {waiting_lines}")
                else:
                    env.log_debug(f"  [库存统计] 等待入库商品总数: 0")
                
                # 统计 expired_discount_by_sku
                expired_by_sku = step_payload.get("expired_discount_by_sku", {})
                if expired_by_sku:
                    total_expired = sum(expired_by_sku.values())
                    env.log_debug(f"  [库存统计] 过期清仓商品总数: {total_expired}")
                    expired_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(expired_by_sku.items()))
                    env.log_debug(f"  [库存统计] 过期清仓商品按SKU分布: {expired_lines}")
                else:
                    env.log_debug(f"  [库存统计] 过期清仓商品总数: 0")
                
                # 统计 returns_by_sku
                returns_by_sku = step_payload.get("returns_by_sku", {})
                if returns_by_sku:
                    total_returns = sum(returns_by_sku.values())
                    env.log_debug(f"  [库存统计] 退货商品总数: {total_returns}")
                    return_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(returns_by_sku.items()))
                    env.log_debug(f"  [库存统计] 退货商品按SKU分布: {return_lines}")
                else:
                    env.log_debug(f"  [库存统计] 退货商品总数: 0")

            ratings = env.get_sku_rating_report()

            if ratings:
                for rating, rating_item in ratings.items():
                    env.log_debug(
                        f"SKU: {rating}, InitialRating: {rating_item['initial_rating']}, AvgRating: {rating_item['avg_rating']}"
                    )

            # 4. 查看当前订单数量
            orders_info = env.exec_tools("view_inventory")
            open_orders = orders_info.get("result", {}).get("pending_order_count")
            env.log_debug(f"当前未完成订单数：{open_orders}")

            end_time = time.time()
            env.log_info(
                f"第 {day} 天模拟结束，耗时 {end_time - start_time:.4f} 秒"
            )

        env.log_debug("\n====== 模拟结束 ======")
        env.log_debug("最终资金 & 日期：" + env.exec_tools("view_funds_and_date")['formatted'])
        ratings = env.get_sku_rating_report()
        
        final_inv = env.exec_tools("view_inventory")
        final_total_skus = final_inv.get("result", {}).get("total_skus")
        env.log_debug(f"最终总 SKU 数：{final_total_skus}")
        
        # ========== 输出新闻影响倍数统计 ==========
        env.log_debug("\n" + "="*80)
        env.log_debug("新闻影响倍数统计（整个模拟过程）")
        env.log_debug("="*80)
        
        if multiplier_tracking:
            # 按 SKU 聚合统计
            sku_stats: Dict[str, Dict[str, List[float]]] = {}
            all_need_mults: List[float] = []
            all_supply_mults: List[float] = []
            all_ratios: List[float] = []
            
            for day, sku_data in multiplier_tracking.items():
                for sku_id, data in sku_data.items():
                    if sku_id not in sku_stats:
                        sku_stats[sku_id] = {
                            "need_mults": [],
                            "supply_mults": [],
                            "ratios": [],
                        }
                    sku_stats[sku_id]["need_mults"].append(data["need_mult"])
                    sku_stats[sku_id]["supply_mults"].append(data["supply_mult"])
                    if data["ratio"] != float('inf'):
                        sku_stats[sku_id]["ratios"].append(data["ratio"])
                        all_ratios.append(data["ratio"])
                    
                    all_need_mults.append(data["need_mult"])
                    all_supply_mults.append(data["supply_mult"])
            
            # 输出总体统计
            if all_need_mults and all_supply_mults:
                avg_need_mult = sum(all_need_mults) / len(all_need_mults)
                avg_supply_mult = sum(all_supply_mults) / len(all_supply_mults)
                avg_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0.0
                min_need_mult = min(all_need_mults)
                max_need_mult = max(all_need_mults)
                min_supply_mult = min(all_supply_mults)
                max_supply_mult = max(all_supply_mults)
                min_ratio = min(all_ratios) if all_ratios else 0.0
                max_ratio = max(all_ratios) if all_ratios else 0.0
                
                env.log_debug(f"\n【总体统计】（共 {len(all_need_mults)} 次计算）")
                env.log_debug(f"  need_multiplier: 平均={avg_need_mult:.4f}, 最小={min_need_mult:.4f}, 最大={max_need_mult:.4f}")
                env.log_debug(f"  supply_multiplier: 平均={avg_supply_mult:.4f}, 最小={min_supply_mult:.4f}, 最大={max_supply_mult:.4f}")
                env.log_debug(f"  ratio(need/supply): 平均={avg_ratio:.4f}, 最小={min_ratio:.4f}, 最大={max_ratio:.4f}")
                env.log_debug(f"  比例分布: need_mult 占主导(ratio>1.2)={sum(1 for r in all_ratios if r > 1.2)}, "
                            f"supply_mult 占主导(ratio<0.8)={sum(1 for r in all_ratios if r < 0.8)}, "
                            f"相对平衡(0.8<=ratio<=1.2)={sum(1 for r in all_ratios if 0.8 <= r <= 1.2)}")
            
            # 按 SKU 输出详细统计
            env.log_debug(f"\n【按 SKU 统计】（共 {len(sku_stats)} 个 SKU）")
            for sku_id in sorted(sku_stats.keys()):
                stats = sku_stats[sku_id]
                if stats["need_mults"]:
                    avg_need = sum(stats["need_mults"]) / len(stats["need_mults"])
                    avg_supply = sum(stats["supply_mults"]) / len(stats["supply_mults"])
                    avg_ratio_sku = sum(stats["ratios"]) / len(stats["ratios"]) if stats["ratios"] else 0.0
                    count = len(stats["need_mults"])
                    env.log_debug(f"  SKU {sku_id}: 计算次数={count}, "
                                f"need_mult_avg={avg_need:.4f}, supply_mult_avg={avg_supply:.4f}, "
                                f"ratio_avg={avg_ratio_sku:.4f}")
        else:
            env.log_debug("未收集到新闻影响倍数数据")
        
        env.log_debug("="*80 + "\n")

        if enable_checkpoint_test:
            _run_checkpoint_consistency_test(
                env,
                label="NewsAware",
                checkpoint_prefix="simulate_news_aware_ckpt_",
                checkpoint_filename="news_aware_env.json",
            )
    finally:
        # 结束时清理容器
        pass

def simulate_quality_based_environment(
    days: int = 7,
    sample_size: Optional[int] = None,
    db_path: str = 'data/simulate_data/15/records_no_review/',
    config_type: str = "dynamic_hard",
    bulk_qty_multiplier: Optional[int] = None,
    log_dir: Optional[str] = None,
    enable_checkpoint_test: bool = False,
):
    """
    基于 simulate_news_aware_environment，但每次订货时直接选择 quality_score 最高的供应商。
    
    主要特点：
    1. 每次订货时，获取所有供应商的 quality_score
    2. 选择 quality_score 最高的供应商进行订货
    3. 如果多个供应商 quality_score 相同，选择价格更低的作为 tie-breaker
    4. 保持新闻影响的需求预测逻辑
    
    Args:
        days: 模拟天数
        sample_size: 每个品类选择的 SKU 数量
        db_path: 数据库路径
        config_type: 配置类型 ("dynamic_hard", "dynamic_middle", "still_hard", "still_middle")
        log_dir: 可选日志目录；设置后写入 environment.log 和 tool_calls.jsonl
        enable_checkpoint_test: 保留统一 runner 参数；quality baseline 默认不执行 checkpoint 自测
    """
    _ = enable_checkpoint_test

    print(
        "========================simulate_quality_based_environment =========================="
    )
    # 根据 config_type 选择配置
    
    config = get_config_by_name(config_type)
    
    # 设置 order_record_dir
    config["order_record_dir"] = db_path if db_path else 'model_run_time'
    if log_dir:
        config["log_dir"] = log_dir
        config["tool_log_path"] = os.path.join(log_dir, "tool_calls.jsonl")
        config["env_log_filename"] = "environment.log"
    env = RetailEnvironment(config)

    # 订货相关参数：按配置使用当前验证过的 non-LLM 最佳参数。
    # easy: bulk=4, sample=6; middle: bulk=3, sample=4; hard: bulk=3, sample=5
    # hard_v2: bulk=6, sample=2 (180-day SKU2 search, final_networth=131,510.42)
    config_type_key = (config_type or "").lower()
    is_hard_v2 = "hard_v2" in config_type_key or config_type_key == "stress_hard"
    if sample_size is None:
        if is_hard_v2:
            sample_size = 2
        elif "hard" in config_type_key:
            sample_size = 5
        elif "middle" in config_type_key:
            sample_size = 4
        else:
            sample_size = 6
    if bulk_qty_multiplier is None:
        if is_hard_v2:
            bulk_qty_multiplier = 6
        elif "hard" in config_type_key:
            bulk_qty_multiplier = 3
        elif "middle" in config_type_key:
            bulk_qty_multiplier = 3
        else:
            bulk_qty_multiplier = 4
    
    # 追踪新闻影响倍数：用于统计整个模拟过程中的比例
    multiplier_tracking: Dict[int, Dict[str, Dict[str, float]]] = {}

    def parse_date_safe(raw: str) -> date:
        try:
            return datetime.strptime(raw, "%m/%d/%y").date()
        except Exception:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except Exception:
                return date.today()

    def avg_profit_for_sku(sku_id: str, start_dt: date, end_dt: date) -> float:
        sales = env.record_manager.read_sku(sku_id, start_dt, end_dt)
        all_records = [rec for recs in sales.values() for rec in recs]
        if not all_records:
            return float("-inf")
        price_rows = env.record_manager.read_supplier_prices(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
        )
        min_price_by_date: Dict[str, float] = {}
        for row in price_rows:
            d = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
            min_price_by_date[d] = min(min_price_by_date.get(d, row.price), row.price)

        total_profit = 0.0
        total_units = 0
        for rec in all_records:
            d = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            cost = min_price_by_date.get(d, rec.price)
            profit_per_unit = rec.price - cost
            total_profit += profit_per_unit * rec.move
            total_units += rec.move

        return total_profit

    def select_skus_per_category(n: int) -> List[str]:
        selected: List[str] = []
        begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
        start_dt = parse_date_safe(begin_raw)
        end_dt = env.current_date

        for category, sku_objs in env.skus_category_map.items():
            scores = []
            for sku_obj in sku_objs:
                score = avg_profit_for_sku(sku_obj.sku_id, start_dt, end_dt)
                scores.append((score, sku_obj.sku_id))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [sku_id for _, sku_id in scores[:n] if _ != float("-inf")]
            if len(top) < n:
                remaining = [s.sku_id for s in sku_objs if s.sku_id not in top]
                top.extend(remaining[: max(0, n - len(top))])
            selected.extend(top)
        return selected

    def select_supplier_by_quality_score(supplier_quotes: List[Dict[str, Any]], sku_id: str) -> tuple:
        """
        根据 quality_score 选择供应商。
        返回: (supplier_id, price, quality_score) 或 (None, None, None)
        """
        if not supplier_quotes:
            return None, None, None
        
        # 为每个供应商获取 quality_score
        candidates: List[Tuple[float, float, str]] = []  # (quality_score, price, supplier_id)
        
        for quote in supplier_quotes:
            supplier_id = quote.get("supplier_id")
            price = quote.get("price", float("inf"))
            
            if not supplier_id:
                continue
            
            # 获取该供应商的 quality_score
            quality_score = env.supplier_manager.get_quality_score(
                supplier_id=supplier_id,
                sku_id=sku_id,
                target_date=env.current_date,
            )
            
            # 如果没有 quality_score，使用 0.0 作为默认值
            if quality_score is None:
                quality_score = 0.0
            
            candidates.append((quality_score, price, supplier_id))
        
        if not candidates:
            return None, None, None
        
        # 按 quality_score 降序排序，如果 quality_score 相同则按价格升序排序
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        best_quality, best_price, best_supplier = candidates[0]
        
        env.log_debug(
            f"[质量优先供应商选择] SKU {sku_id} 选择供应商 {best_supplier}，"
            f"quality_score={best_quality:.4f}，价格={best_price:.4f}"
        )
        
        return best_supplier, best_price, best_quality

    try:
        # 按品类选择 SKU：每个品类挑选 sample_size 个平均收益最高的 SKU
        if env.skus_category_map:
            sampled_skus = select_skus_per_category(sample_size)
        else:
            sampled_skus = []
        sampled_skus = _apply_baseline_shelf_assortment(env, sampled_skus, "Quality")

        env.log_debug(f"开始模拟 {days} 天经营（质量优先版本）")
        env.log_debug("初始资金 & 日期：" + env.view_funds_and_date()['formatted'])
        env.log_debug(f"测试用 SKU: {sampled_skus}\n")

        consecutive_negative_days = 0  # 跟踪连续负资金天数
        total_sold = 0
        total_expired = 0
        total_returns = 0
        total_money_earned = 0.0
        total_ordered = 0
        run_days = 0

        for day in range(1, days + 1):
            start_time = time.time()

            # 检查资金状态
            if env.funds < 0:
                consecutive_negative_days += 1
                if consecutive_negative_days >= 10:
                    env.log_debug(f"连续 {consecutive_negative_days} 天资金为负数，模拟结束。")
                    break
            else:
                consecutive_negative_days = 0

            env.log_debug(f"\n====== 第 {day} 天 ======")

            # 0. 查看当前日期与资金
            funds_snapshot = env.view_funds_and_date()
            env.log_debug(f"资金与日期：{funds_snapshot['formatted']}")

            # 0.0 生成并展示今日新闻（如果启用）
            if env.config.get("enable_new") and env.news_manager:
                try:
                    today_news = env.exec_tools("view_today_news")
                    env.log_debug("今日新闻：")
                    env.log_debug(today_news.get("formatted", ""))
                except Exception as e:
                    env.log_debug(f"今日新闻获取失败: {e}")

            # 1. 查看当前库存
            inv_info = env.view_inventory()
            inv_result = inv_info.get("result", {})
            env.log_debug(f"当前库存: {inv_info['formatted']}")
            inventory_map = inv_result.get("inventory", {})
            if not sampled_skus:
                env.log_debug("没有找到可用的测试 SKU，跳过补货逻辑。")
            else:
                for sku_id_for_test in sampled_skus:
                    # 计算当前库存：包括已入库的商品和等待入库的商品
                    sku_inv_info = inventory_map.get(sku_id_for_test, {})
                    base_quantity = sku_inv_info.get("quantity", 0)
                    waiting_count = sku_inv_info.get("waiting", 0)
                    current_quantity = base_quantity + waiting_count
                    env.log_debug(f"SKU {sku_id_for_test} 当前库存：{current_quantity} (已入库: {base_quantity}, 等待入库: {waiting_count})")

                    # 1.1 查看历史销售数据（用于定价 & 需求估计）
                    sales_resp = env.exec_tools(
                        "view_sales_profit_history",
                        sku_ids=[sku_id_for_test],
                        start_date=(env.current_date - timedelta(days=30)).isoformat(),
                        end_date=env.current_date.isoformat(),
                    )

                    env.log_debug(f"历史销售数据: {sales_resp['formatted']}")
                    sales_records = sales_resp.get("result", {}).get(sku_id_for_test, {}).get("records", {})

                    # 1.2 查看当前所有供给商报价
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test],
                    )
                    env.log_debug(f"供应商报价: {quotes_resp['formatted']}")
                    supplier_quotes: List[Dict[str, Any]] = quotes_resp.get("result", {}).get(sku_id_for_test, []) or []

                    if not supplier_quotes:
                        env.log_debug("今日该 SKU 无供应商报价，跳过。")
                        continue

                    # 2. 基于 quality_score 选择供应商（核心修改点）
                    chosen_supplier, chosen_supplier_price, chosen_quality_score = select_supplier_by_quality_score(
                        supplier_quotes, sku_id_for_test
                    )

                    if not chosen_supplier:
                        env.log_debug("未能选出供应商，跳过后续补货逻辑。")
                        continue

                    # 4. 基于历史销售选择最优售价（总利润最大）
                    revenue_by_price: Dict[float, List[float]] = {}
                    moves_by_price: Dict[float, List[int]] = {}
                    cost_price = chosen_supplier_price if chosen_supplier_price is not None else 0.0

                    for day_records in sales_records.values():
                        for rec in day_records:
                            price = float(rec.get("price", 0))
                            move = int(rec.get("move", 0))
                            profit = (price - cost_price) * move
                            revenue_by_price.setdefault(price, []).append(profit)
                            moves_by_price.setdefault(price, []).append(move)

                    best_price: Optional[float] = None
                    expected_move = 0.0
                    if revenue_by_price:
                        def avg_profit(item):
                            price, profits = item
                            if not profits:
                                return float("-inf")
                            return sum(profits) / len(profits)

                        best_price = max(revenue_by_price.items(), key=avg_profit)[0]
                        moves = moves_by_price.get(best_price, [])
                        expected_move = (sum(moves) / len(moves)) if moves else 0.0
                        env.log_debug(f"历史最优售价 {best_price}，基于历史预期日销量 {expected_move}")

                    # 5. 结合新闻影响调整对未来需求的预估
                    need_effect = 0.0
                    supply_effect = 0.0
                    if env.config.get("enable_new") and getattr(env, "news_manager", None):
                        try:
                            sku_obj = env.skus_id_map.get(sku_id_for_test)
                            sku_category = getattr(sku_obj, "category", None) if sku_obj else None

                            need_info = env.news_manager.evaluate_impact_for_sku(
                                sku_id=sku_id_for_test,
                                sku_category=sku_category,
                                impact_factors=["need"],
                            )
                            need_effect = float(need_info.get("total_effect", 0.0) or 0.0)

                            supply_info = env.news_manager.evaluate_impact_for_sku(
                                sku_id=sku_id_for_test,
                                sku_category=sku_category,
                                impact_factors=["supply"],
                            )
                            supply_effect = float(supply_info.get("total_effect", 0.0) or 0.0)
                        except Exception as e:
                            env.log_debug(f"[NewsImpact] 计算 SKU {sku_id_for_test} 新闻影响失败: {e}")

                    base_expected = expected_move if expected_move > 0 else 10.0
                    need_multiplier = 1.0 + need_effect
                    supply_multiplier = 1.0 + supply_effect
                    total_multiplier = max(0.5, min(2.0, need_multiplier * supply_multiplier))
                    effective_expected = base_expected * total_multiplier
                    
                    if day not in multiplier_tracking:
                        multiplier_tracking[day] = {}
                    multiplier_tracking[day][sku_id_for_test] = {
                        "need_mult": need_multiplier,
                        "supply_mult": supply_multiplier,
                        "need_effect": need_effect,
                        "supply_effect": supply_effect,
                    }
                    
                    env.log_debug(
                        f"[NewsImpact] SKU {sku_id_for_test} base_expected={base_expected:.2f}, "
                        f"effective_expected={effective_expected:.2f}"
                    )

                    # 6. 修改零售价
                    if best_price is not None and best_price > 0:
                        try:
                            res = env.exec_tools(
                                "modify_sku_price",
                                sku_id=sku_id_for_test,
                                new_price=best_price,
                            )
                            env.log_debug(f"修改价格结果: {res['formatted']}")
                        except Exception as e:
                            env.log_debug(f"修改价格失败: {e}")

                    # 7. 库存不足时触发大单补货
                    target_move = effective_expected
                    reorder_threshold = target_move * bulk_qty_multiplier
                    if current_quantity <= reorder_threshold:
                        open_orders_info = env.exec_tools("view_inventory")
                        open_orders = open_orders_info.get("result", {}).get("pending_orders", []) or []
                        
                        pending_qty = 0
                        for order in open_orders:
                            items = order.get("items", []) or []
                            for item in items:
                                if item.get("sku_id") == sku_id_for_test:
                                    pending_qty += item.get("count", item.get("quantity", 0))
                        
                        order_qty = max(1, int(target_move * bulk_qty_multiplier))
                        
                        current_total_inventory = sum(len(items) for items in env.inventory.items_by_sku.values()) + len(env.inventory.waiting_items)
                        available_capacity = env.inventory.capacity - current_total_inventory if env.inventory.capacity is not None else float('inf')
                        max_order_qty_considering_capacity = max(0, available_capacity - 100)
                        
                        if order_qty - pending_qty > max_order_qty_considering_capacity:
                            env.log_debug(
                                f"[库存容量限制] SKU {sku_id_for_test} 计划订货 {order_qty - pending_qty}，但可用容量仅 {max_order_qty_considering_capacity}，跳过或减少订货。"
                            )
                            order_qty = max(0, max_order_qty_considering_capacity + pending_qty)
                            if order_qty <= pending_qty:
                                env.log_debug(f"[库存容量限制] SKU {sku_id_for_test} 调整后无需新订货。")
                                continue
                        
                        if pending_qty >= order_qty:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，数量 {pending_qty} >= 需要补货数量 {order_qty}，跳过大单补货。"
                            )
                            continue
                        elif pending_qty > 0:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，但数量 {pending_qty} < 需要补货数量 {order_qty}，继续补货。"
                            )
                        
                        place_kwargs = {
                            "items": [{"sku_id": sku_id_for_test, "quantity": order_qty - pending_qty}],
                            "supplier_id": chosen_supplier,
                        }

                        try:
                            order_res = env.exec_tools("place_order", **place_kwargs)
                            env.log_debug(
                                f"[大单补货-质量优先] 触发补货，供应商 {chosen_supplier} (quality_score={chosen_quality_score:.4f})，"
                                f"下单结果：{order_res['formatted']} "
                                f"(基于新闻修正后的预期销量: {effective_expected:.2f})"
                            )
                            total_ordered += order_qty - pending_qty
                        except Exception as e:
                            env.log_debug(f"[大单补货] 下单失败：{e}")

            # 3. 推进一天
            step_res = env.exec_tools("end_today")
            step_payload = step_res.get("result", {})
            env.log_debug("step 结果：")
            env.log_debug(step_res.get("formatted", ""))

            # 收集每日指标
            run_days += 1
            total_money_earned += float(step_payload.get("money_earned", 0))
            sales_by_sku = step_payload.get("sales_by_sku", {})
            if isinstance(sales_by_sku, dict):
                total_sold += sum(v for v in sales_by_sku.values() if isinstance(v, (int, float)))
            expired_by_sku = step_payload.get("expired_discount_by_sku", {})
            if isinstance(expired_by_sku, dict):
                total_expired += sum(v for v in expired_by_sku.values() if isinstance(v, (int, float)))
            returns_by_sku = step_payload.get("returns_by_sku", {})
            if isinstance(returns_by_sku, dict):
                total_returns += sum(v for v in returns_by_sku.values() if isinstance(v, (int, float)))

            end_time = time.time()
            env.log_info(
                f"第 {day} 天模拟结束，耗时 {end_time - start_time:.4f} 秒"
            )

        env.log_debug("\n====== 模拟结束 ======")
        funds_info = env.exec_tools('view_funds_and_date')
        env.log_debug("最终资金 & 日期：" + funds_info['formatted'])

        final_inv = env.exec_tools("view_inventory")
        final_total_skus = final_inv.get("result", {}).get("total_skus")
        env.log_debug(f"最终总 SKU 数：{final_total_skus}")

        # ------------ 输出汇总指标 ------------
        final_net_worth = env.inventory.compute_net_worth(env.current_date) + env.funds + sum([order.cost for order in env.order_manager.get_current_orders()])
        expired_ratio = total_expired / total_ordered if total_ordered > 0 else 0.0
        return_ratio = total_returns / total_sold if total_sold > 0 else 0.0
        avg_daily_sold = total_sold / run_days if run_days > 0 else 0
        avg_daily_profit = total_money_earned / run_days if run_days > 0 else 0
        print(f"\n{'='*60}")
        print(f"  Simulation Summary: Quality Based")
        print(f"  Config Type: {config_type}, bulk_qty_multiplier={bulk_qty_multiplier}")
        print(f"{'='*60}")
        print(f"  Run Days:              {run_days}")
        print(f"  Final Net Worth:       {final_net_worth:,.2f}")
        print(f"  Final Funds:           {env.funds:,.2f}")
        print(f"  Avg Daily Sold:        {avg_daily_sold:,.1f}")
        print(f"  Avg Daily Profit:      {avg_daily_profit:,.2f}")
        print(f"  Total Sold:            {total_sold:,}")
        print(f"  Total Ordered:         {total_ordered:,}")
        print(f"  Total Expired:         {total_expired:,}")
        print(f"  Total Returns:         {total_returns:,}")
        print(f"  Expired Ratio:         {expired_ratio:.4f}")
        print(f"  Return Ratio:          {return_ratio:.4f}")
        print(f"{'='*60}")

        return {
            "config_type": config_type,
            "bulk_qty_multiplier": bulk_qty_multiplier,
            "sample_size": sample_size,
            "run_days": run_days,
            "final_net_worth": round(final_net_worth, 2),
            "final_funds": round(float(env.funds), 2),
            "total_sold": total_sold,
            "total_ordered": total_ordered,
            "total_expired": total_expired,
            "total_returns": total_returns,
            "expired_ratio": round(expired_ratio, 4),
            "return_ratio": round(return_ratio, 4),
            "avg_daily_sold": round(avg_daily_sold, 2),
            "avg_daily_profit": round(avg_daily_profit, 2),
        }
    finally:
        # 结束时清理容器
        pass

# Environment mapping for reference
ENVIRONMENT_MAPPING = {
    "easy": {
        "function": simulate_quality_based_environment,
        "description": "Easy quality-based baseline",
        "config_types": ["easy"],
    },
    "middle": {
        "function": simulate_quality_based_environment,
        "description": "Middle quality-based baseline",
        "config_types": ["middle"],
    },
    "hard": {
        "function": simulate_quality_based_environment,
        "description": "Hard quality-based baseline",
        "config_types": ["hard"],
    },
    "hard_v2": {
        "function": simulate_quality_based_environment,
        "description": "Hard v2 quality-based baseline with shelf constraint",
        "config_types": ["hard_v2"],
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Non-LLM Simulation Runner for RetailBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config Types:
  easy    - Small SKU/category set, no rent
  middle  - Larger SKU/category set with rent
  hard    - Dynamic data with news and higher capacity/rent
  hard_v2 - Hard setting with power category attraction and shelf capacity

Example usage:
  python agents/run_non_llm_simulation.py --days 180 --sample-size 6 --config-type easy
  python agents/run_non_llm_simulation.py --days 180 --sample-size 5 --config-type hard
  python agents/run_non_llm_simulation.py --days 180 --config-type hard_v2
        """
    )

    parser.add_argument("--days", type=int, default=180, help="Number of days to simulate")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="SKU sample size per category; omit to use tuned default for the config",
    )
    parser.add_argument("--config-type", type=str, default="hard_v2", choices=CONFIG_TYPE_CHOICES,
                        help="Configuration type")
    parser.add_argument("--db-path", type=str, default=None, help="Database path (optional)")
    parser.add_argument(
        "--enable-checkpoint-test",
        action="store_true",
        help="Run checkpoint save/recover consistency test at the end of supported baselines",
    )

    args = parser.parse_args()

    args.db_path = args.db_path or ("model_run_time/" + args.config_type)

    env_info = ENVIRONMENT_MAPPING.get(args.config_type)
    if not env_info:
        valid = ", ".join(sorted(ENVIRONMENT_MAPPING))
        print(f"Unknown config_type: {args.config_type}. Valid options: {valid}")
        exit(1)

    sim_func = env_info["function"]
    sample_size_label = args.sample_size if args.sample_size is not None else "auto"
    print(f"Running environment: days={args.days}, sample_size={sample_size_label}, config_type={args.config_type}")
    
    sim_func(
        days=args.days,
        sample_size=args.sample_size,
        db_path=args.db_path or 'data/simulate_data/15/records_no_review/',
        config_type=args.config_type,
        enable_checkpoint_test=args.enable_checkpoint_test,
    )

    print("Simulation complete!")
