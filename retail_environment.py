from __future__ import annotations
import ast
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "module"))
from collections import defaultdict
from datetime import date, datetime, timedelta
import random
from typing import Any, Dict, List, Optional, Set
import json
import time
import argparse
from pathlib import Path
from uuid import uuid4
from util.config_registry import CONFIG_TYPE_CHOICES, load_config_by_type
from util.file import load_json
from util.sql_formatter import format_sql_rows
from util.logger import configure_logger, get_logger, set_logger_level
from module.customer_manager import CustomerManager
from module.supplier_manager import SupplierManager
from module.inventory import Inventory
from module.order_manager import OrderManager, Order
from module.news_manager import NewsManager
from module.record_manager import (
    RecordManager,
    SupplierOrderRecord,
    ProductLifecycleRecord,
)
from module.quality_first_supplier_guard import QualityFirstSupplierGuard
from module.review_manager import WINDOW_DAYS, ReviewManager
from module.sku import SKU, Merchandise

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# raise ValueError("Unable to find legal sku")
# ValueError(f"No rating found for SKU {sku.sku_id}")
#  raise ValueError("limit must be positive")

class RetailEnvironment:
    """
    零售环境入口。
    只实现图中给出的 5 个方法：
    - __init__
    - reset
    - step
    - get_tools
    - exec_tools
    """

    def __init__(self, config) -> None:
        self.config = config

        # 可选：锁定全局随机种子，保证同一配置下运行可复现
        seed = self.config.get("global_random_seed", None)
        if seed is not None:
            try:
                random.seed(int(seed))
            except Exception:
                # 如果配置不合法，忽略种子设置，继续运行
                pass

        self.debug = bool(self.config.get("debug", False))

        configure_logger(
            log_dir=self.config.get("log_dir"),
            filename=self.config.get("env_log_filename", "environment.log"),
            debug=self.debug,
        )
        self.logger = get_logger()

        self.inventory = Inventory(
            capacity=config.get("inventory_capacity")
        )
        self.order_manager = OrderManager()

        self.customer_manager = CustomerManager(
            begin_time=config['data_begin_time'],
            end_time=config['data_end_time'],
            data_path=config['customer_data_path'],
            store_id=config.get("store_id", ""),
        )

        self.record_manager = RecordManager(
            data_dir=config["order_record_dir"],
            init_sql_path=config.get("init_sql_path", None),
        )

        self.store_id = config.get("store_id", "")
        self.funds = config.get("initial_funds", 0)

        self.current_date = datetime.strptime(
            config.get("store_begin_time", date.today()),
            "%m/%d/%y"
        ).date()

        store_data_path = config.get("data_dir")

        # SKU 初始化：读取参数与 UPC 元信息
        self.skus_category_map = self._load_sku_data(store_data_path, self.store_id)

        self.skus_list = []
        for category, sku_list in self.skus_category_map.items():
            self.skus_list.extend(sku_list)

        # self._initialize_inventory_with_skus()

        self.skus_id_map: Dict[str, SKU] = {sku.sku_id: sku for sku in self.skus_list}
        self.enable_shelf_constraint = bool(self.config.get("enable_shelf_constraint", False))
        self.shelf_sku_capacity = self.config.get("shelf_sku_capacity")
        self.shelf_sku_ids: Set[str] = set()

        self.rent = self.config.get("everyday_rent")

        # Initialize inventory with SKU information
        self.review_manager = ReviewManager(
            record_manager=self.record_manager,
            review_model_path=config.get("review_model_path"),
            review_source_path=config.get("review_source_path"),
            enabled=config.get("enable_review", False),
            gen_prob=config.get("review_ratio", 0.1),
        )

        self._apply_initial_ratings(
            review_manager=self.review_manager
        )

        self.news_manager = (
            NewsManager(
                news_source_path=config.get("news_source_path"),
                random_seed=config.get("news_random_seed"),
                sample_ratios=config.get("news_sample_ratios"),
                daily_news_count=config.get("news_daily_count", 0),
                allowed_categories=list(self.skus_category_map.keys()),
                allowed_sku_ids=list(self.skus_id_map.keys()),
                current_date=self.current_date,
                impact_mode_weights=config.get("news_impact_mode_weights"),
                impact_base_scale=config.get("news_impact_base_scale", 0.2),
                impact_abs_cap=config.get("news_impact_abs_cap"),
            )
            if config.get("enable_new")
            else None
        )

        self.supplier_manager = SupplierManager(
            store_root=config.get("supplier_data_dir", config.get("data_dir", "")),
            store_id=config.get("store_id", ""),
            begin_date=config.get("data_begin_time"),
            end_date=config.get("data_end_time"),
            news_manager=self.news_manager if self.config.get("enable_new") else None,
            allowed_sku_ids=list(self.skus_id_map.keys()),
        )

        # Initialize notes storage
        self.notes: List[Dict[str, Any]] = []
        self.quality_first_guard = QualityFirstSupplierGuard(self, config)

        

    def _load_sku_data(self, store_data_path: str, store_id: str = "") -> Dict[str, List[SKU]]:
        """
        从 sku_model_parameter.json 和 upc 元数据构建 SKU 实例。

        - 参数文件：store_data_path/store_id/sku_model_parameter.json
        - 元信息：config['upc_meta_path']（默认 data/upc/upc.json），包含 DESCRIP/COM_CODE 等
        """
        param_path = Path(store_data_path) / str(store_id) / "sku_model_parameter.json"
        if not param_path.exists():
            self.log_debug(f"[WARN] sku_model_parameter.json not found: {param_path}")
            return {}
        sku_params = load_json(str(param_path))
        root = Path(
            self.config.get('data_dir')
        )
            
        upc_meta_path = root / f"{self.config.get('store_id', '')}" / f"upc.json"
        upc_meta = load_json(upc_meta_path) if Path(upc_meta_path).exists() else {}

        self.skus_category_map: Dict[str, str] = {}

        for meta in upc_meta:
            category = meta.get("CATEGORY") or meta.get("category")
            sku_id = meta.get("UPC")
            self.skus_category_map[sku_id] = category

        # 仅保留指定 sku_ids（若配置）
        sku_ids_filter = set(self.config.get("sku_ids", []))
        selected_categories = {c.replace("_", " "): True for c in self.config.get("selected_categories", [])}
        def allow_sku(sku_id: str, category: str) -> bool:
            if sku_ids_filter and sku_id not in sku_ids_filter:
                return False
            if selected_categories and category not in selected_categories:
                return False
            return True

        category_to_skus: Dict[str, List[SKU]] = {}

        for sku_id, params in sku_params.items():
            category = self.skus_category_map[sku_id]
            if not allow_sku(sku_id, category):
                continue

            meta = [item for item in upc_meta if item["UPC"] == sku_id][0]
            category = meta.get("CATEGORY") or meta.get("category") or meta.get("COM_CODE") or ""
            price_range = [meta.get('price_min'), meta.get('price_max')]
           
            category_effect = self.config.get("category_effects", {}).get(category, 0.0)

            init_price = random.uniform(float(price_range[0]), float(price_range[1]))

            description = {
                "description": meta.get("DESCRIPTION"),
                "extra": meta,
            }

            brand = meta.get("BRAND")

            sku_obj = SKU(
                sku_id=sku_id,
                description=description,
                category=category,
                model_parameters=params,
                init_price=init_price,
                category_effect=category_effect,
                brand=brand,
                promotion_day=meta.get('PROMOTION_TIME'),
            )

            category_to_skus.setdefault(category, []).append(sku_obj)

        # 同类目互相关联
        for skus in category_to_skus.values():
            for sku_obj in skus:
                sku_obj.set_same_category_skus([s for s in skus if s is not sku_obj])

        return category_to_skus
    
    def _apply_initial_ratings(self, review_manager=None) -> None:
        """
        为 SKU 注入初始评分：
        - 优先使用 review_manager 已有的历史评分均值；
        - 使用滑动时间窗口向更久之前回溯，尽量找到可用评分；
        - 如仍找不到，则保持默认（由 SKU 自身或其他配置决定）。
        """
        if not review_manager or not getattr(review_manager, "enabled", False):
            return

        window_days = WINDOW_DAYS
        max_windows = 20

        for sku in self.skus_list:
            rating_value = None
            window_end = datetime.strptime(
                self.config.get("store_begin_time", date.today()),
                "%m/%d/%y"
            ).date()
            
            window_start = window_end - timedelta(days=window_days)

            # 模拟 review_manager.compute_sales_impact 中的滑动窗口逻辑：
            # 先查最近 WINDOW_DAYS 天的平均评分，如果没有，就继续向前滚动窗口，
            # 最多尝试 max_windows 次，直到找到任意一个可用均值。
            for _ in range(max_windows):
                rating_value = review_manager.get_average_rating(
                    sku_id=sku.sku_id,
                    start_date=window_start,
                    end_date=window_end,
                )
                if rating_value is not None:
                    break
                window_start -= timedelta(days=window_days)

            if rating_value is not None:
                setattr(sku, "initial_rating", rating_value)
                if isinstance(sku.attributes, dict):
                    print(f"Setting initial rating for SKU {sku.sku_id} to {rating_value}")
                    sku.attributes["initial_rating"] = rating_value
            else:
                raise ValueError(f"No rating found for SKU {sku.sku_id}")
   
    def reset(self) -> Dict[str, Any]:
        """重置整个环境。"""
        config = dict(self.config)
        self.__init__(config)
        return {
            "funds": self.funds,
            "current_date": self.current_date,
            "num_skus": len(self.skus_list)
        }

    def step(self) -> Dict[str, Any]:
        """
        环境前进一步：推进一天，并让各 manager 执行各自的 step。
        """

        money_earned, insufficient_skus, sales_by_sku, returns_by_sku, expired_discount_by_sku, waiting_items, cost_of_goods_sold = self.inventory.step(
            self.current_date,
            self.customer_manager.get_customer_count(self.current_date),
            self.skus_id_map,
            record_manager=self.record_manager,
            review_manager=self.review_manager if self.review_manager.enabled else None,
            new_manager=self.news_manager if self.config.get('enable_new') and self.news_manager else None,
            shelf_sku_ids=self._effective_shelf_sku_ids(),
            category_attraction_aggregation=self.config.get("category_attraction_aggregation", "legacy"),
            category_attraction_rho=float(self.config.get("category_attraction_rho", 1.0)),
        )

        self.supplier_manager.step(
            current_date=self.current_date,
            record_manager=self.record_manager
        )

        if self.news_manager and self.config.get("enable_new"):
            self.news_manager.step(
                record_manager=self.record_manager,
                current_date=self.current_date
            )

        self.funds += money_earned
        self.current_date += timedelta(days=1)

        arrived_merchanises = self.order_manager.step(
            current_date=self.current_date,
            record_manager=self.record_manager,
        )

        for merchandise in arrived_merchanises:
            self.inventory.add_item(merchandise)

        self.funds -= self.rent

        # 使用 view_inventory 获取库存信息
        inventory_result = self.view_inventory()
        inventory_data = inventory_result.get("result", {})

        step_result = {
            "funds": self.funds,
            "net_worth": self.inventory.compute_net_worth(self.current_date) + self.funds + sum([order.cost for order in self.order_manager.get_current_orders()]),
            "current_date": self.current_date,
            "money_earned": money_earned,
            "insufficient_skus": insufficient_skus,
            "sales_by_sku": sales_by_sku,
            "returns_by_sku": returns_by_sku,
            "expired_discount_by_sku": expired_discount_by_sku,
            "waiting_items": waiting_items,
            "cost_of_goods_sold": cost_of_goods_sold,
            "inventory": inventory_data,  # 直接使用 view_inventory 返回的数据
            "pending_order_count": inventory_data.get("pending_order_count", 0),
            "incoming_items": inventory_data.get("incoming_items", 0),
            "incoming_by_sku": inventory_data.get("incoming_by_sku", {}),
            "pending_orders": inventory_data.get("pending_orders", []),
        }
        return self._wrap_tool_result(step_result, self._format_step_result(step_result))

    # =========================
    # 工具函数与 MCP 工具定义
    # =========================

    @staticmethod
    def _wrap_tool_result(result: Any, formatted: str) -> Dict[str, Any]:
        """统一返回格式：同时带原始结果与格式化字符串。"""
        return {"result": result, "formatted": formatted}

    def log_debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def log_info(self, msg: str) -> None:
        self.logger.info(msg)

    def _json_safe(self, obj: Any) -> Any:
        """递归转为可 JSON 序列化的结构，并确保键为字符串。"""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(v) for v in obj]
        return str(obj)

    def _log_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        elapsed_time: float,
        *,
        source: Optional[str] = None,
        parent_tool: Optional[str] = None,
        tag: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录工具调用日志（JSONL 格式），包含返回结果与资金/净值快照。
        每行一个 JSON 对象，使用 append 模式写入。
        
        Args:
            tool_name: 工具名称
            args: 工具参数
            result: 工具返回结果
            elapsed_time: 工具执行耗时（秒）
        """

        tool_log_path = self.config.get("tool_log_path")
        if tool_log_path:
            log_path = Path(tool_log_path)
        else:
            log_dir = Path(self.config.get("log_dir", "logs"))
            log_path = log_dir / "tool_calls.jsonl"

        log_path.parent.mkdir(parents=True, exist_ok=True)

        inventory_worth = self.inventory.compute_net_worth(current_date=self.current_date)
        pending_order_worth = sum(order.cost for order in self.order_manager.get_current_orders())
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tool": tool_name,
            "call_source": source or "direct",
            "args": self._json_safe(args),
            "result": self._json_safe(result),
            "funds": self.funds,
            "net_worth": inventory_worth + pending_order_worth + self.funds,
            "current_date": self.current_date.isoformat() if isinstance(self.current_date, date) else str(self.current_date),
            "elapsed_time": round(elapsed_time, 4),  # 保留4位小数
        }
        payload = result.get("result") if isinstance(result, dict) else None
        if isinstance(payload, dict):
            if payload.get("status"):
                entry["result_status"] = str(payload.get("status"))
            if "action_executed" in payload:
                entry["action_executed"] = bool(payload.get("action_executed"))
            elif tool_name in {"place_order", "modify_sku_price", "end_today"}:
                entry["action_executed"] = not bool(payload.get("error"))
        if source:
            entry["source"] = source
        if parent_tool:
            entry["parent_tool"] = parent_tool
        if tag:
            entry["tag"] = tag
        if extra:
            entry.update(self._json_safe(extra))
        
        # 使用 append 模式写入 JSONL 格式（每行一个 JSON 对象）
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _format_step_result(self, result: Dict[str, Any]) -> str:
        """格式化 step 返回结果。"""
        date_str = result.get("current_date")
        funds = result.get("funds")
        networth = result.get("net_worth")
        money = result.get("money_earned")
        insufficient = result.get("insufficient_skus") or []
        insufficient_skus = ' , '.join([sku.sku_id for sku in insufficient])
        sales_by_sku = result.get("sales_by_sku") or {}
        returns_by_sku = result.get("returns_by_sku") or {}
        expired_by_sku = result.get("expired_discount_by_sku") or {}

        waiting_by_sku = defaultdict(int)
        for merch in self.inventory.waiting_items:
            waiting_by_sku[merch.sku.sku_id] += 1

        # 从结果中获取库存信息（来自 view_inventory）
        inventory_data = result.get("inventory", {})
        total_items = inventory_data.get("total_items", 0)
        total_waiting = inventory_data.get("waiting_items", 0)
        incoming_items = inventory_data.get("incoming_items", 0)
        pending_order_count = inventory_data.get("pending_order_count", 0)
        pending_orders = inventory_data.get("pending_orders", [])
        inventory_by_sku = inventory_data.get("inventory", {})

        lines = [
            f"Current Date: {date_str}",
            f"Current Funds: {funds}",
            f"Current Networth: {networth}",
            f"Money earned yestarday: {money}",
        ]
        
        # 添加库存摘要信息
        if self.inventory.capacity is not None:
            capacity_used = total_items + total_waiting
            capacity_available = self.inventory.capacity - capacity_used
            lines.append(
                f"Inventory: {total_items} items in stock, {total_waiting} capacity-waiting, "
                f"{incoming_items} incoming in {pending_order_count} pending supplier orders "
                f"({capacity_used}/{self.inventory.capacity} used, {capacity_available} available)"
            )
        else:
            lines.append(
                f"Inventory: {total_items} items in stock, {total_waiting} capacity-waiting, "
                f"{incoming_items} incoming in {pending_order_count} pending supplier orders"
            )
        
        if insufficient:
            lines.append(f"Insufficient SKUs: {insufficient_skus}")
        else:
            lines.append("No stockouts today.")
        if sales_by_sku:
            sales_lines = ", ".join(f"{k}={v}" for k, v in sales_by_sku.items())
            lines.append(f"Sales by SKU: {sales_lines}")
        if returns_by_sku:
            return_lines = ", ".join(f"{k}={v}" for k, v in returns_by_sku.items())
            lines.append(f"Returns by SKU: {return_lines}")
        if expired_by_sku:
            expired_lines = ", ".join(f"{k}={v}" for k, v in expired_by_sku.items())
            lines.append(f"Expired clearance sold: {expired_lines}")
        if waiting_by_sku:
            promo_lines = ", ".join(f"{k}={v}" for k, v in waiting_by_sku.items())
            lines.append(f"Capacity-waiting SKU Items: {promo_lines}")

        if pending_orders:
            order_lines = []
            for order in pending_orders[:5]:
                order_lines.append(
                    f"{order.get('order_id')} from {order.get('supplier_id')} "
                    f"arrives {order.get('expected_delivery_date')} "
                    f"({order.get('total_items')} items, cost {order.get('cost')})"
                )
            lines.append(
                "Pending supplier orders: "
                + "; ".join(order_lines)
                + ("..." if len(pending_orders) > 5 else "")
            )
        
        # 添加库存详情（只显示有库存的 SKU，最多10个）
        inventory_summary = []
        for sku_id, data in inventory_by_sku.items():
            qty = data.get("quantity", 0)
            waiting = data.get("waiting", 0)
            incoming = data.get("incoming", 0)
            if qty > 0 or waiting > 0 or incoming > 0:
                extras = []
                if waiting > 0:
                    extras.append(f"capacity-waiting {waiting}")
                if incoming > 0:
                    extras.append(f"incoming {incoming}")
                inventory_summary.append(
                    f"{sku_id}: {qty}" + (f" ({', '.join(extras)})" if extras else "")
                )
        
        if inventory_summary:
            lines.append(f"Inventory details: {', '.join(inventory_summary[:10])}" + ("..." if len(inventory_summary) > 10 else ""))
        
        return "\n".join(lines)

    def _summarize_pending_supplier_orders(self) -> Dict[str, Any]:
        """Return open supplier orders that have been placed but not delivered."""
        pending_orders = []
        incoming_by_sku: Dict[str, int] = defaultdict(int)
        total_incoming_items = 0

        for order in self.order_manager.get_current_orders():
            ordered_sku = {
                sku.sku_id: qty for sku, qty in getattr(order, "ordered_sku", {}).items()
            }
            order_total_items = sum(ordered_sku.values())
            total_incoming_items += order_total_items

            for sku_id, qty in ordered_sku.items():
                incoming_by_sku[sku_id] += qty

            expected_date = getattr(order, "expected_delivery_date", None)
            days_until_arrival = None
            if isinstance(expected_date, date):
                days_until_arrival = (expected_date - self.current_date).days

            order_items = [
                {"sku_id": sku_id, "count": qty}
                for sku_id, qty in ordered_sku.items()
            ]
            created_at = (
                order.created_at.isoformat()
                if isinstance(order.created_at, date)
                else str(order.created_at)
            )
            expected_delivery_date = (
                expected_date.isoformat()
                if isinstance(expected_date, date)
                else str(expected_date)
            )

            pending_orders.append(
                {
                    "order_id": order.order_id,
                    "supplier_id": order.customer_id,
                    "customer_id": order.customer_id,
                    "ordered_sku": ordered_sku,
                    "created_at": created_at,
                    "delivery_time": order.delivery_time,
                    "expected_delivery_date": expected_delivery_date,
                    "days_until_arrival": days_until_arrival,
                    "cost": order.cost,
                    "total_items": order_total_items,
                    "items": order_items,
                }
            )

        pending_orders.sort(key=lambda item: (item["expected_delivery_date"], item["supplier_id"], item["order_id"]))
        return {
            "pending_order_count": len(pending_orders),
            "incoming_items": total_incoming_items,
            "incoming_by_sku": dict(incoming_by_sku),
            "pending_orders": pending_orders,
        }

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        """允许 date 或常见字符串格式，其他返回 None。"""
        if value is None:
            return None
        if isinstance(value, date):
            return value
        for fmt in ("%Y-%m-%d", "%m/%d/%y"):
            try:
                return datetime.strptime(str(value), fmt).date()
            except Exception:
                continue
        return None

    def get_sku_rating_report(
        self,
        sku_ids: List[str] = None,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return initial and average ratings for given SKUs.
        Not registered as a tool; callable directly.
        """
        try:
            self._ensure_review_mgr()
            target_ids = self._filter_allowed_skus(sku_ids) if sku_ids else list(self.skus_id_map.keys())
            start = self._parse_date(start_date)
            end = self._parse_date(end_date)

            def initial_rating(sku_obj: SKU) -> Optional[float]:
                if hasattr(sku_obj, "initial_rating"):
                    return getattr(sku_obj, "initial_rating")
                if isinstance(getattr(sku_obj, "attributes", None), dict):
                    return sku_obj.attributes.get("initial_rating")
                return None

            report: Dict[str, Dict[str, Any]] = {}
            for sid in target_ids:
                sku_obj = self.skus_id_map.get(sid)
                if not sku_obj:
                    continue
                init_rate = initial_rating(sku_obj)
                avg_rate = self.review_manager.get_average_rating(
                    sku_id=sid,
                    start_date=self.current_date - timedelta(days=WINDOW_DAYS),
                    end_date=self.current_date
                )

                report[sid] = {
                    "initial_rating": init_rate,
                    "avg_rating": avg_rate,
                }
            return report
        except Exception as e:
            err_msg = f"Error executing get_sku_rating_report: {type(e).__name__}: {e}"
            # 这个方法不是工具方法，直接返回错误字典
            return {"error": err_msg}
    # ---------- 工具实现 ----------
    def view_funds_and_date(self) -> Dict[str, Any]:
        payload = {
            "funds": self.funds,
            "current_date": self.current_date,
        }
        formatted = f"Current date: {self.current_date}, funds balance: {self.funds:.2f}"
        return self._wrap_tool_result(payload, formatted)

    def view_inventory(self) -> Dict[str, Any]:
        inventory_info: Dict[str, Dict[str, Any]] = {}
        total_items = 0
        waiting_by_sku = defaultdict(int)
        pending_summary = self._summarize_pending_supplier_orders()
        incoming_by_sku = pending_summary["incoming_by_sku"]

        for merch in self.inventory.waiting_items:
            waiting_by_sku[merch.sku.sku_id] += 1

        total_waiting = sum(waiting_by_sku.values())

        for sku_id, value in self.skus_id_map.items():
            inventory_info[sku_id] = {
                "quantity": 0,
                "waiting": waiting_by_sku.get(sku_id, 0),
                "incoming": incoming_by_sku.get(sku_id, 0),
                "on_shelf": sku_id in self.shelf_sku_ids,
            }

        for sku_id, items in self.inventory.items_by_sku.items():
            sku_obj = self.skus_id_map.get(sku_id)
            quantities = len(items)
            total_items += quantities
            inventory_info[sku_id] = {
                "quantity": quantities,
                "waiting": waiting_by_sku.get(sku_id, 0),
                "incoming": incoming_by_sku.get(sku_id, 0),
                "on_shelf": sku_id in self.shelf_sku_ids,
            }

        formatted_lines = [
            f"Inventory summary: {len(inventory_info)} SKUs, {total_items} total items"
        ]
        if total_waiting:
            formatted_lines[0] += f" (+{total_waiting} capacity-waiting)"
        if pending_summary["incoming_items"]:
            formatted_lines[0] += (
                f", {pending_summary['incoming_items']} incoming in "
                f"{pending_summary['pending_order_count']} pending supplier orders"
            )

        for sku_id, data in list(inventory_info.items()):
            qty = data["quantity"]
            waiting = data.get("waiting", 0)
            incoming = data.get("incoming", 0)
            if qty == 0 and waiting == 0 and incoming == 0:
                continue
            extras = []
            if waiting:
                extras.append(f"capacity-waiting {waiting}")
            if incoming:
                extras.append(f"incoming {incoming}")
            if data.get("on_shelf"):
                extras.append("on shelf")
            formatted_lines.append(
                f"- {sku_id}: quantity {qty}" + (f" ({', '.join(extras)})" if extras else "")
            )

        if len(formatted_lines) == 1:
            formatted_lines.append("Inventory is empty.")

        if pending_summary["pending_orders"]:
            formatted_lines.append("Pending supplier orders:")
            for order in pending_summary["pending_orders"][:8]:
                formatted_lines.append(
                    f"- {order['order_id']} from {order['supplier_id']}: "
                    f"{order['total_items']} items, arrives {order['expected_delivery_date']}, "
                    f"cost {order['cost']:.2f}"
                )
            if len(pending_summary["pending_orders"]) > 8:
                formatted_lines.append(f"- ... {len(pending_summary['pending_orders']) - 8} more pending orders")
        
        return self._wrap_tool_result(
            {
                "inventory": inventory_info,
                "total_items": total_items,
                "total_skus": len(inventory_info),
                "waiting_items": total_waiting,
                "pending_order_count": pending_summary["pending_order_count"],
                "incoming_items": pending_summary["incoming_items"],
                "incoming_by_sku": pending_summary["incoming_by_sku"],
                "pending_orders": pending_summary["pending_orders"],
                "shelf": {
                    "enabled": self.enable_shelf_constraint,
                    "capacity": self.shelf_sku_capacity,
                    "used": len(self.shelf_sku_ids),
                    "sku_ids": sorted(self.shelf_sku_ids),
                },
            },
            "\n".join(formatted_lines),
        )

    def _effective_shelf_sku_ids(self) -> Optional[Set[str]]:
        if not self.enable_shelf_constraint:
            return None
        return set(self.shelf_sku_ids)

    def view_shelf_status(self) -> Dict[str, Any]:
        inventory_quantities = {
            sku_id: len(items)
            for sku_id, items in self.inventory.items_by_sku.items()
        }
        shelf_entries: Dict[str, Dict[str, Any]] = {}
        for sku_id in sorted(self.shelf_sku_ids):
            sku_obj = self.skus_id_map.get(sku_id)
            if sku_obj is None:
                continue
            quantity = inventory_quantities.get(sku_id, 0)
            shelf_entries[sku_id] = {
                "quantity": quantity,
                "category": sku_obj.category,
                "price": sku_obj.price,
                "in_stock": quantity > 0,
            }

        capacity = self.shelf_sku_capacity
        used = len(self.shelf_sku_ids)
        available = None if capacity is None else max(int(capacity) - used, 0)
        payload = {
            "enabled": self.enable_shelf_constraint,
            "capacity": capacity,
            "used": used,
            "available": available,
            "shelf_skus": sorted(self.shelf_sku_ids),
            "on_shelf": shelf_entries,
        }

        lines = [
            "## Shelf status",
            f"- enabled: {self.enable_shelf_constraint}",
            f"- capacity: {capacity if capacity is not None else 'unlimited'}",
            f"- used: {used}",
        ]
        if available is not None:
            lines.append(f"- available: {available}")
        if shelf_entries:
            lines.extend(["", "| SKU | Category | Qty | Price | Status |", "|---|---|---|---|---|"])
            for sku_id, row in shelf_entries.items():
                status = "in stock" if row["in_stock"] else "out of stock"
                lines.append(
                    f"| {sku_id} | {row['category']} | {row['quantity']} | {row['price']} | {status} |"
                )
        else:
            lines.append("No SKUs are currently on shelf.")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def set_shelf_skus(self, sku_ids: List[str]) -> Dict[str, Any]:
        self._validate_sku_list(sku_ids, allow_empty=True)
        deduped = list(dict.fromkeys(sid.strip() for sid in sku_ids))
        capacity = self.shelf_sku_capacity
        if capacity is not None and len(deduped) > int(capacity):
            raise ValueError(f"shelf accepts at most {capacity} SKUs")

        previous = set(self.shelf_sku_ids)
        self.shelf_sku_ids = set(deduped)
        added = sorted(self.shelf_sku_ids - previous)
        removed = sorted(previous - self.shelf_sku_ids)
        result = {
            "capacity": capacity,
            "used": len(self.shelf_sku_ids),
            "shelf_skus": sorted(self.shelf_sku_ids),
            "added": added,
            "removed": removed,
        }
        formatted = (
            f"Shelf updated: {len(self.shelf_sku_ids)}/"
            f"{capacity if capacity is not None else 'unlimited'} SKUs on shelf."
        )
        if added:
            formatted += f"\nAdded: {', '.join(added)}"
        if removed:
            formatted += f"\nRemoved: {', '.join(removed)}"
        return self._wrap_tool_result(result, formatted)

    def _ensure_review_mgr(self) -> None:
        if not getattr(self, "review_manager", None):
            raise ValueError("Review manager not initialized.")
        if not self.review_manager.enabled:
            raise ValueError("Reviews feature disabled.")

    def _filter_allowed_skus(self, sku_ids: Optional[List[str]]) -> List[str]:
        """Return only SKUs present in the environment; used to guard tools."""
        if not sku_ids:
            return []
        allowed = set(self.skus_id_map.keys())
        return [sid for sid in sku_ids if sid in allowed]

    def _validate_sku_list(
        self,
        sku_ids: Any,
        allow_empty: bool = False,
    ) -> List[str]:
        """Validate sku_ids input and return the list; raises on invalid input."""
        if sku_ids is None:
            if allow_empty:
                return []
            raise ValueError("sku_ids is required")
        if not isinstance(sku_ids, list):
            raise ValueError("sku_ids must be an array")
        if not sku_ids and not allow_empty:
            raise ValueError("sku_ids cannot be empty")
        for idx, sid in enumerate(sku_ids):
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError(f"sku_ids[{idx}] is invalid")
            if sid not in self.skus_id_map:
                raise ValueError(f"Unknown SKU: {sid}")
        return sku_ids

    def _validate_dates(
        self,
        start_date: Any = None,
        end_date: Any = None,
        allow_none: bool = False,
    ) -> Any:
        """
        Validate date inputs; returns parsed (start, end).
        - When allow_none is False, both start_date and end_date are required.
        - Accepts date objects or strings in YYYY-MM-DD / MM/DD/YY.
        """
        if not allow_none:
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date are required")

        start = self._parse_date(start_date) if start_date is not None else None
        end = self._parse_date(end_date) if end_date is not None else None

        if start_date is not None and start is None:
            raise ValueError("Invalid start_date format; use YYYY-MM-DD or MM/DD/YY.")
        if end_date is not None and end is None:
            raise ValueError("Invalid end_date format; use YYYY-MM-DD or MM/DD/YY.")

        if start and end and start > end:
            raise ValueError("start_date cannot be after end_date")

        return start, end

    def view_sku_reviews(
        self,
        sku_ids: List[str],
        start_date: str = None,
        end_date: str = None,
        ratings: List[int] = [1, 2, 3, 4, 5],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._ensure_review_mgr()
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("No allowed sku_ids to query.")
        start, end = self._validate_dates(start_date, end_date, allow_none=True)
        if ratings is None:
            ratings = [1, 2, 3, 4, 5]
        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("limit must be an integer")
            if limit <= 0:
                raise ValueError("limit must be positive")
        else:
            limit = 20  # align with tool schema default

        payload: Dict[str, Any] = {}
        lines = ["## Reviews"]
        for sku_id in sku_ids:
            reviews = self.review_manager.get_reviews_in_range(
                sku_id=sku_id,
                start_date=start,
                end_date=end,
                ratings=ratings,
            )
            if limit is not None:
                if len(reviews) <= limit:
                    pass
                else:
                    seed = f"{sku_id}|{start}|{end}|{ratings}|{limit}"
                    reviews = random.Random(seed).sample(reviews, limit)

            payload[sku_id] = [r.to_dict() for r in reviews]
            lines.append(f"### {sku_id} ({len(reviews)})")
            lines.append("| Date | Rating | Category | Dimension | Supplier | Merchandise | Comment |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in reviews:
                lines.append(
                    f"| {r.date} | {r.rating} | {r.category or '-'} | {r.dimension or '-'} | "
                    f"{getattr(r, 'supplier_id', '-') or '-'} | {getattr(r, 'merchandise_id', '-') or '-'} | "
                    f"{r.comment or '-'} |"
                )
            lines.append("")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_sku_avg_ratings(
        self,
        sku_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        self._ensure_review_mgr()
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("No allowed sku_ids to query.")
        start, end = self._validate_dates(start_date, end_date, allow_none=True)
        payload: Dict[str, Any] = {}
        lines = ["## Average ratings"]
        for sku_id in sku_ids:
            avg = self.review_manager.get_average_rating(
                sku_id=sku_id,
                start_date=start,
                end_date=end,
            )
            payload[sku_id] = avg
            lines.append(f"- {sku_id}: {avg if avg is not None else 'N/A'}")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_sales_profit_history(
        self,
        sku_ids: List[str],
        start_date: str,
        end_date: str,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """View daily sales price, sold units, and realized net profit for SKUs."""
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("Unable to find legal sku")
        
        start, end = self._validate_dates(start_date, end_date, allow_none=False)

        supplier_filter: Optional[str] = None
        if supplier_id is not None:
            if not isinstance(supplier_id, str) or not supplier_id.strip():
                raise ValueError("supplier_id must be a non-empty string when provided")
            supplier_filter = supplier_id.strip()

        day_values = [
            start + timedelta(days=offset)
            for offset in range((end - start).days + 1)
        ]
        day_keys = [day.isoformat() for day in day_values]
        sale_price_cache: Dict[tuple[str, str], Optional[float]] = {}

        def _sale_price_for_return(sku_id: str, day: date, fallback: Optional[float]) -> float:
            cache_key = (sku_id, day.isoformat())
            if cache_key not in sale_price_cache:
                records_by_day = self.record_manager.read_sku(sku_id, day, day)
                records = records_by_day.get(day.isoformat(), [])
                total_units = sum(max(int(record.move or 0), 0) for record in records)
                if total_units:
                    weighted = sum(float(record.price) * int(record.move or 0) for record in records)
                    sale_price_cache[cache_key] = weighted / total_units
                else:
                    sale_price_cache[cache_key] = fallback
            cached = sale_price_cache[cache_key]
            return float(cached or 0.0)

        payload: Dict[str, Any] = {}
        formatted_sections: List[str] = []

        for sku_id in sku_ids:
            daily_stats: Dict[str, Dict[str, Any]] = {
                day: {
                    "date": day,
                    "gross_sales_revenue": 0.0,
                    "cogs": 0.0,
                    "return_refunds": 0.0,
                    "units_sold": 0,
                }
                for day in day_keys
            }

            sold_records = self.record_manager.read_product_lifecycle(
                sku_id=sku_id,
                status="sold",
                start_sold_date=start,
                end_sold_date=end,
            )
            if supplier_filter:
                sold_records = [
                    record for record in sold_records
                    if record.supplier_id == supplier_filter
                ]

            valid_sold_records = [
                record for record in sold_records
                if record.sold_date is not None and record.selling_price is not None
            ]
            total_gross_sales_revenue = 0.0
            total_cogs = 0.0
            for record in valid_sold_records:
                day = record.sold_date.isoformat()
                if day not in daily_stats:
                    continue
                selling_price = float(record.selling_price or 0.0)
                buy_price = float(record.buy_price or 0.0)
                daily_stats[day]["gross_sales_revenue"] += selling_price
                daily_stats[day]["cogs"] += buy_price
                daily_stats[day]["units_sold"] += 1
                total_gross_sales_revenue += selling_price
                total_cogs += buy_price

            avg_selling_price = (
                total_gross_sales_revenue / len(valid_sold_records)
                if valid_sold_records
                else None
            )
            return_records = self.record_manager.read_returns(
                supplier_id=supplier_filter,
                sku_id=sku_id,
                start_date=start,
                end_date=end,
            )
            for record in return_records:
                day = record.date.isoformat()
                if day not in daily_stats:
                    continue
                daily_stats[day]["return_refunds"] += _sale_price_for_return(
                    sku_id,
                    record.date,
                    avg_selling_price,
                )

            daily_rows: List[Dict[str, Any]] = []
            records: Dict[str, List[Dict[str, Any]]] = {}
            total_units = 0
            total_net_profit = 0.0
            for day in day_keys:
                stats = daily_stats[day]
                units_sold = int(stats["units_sold"])
                gross_revenue = float(stats["gross_sales_revenue"])
                cogs = float(stats["cogs"])
                refunds = float(stats["return_refunds"])
                net_profit = gross_revenue - refunds - cogs
                selling_price = gross_revenue / units_sold if units_sold else None
                row = {
                    "date": day,
                    "selling_price": selling_price,
                    "units_sold": units_sold,
                    "net_profit": round(net_profit, 4),
                }
                daily_rows.append(row)
                records[day] = []
                if units_sold:
                    records[day].append(
                        {
                            "sku_id": sku_id,
                            "date": day,
                            "price": selling_price,
                            "move": units_sold,
                            "net_profit": round(net_profit, 4),
                        }
                    )
                total_units += units_sold
                total_net_profit += net_profit

            payload[sku_id] = {
                "daily": daily_rows,
                "records": records,
                "total_units": total_units,
                "days": len(day_keys),
                "net_profit": round(total_net_profit, 4),
                "avg_selling_price": avg_selling_price,
            }

            table_lines = [
                f"### SKU: {sku_id}",
                "",
                "| Date | Selling Price | Units Sold | Net Profit |",
                "|------|---------------|------------|------------|",
            ]

            for row in daily_rows:
                price = "-" if row["selling_price"] is None else f"{row['selling_price']:.4f}"
                table_lines.append(
                    f"| {row['date']} | {price} | {row['units_sold']} | {row['net_profit']:.4f} |"
                )

            formatted_sections.append("\n".join(table_lines))

        formatted = "\n\n".join(formatted_sections)
        return self._wrap_tool_result(payload, formatted)

    def view_current_date_supplier_prices(
        self,
        sku_ids: List[str],
    ) -> Dict[str, Any]:
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("Unable to find legal sku")
        dt = self.current_date

        payload: Dict[str, Any] = {}
        supplier_groups = defaultdict(list)

        for sku_id in sku_ids:
            entries = self.supplier_manager.get_sku_date(sku_id, dt)
            public_entries = [self._public_supplier_price_entry(entry) for entry in entries]
            payload[sku_id] = public_entries

            for entry in public_entries or []:
                supplier_groups[entry['supplier_id']].append(entry)

        # ------- 按 supplier 分组输出 -------
        lines = [
            f"## Supplier prices on {dt}",
            "",
            "| Supplier | SKU | Price |",
            "|---|---|---|",
        ]
        if not supplier_groups:
            lines.append("| - | - | - |")

        for supplier_id, entries in supplier_groups.items():
            for entry in entries:
                sku = entry["sku_id"]
                price = entry["price"]
                lines.append(f"| {supplier_id} | {sku} | {price} |")

        return self._wrap_tool_result(payload, "\n".join(lines))

    def _public_supplier_price_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Return supplier quote fields visible to agents; hide internal quality labels."""
        allowed_keys = {
            "supplier_id",
            "sku_id",
            "date",
            "price",
            "category",
            "transport_days_min",
            "transport_days_max",
        }
        return {key: entry.get(key) for key in allowed_keys if key in entry}

    def view_supplier_price_history(
        self,
        sku_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if sku_id:
            self._validate_sku_list([sku_id])
        sku_ids = self._filter_allowed_skus([sku_id])
        if not sku_ids:
            raise ValueError("Unable to find legal sku")
        start, end = self._validate_dates(start_date, end_date, allow_none=True)
        records = self.record_manager.read_supplier_prices(
            supplier_id=supplier_id,
            sku_id=sku_id,
            start_date=start,
            end_date=end,
        )

        payload = [
            {
                "supplier_id": r.supplier_id,
                "sku_id": r.sku_id,
                "date": r.date,
                "price": r.price,
            }
            for r in records
        ]

        lines = [
            f"## Historical supplier price records ({len(payload)} rows)",
            f"- filters: supplier_id={supplier_id}, sku_id={sku_id}, start={start}, end={end}",
            "",
            "| # | Supplier | SKU | Date | Price |",
            "|---|---|---|---|---|",
        ]
        for idx, row in enumerate(payload):
            lines.append(
                f"| {idx} | {row['supplier_id']} | {row['sku_id']} | {row['date']} | {row['price']} |"
            )

        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_return_rates(
        self,
        sku_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """查看 SKU 的退货率记录。"""
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("Unable to find legal sku")
        start, end = self._validate_dates(start_date, end_date, allow_none=False)

        payload: Dict[str, List[Dict[str, Any]]] = {}
        lines = [f"## Return rates {start} ~ {end}"]
        lines.append("| SKU | Date | Return Rate | Return Number |")
        lines.append("|---|---|---|---|")

        for sku_id in sku_ids:
            recs = self.record_manager.read_return_rates(
                sku_id=sku_id,
                start_date=start,
                end_date=end,
            )
            payload[sku_id] = [r.to_dict() for r in recs]
            if not recs:
                lines.append(f"| {sku_id} | - | - | - |")
            else:
                for r in recs:
                    lines.append(f"| {sku_id} | {r.date} | {r.return_rate} | {getattr(r, 'return_number', '-') or '-'} |")

        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_supplier_returns_avg_rate(
        self,
        supplier_id: str,
        start_date: str,
        end_date: str,
        sku_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """查看 supplier 在日期范围内的平均退货率。"""
        if not isinstance(supplier_id, str) or not supplier_id.strip():
            raise ValueError("supplier_id is required")
        supplier_id = supplier_id.strip()

        if sku_ids is None:
            target_sku_ids = list(self.skus_id_map.keys())
        else:
            self._validate_sku_list(sku_ids)
            target_sku_ids = self._filter_allowed_skus(sku_ids)
        if sku_ids is not None and not target_sku_ids:
            raise ValueError("Unable to find legal sku")
        start, end = self._validate_dates(start_date, end_date, allow_none=False)

        supplier_orders = self.record_manager.read_supplier_orders(
            supplier_id=supplier_id,
            start_order_date=start,
            end_order_date=end,
        )
        target_sku_set = set(target_sku_ids)
        ordered_by_sku: Dict[str, int] = {sku_id: 0 for sku_id in target_sku_ids}
        for order in supplier_orders:
            for sku_id, qty in (order.items or {}).items():
                if sku_id not in target_sku_set:
                    continue
                try:
                    ordered_by_sku[sku_id] += int(qty)
                except (TypeError, ValueError):
                    continue

        per_sku: Dict[str, Dict[str, Any]] = {}
        total_sold = 0
        total_ordered = 0
        total_denominator = 0
        total_returns = 0
        sku_rates: List[float] = []

        for sku_id in target_sku_ids:
            sold_records = [
                record
                for record in self.record_manager.read_product_lifecycle(
                    sku_id=sku_id,
                    status="sold",
                )
                if record.supplier_id == supplier_id
                and record.sold_date is not None
                and start <= record.sold_date <= end
            ]
            return_records = self.record_manager.read_returns(
                supplier_id=supplier_id,
                sku_id=sku_id,
                start_date=start,
                end_date=end,
            )
            sold_count = len(sold_records)
            ordered_count = ordered_by_sku.get(sku_id, 0)
            return_count = len(return_records)
            denominator_count = sold_count if sold_count else ordered_count
            if sold_count:
                denominator_source = "sold_count"
            elif ordered_count:
                denominator_source = "ordered_count_fallback"
            else:
                denominator_source = "none"
            return_rate = return_count / denominator_count if denominator_count else None
            if return_rate is not None:
                sku_rates.append(return_rate)
            total_sold += sold_count
            total_ordered += ordered_count
            total_denominator += denominator_count
            total_returns += return_count
            per_sku[sku_id] = {
                "sku_id": sku_id,
                "sold_count": sold_count,
                "ordered_count": ordered_count,
                "denominator_count": denominator_count,
                "denominator_source": denominator_source,
                "return_count": return_count,
                "return_rate": return_rate,
            }

        if total_denominator == 0:
            denominator_source = "none"
        elif total_sold == total_denominator:
            denominator_source = "sold_count"
        elif total_ordered == total_denominator:
            denominator_source = "ordered_count_fallback"
        else:
            denominator_source = "mixed_sold_and_ordered"

        average_return_rate = total_returns / total_denominator if total_denominator else None
        sold_based_return_rate = total_returns / total_sold if total_sold else None
        order_based_return_rate = total_returns / total_ordered if total_ordered else None
        average_sku_return_rate = sum(sku_rates) / len(sku_rates) if sku_rates else None
        payload = {
            "supplier_id": supplier_id,
            "sku_ids": target_sku_ids,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "sold_count": total_sold,
            "ordered_count": total_ordered,
            "denominator_count": total_denominator,
            "denominator_source": denominator_source,
            "return_count": total_returns,
            "average_return_rate": average_return_rate,
            "sold_based_return_rate": sold_based_return_rate,
            "order_based_return_rate": order_based_return_rate,
            "average_sku_return_rate": average_sku_return_rate,
            "per_sku": per_sku,
        }

        display_avg = "-" if average_return_rate is None else f"{average_return_rate:.6f}"
        display_sold_avg = "-" if sold_based_return_rate is None else f"{sold_based_return_rate:.6f}"
        display_order_avg = "-" if order_based_return_rate is None else f"{order_based_return_rate:.6f}"
        display_sku_avg = "-" if average_sku_return_rate is None else f"{average_sku_return_rate:.6f}"
        lines = [
            f"## Supplier average return rate {start} ~ {end}",
            f"- supplier_id: {supplier_id}",
            f"- sku_count: {len(target_sku_ids)}",
            f"- sold_count: {total_sold}",
            f"- ordered_count: {total_ordered}",
            f"- denominator_count: {total_denominator}",
            f"- denominator_source: {denominator_source}",
            f"- return_count: {total_returns}",
            f"- average_return_rate: {display_avg}",
            f"- sold_based_return_rate: {display_sold_avg}",
            f"- order_based_return_rate: {display_order_avg}",
            f"- average_sku_return_rate: {display_sku_avg}",
            "",
            "| SKU | Sold Count | Ordered Count | Denominator | Source | Return Count | Return Rate |",
            "|---|---|---|---|---|---|---|",
        ]
        for sku_id, row in per_sku.items():
            display_rate = "-" if row["return_rate"] is None else f"{row['return_rate']:.6f}"
            lines.append(
                f"| {sku_id} | {row['sold_count']} | {row['ordered_count']} | "
                f"{row['denominator_count']} | {row['denominator_source']} | "
                f"{row['return_count']} | {display_rate} |"
            )

        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_returns(
        self,
        supplier_id: Optional[str] = None,
        sku_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """查看退货记录，可按 supplier_id / sku_id / 日期范围过滤。"""
        if sku_id:
            self._validate_sku_list([sku_id])
        sku_ids = self._filter_allowed_skus([sku_id]) if sku_id else []
        if sku_id and not sku_ids:
            raise ValueError("Unable to find legal sku")
        start, end = self._validate_dates(start_date, end_date, allow_none=True)

        records = self.record_manager.read_returns(
            supplier_id=supplier_id,
            sku_id=sku_id,
            start_date=start,
            end_date=end,
        )

        payload = [
            {
                "supplier_id": r.supplier_id,
                "sku_id": r.sku_id,
                "date": r.date,
            }
            for r in records
        ]

        lines = [
            f"## Return records ({len(payload)} rows)",
            f"- filters: supplier_id={supplier_id}, sku_id={sku_id}, start={start}, end={end}",
            "",
            "| # | Supplier | SKU | Date |",
            "|---|---|---|---|",
        ]
        for idx, row in enumerate(payload):
            lines.append(
                f"| {idx} | {row['supplier_id']} | {row['sku_id']} | {row['date']} |"
            )

        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_sku_prices(self, sku_ids: List[str]) -> Dict[str, Any]:
        if sku_ids is not None:
            self._validate_sku_list(sku_ids, allow_empty=True)
        target_ids = self._filter_allowed_skus(sku_ids)
        if sku_ids is not None and not target_ids:
            raise ValueError("Unable to find legal sku")

        prices = {}
        for sku_id in target_ids:
            sku_obj = self.skus_id_map.get(sku_id)
            if sku_obj:
                prices[sku_id] = sku_obj.price
        lines = [
            "## Current prices",
            "",
            "| SKU | Price |",
            "|---|---|",
        ]
        for sku_id, price in list(prices.items()):
            lines.append(f"| {sku_id} | {price} |")
        return self._wrap_tool_result(prices, "\n".join(lines))

    def view_sku_inventory_cost(self, sku_ids: List[str]) -> Dict[str, Any]:
        """
        查询当前库存中指定 SKU 的平均购入成本和平均入库时间。

        Args:
            sku_ids: 要查询的 SKU ID 列表

        Returns:
            每个 SKU 的库存数量、平均购入成本、平均入库天数
        """
        if sku_ids is not None:
            self._validate_sku_list(sku_ids, allow_empty=True)
        target_ids = self._filter_allowed_skus(sku_ids)
        if sku_ids is not None and not target_ids:
            raise ValueError("Unable to find legal sku")

        result = {}
        lines = [
            "## SKU Inventory Cost & Warehousing Time",
            "",
            "| SKU | Qty | Avg Buy Cost | Avg Days in Stock | Min Buy Cost | Max Buy Cost |",
            "|---|---|---|---|---|---|",
        ]

        for sku_id in target_ids:
            items = self.inventory.items_by_sku.get(sku_id, [])
            if not items:
                result[sku_id] = {
                    "quantity": 0,
                    "avg_buy_cost": 0.0,
                    "avg_days_in_stock": 0.0,
                    "min_buy_cost": 0.0,
                    "max_buy_cost": 0.0,
                }
                lines.append(f"| {sku_id} | 0 | - | - | - | - |")
                continue

            buy_prices = [item.buy_price for item in items]
            days_in_stock = [(self.current_date - item.begin_time).days for item in items]

            avg_buy_cost = sum(buy_prices) / len(buy_prices)
            avg_days = sum(days_in_stock) / len(days_in_stock)
            min_buy_cost = min(buy_prices)
            max_buy_cost = max(buy_prices)

            result[sku_id] = {
                "quantity": len(items),
                "avg_buy_cost": round(avg_buy_cost, 4),
                "avg_days_in_stock": round(avg_days, 1),
                "min_buy_cost": round(min_buy_cost, 4),
                "max_buy_cost": round(max_buy_cost, 4),
            }
            lines.append(
                f"| {sku_id} | {len(items)} | {avg_buy_cost:.2f} | {avg_days:.1f} | {min_buy_cost:.2f} | {max_buy_cost:.2f} |"
            )

        return self._wrap_tool_result(result, "\n".join(lines))

    def modify_sku_price(
        self,
        sku_id: str,
        new_price: float,
    ) -> Dict[str, Any]:
        if not isinstance(new_price, (int, float)):
            raise ValueError("new_price must be a number")
        if new_price <= 0:
            raise ValueError("new_price must be positive")
        sku_obj = self.skus_id_map.get(sku_id)
        if sku_obj is None:
            raise ValueError(f"SKU {sku_id} 不存在")
        old_price = sku_obj.price
        sku_obj.set_price(new_price)
        formatted = f"Updated SKU {sku_id} price: {old_price} -> {new_price}"
        return self._wrap_tool_result(
            {"sku_id": sku_id, "old_price": old_price, "new_price": new_price},
            formatted,
        )

    def _news_tools_enabled(self) -> bool:
        return bool(getattr(self, "news_manager", None) and self.config.get("enable_new"))

    def _news_unavailable_result(self, tool_name: str) -> Dict[str, Any]:
        message = f"{tool_name} is unavailable because news is disabled for this config."
        return self._wrap_tool_result(
            {"error": message, "status": "unavailable", "tool": tool_name},
            message,
        )

    def view_today_news(self) -> Dict[str, Any]:
        if not self._news_tools_enabled():
            return self._news_unavailable_result("view_today_news")
        news = self.news_manager.get_today_news()
        payload = [{"id": n.get("id") or n.get("record_id"), "record_id": n.get("record_id"), "title": n.get("title", "")} for n in news]
        lines = ["## Today's news"]
        for item in payload:
            lines.append(f"- {item['id']}: {item['title']}")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_news_detail(self, news_id: str) -> Dict[str, Any]:
        if not isinstance(news_id, str) or not news_id.strip():
            raise ValueError("news_id is required")
        if not self._news_tools_enabled():
            return self._news_unavailable_result("view_news_detail")
        news = self.news_manager.fetch_news_detail(self.record_manager, news_id, current_date=self.current_date)
        if not news:
            raise ValueError(f"News id not found: {news_id}")
        return self._wrap_tool_result(news, json.dumps(news, ensure_ascii=False, indent=2, default=str))

    def view_news_history(self, start_date: str, end_date: str) -> Dict[str, Any]:
        if not self._news_tools_enabled():
            return self._news_unavailable_result("view_news_history")
        start, end = self._validate_dates(start_date, end_date, allow_none=False)

        payload = self.news_manager.fetch_news_history(self.record_manager, start, end)
        lines = [f"## News {start} ~ {end} ({len(payload)} items)"]
        for item in payload:
            lines.append(f"- {item['id']}: {item['title']}")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def add_note(self, content: str) -> Dict[str, Any]:
        """
        Add a note to the store's note system.
        
        Args:
            content: The content of the note to add.
            
        Returns:
            Dict containing the note information.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Note content must be a non-empty string")
        
        note = {
            "id": uuid4().hex,
            "content": content.strip(),
            "date": self.current_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
        }
        
        self.notes.append(note)
        
        formatted = f"Note added successfully.\n- ID: {note['id']}\n- Date: {note['date']}\n- Content: {content.strip()}"
        return self._wrap_tool_result(note, formatted)

    def view_notes(self) -> Dict[str, Any]:
        """
        View all notes stored in the system.
        
        Returns:
            Dict containing all notes.
        """
        payload = {
            "total_notes": len(self.notes),
            "notes": self.notes,
        }
        
        if not self.notes:
            formatted = "No notes found."
        else:
            lines = [f"## Notes ({len(self.notes)} total)"]
            for idx, note in enumerate(self.notes, 1):
                lines.append(f"\n### Note {idx}")
                lines.append(f"- ID: {note['id']}")
                lines.append(f"- Date: {note['date']}")
                lines.append(f"- Content: {note['content']}")
            formatted = "\n".join(lines)
        
        return self._wrap_tool_result(payload, formatted)

    def remove_note(self, note_id: str) -> Dict[str, Any]:
        """
        Remove a note by its ID.

        Args:
            note_id: The ID of the note to remove.

        Returns:
            Dict containing the removal result.
        """
        if not isinstance(note_id, str) or not note_id.strip():
            raise ValueError("note_id must be a non-empty string")

        original_count = len(self.notes)
        self.notes = [n for n in self.notes if n["id"] != note_id.strip()]
        removed = original_count - len(self.notes)

        if removed == 0:
            formatted = f"No note found with ID: {note_id}"
        else:
            formatted = f"Note removed successfully (ID: {note_id}). {len(self.notes)} notes remaining."

        return self._wrap_tool_result(
            {"note_id": note_id, "removed": removed, "remaining_notes": len(self.notes)},
            formatted,
        )

    _ALLOWED_EXEC_IMPORT_MODULES = {
        "collections",
        "datetime",
        "decimal",
        "functools",
        "itertools",
        "json",
        "math",
        "operator",
        "statistics",
    }
    _INTERNAL_EXEC_IMPORT_MODULES = {
        # datetime.strptime lazily imports this stdlib helper.
        "_strptime",
        # _strptime imports time internally.
        "time",
    }

    @classmethod
    def _validate_execute_code_ast(cls, code: str) -> None:
        """Reject Python constructs that can escape the constrained code sandbox."""
        banned_nodes = (
            ast.Global,
            ast.Nonlocal,
            ast.ClassDef,
            ast.AsyncFunctionDef,
            ast.With,
            ast.AsyncWith,
            ast.Raise,
        )
        banned_names = {
            "__builtins__",
            "__import__",
            "eval",
            "exec",
            "compile",
            "open",
            "globals",
            "locals",
            "vars",
            "dir",
            "help",
            "input",
            "super",
            "place_order",
            "modify_sku_price",
        }

        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, banned_nodes):
                raise ValueError(f"Unsupported code construct: {type(node).__name__}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".", 1)[0]
                    if root_module not in cls._ALLOWED_EXEC_IMPORT_MODULES:
                        raise ValueError(f"Unsupported import module in execute_code: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                if node.level != 0 or not node.module:
                    raise ValueError("Unsupported relative import in execute_code")
                root_module = node.module.split(".", 1)[0]
                if root_module not in cls._ALLOWED_EXEC_IMPORT_MODULES:
                    raise ValueError(f"Unsupported import module in execute_code: {node.module}")
                if any(alias.name == "*" for alias in node.names):
                    raise ValueError("Unsupported star import in execute_code")
            if isinstance(node, ast.Name):
                if node.id in banned_names or "__" in node.id:
                    raise ValueError(f"Unsupported name in execute_code: {node.id}")
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_") or "__" in node.attr:
                    raise ValueError(f"Unsupported attribute access in execute_code: {node.attr}")

    class _CodeToolProxy:
        """Callable proxy for tools invoked from inside execute_code."""

        _POSITIONAL_ARG_NAMES = {
            "view_sales_profit_history": ["sku_ids", "start_date", "end_date", "supplier_id"],
            "view_current_date_supplier_prices": ["sku_ids"],
            "view_supplier_price_history": ["sku_id", "start_date", "end_date", "supplier_id"],
            "view_sku_prices": ["sku_ids"],
            "view_sku_inventory_cost": ["sku_ids"],
            "set_shelf_skus": ["sku_ids"],
            "view_sku_avg_ratings": ["sku_ids", "start_date", "end_date"],
            "view_supplier_returns_avg_rate": ["supplier_id", "start_date", "end_date", "sku_ids"],
            "view_news_detail": ["news_id"],
            "view_news_history": ["start_date", "end_date"],
        }

        __slots__ = ("_env", "_name", "_fn", "_json_safe", "_collector")

        def __init__(self, env, name: str, fn, json_safe, collector: Optional[List[Dict[str, Any]]] = None) -> None:
            self._env = env
            self._name = name
            self._fn = fn
            self._json_safe = json_safe
            self._collector = collector

        def __call__(self, *args, **kwargs):
            start_time = time.time()
            call_args = self._format_call_args(args, kwargs)
            try:
                res = self._fn(*args, **kwargs)
            except Exception as e:
                err_msg = f"Error executing {self._name}: {type(e).__name__}: {e}"
                result = self._env._wrap_tool_result({"error": err_msg}, err_msg)
                self._env._log_tool_call(
                    self._name,
                    call_args,
                    result,
                    time.time() - start_time,
                    source="execute_code",
                    parent_tool="execute_code",
                )
                self._record_inner_call(call_args, result, success=False)
                raise

            self._env._log_tool_call(
                self._name,
                call_args,
                res,
                time.time() - start_time,
                source="execute_code",
                parent_tool="execute_code",
            )
            self._record_inner_call(call_args, res, success=True)
            if isinstance(res, dict) and "result" in res:
                return self._json_safe(res["result"])
            return self._json_safe(res)

        def __repr__(self) -> str:
            return f"<execute_code tool {self._name}>"

        def _format_call_args(self, args, kwargs) -> Dict[str, Any]:
            call_args = dict(kwargs)
            positional_names = self._POSITIONAL_ARG_NAMES.get(self._name, [])
            extra_args = []
            for idx, value in enumerate(args):
                if idx < len(positional_names) and positional_names[idx] not in call_args:
                    call_args[positional_names[idx]] = value
                else:
                    extra_args.append(value)
            if extra_args:
                call_args["args"] = extra_args
            return call_args

        def _record_inner_call(self, args: Dict[str, Any], result: Any, success: bool) -> None:
            if self._collector is None:
                return
            self._collector.append(
                {
                    "tool_name": self._name,
                    "args": self._json_safe(args),
                    "result": self._json_safe(result),
                    "success": success,
                }
            )

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        在沙盒中执行 Python 代码，只能调用只读环境工具函数进行数据分析。
        可用函数：view_funds_and_date, view_inventory,
                  view_sales_profit_history, view_current_date_supplier_prices,
                  view_supplier_price_history, view_sku_prices,
                  view_sku_inventory_cost, view_shelf_status, view_notes。
        启用 review 时还可用 view_sku_avg_ratings,
                  view_supplier_returns_avg_rate。
        启用 news 时还可用 view_today_news, view_news_detail,
                  view_news_history。
        不允许在 execute_code 内调用 place_order、modify_sku_price 或 set_shelf_skus；
        这些写操作必须通过普通工具调用执行。
        函数返回值自动提取 result 字段，直接可用。
        """
        import io
        import contextlib

        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if len(code) > 20000:
            raise ValueError("code is too long; keep execute_code snippets under 20000 characters")
        validation_error = None
        try:
            self._validate_execute_code_ast(code)
        except Exception as e:
            validation_error = f"{type(e).__name__}: {e}"
            return self._wrap_tool_result(
                {"stdout": "", "result": None, "error": validation_error},
                f"## Code Execution Result\n### Error\n```\n{validation_error}\n```",
            )

        code_tools = {
            "view_funds_and_date": self.view_funds_and_date,
            "view_inventory": self.view_inventory,
            "view_sales_profit_history": self.view_sales_profit_history,
            "view_current_date_supplier_prices": self.view_current_date_supplier_prices,
            "view_supplier_price_history": self.view_supplier_price_history,
            "view_sku_prices": self.view_sku_prices,
            "view_sku_inventory_cost": self.view_sku_inventory_cost,
            "view_shelf_status": self.view_shelf_status,
            "view_notes": self.view_notes,
        }

        if getattr(self, "review_manager", None) and self.review_manager.enabled:
            code_tools["view_sku_avg_ratings"] = self.view_sku_avg_ratings
            code_tools["view_supplier_returns_avg_rate"] = self.view_supplier_returns_avg_rate

        if self._news_tools_enabled():
            code_tools["view_today_news"] = self.view_today_news
            code_tools["view_news_detail"] = self.view_news_detail
            code_tools["view_news_history"] = self.view_news_history

        inner_tool_calls: List[Dict[str, Any]] = []
        wrapped_functions = {
            name: self._CodeToolProxy(self, name, fn, self._json_safe, inner_tool_calls)
            for name, fn in code_tools.items()
        }

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level != 0:
                raise ImportError("Relative imports are not allowed in execute_code")
            root_module = name.split(".", 1)[0]
            allowed_modules = self._ALLOWED_EXEC_IMPORT_MODULES | self._INTERNAL_EXEC_IMPORT_MODULES
            if root_module not in allowed_modules:
                raise ImportError(f"Import module {name} is not allowed in execute_code")
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "filter": filter, "float": float, "int": int,
            "isinstance": isinstance, "len": len, "list": list, "map": map,
            "max": max, "min": min, "print": print, "range": range, "round": round,
            "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
            "zip": zip, "reversed": reversed, "__import__": safe_import,
        }

        namespace = {
            "__builtins__": safe_builtins,
            "funds": self.funds,
            "current_date": self.current_date.isoformat()
            if hasattr(self.current_date, "isoformat")
            else str(self.current_date),
            "rent": self.rent,
            "sku_ids": list(self.skus_id_map.keys()),
            **wrapped_functions,
            "json": json,
            "math": __import__("math"),
            "datetime": datetime,
            "date": date,
            "timedelta": timedelta,
            "defaultdict": defaultdict,
        }

        stdout_capture = io.StringIO()
        result_value = None
        error = None

        try:
            with contextlib.redirect_stdout(stdout_capture):
                try:
                    result_value = eval(compile(code, "<execute_code>", "eval"), namespace)
                except SyntaxError:
                    exec(compile(code, "<execute_code>", "exec"), namespace)
                    result_value = namespace.get("result", namespace.get("output", None))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        stdout_output = stdout_capture.getvalue()

        payload = {
            "stdout": stdout_output,
            "result": self._json_safe(result_value) if result_value is not None else None,
            "error": error,
            "tool_calls": inner_tool_calls,
        }

        lines = ["## Code Execution Result"]
        if stdout_output:
            stdout_text = stdout_output.rstrip()
            if len(stdout_text) > 10000:
                stdout_text = stdout_text[:10000] + "\n... (truncated)"
            lines.append(f"### Output\n```\n{stdout_text}\n```")
        if result_value is not None:
            if isinstance(result_value, (dict, list)):
                result_str = json.dumps(self._json_safe(result_value), indent=2, ensure_ascii=False, default=str)
            else:
                result_str = str(result_value)
            if len(result_str) > 5000:
                result_str = result_str[:5000] + "\n... (truncated)"
            lines.append(f"### Return Value\n```\n{result_str}\n```")
        if error:
            lines.append(f"### Error\n```\n{error}\n```")
        if len(lines) == 1:
            lines.append("(No output)")

        return self._wrap_tool_result(payload, "\n".join(lines))

    def place_order(
        self,
        items: List[Dict[str, Any]],
        supplier_id: str,
    ) -> Dict[str, Any]:
        """
        Place one order containing multiple SKUs (items = [{sku_id, quantity}, ...]).
        supplier_id is required and every ordered SKU must have a supplier quote for today.
        """
        if not isinstance(items, list):
            raise ValueError("items must be an array")
        if not items:
            raise ValueError("items cannot be empty")
        if not isinstance(supplier_id, str) or not supplier_id.strip():
            raise ValueError("supplier_id is required")
        supplier_id = supplier_id.strip()

        normalized_items: List[Dict[str, Any]] = []
        for idx, line in enumerate(items):
            if not isinstance(line, dict):
                raise ValueError(f"items[{idx}] must be an object")

            sku_id = line.get("sku_id")
            qty = line.get("quantity")

            if sku_id is None or qty is None:
                raise ValueError(f"items[{idx}] must include sku_id and quantity")

            if not isinstance(sku_id, str) or not sku_id.strip():
                raise ValueError(f"items[{idx}].sku_id is invalid")

            if not isinstance(qty, int):
                raise ValueError(f"items[{idx}].quantity must be an integer")

            if qty <= 0:
                raise ValueError(f"items[{idx}].quantity must be a positive integer")

            if sku_id not in self.skus_id_map:
                raise ValueError(f"Unknown SKU: {sku_id}")

            normalized_items.append({"sku_id": sku_id, "quantity": qty})
            
        if supplier_id not in self.supplier_manager.get_available_suppliers():
            raise ValueError(f"Unknown supplier: {supplier_id}")

        line_plans = []
        total_cost = 0.0
        total_quantity = 0
        supplier_used_set = set()
        for line in normalized_items:
            sku_id = line["sku_id"]
            qty = line["quantity"]
            today_quotes = self.supplier_manager.get_sku_date(sku_id, self.current_date)
            chosen = next((q for q in today_quotes if q.get("supplier_id") == supplier_id), None)
            if chosen is None:
                raise ValueError(f"No quote found for supplier {supplier_id} and SKU {sku_id} on {self.current_date}")

            unit_price_raw = chosen.get("price")
            if not isinstance(unit_price_raw, (int, float)):
                raise ValueError(f"Invalid quote price for supplier {supplier_id} and SKU {sku_id} on {self.current_date}")
            unit_price = float(unit_price_raw)
            if unit_price < 0:
                raise ValueError(f"Quote price must be non-negative for supplier {supplier_id} and SKU {sku_id}")

            supplier_used = chosen["supplier_id"]
            supplier_used_set.add(supplier_used)
            quantity_cost = unit_price * qty
            total_cost += quantity_cost
            total_quantity += qty
            line_plans.append(
                {
                    "sku_id": sku_id,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "supplier_id": supplier_used,
                    "line_cost": quantity_cost,
                    "quality_score": self.supplier_manager.get_quality_score(
                        supplier_id=supplier_used,
                        sku_id=sku_id,
                        target_date=self.current_date,
                    ),
                    "transport_range": self.supplier_manager.get_transport_range(
                        supplier_id=supplier_used,
                        sku_id=sku_id,
                        target_date=self.current_date,
                    ),
                }
            )

        configured_max_units = self.config.get("max_order_units_per_order")
        if configured_max_units is not None:
            max_order_units = int(configured_max_units)
        elif self.inventory.capacity is not None:
            max_order_units = max(int(self.inventory.capacity), 10000)
        else:
            max_order_units = 100000
        if max_order_units > 0 and total_quantity > max_order_units:
            raise ValueError(
                f"Order quantity {total_quantity} exceeds max_order_units_per_order {max_order_units}"
            )

        if self.funds < total_cost:
            return self._wrap_tool_result(
                {"error": "Insufficient funds", "required": total_cost, "available": self.funds},
                f"Insufficient funds to place the order. Required: {total_cost:.2f}, Available: {self.funds:.2f}",
            )

        if getattr(self, "quality_first_guard", None):
            quality_first_result = self.quality_first_guard.before_place_order(
                normalized_items,
                supplier_id,
            )
            if quality_first_result is not None:
                return quality_first_result

        merchandises: List[Merchandise] = []
        ordered_sku_map: Dict[SKU, int] = {}
        line_results = []
        delivery_days: Optional[int] = None
        arrival_date: Optional[date] = None

        for line in line_plans:
            sku_id = line["sku_id"]
            qty = line["quantity"]
            unit_price = line["unit_price"]
            supplier_used = line["supplier_id"]
            quality_score = line["quality_score"]
            transport_range = line["transport_range"]
            if transport_range:
                shipping_days = random.randint(int(transport_range[0]), int(transport_range[1]))
            else:
                shipping_days = random.randint(3, 7)
            arrival_line = self.current_date + timedelta(days=shipping_days)
            if arrival_date is None or arrival_line > arrival_date:
                delivery_days = shipping_days
                arrival_date = arrival_line
            line_results.append(
                {
                    "sku_id": sku_id,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "supplier_id": supplier_used,
                    "line_cost": line["line_cost"],
                    "quality_score": quality_score,
                    "sampled_shipping_days": shipping_days,
                    "sampled_arrival_date": arrival_line,
                }
            )

        order_supplier = supplier_id or (supplier_used_set.pop() if len(supplier_used_set) == 1 else "mixed")
        if delivery_days is None or arrival_date is None:
            delivery_days = random.randint(3, 7)
            arrival_date = self.current_date + timedelta(days=delivery_days)
        order_id = uuid4().hex

        for line in line_results:
            sku_id = line["sku_id"]
            qty = line["quantity"]
            unit_price = line["unit_price"]
            supplier_used = line["supplier_id"]
            sku_obj = self.skus_id_map[sku_id]
            shelf_life = sku_obj.promotion_day
            for _ in range(qty):
                merchandises.append(
                    Merchandise(
                        sku=sku_obj,
                        begin_time=arrival_date,
                        expired_time=arrival_date + timedelta(days=shelf_life),
                        buy_price=unit_price,
                        merch_id=uuid4().hex,
                        quality_score=line["quality_score"],
                        supplier_id=supplier_used,
                    )
                )

            ordered_sku_map[sku_obj] = ordered_sku_map.get(sku_obj, 0) + qty
            line["shipping_days"] = delivery_days
            line["arrival_date"] = arrival_date

        order = Order(
            order_id=order_id,
            ordered_sku=ordered_sku_map,
            customer_id=order_supplier,
            items=merchandises,
            created_at=self.current_date,
            delivery_time=delivery_days,
            cost=total_cost,
            supplier_order_recorded=True,
        )

        # 扣除资金并添加订单
        self.funds -= total_cost
        self.order_manager.add_order(order)

        # 记录供应商订单
        supplier_order_record = SupplierOrderRecord(
            supplier_id=order_supplier,
            order_date=self.current_date,
            arrival_date=arrival_date,
            shipping_days=delivery_days,
            items={sku.sku_id: qty for sku, qty in ordered_sku_map.items()},
            cost=total_cost,
        )
        self.record_manager.add_supplier_order(supplier_order_record)

        # 记录每个商品的生命周期
        lifecycle_records = []
        for merch in merchandises:
            lifecycle_records.append(
                ProductLifecycleRecord(
                    merch_id=merch.merch_id,
                    sku_id=merch.sku.sku_id,
                    supplier_id=merch.supplier_id,
                    order_id=order_id,
                    buy_price=merch.buy_price,
                    order_date=self.current_date,
                    arrival_date=merch.begin_time,
                    expired_time=merch.expired_time,
                )
            )
        self.record_manager.batch_add_product_lifecycle(lifecycle_records)

        formatted_lines = [
            f"**Order placed**",
            f"- ID: `{order_id}`",
            f"- Expected arrival: {arrival_date}",
            f"- Supplier: {order_supplier}",
            f"- Total cost: {total_cost:.2f}",
            f"- Funds after: {self.funds:.2f}",
            "",
            "| SKU | Qty | Unit Price | Supplier | Line Cost |",
            "|---|---|---|---|---|",
        ]
        for line in line_results:
            formatted_lines.append(
                f"| {line['sku_id']} | {line['quantity']} | {line['unit_price']} | {line['supplier_id']} | {line['line_cost']:.2f} |"
            )

        result = {
            "order_id": order_id,
            "arrival_date": arrival_date,
            "supplier_id": order_supplier,
            "total_cost": total_cost,
            "funds_after": self.funds,
            "lines": line_results,
        }
        return self._wrap_tool_result(result, "\n".join(formatted_lines))
    
    def end_of_day(self) -> Dict[str, Any]:
        return self.step()
    
    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """
        保存当前环境状态到 checkpoint 文件。
        
        Args:
            checkpoint_path: checkpoint 文件路径
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化 checkpoint 数据
        checkpoint_data = {
            "funds": self.funds,
            "current_date": self.current_date.isoformat(),
            "sku_prices": {sku_id: sku.price for sku_id, sku in self.skus_id_map.items()},
            "shelf_sku_ids": sorted(self.shelf_sku_ids),
            "config": self.config,  # 保存配置以便恢复
        }
        
        # 调用各个 manager 的 save_checkpoint 方法
        checkpoint_data = self.inventory.save_checkpoint(checkpoint_data)
        checkpoint_data = self.order_manager.save_checkpoint(checkpoint_data)
        
        # 保存 news_manager 状态（如果启用）
        if self.news_manager and self.config.get("enable_new"):
            checkpoint_data = self.news_manager.save_checkpoint(checkpoint_data)
        
        # 保存 record_manager 数据库到 SQL 文件
        checkpoint_dir = checkpoint_path.parent
        sql_dump_path = checkpoint_dir / f"{checkpoint_path.stem}_records.sql"
        self.record_manager.dump_to_sql(sql_dump_path)
        checkpoint_data["record_manager_sql_path"] = str(sql_dump_path.relative_to(checkpoint_dir))
        
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    @classmethod
    def recover_from_checkpoint(cls, checkpoint_path: Path) -> "RetailEnvironment":
        """
        从 checkpoint 文件恢复环境状态。
        
        Args:
            checkpoint_path: checkpoint 文件路径
            
        Returns:
            恢复后的 RetailEnvironment 实例
        """
        with checkpoint_path.open("r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        
        config = checkpoint_data["config"]
        env = cls(config)
        
        # 恢复基本状态
        env.funds = checkpoint_data["funds"]
        env.current_date = datetime.strptime(checkpoint_data["current_date"], "%Y-%m-%d").date()
        
        # 恢复 SKU 价格
        sku_prices = checkpoint_data.get("sku_prices", {})
        for sku_id, price in sku_prices.items():
            if sku_id in env.skus_id_map:
                env.skus_id_map[sku_id].set_price(price)
        env.shelf_sku_ids = {
            sku_id for sku_id in checkpoint_data.get("shelf_sku_ids", [])
            if sku_id in env.skus_id_map
        }
        
        # 调用各个 manager 的 recover_from_checkpoint 方法
        env.inventory.recover_from_checkpoint(checkpoint_data, env.skus_id_map)
        env.order_manager.recover_from_checkpoint(checkpoint_data, env.skus_id_map)
        
        # 恢复 news_manager 状态（如果启用）
        if env.news_manager and env.config.get("enable_new"):
            env.news_manager.recover_from_checkpoint(checkpoint_data)
        
        # 恢复 record_manager 数据库（从 SQL 文件）
        record_manager_sql_path = checkpoint_data.get("record_manager_sql_path")
        if record_manager_sql_path:
            sql_file_path = checkpoint_path.parent / record_manager_sql_path
            if sql_file_path.exists():
                env.record_manager.restore_from_sql(sql_file_path)
                env.logger.info(f"Record manager database restored from {sql_file_path}")
            else:
                env.logger.warning(f"Record manager SQL file not found: {sql_file_path}")
        
        env.logger.info(f"Environment recovered from checkpoint {checkpoint_path}")
        return env
        

    # ---------- MCP 工具声明 ----------
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        tools: Dict[str, Dict[str, Any]] = {
            "place_order": {
                "name": "place_order",
                "description": "Place an order: multiple SKUs supported; supplier_id required.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku_id": {"type": "string", "description": "SKU to purchase"},
                                    "quantity": {"type": "integer", "minimum": 1, "description": "Units to order"},
                                },
                                "required": ["sku_id", "quantity"],
                            },
                            "description": "Order lines array",
                        },
                        "supplier_id": {"type": "string", "description": "Supplier id to use"},
                    },
                    "required": ["items", "supplier_id"],
                },
            },
            "view_sales_profit_history": {
                "name": "view_sales_profit_history",
                "description": (
                    "View daily SKU sales, average selling price, and realized net profit. "
                    "Optionally restrict sold lifecycle rows and returns to one supplier."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Non-empty list of legal SKU ids to query. Do not pass an empty list.",
                        },
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD or MM/DD/YY"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD or MM/DD/YY"},
                        "supplier_id": {
                            "type": "string",
                            "description": "Optional supplier id; omit to aggregate all suppliers.",
                        },
                    },
                    "required": ["sku_ids", "start_date", "end_date"],
                },
            },
            "view_current_date_supplier_prices": {
                "name": "view_current_date_supplier_prices",
                "description": "View supplier quotes for the current date.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Non-empty list of legal SKU ids to check. Do not pass an empty list.",
                        },
                    },
                    "required": ["sku_ids"],
                },
            },
            "view_supplier_price_history": {
                "name": "view_supplier_price_history",
                "description": "Query historical supplier price records for one SKU, optionally filtered by supplier.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "supplier_id": {"type": "string", "description": "Optional supplier id"},
                        "sku_id": {"type": "string", "description": "SKU id to query"},
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD or MM/DD/YY"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD or MM/DD/YY"},
                    },
                    "required": ["sku_id", "start_date", "end_date"],
                },
            },
            "view_inventory": {
                "name": "view_inventory",
                "description": "View current inventory, waiting capacity items, incoming quantities, and open supplier orders.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "view_shelf_status": {
                "name": "view_shelf_status",
                "description": "View current shelf assortment and capacity. Only on-shelf SKUs can sell.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "set_shelf_skus": {
                "name": "set_shelf_skus",
                "description": "Replace the current shelf assortment with the provided SKU list. SKUs not on shelf cannot sell.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Legal SKU ids to put on shelf. The list replaces the entire shelf assortment.",
                        },
                    },
                    "required": ["sku_ids"],
                },
            },
            "view_funds_and_date": {
                "name": "view_funds_and_date",
                "description": "View current date and funds balance.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "view_sku_prices": {
                "name": "view_sku_prices",
                "description": "View current selling prices for specified SKUs. Pass a non-empty list of legal SKU ids; do not pass an empty list.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Non-empty list of legal SKU ids. In execute_code, use the provided sku_ids variable or a non-empty slice such as sku_ids[:10].",
                        }
                    },
                    "required": ["sku_ids"],
                },
            },
            "modify_sku_price": {
                "name": "modify_sku_price",
                "description": "Update the selling price of a SKU.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_id": {"type": "string", "description": "SKU id to update"},
                        "new_price": {"type": "number", "description": "New price"},
                    },
                    "required": ["sku_id", "new_price"],
                },
            },
            "end_today": {
                "name": "end_today",
                "description": "End today's operations and advance the store to the next day (calls step).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "add_note": {
                "name": "add_note",
                "description": "Add a note to the store's note system for recording important information, observations, or reminders.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content of the note to add.",
                        },
                    },
                    "required": ["content"],
                },
            },
            "view_notes": {
                "name": "view_notes",
                "description": "View all notes stored in the system.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "remove_note": {
                "name": "remove_note",
                "description": "Remove a note by its ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "note_id": {
                            "type": "string",
                            "description": "The ID of the note to remove.",
                        },
                    },
                    "required": ["note_id"],
                },
            },
            "view_sku_inventory_cost": {
                "name": "view_sku_inventory_cost",
                "description": "查询当前库存中指定 SKU 的平均购入成本和平均入库时间（即已在仓库存放的天数）。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Non-empty list of legal SKU ids to query. Do not pass an empty list.",
                        },
                    },
                    "required": ["sku_ids"],
                },
            },
            "execute_code": {
                "name": "execute_code",
                "description": (
                    "Execute Python code in a constrained sandbox for read-only data analysis and environment tool calls. "
                    "Always available functions: view_funds_and_date(), view_inventory(), "
                    "view_sales_profit_history(sku_ids, start_date, end_date, supplier_id=None), "
                    "view_current_date_supplier_prices(sku_ids), "
                    "view_supplier_price_history(sku_id, start_date, end_date, supplier_id=None), "
                    "view_sku_prices(sku_ids), view_sku_inventory_cost(sku_ids), view_shelf_status(), "
                    "view_notes(). "
                    "When reviews are enabled, also available: view_sku_avg_ratings(sku_ids, start_date, end_date), "
                    "view_supplier_returns_avg_rate(supplier_id, start_date, end_date, sku_ids). "
                    "When news is enabled, also available: view_today_news(), view_news_detail(news_id), "
                    "view_news_history(start_date, end_date). "
                    "Mutation tools are not available inside execute_code: do not call place_order(), modify_sku_price(), or set_shelf_skus() in code. "
                    "Use place_order, modify_sku_price, and set_shelf_skus only as normal top-level tool calls after analysis. "
                    "Variables: funds, current_date, rent, sku_ids. "
                    "Libraries/objects: json, math, datetime, date, timedelta, defaultdict. "
                    "Allowed imports: datetime, collections, math, statistics, itertools, functools, decimal, operator, json. "
                    "Do not import OS, filesystem, subprocess, network, pickle, or importlib modules. "
                    "Each execute_code call is isolated: do not reuse variables such as inv, sales, result, or velocity from earlier code calls; recompute them in the same snippet. "
                    "current_date is a YYYY-MM-DD string; use date.fromisoformat(current_date) or datetime.strptime(current_date, '%Y-%m-%d') before date arithmetic. "
                    "Prefer simple loops over lambda/key functions to maximize compatibility with the sandbox validator. "
                    "Functions auto-extract result dict. Use print() to output. "
                    "For statements, set variable 'result' as return value.\n\n"
                    "Examples:\n"
                    "1) Find low-stock SKUs needing reorder:\n"
                    "inv = view_inventory()\n"
                    "low = [s for s in sku_ids if inv['inventory'].get(s,{}).get('quantity',0) + inv['inventory'].get(s,{}).get('waiting',0) + inv['inventory'].get(s,{}).get('incoming',0) < 5]\n"
                    "print(f'{len(low)} SKUs need reorder')\n"
                    "result = low\n\n"
                    "2) Compute sales velocity and profit margin:\n"
                    "today = date.fromisoformat(current_date)\n"
                    "start = str(today - timedelta(days=14))\n"
                    "sales = view_sales_profit_history(sku_ids[:10], start, current_date)\n"
                    "for s in sku_ids[:10]:\n"
                    "    d = sales.get(s,{})\n"
                    "    avg = d.get('total_units',0)/max(d.get('days',1),1) if d else 0\n"
                    "    print(f\"{s}: {avg:.1f}/day, net_profit={d.get('net_profit',0):.2f}, avg_price={d.get('avg_selling_price')}\")\n\n"
                    "3) Find cheapest supplier for target SKUs:\n"
                    "prices = view_current_date_supplier_prices(sku_ids[:5])\n"
                    "for s in sku_ids[:5]:\n"
                    "    quotes = [q for q in prices.get(s,[]) if q]\n"
                    "    best = None\n"
                    "    for q in quotes:\n"
                    "        if best is None or q['price'] < best['price']:\n"
                    "            best = q\n"
                    "    if best: print(f\"{s}: {best['supplier_id']} @ ${best['price']:.2f}\")"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute for data analysis",
                        },
                    },
                    "required": ["code"],
                },
            },
        }

        enable_news = self._news_tools_enabled()

        if enable_news:
            
            tools["view_today_news"] = {
                "name": "view_today_news",
                "description": "View today's news list (title + id).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }

            tools["view_news_detail"] = {
                "name": "view_news_detail",
                "description": "View a specific news detail by id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "news_id": {"type": "string", "description": "News id to query"},
                    },
                    "required": ["news_id"],
                },
            }

            tools["view_news_history"] = {
                "name": "view_news_history",
                "description": "View news in a date range (inclusive).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                    },
                    "required": ["start_date", "end_date"],
                },
            }

        enable_review = self.config.get("enable_review", False)

        if enable_review:
            tools["view_sku_avg_ratings"] = {
                "name": "view_sku_avg_ratings",
                "description": "Get average rating for SKUs within an optional date range.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "SKU 列表",
                        },
                        "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD 或 MM/DD/YY"},
                        "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD 或 MM/DD/YY"},
                    },
                    "required": ["sku_ids", "start_date", "end_date"],
                },
            }

            tools["view_supplier_returns_avg_rate"] = {
                "name": "view_supplier_returns_avg_rate",
                "description": "Get supplier-level average return rate for a supplier in a date range, optionally restricted to target SKUs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "supplier_id": {
                            "type": "string",
                            "description": "Supplier ID to evaluate.",
                        },
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Optional non-empty list of legal SKU ids. Omit to include all SKUs.",
                        },
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD or MM/DD/YY"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD or MM/DD/YY"},
                    },
                    "required": ["supplier_id", "start_date", "end_date"],
                },
            }


        supplier_quote_schema = {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "sku_id": {"type": "string"},
                "price": {"type": "number"},
            },
            "additionalProperties": True,
        }
        note_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "date": {"type": "string"},
                "timestamp": {"type": "string"},
            },
            "required": ["id", "content", "date", "timestamp"],
            "additionalProperties": True,
        }
        sku_number_map_schema = {
            "type": "object",
            "additionalProperties": {"type": ["number", "null"]},
        }
        order_item_schema = {
            "type": "object",
            "properties": {
                "sku_id": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["sku_id", "count"],
            "additionalProperties": True,
        }
        current_order_schema = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "ordered_sku": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                    "description": "Map of sku_id to ordered unit count.",
                },
                "customer_id": {
                    "type": "string",
                    "description": "Backward-compatible supplier id alias; prefer supplier_id.",
                },
                "supplier_id": {
                    "type": "string",
                    "description": "Supplier id for this open supplier order.",
                },
                "created_at": {"type": "string", "description": "Order creation date YYYY-MM-DD."},
                "delivery_time": {"type": "integer", "description": "Lead time in days."},
                "expected_delivery_date": {
                    "type": "string",
                    "description": "Expected arrival date YYYY-MM-DD.",
                },
                "days_until_arrival": {"type": ["integer", "null"]},
                "cost": {"type": "number"},
                "total_items": {"type": "integer"},
                "items": {
                    "type": "array",
                    "items": order_item_schema,
                    "description": "Order line items; each item uses count, not quantity.",
                },
            },
            "required": [
                "order_id",
                "ordered_sku",
                "customer_id",
                "supplier_id",
                "created_at",
                "delivery_time",
                "expected_delivery_date",
                "cost",
                "total_items",
                "items",
            ],
            "additionalProperties": True,
        }
        supplier_return_rate_schema = {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "sku_ids": {"type": "array", "items": {"type": "string"}},
                "average_return_rate": {"type": ["number", "null"]},
                "sold_based_return_rate": {"type": ["number", "null"]},
                "order_based_return_rate": {"type": ["number", "null"]},
                "average_sku_return_rate": {"type": ["number", "null"]},
                "sold_count": {"type": "integer"},
                "ordered_count": {"type": "integer"},
                "denominator_count": {"type": "integer"},
                "denominator_source": {"type": "string"},
                "return_count": {"type": "integer"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "per_sku": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "sku_id": {"type": "string"},
                            "sold_count": {"type": "integer"},
                            "ordered_count": {"type": "integer"},
                            "denominator_count": {"type": "integer"},
                            "denominator_source": {"type": "string"},
                            "return_count": {"type": "integer"},
                            "return_rate": {"type": ["number", "null"]},
                        },
                        "required": [
                            "sku_id",
                            "sold_count",
                            "ordered_count",
                            "denominator_count",
                            "denominator_source",
                            "return_count",
                            "return_rate",
                        ],
                        "additionalProperties": True,
                    },
                },
            },
            "required": [
                "supplier_id",
                "sku_ids",
                "average_return_rate",
                "sold_based_return_rate",
                "order_based_return_rate",
                "average_sku_return_rate",
                "sold_count",
                "ordered_count",
                "denominator_count",
                "denominator_source",
                "return_count",
                "start_date",
                "end_date",
                "per_sku",
            ],
            "additionalProperties": True,
        }
        output_schemas = {
            "place_order": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "arrival_date": {"type": "string"},
                    "supplier_id": {"type": "string"},
                    "total_cost": {"type": "number"},
                    "funds_after": {"type": "number"},
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku_id": {"type": "string"},
                                "quantity": {"type": "integer"},
                                "unit_price": {"type": "number"},
                                "supplier_id": {"type": "string"},
                                "line_cost": {"type": "number"},
                            },
                            "required": ["sku_id", "quantity", "unit_price", "supplier_id", "line_cost"],
                            "additionalProperties": True,
                        },
                    },
                    "error": {"type": "string"},
                    "required": {"type": "number"},
                    "available": {"type": "number"},
                },
                "additionalProperties": True,
            },
            "view_sales_profit_history": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "daily": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date": {"type": "string"},
                                    "selling_price": {"type": ["number", "null"]},
                                    "units_sold": {"type": "integer"},
                                    "net_profit": {"type": "number"},
                                },
                                "required": ["date", "selling_price", "units_sold", "net_profit"],
                                "additionalProperties": True,
                            },
                        },
                        "records": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "sku_id": {"type": "string"},
                                        "date": {"type": "string"},
                                        "move": {"type": "number"},
                                        "price": {"type": ["number", "null"]},
                                        "net_profit": {"type": "number"},
                                    },
                                    "required": ["sku_id", "date", "move", "price", "net_profit"],
                                    "additionalProperties": True,
                                },
                            },
                        },
                        "total_units": {"type": "number"},
                        "days": {"type": "integer"},
                        "net_profit": {"type": "number"},
                        "avg_selling_price": {"type": ["number", "null"]},
                    },
                    "required": ["daily", "records", "total_units", "days", "net_profit", "avg_selling_price"],
                    "additionalProperties": True,
                },
            },
            "view_current_date_supplier_prices": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": supplier_quote_schema},
            },
            "view_supplier_price_history": {"type": "array", "items": supplier_quote_schema},
            "view_inventory": {
                "type": "object",
                "properties": {
                    "inventory": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "quantity": {"type": "integer"},
                                "waiting": {"type": "integer"},
                                "incoming": {"type": "integer"},
                            },
                            "required": ["quantity", "waiting", "incoming"],
                            "additionalProperties": True,
                        },
                    },
                    "total_items": {"type": "integer"},
                    "total_skus": {"type": "integer"},
                    "waiting_items": {"type": "integer"},
                    "pending_order_count": {"type": "integer"},
                    "incoming_items": {"type": "integer"},
                    "incoming_by_sku": {"type": "object", "additionalProperties": {"type": "integer"}},
                    "pending_orders": {"type": "array", "items": current_order_schema},
                },
                "required": [
                    "inventory",
                    "total_items",
                    "total_skus",
                    "waiting_items",
                    "pending_order_count",
                    "incoming_items",
                    "incoming_by_sku",
                    "pending_orders",
                ],
            },
            "view_shelf_status": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "capacity": {"type": ["integer", "null"]},
                    "used": {"type": "integer"},
                    "available": {"type": ["integer", "null"]},
                    "shelf_skus": {"type": "array", "items": {"type": "string"}},
                    "on_shelf": {"type": "object", "additionalProperties": True},
                },
                "required": ["enabled", "capacity", "used", "available", "shelf_skus", "on_shelf"],
            },
            "view_funds_and_date": {
                "type": "object",
                "properties": {
                    "funds": {"type": "number"},
                    "current_date": {"type": "string"},
                },
                "required": ["funds", "current_date"],
            },
            "view_sku_prices": {"type": "object", "additionalProperties": {"type": "number"}},
            "set_shelf_skus": {
                "type": "object",
                "properties": {
                    "capacity": {"type": ["integer", "null"]},
                    "used": {"type": "integer"},
                    "shelf_skus": {"type": "array", "items": {"type": "string"}},
                    "added": {"type": "array", "items": {"type": "string"}},
                    "removed": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["capacity", "used", "shelf_skus", "added", "removed"],
            },
            "modify_sku_price": {
                "type": "object",
                "properties": {
                    "sku_id": {"type": "string"},
                    "old_price": {"type": "number"},
                    "new_price": {"type": "number"},
                },
                "required": ["sku_id", "old_price", "new_price"],
            },
            "end_today": {
                "type": "object",
                "properties": {
                    "funds": {"type": "number"},
                    "net_worth": {"type": "number"},
                    "current_date": {"type": "string"},
                    "money_earned": {"type": "number"},
                    "sales_by_sku": {"type": "object", "additionalProperties": {"type": "number"}},
                    "returns_by_sku": {"type": "object", "additionalProperties": {"type": "number"}},
                    "expired_discount_by_sku": {"type": "object", "additionalProperties": {"type": "number"}},
                    "inventory": {"type": "object"},
                    "pending_order_count": {"type": "integer"},
                    "incoming_items": {"type": "integer"},
                    "incoming_by_sku": {"type": "object", "additionalProperties": {"type": "integer"}},
                    "pending_orders": {"type": "array", "items": current_order_schema},
                },
                "additionalProperties": True,
            },
            "add_note": note_schema,
            "view_notes": {
                "type": "object",
                "properties": {
                    "total_notes": {"type": "integer"},
                    "notes": {"type": "array", "items": note_schema},
                },
                "required": ["total_notes", "notes"],
            },
            "remove_note": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "removed": {"type": "integer"},
                    "remaining_notes": {"type": "integer"},
                },
                "required": ["note_id", "removed", "remaining_notes"],
            },
            "view_sku_inventory_cost": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "quantity": {"type": "integer"},
                        "avg_buy_cost": {"type": "number"},
                        "avg_days_in_stock": {"type": "number"},
                        "min_buy_cost": {"type": "number"},
                        "max_buy_cost": {"type": "number"},
                    },
                    "required": [
                        "quantity",
                        "avg_buy_cost",
                        "avg_days_in_stock",
                        "min_buy_cost",
                        "max_buy_cost",
                    ],
                    "additionalProperties": True,
                },
            },
            "execute_code": {
                "type": "object",
                "properties": {
                    "stdout": {"type": "string"},
                    "result": {},
                    "error": {"type": ["string", "null"]},
                },
                "required": ["stdout", "result", "error"],
            },
            "view_today_news": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "record_id": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "view_news_detail": {"type": "object", "additionalProperties": True},
            "view_news_history": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "view_sku_avg_ratings": sku_number_map_schema,
            "view_supplier_returns_avg_rate": {
                **supplier_return_rate_schema,
            },
        }
        return_contracts = {
            "view_inventory": "{'inventory': {sku_id: {'quantity': int, 'waiting': int, 'incoming': int}}, 'total_items': int, 'total_skus': int, 'waiting_items': int, 'pending_order_count': int, 'incoming_items': int, 'incoming_by_sku': {sku_id: int}, 'pending_orders': [{'order_id': str, 'supplier_id': str, 'ordered_sku': {sku_id: count}, 'expected_delivery_date': 'YYYY-MM-DD', 'days_until_arrival': int, 'cost': number, 'total_items': int}]}",
            "view_shelf_status": "{'enabled': bool, 'capacity': int_or_null, 'used': int, 'available': int_or_null, 'shelf_skus': [str], 'on_shelf': {...}}",
            "view_sales_profit_history": "{sku_id: {'daily': [{'date': 'YYYY-MM-DD', 'selling_price': number_or_null, 'units_sold': int, 'net_profit': number}], 'records': {date: [{'sku_id': str, 'date': 'YYYY-MM-DD', 'move': int, 'price': number_or_null, 'net_profit': number}]}, 'total_units': int, 'days': int, 'net_profit': number, 'avg_selling_price': number_or_null}}",
            "view_current_date_supplier_prices": "{sku_id: [{'supplier_id': str, 'sku_id': str, 'price': number, ...}]}",
            "view_supplier_price_history": "[{'supplier_id': str, 'sku_id': str, 'date': str, 'price': number}]",
            "view_sku_prices": "{sku_id: current_price_number}. Input sku_ids must be a non-empty list of legal SKU ids; do not pass [].",
            "view_sku_inventory_cost": "{sku_id: {'quantity': int, 'avg_buy_cost': number, 'avg_days_in_stock': number, 'min_buy_cost': number, 'max_buy_cost': number}}",
            "view_funds_and_date": "{'funds': number, 'current_date': 'YYYY-MM-DD'}",
            "view_sku_avg_ratings": "{sku_id: rating_number_or_null}",
            "view_supplier_returns_avg_rate": "{'supplier_id': str, 'sku_ids': [str], 'average_return_rate': number_or_null, 'sold_based_return_rate': number_or_null, 'order_based_return_rate': number_or_null, 'average_sku_return_rate': number_or_null, 'sold_count': int, 'ordered_count': int, 'denominator_count': int, 'denominator_source': str, 'return_count': int, 'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD', 'per_sku': {sku_id: {'sold_count': int, 'ordered_count': int, 'denominator_count': int, 'denominator_source': str, 'return_count': int, 'return_rate': number_or_null}}}. average_return_rate uses sold_count when lifecycle sold rows exist, otherwise falls back to supplier_orders ordered_count synthesized by the historical data generator.",
            "view_today_news": "[{'id': str, 'record_id': str, 'title': str}]",
            "view_news_detail": "news detail object",
            "view_news_history": "[news_object, ...]",
            "execute_code": "{'stdout': str, 'result': any, 'error': str_or_null}",
            "place_order": "{'order_id': str, 'arrival_date': 'YYYY-MM-DD', 'supplier_id': str, 'total_cost': number, 'funds_after': number, 'lines': [...]} or {'status': 'quality_first_supplier_reconsideration_required', 'action_executed': false, 'selected_supplier_id': str, 'per_sku_quality_table': [{'sku_id': str, 'rank': int, 'supplier_id': str, 'average_review_rating': number, 'return_rate': number, 'quality_score': number, ...}], 'violations': [...]} or {'error': 'Insufficient funds', 'required': number, 'available': number}",
            "modify_sku_price": "{'sku_id': str, 'old_price': number, 'new_price': number}",
            "set_shelf_skus": "{'capacity': int_or_null, 'used': int, 'shelf_skus': [str], 'added': [str], 'removed': [str]}",
            "end_today": "{'funds': number, 'net_worth': number, 'current_date': 'YYYY-MM-DD', 'money_earned': number, 'sales_by_sku': {...}, 'inventory': {...}, 'pending_order_count': int, 'incoming_items': int, 'pending_orders': [...], ...}",
            "add_note": "{'id': str, 'content': str, 'date': 'YYYY-MM-DD', 'timestamp': str}",
            "view_notes": "{'total_notes': int, 'notes': [note_object, ...]}",
            "remove_note": "{'note_id': str, 'removed': int, 'remaining_notes': int}",
        }

        # 兼容旧的 OpenAI tool schema，并提供返回结构给 MCP/提示词消费。
        for meta in tools.values():
            name = meta["name"]
            output_schema = output_schemas.get(name, {})
            meta.setdefault("output_schema", output_schema)
            meta.setdefault("outputSchema", output_schema)
            if name in return_contracts:
                meta["description"] = (
                    f"{meta.get('description', '').rstrip()} "
                    f"Returns result as: {return_contracts[name]}."
                )
            meta.setdefault("parameters", meta["input_schema"])
        return tools
    
    def exec_tools(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        dispatch = {
            "place_order": self.place_order,
            "modify_sku_price": self.modify_sku_price,
            "set_shelf_skus": self.set_shelf_skus,
            "end_today": self.end_of_day,
            "view_sales_profit_history": self.view_sales_profit_history,
            "view_current_date_supplier_prices": self.view_current_date_supplier_prices,
            "view_supplier_price_history": self.view_supplier_price_history,
            "view_inventory": self.view_inventory,
            "view_sku_avg_ratings": self.view_sku_avg_ratings,
            "view_supplier_returns_avg_rate": self.view_supplier_returns_avg_rate,
            "view_funds_and_date": self.view_funds_and_date,
            "view_sku_prices": self.view_sku_prices,
            "view_today_news": self.view_today_news,
            "view_news_detail": self.view_news_detail,
            "view_news_history": self.view_news_history,
            "add_note": self.add_note,
            "view_notes": self.view_notes,
            "remove_note": self.remove_note,
            "view_sku_inventory_cost": self.view_sku_inventory_cost,
            "view_shelf_status": self.view_shelf_status,
            "execute_code": self.execute_code,
        }
        # 记录开始时间
        start_time = time.time()
        
        try:
            if tool_name not in dispatch:
                raise ValueError(f"未知工具：{tool_name}")
            
            result = dispatch[tool_name](**kwargs)
            
            # 验证返回格式
            if not isinstance(result, dict) or "result" not in result or "formatted" not in result:
                self.logger.warning(f"Tool {tool_name} returned unexpected format: {type(result)}")
                # 尝试包装结果
                if isinstance(result, dict):
                    result = self._wrap_tool_result(result, str(result))
                else:
                    result = self._wrap_tool_result({"value": result}, str(result))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = f"Error executing {tool_name}: {type(e).__name__}: {e}"
            result = self._wrap_tool_result({"error": err_msg}, err_msg)
        finally:
            # 计算耗时
            pass
            

        # Make the payload JSON safe for downstream logging
        if isinstance(result, dict) and "result" in result:
            result["result"] = self._json_safe(result.get("result"))

        self._log_tool_call(tool_name, kwargs, result, time.time() - start_time)
        elapsed_time = time.time() - start_time
        self.logger.debug(f"[工具耗时] {tool_name}: {elapsed_time:.4f} 秒")
            
        # 在调试模式下输出耗时信息
        # self.logger.info(f"[工具耗时] {tool_name}: {elapsed_time:.4f} 秒")
        
        return result
    

from datetime import date
from retail_environment import RetailEnvironment  # 注意根据你的文件名调整导入


def _snapshot_env_state_for_test(env: RetailEnvironment) -> Dict[str, Any]:
    """
    对环境做一次深度快照，用于和 checkpoint 恢复后的环境做对比。
    只包含可序列化的基础类型，方便直接做 == 比较。
    """
    state: Dict[str, Any] = {}

    # 基本信息
    state["funds"] = round(float(env.funds), 6)
    state["current_date"] = env.current_date.isoformat()

    # SKU 价格
    sku_prices: Dict[str, float] = {}
    for sku_id, sku in env.skus_id_map.items():
        sku_prices[sku_id] = round(float(sku.price), 6)
    state["sku_prices"] = sku_prices
    state["shelf_sku_ids"] = sorted(getattr(env, "shelf_sku_ids", set()))

    # 库存（items_by_sku + waiting_items）
    inv_snapshot: Dict[str, Any] = {"capacity": env.inventory.capacity, "items_by_sku": {}, "waiting_items": []}

    for sku_id, items in env.inventory.items_by_sku.items():
        items_data = []
        for m in items:
            items_data.append(
                {
                    "sku_id": m.sku.sku_id,
                    "begin_time": m.begin_time.isoformat(),
                    "expired_time": m.expired_time.isoformat(),
                    "buy_price": round(float(m.buy_price), 6),
                    "merch_id": m.merch_id,
                    "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
                    "supplier_id": getattr(m, "supplier_id", None),
                }
            )
        # 按 merch_id 排序，避免顺序差异
        items_data.sort(key=lambda x: x["merch_id"])
        inv_snapshot["items_by_sku"][sku_id] = items_data

    waiting_data = []
    for m in env.inventory.waiting_items:
        waiting_data.append(
            {
                "sku_id": m.sku.sku_id,
                "begin_time": m.begin_time.isoformat(),
                "expired_time": m.expired_time.isoformat(),
                "buy_price": round(float(m.buy_price), 6),
                "merch_id": m.merch_id,
                "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
                "supplier_id": getattr(m, "supplier_id", None),
            }
        )
    waiting_data.sort(key=lambda x: x["merch_id"])
    inv_snapshot["waiting_items"] = waiting_data
    state["inventory"] = inv_snapshot

    # 订单
    orders_snapshot: Dict[str, Any] = {}
    for order_id, order in env.order_manager.orders.items():
        items_data = []
        for m in order.items:
            items_data.append(
                {
                    "sku_id": m.sku.sku_id,
                    "begin_time": m.begin_time.isoformat(),
                    "expired_time": m.expired_time.isoformat(),
                    "buy_price": round(float(m.buy_price), 6),
                    "merch_id": m.merch_id,
                    "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
                    "supplier_id": getattr(m, "supplier_id", None),
                }
            )
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

    # 从 record_manager 中抽样一个 SKU 的销售记录快照（如果有 SKU）
    if env.skus_list:
        sample_sku_id = env.skus_list[0].sku_id
        # 取从 data_begin_time 到 current_date 的区间
        cfg_begin = env.config.get("data_begin_time") or env.config.get("begin_time")
        try:
            start_dt = datetime.strptime(cfg_begin, "%m/%d/%y").date()
        except Exception:
            start_dt = env.current_date
        sales = env.record_manager.read_sku(sample_sku_id, start_dt, env.current_date)
        # 按日期聚合 (move_sum, last_price, last_customer_count)
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

    # 新闻管理器（如启用）
    if getattr(env, "news_manager", None) and env.config.get("enable_new"):
        today_news = env.news_manager.get_today_news()
        rolling = getattr(env.news_manager, "_rolling", [])
        state["news_manager"] = {
            "today_news_count": len(today_news),
            "rolling_count": len(rolling),
        }

    return state


def run_all_tools_once(env = None):
    if env is None:
        config = load_config_by_type("hard")
        env = RetailEnvironment(config)

    ratings = env.get_sku_rating_report()

    if ratings:
        for rating, rating_item in ratings.items():
            env.log_debug(
                f"SKU: {rating}, InitialRating: {rating_item['initial_rating']}, AvgRating: {rating_item['avg_rating']}"
            )
    tools = env.get_tools()
    env.log_debug(f"发现工具 {len(tools)} 个：{list(tools.keys())}")

    skus = env.skus_id_map
    env.log_debug(f"发现 SKU {len(skus)} 个：{list(skus.keys())}")

    # 先准备一个可用的 sku_id（如果有的话）
    sku_id_for_test = None
    if env.skus_list:
        sku_id_for_test = env.skus_list[0].sku_id
        env.log_debug(f"用于测试的 sku_id: {sku_id_for_test}")
    else:
        env.log_debug("警告：当前环境中没有任何 SKU，相关工具会返回报错信息。")

    env.log_debug("\n===== 开始逐个执行工具 =====\n")

    for tool_name in tools.keys():
        env.log_debug(f"\n---- 执行工具：{tool_name} ----")

        try:
            # 根据不同工具准备参数
            if tool_name == "view_funds_and_date":
                result = env.exec_tools(tool_name)

            elif tool_name == "view_sales_profit_history":
                # 需要 sku_ids
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(
                    tool_name,
                    sku_ids=target_skus,
                    start_date="06/06/91",
                    end_date="06/08/91",
                )

            elif tool_name == "view_current_date_supplier_prices":
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(tool_name, sku_ids=target_skus)

            elif tool_name == "view_sku_prices":
                target_skus = [sku_id_for_test] if sku_id_for_test else []
                result = env.exec_tools(tool_name, sku_ids=target_skus if target_skus else None)

            elif tool_name == "view_supplier_price_history":
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(
                    tool_name,
                    sku_id=target_skus[0],
                    start_date="06/06/91",
                    end_date="06/08/91",
                )

            elif tool_name == "view_sku_avg_ratings":
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(
                    tool_name,
                    sku_ids=target_skus,
                    start_date="06/06/91",
                    end_date="06/08/91",
                )

            elif tool_name == "view_supplier_returns_avg_rate":
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(
                    tool_name,
                    supplier_id="supplier_1",
                    sku_ids=target_skus,
                    start_date="06/06/91",
                    end_date="06/08/91",
                )

            elif tool_name == "view_inventory":
                result = env.exec_tools(tool_name)

            elif tool_name == "view_shelf_status":
                result = env.exec_tools(tool_name)

            elif tool_name == "set_shelf_skus":
                target_skus = [sku_id_for_test] if sku_id_for_test else []
                result = env.exec_tools(tool_name, sku_ids=target_skus)

            elif tool_name == "place_order":
                # 需要 sku_id 和 quantity
                result = env.exec_tools(
                    tool_name,
                    items=[{"sku_id": sku_id_for_test or "DUMMY_SKU", "quantity": 5}],
                    supplier_id="supplier_1",
                )

            elif tool_name == "modify_sku_price":
                # 需要 sku_id 和 new_price
                new_price = 9.99
                if sku_id_for_test and sku_id_for_test in env.skus_id_map:
                    new_price = env.skus_id_map[sku_id_for_test].price * 1.1
                result = env.exec_tools(
                    tool_name,
                    sku_id=sku_id_for_test or "DUMMY_SKU",
                    new_price=new_price,
                )

            else:
                # 兜底：如果以后新增工具但这里没特殊处理，就尝试裸调
                result = env.exec_tools(tool_name)

            env.log_debug("执行结果：")
            env.log_debug(result)

        except Exception as e:
            env.log_debug(f"工具 {tool_name} 执行时发生异常：{e}")

    env.log_debug("\n===== 所有工具执行完毕 =====")

if __name__ == "__main__":
    # Import simulation functions from the dedicated module
    from agents.run_non_llm_simulation import (
        simulate_simple_logic_environment,
        simulate_simple_review_environment,
        simulate_news_aware_environment,
        simulate_quality_based_environment,
        ENVIRONMENT_MAPPING,
    )

    parser = argparse.ArgumentParser(
        description="零售环境模拟器 - 选择要执行的模拟函数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行简单逻辑环境（365天）
  python retail_environment.py --mode logic --days 365 --sample-size 2 --config-type hard

  # 运行评价感知环境（180天）
  python retail_environment.py --mode review --days 180 --sample-size 2 --config-type middle --db-path middle/simulate_data/15/records_review/

  # 运行新闻感知环境（180天）
  python retail_environment.py --mode news --days 180 --sample-size 2 --config-type middle --db-path middle/simulate_data/15/records_review_news/

  # 运行质量优先环境（180天）
  python retail_environment.py --mode quality --days 180 --sample-size 2 --config-type hard --db-path data/simulate_data/15/records_no_review/

  # 运行工具自检
  python retail_environment.py --mode tools

  # 查看环境映射关系
  python retail_environment.py --mode help
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=list(ENVIRONMENT_MAPPING.keys()) + ["tools", "help"],
        required=True,
        help="选择模拟模式: logic=简单逻辑环境, review=评价感知环境, news=新闻感知环境, quality=质量优先环境, tools=工具自检, help=显示环境映射"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="模拟天数（默认: 180）"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=2,
        help="每个品类选择的 SKU 数量（默认: 2）"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="数据库路径（可选，不同模式有默认值）"
    )

    parser.add_argument(
        "--config-type",
        type=str,
        choices=CONFIG_TYPE_CHOICES,
        default="hard",
        help="配置类型，支持 easy / middle / hard"
    )

    args = parser.parse_args()

    # 显示环境映射关系
    if args.mode == "help":
        print("\n" + "=" * 80)
        print("RetailBench 环境映射关系")
        print("=" * 80)
        for mode, info in ENVIRONMENT_MAPPING.items():
            print(f"\n{mode}:")
            print(f"  描述: {info['description']}")
            print(f"  支持的配置类型: {', '.join(info['config_types'])}")
        print("\n" + "=" * 80)
        exit(0)

    # 根据模式执行相应的函数
    if args.mode == "tools":
        print(f"运行工具自检 (config_type={args.config_type})...")
        config = load_config_by_type(args.config_type)
        env = RetailEnvironment(config)
        run_all_tools_once(env)
    else:
        # 获取对应的模拟函数
        env_info = ENVIRONMENT_MAPPING.get(args.mode)
        sim_func = env_info["function"]

        if args.mode == "logic":
            db_path = args.db_path or 'model_run_time'
            print(f"运行简单逻辑环境: days={args.days}, sample_size={args.sample_size}, config_type={args.config_type}, db_path={db_path}")
            sim_func(days=args.days, sample_size=args.sample_size, config_type=args.config_type, db_path=db_path)
        elif args.mode == "review":
            db_path = args.db_path or 'middle/simulate_data/15/records_review/'
            print(f"运行评价感知环境: days={args.days}, sample_size={args.sample_size}, config_type={args.config_type}, db_path={db_path}")
            sim_func(days=args.days, sample_size=args.sample_size, db_path=db_path, config_type=args.config_type)
        elif args.mode == "news":
            db_path = args.db_path or 'middle/simulate_data/15/records_review_news/'
            print(f"运行新闻感知环境: days={args.days}, sample_size={args.sample_size}, config_type={args.config_type}, db_path={db_path}")
            sim_func(days=args.days, sample_size=args.sample_size, db_path=db_path, config_type=args.config_type)
        elif args.mode == "quality":
            db_path = args.db_path or 'data/simulate_data/15/records_no_review/'
            print(f"运行质量优先环境: days={args.days}, sample_size={args.sample_size}, config_type={args.config_type}, db_path={db_path}")
            sim_func(days=args.days, sample_size=args.sample_size, db_path=db_path, config_type=args.config_type)
