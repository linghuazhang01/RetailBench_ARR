from __future__ import annotations
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from model.return_rate_model import ReturnRateModel
from module.news_manager import NewsManager
from module.review_manager import ReviewManager
from sku import SKU, Merchandise
from module.record_manager import ReturnRateRecord, ReturnRecord, SaleRecord, RecordManager
from util.logger import get_logger


class Inventory:
    """
    库存管理。
    基础的 5 个方法：
    - __init__
    - add_item
    - step
    - remove_item (修改：现在接收 Merchandise 对象而不是 merch_id)
    - reset
    扩展方法：
    - consume_by_category
    - consume_by_sku
    - destroy_items
    """

    def __init__(self, capacity: Optional[int] = None) -> None:
        # Only organize by SKU
        self.items_by_sku: Dict[str, List[Merchandise]] = {}
        self.waiting_items: List[Merchandise] = []
        self.capacity = capacity
        self.logger = get_logger()

    def _current_size(self) -> int:
        return sum(len(items) for items in self.items_by_sku.values())

    def _can_accept(self, incoming_count: int = 1) -> bool:
        if self.capacity is None:
            return True
        return self._current_size() + incoming_count <= self.capacity

    def _store_merchandise(self, merchandise: Merchandise) -> None:
        sku_id = merchandise.sku.sku_id
        if sku_id not in self.items_by_sku:
            self.items_by_sku[sku_id] = []
        self.items_by_sku[sku_id].append(merchandise)

    def _admit_waiting_items(self) -> None:
        """Move waiting items into inventory while capacity allows."""
        while self.waiting_items and self._can_accept():
            merchandise = self.waiting_items.pop(0)
            self._store_merchandise(merchandise)

    def add_item(self, merchandise: Merchandise) -> None:
        """新增或覆盖一个商品。超过容量时先放入 waiting_items。"""
        if self._can_accept():
            self._store_merchandise(merchandise)
        else:
            self.waiting_items.append(merchandise)

    def step(
            self,
            current_date,
            customer_count,
            skus_id_maps,
            record_manager: RecordManager,
            review_manager: ReviewManager = None,
            new_manager: NewsManager = None,
            shelf_sku_ids: Optional[Set[str]] = None,
            category_attraction_aggregation: str = "legacy",
            category_attraction_rho: float = 1.0,
        ) -> tuple:
        """
        库存的逐步更新逻辑。
        处理商品过期、销量更新等。

        Returns:
            tuple: (money_earned, insufficient_skus, sales_by_sku, returns_by_sku,
                    expired_discount_by_sku, waiting_items, cost_of_goods_sold)
                - money_earned: float, the total money earned from sales
                - insufficient_skus: List[str], SKUs that had insufficient stock for demand
                - sales_by_sku: Dict[str, int], actual units sold per SKU
                - returns_by_sku: Dict[str, int], units returned per SKU
                - expired_discount_by_sku: Dict[str, int], units expired per SKU
                - waiting_items: List[Merchandise], items waiting to be admitted
                - cost_of_goods_sold: float, sum of buy_price for all items sold
        """
        money_earned = 0.0
        insufficient_skus = []
        sales_by_sku: Dict[str, int] = {}
        returns_by_sku: Dict[str, int] = {}
        expired_discount_by_sku: Dict[str, int] = {}

        # Try to move any waiting items in before sales occur
        self._admit_waiting_items()

        # Iterate through all items by SKU to process sales
        skus_to_check = list(self.items_by_sku.keys())

        current_inventory_skus = list([item for item in self.items_by_sku if len(self.items_by_sku[item]) > 0])

        cost_of_goods_sold = 0.0

        def _compute_attribute_attraction(sku_id: str, sku_obj: SKU) -> float:
            exterinal_attracrtion_rate = 0
            if review_manager:
                exterinal_attracrtion_rate += review_manager.compute_sales_impact(
                    sku_obj,
                    current_date=current_date
                )

                self.logger.debug(
                    f"sku_id: {sku_id}, exterinal_attracrtion_review: {exterinal_attracrtion_rate}"
                )

            if new_manager:
                exterinal_new_attracrtion_rate = new_manager.evaluate_impact_for_sku(
                    sku_id=sku_obj.sku_id,
                    sku_category=sku_obj.category,
                    impact_factors=['need']
                )

                exterinal_attracrtion_rate += exterinal_new_attracrtion_rate['total_effect']

                self.logger.debug(
                    f"sku_id: {sku_id}, exterinal_attracrtion_news: {exterinal_new_attracrtion_rate['total_effect']}"
                )

                matched_news = exterinal_new_attracrtion_rate.get("matched_news") or []
                for news in matched_news:
                    self.logger.debug(
                        f"sku_id: {sku_id}, matched_news_id: {news.get('id', '')}, "
                        f"title: {news.get('title', '')}, content: {news.get('content', '')}"
                    )

            return sku_obj.compute_attribute_attraction(
                date=current_date,
                external_attraction_rate=exterinal_attracrtion_rate,
            )

        def _sell_sku_units(sku_id: str, expected_sales: int) -> None:
            nonlocal money_earned, cost_of_goods_sold

            if not self.items_by_sku.get(sku_id):
                return
            sku_obj: SKU = skus_id_maps.get(sku_id)
            if sku_obj is None:
                return

            available_items = len(self.items_by_sku[sku_id])
            self.logger.debug(
                f"sku_id: {sku_id}, expected_sales: {expected_sales}, available_items: {available_items}"
            )

            actual_sales = min(expected_sales, available_items)
            if expected_sales > available_items and sku_obj not in insufficient_skus:
                insufficient_skus.append(sku_obj)

            items_to_sell = list(self.items_by_sku[sku_id][:actual_sales])
            for item in items_to_sell:
                cost_of_goods_sold += item.buy_price

            if actual_sales > 0 and record_manager:
                sales_record = SaleRecord(
                    upc=sku_obj.sku_id,
                    date_obj=current_date,
                    move=actual_sales,
                    price=sku_obj.price,
                    customer_count=customer_count,
                )
                record_manager.add_record(sku_obj.sku_id, sales_record)
                for item in items_to_sell:
                    record_manager.update_lifecycle_sold(item.merch_id, sku_obj.price, current_date)

            money_earned += sku_obj.price * actual_sales
            sales_by_sku[sku_id] = actual_sales

            return_count = 0
            for item in items_to_sell:
                self.remove_item(item)

                if review_manager:
                    return_prob = ReturnRateModel.from_quality(item.quality_score)
                    if random.random() < return_prob:
                        return_count += 1
                        if record_manager and hasattr(item, 'supplier_id') and item.supplier_id:
                            record_manager.add_return(
                                ReturnRecord(
                                    supplier_id=item.supplier_id,
                                    sku_id=sku_id,
                                    date_obj=current_date,
                                )
                            )

                    review_manager.maybe_generate_review_for_merchandise(
                        merchandise=item,
                        date_obj=current_date
                    )

            if review_manager:
                returns_by_sku[sku_id] = return_count

                if actual_sales > 0 and record_manager:
                    record_manager.add_return_rate(
                        ReturnRateRecord(
                            sku_id=sku_id,
                            return_rate=return_count / actual_sales,
                            return_number=return_count,
                            date_obj=current_date,
                        )
                    )

                money_earned -= sku_obj.price * return_count

        if category_attraction_aggregation == "power":
            visible_sku_set = set(shelf_sku_ids) if shelf_sku_ids is not None else set(current_inventory_skus)
            category_groups: Dict[str, List[tuple[str, SKU, float]]] = defaultdict(list)

            for sku_id in current_inventory_skus:
                sku_obj = skus_id_maps[sku_id]
                attraction = _compute_attribute_attraction(sku_id, sku_obj)
                sku_obj.today_attraction = attraction
                if sku_id not in visible_sku_set:
                    continue
                if attraction <= 0:
                    continue
                category_groups[sku_obj.category].append((sku_id, sku_obj, attraction))

            rho = float(category_attraction_rho)
            if rho <= 0:
                raise ValueError("category_attraction_rho must be positive")

            customer_count_int = int(customer_count or 0)
            for category, entries in category_groups.items():
                attractions = [attraction for _, _, attraction in entries]
                category_attraction = sum(a ** rho for a in attractions) ** (1.0 / rho)
                purchase_prob = category_attraction / (1.0 + category_attraction)
                category_demand = int(np.random.binomial(customer_count_int, purchase_prob))
                share_total = sum(attractions)
                if category_demand <= 0 or share_total <= 0:
                    for sku_id, _, _ in entries:
                        sales_by_sku[sku_id] = 0
                    continue

                shares = [attraction / share_total for attraction in attractions]
                sku_demands = np.random.multinomial(category_demand, shares)

                self.logger.debug(
                    f"category: {category}, category_attraction: {category_attraction}, "
                    f"rho: {rho}, category_demand: {category_demand}"
                )

                for (sku_id, _, _), sku_demand in zip(entries, sku_demands):
                    _sell_sku_units(sku_id, int(sku_demand))

        else:
            for sku_id in current_inventory_skus:
                sku_obj = skus_id_maps[sku_id]
                _compute_attribute_attraction(sku_id, sku_obj)

            for sku_id in current_inventory_skus:
                sku_obj = skus_id_maps[sku_id]
                sku_obj.compute_attraction(
                    currnet_market_skus_ids=current_inventory_skus,
                )

            for sku_id in skus_to_check:
                if self.items_by_sku[sku_id]:
                    sku_obj: SKU = skus_id_maps.get(sku_id)
                    if sku_obj is None:
                        continue  # Skip if SKU object doesn't exist
                    
                    # Calculate sales for this SKU based on customer count
                    expected_sales = sku_obj.get_sales(
                        currnet_market_skus_ids=current_inventory_skus,
                        customer_count=customer_count,
                    )

                    # Check if we have enough inventory for the sales demand
                    available_items = len(self.items_by_sku[sku_id])

                    self.logger.debug(
                        f"sku_id: {sku_id}, expected_sales: {expected_sales}, available_items: {available_items}"
                    )

                    actual_sales = min(expected_sales, available_items)
                    if expected_sales > available_items:
                        # Not enough inventory - add to insufficient SKUs list
                        if sku_obj not in insufficient_skus:
                            insufficient_skus.append(sku_obj)

                    # Work on a copy to avoid mutating while iterating
                    items_to_sell = list(self.items_by_sku[sku_id][:actual_sales])

                    # Track cost of goods sold (sum of buy_price for sold items)
                    for item in items_to_sell:
                        cost_of_goods_sold += item.buy_price

                    # Record the sales data
                    if actual_sales > 0 and record_manager:
                        sales_record = SaleRecord(
                            upc=sku_obj.sku_id,
                            date_obj=current_date,
                            move=actual_sales,
                            price=sku_obj.price,
                            customer_count=customer_count,
                        )
                        record_manager.add_record(sku_obj.sku_id, sales_record)
                        # 更新商品生命周期为已售出
                        for item in items_to_sell:
                            record_manager.update_lifecycle_sold(item.merch_id, sku_obj.price, current_date)

                    money_earned += sku_obj.price * actual_sales

                    sales_by_sku[sku_id] = actual_sales
                    
                    if review_manager:

                        return_count = 0
                        
                        # Remove sold items from inventory
                        for item in items_to_sell:
                            self.remove_item(item)

                            return_prob = ReturnRateModel.from_quality(item.quality_score)

                            if random.random() < return_prob:
                                return_count += 1
                                # 记录退货信息
                                if record_manager and hasattr(item, 'supplier_id') and item.supplier_id:
                                    record_manager.add_return(
                                        ReturnRecord(
                                            supplier_id=item.supplier_id,
                                            sku_id=sku_id,
                                            date_obj=current_date,
                                        )
                                    )

                            ## Generate review
                            if review_manager:
                                review_manager.maybe_generate_review_for_merchandise(
                                    merchandise=item,
                                    date_obj=current_date
                                )
                        
                        returns_by_sku[sku_id] = return_count

                        if sales_by_sku[sku_id] > 0:
                            record_manager.add_return_rate(
                                ReturnRateRecord(
                                    sku_id=sku_id,
                                    return_rate=returns_by_sku[sku_id] / sales_by_sku[sku_id],
                                    return_number=return_count,
                                    date_obj=current_date,
                                )
                            )

                        money_earned -= sku_obj.price * return_count

        # Now handle expired products that remain in inventory - sell at 0.6x buy price
        skus_to_check = list(self.items_by_sku.keys())
        for sku_id in skus_to_check:
            if self.items_by_sku[sku_id]:
                items_to_remove = []
                for merchandise in self.items_by_sku[sku_id]:
                    if merchandise.judge_expired(current_date):
                        # Sell expired items at 0.6x buy price
                        money_earned += merchandise.buy_price * 0
                        items_to_remove.append(merchandise)
                        expired_discount_by_sku[sku_id] = expired_discount_by_sku.get(sku_id, 0) + 1
                        # 更新商品生命周期为已损耗
                        if record_manager:
                            record_manager.update_lifecycle_expired(merchandise.merch_id, current_date)

                # Remove expired items from this SKU's list
                for merchandise in items_to_remove:
                    self.items_by_sku[sku_id].remove(merchandise)

        # Promo clearance for waiting items (sell at 80% of price/buy price)
        if self.waiting_items:
            for merchandise in list(self.waiting_items):
                if merchandise.judge_expired(current_date):
                        # Sell expired items at 0.6x buy price
                        money_earned += merchandise.buy_price * 0
                        self.waiting_items.remove(merchandise)
                        expired_discount_by_sku[merchandise.sku.sku_id] = expired_discount_by_sku.get(merchandise.sku.sku_id, 0) + 1
                        # 更新商品生命周期为已损耗
                        if record_manager:
                            record_manager.update_lifecycle_expired(merchandise.merch_id, current_date)

        # After clearing sold/expired items, try to admit waiting items again
        self._admit_waiting_items()
            
        return money_earned, insufficient_skus, sales_by_sku, returns_by_sku, expired_discount_by_sku, self.waiting_items, cost_of_goods_sold

    def remove_item(self, merchandise: Merchandise) -> None:
        """移除某个商品。"""
        # Find the SKU list containing this merchandise
        sku_id = merchandise.sku.sku_id
        if sku_id in self.items_by_sku:
            if merchandise in self.items_by_sku[sku_id]:
                self.items_by_sku[sku_id].remove(merchandise)
                # If the SKU list is now empty, remove the key
                if not self.items_by_sku[sku_id]:
                    del self.items_by_sku[sku_id]

    def destroy_items(self, quantity: int, sku_id: Optional[str] = None) -> int:
        """
        销毁库存中的商品以腾出容量。

        Args:
            quantity: 计划销毁的数量。
            sku_id: 可选，仅销毁特定 SKU 的库存。

        Returns:
            int: 实际销毁的数量。
        """
        if quantity <= 0:
            return 0

        destroyed = 0
        targets = [sku_id] if sku_id else list(self.items_by_sku.keys())

        for target_sku in targets:
            items = self.items_by_sku.get(target_sku, [])
            while items and destroyed < quantity:
                items.pop(0)
                destroyed += 1
            if not items and target_sku in self.items_by_sku:
                break
            if destroyed >= quantity:
                break

        if destroyed:
            # Try to bring in waiting items after freeing space
            self._admit_waiting_items()

        return destroyed

    def reset(self) -> None:
        """清空库存。"""
        self.items_by_sku.clear()
        self.waiting_items.clear()

    def consume_by_sku(self, consumed_sku_id: str, quantity: int) -> int:
        """
        消耗指定品类的指定数量商品。
        返回实际消耗的数量。
        """
        # Find all items that belong to the specified category
        items_to_consume = []
        
        # Iterate through all items by SKU
        for sku_id, merchandise_list in self.items_by_sku.items():
            if sku_id == consumed_sku_id:
                if len(merchandise_list) < quantity:
                    items_to_consume.extend(merchandise_list)
                else:
                    items_to_consume.extend(merchandise_list[:quantity])
        
        consumed_count = 0
        for merchandise in items_to_consume: 
            self.remove_item(merchandise)
            consumed_count += 1
        
        return consumed_count
    
    def compute_net_worth(self, current_date: Optional[date] = None) -> float:
        """
        计算库存净值，按线性折旧价格：
        当前价格 = 进货价 * 剩余时间 / 总保质期。
        过期则计为 0，总保质期为 0 时按进货价计。
        """
        total_worth = 0.0
        current_dt: date = current_date 

        def _calc_value(merch: Merchandise) -> float:
            total_span = (merch.expired_time - merch.begin_time).days
            remaining = (merch.expired_time - current_dt).days

            if remaining <= 0:
                return 0.0
            if total_span <= 0:
                return merch.buy_price
            ratio = remaining / total_span
            return merch.buy_price * ratio

        for sku_items in self.items_by_sku.values():
            for merch in sku_items:
                total_worth += _calc_value(merch)

        for merch in self.waiting_items:
            total_worth += _calc_value(merch)

        return total_worth
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将库存状态序列化为字典，用于 checkpoint。
        
        Returns:
            包含库存状态的字典
        """
        inventory_state = {
            "items_by_sku": {},
            "waiting_items": [],
            "capacity": self.capacity,
        }
        
        # 序列化 items_by_sku
        for sku_id, items in self.items_by_sku.items():
            inventory_state["items_by_sku"][sku_id] = []
            for merch in items:
                merch_dict = {
                    "sku_id": merch.sku.sku_id,
                    "begin_time": merch.begin_time.isoformat(),
                    "expired_time": merch.expired_time.isoformat(),
                    "buy_price": merch.buy_price,
                    "merch_id": merch.merch_id,
                    "quality_score": merch.quality_score,
                    "supplier_id": getattr(merch, "supplier_id", None),
                }
                inventory_state["items_by_sku"][sku_id].append(merch_dict)
        
        # 序列化 waiting_items
        for merch in self.waiting_items:
            merch_dict = {
                "sku_id": merch.sku.sku_id,
                "begin_time": merch.begin_time.isoformat(),
                "expired_time": merch.expired_time.isoformat(),
                "buy_price": merch.buy_price,
                "merch_id": merch.merch_id,
                "quality_score": merch.quality_score,
                "supplier_id": getattr(merch, "supplier_id", None),
            }
            inventory_state["waiting_items"].append(merch_dict)
        
        checkpoint_data["inventory"] = inventory_state
        return checkpoint_data
    
    def recover_from_checkpoint(self, checkpoint_data: Dict[str, Any], skus_id_map: Dict[str, SKU]) -> None:
        """
        从 checkpoint 数据恢复库存状态。
        
        Args:
            checkpoint_data: checkpoint 数据字典
            skus_id_map: SKU ID 到 SKU 对象的映射
        """
        inventory_state = checkpoint_data.get("inventory", {})
        
        # 清空当前库存
        self.items_by_sku.clear()
        self.waiting_items.clear()
        
        # 恢复 capacity
        if "capacity" in inventory_state:
            self.capacity = inventory_state["capacity"]
        
        # 恢复 items_by_sku
        for sku_id, items_data in inventory_state.get("items_by_sku", {}).items():
            if sku_id not in skus_id_map:
                continue
            sku_obj = skus_id_map[sku_id]
            self.items_by_sku[sku_id] = []
            for merch_data in items_data:
                merch = Merchandise(
                    sku=sku_obj,
                    begin_time=datetime.strptime(merch_data["begin_time"], "%Y-%m-%d").date(),
                    expired_time=datetime.strptime(merch_data["expired_time"], "%Y-%m-%d").date(),
                    buy_price=merch_data["buy_price"],
                    merch_id=merch_data["merch_id"],
                    quality_score=merch_data.get("quality_score", 0.5),
                    supplier_id=merch_data.get("supplier_id"),
                )
                self.items_by_sku[sku_id].append(merch)
        
        # 恢复 waiting_items
        for merch_data in inventory_state.get("waiting_items", []):
            sku_id = merch_data["sku_id"]
            if sku_id not in skus_id_map:
                continue
            sku_obj = skus_id_map[sku_id]
            merch = Merchandise(
                sku=sku_obj,
                begin_time=datetime.strptime(merch_data["begin_time"], "%Y-%m-%d").date(),
                expired_time=datetime.strptime(merch_data["expired_time"], "%Y-%m-%d").date(),
                buy_price=merch_data["buy_price"],
                merch_id=merch_data["merch_id"],
                quality_score=merch_data.get("quality_score", 0.5),
                supplier_id=merch_data.get("supplier_id"),
            )
            self.waiting_items.append(merch)
