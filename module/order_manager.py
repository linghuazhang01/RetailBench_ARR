from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from sku import SKU, Merchandise

class Order:
    """
    订单对象。
    只实现图中给出的 2 个方法：
    - __init__
    - get_detail
    """

    def __init__(
        self,
        order_id: str,
        ordered_sku: Dict[SKU, int],
        customer_id: str,
        items: List[Merchandise],
        created_at: Optional[date] = None,
        delivery_time: int = 0,
        cost: float = 0.0,
        supplier_order_recorded: bool = False,
    ) -> None:
        self.order_id = order_id
        self.ordered_sku = ordered_sku
        self.customer_id = customer_id
        self.items = items
        self.cost = cost
        self.supplier_order_recorded = supplier_order_recorded

        self.created_at: date = created_at or date.today()
        self.delivery_time: int = delivery_time
        self.expected_delivery_date: date = self.created_at + timedelta(days=delivery_time)

    def get_detail(self) -> Dict[str, Any]:
        """
        返回订单的简单结构化信息。
        外部如果需要更复杂的逻辑，可以在环境层组合使用。
        """
        # 统计每个 SKU 数量（使用 sku_id 作为 key，便于序列化）
        sku_count: Dict[str, int] = {}
        for item in self.items:
            sku_id = item.sku.sku_id
            sku_count[sku_id] = sku_count.get(sku_id, 0) + 1

        return {
            "order_id": self.order_id,
            # 使用 sku_id 映射数量，避免 SKU 对象出现在 dict key 中
            "ordered_sku": {sku.sku_id: qty for sku, qty in self.ordered_sku.items()},
            "customer_id": self.customer_id,
            "created_at": self.created_at.isoformat(),
            "delivery_time": self.delivery_time,
            "expected_delivery_date": self.expected_delivery_date.isoformat(),
            "cost": self.cost,
            "items": [
                {"sku_id": sku_id, "count": count}
                for sku_id, count in sku_count.items()
            ],
        }
    


class OrderManager:
    """
    订单管理器。
    只实现图中给出的 5 个方法：
    - __init__
    - step
    - reset
    - add_order
    - remove_order
    """

    def __init__(self) -> None:
        # order_id -> Order
        self.orders: Dict[str, Order] = {}

    def step(self, current_date, record_manager=None) -> List[Merchandise]:
        """
        订单随时间推进的占位方法。
        推进一天，返回当天到期的订单中的商品。
        """

        delivered_items = []
        orders_to_remove = []
        
        for order_id, order in self.orders.items():
            if order.expected_delivery_date <= current_date:
                delivered_items.extend(order.items)
                orders_to_remove.append(order_id)
        
        # Remove delivered orders
        for order_id in orders_to_remove:
            order = self.orders.pop(order_id)
            if record_manager is not None and not getattr(order, "supplier_order_recorded", False):
                from module.record_manager import SupplierOrderRecord
                items = {}
                for merch in order.items:
                    sku_id = merch.sku.sku_id
                    items[sku_id] = items.get(sku_id, 0) + 1
                record_manager.add_supplier_order(
                    SupplierOrderRecord(
                        supplier_id=order.customer_id,
                        order_date=order.created_at,
                        arrival_date=current_date,
                        shipping_days=(current_date - order.created_at).days,
                        items=items,
                        cost=getattr(order, "cost", 0.0),
                    )
                )
                order.supplier_order_recorded = True
        
        return delivered_items

    def reset(self) -> None:
        """清空所有订单。"""
        self.orders.clear()

    def add_order(self, order: Order) -> None:
        """新增订单。"""
        self.orders[order.order_id] = order

    def remove_order(self, order_id: str) -> None:
        """删除订单。"""
        if order_id in self.orders:
            del self.orders[order_id]

    def get_current_orders(self) -> List[Order]:
        return list(self.orders.values())
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将订单状态序列化为字典，用于 checkpoint。
        
        Returns:
            更新后的 checkpoint_data，包含订单状态
        """
        orders_state = []
        for order in self.orders.values():
            order_dict = {
                "order_id": order.order_id,
                "ordered_sku": {sku.sku_id: qty for sku, qty in order.ordered_sku.items()},
                "customer_id": order.customer_id,
                "created_at": order.created_at.isoformat(),
                "delivery_time": order.delivery_time,
                "expected_delivery_date": order.expected_delivery_date.isoformat(),
                "cost": order.cost,
                "supplier_order_recorded": getattr(order, "supplier_order_recorded", False),
                "items": [
                    {
                        "sku_id": item.sku.sku_id,
                        "begin_time": item.begin_time.isoformat(),
                        "expired_time": item.expired_time.isoformat(),
                        "buy_price": item.buy_price,
                        "merch_id": item.merch_id,
                        "quality_score": item.quality_score,
                        "supplier_id": getattr(item, "supplier_id", None),
                    }
                    for item in order.items
                ],
            }
            orders_state.append(order_dict)
        
        checkpoint_data["orders"] = orders_state
        return checkpoint_data
    
    def recover_from_checkpoint(self, checkpoint_data: Dict[str, Any], skus_id_map: Dict[str, SKU]) -> None:
        """
        从 checkpoint 数据恢复订单状态。
        
        Args:
            checkpoint_data: checkpoint 数据字典
            skus_id_map: SKU ID 到 SKU 对象的映射
        """
        # 清空当前订单
        self.orders.clear()
        
        for order_data in checkpoint_data.get("orders", []):
            ordered_sku = {}
            items = []
            
            # 恢复 ordered_sku
            for sku_id, qty in order_data.get("ordered_sku", {}).items():
                if sku_id in skus_id_map:
                    ordered_sku[skus_id_map[sku_id]] = qty
            
            # 恢复 items
            for item_data in order_data.get("items", []):
                sku_id = item_data["sku_id"]
                if sku_id not in skus_id_map:
                    continue
                sku_obj = skus_id_map[sku_id]
                merch = Merchandise(
                    sku=sku_obj,
                    begin_time=datetime.strptime(item_data["begin_time"], "%Y-%m-%d").date(),
                    expired_time=datetime.strptime(item_data["expired_time"], "%Y-%m-%d").date(),
                    buy_price=item_data["buy_price"],
                    merch_id=item_data["merch_id"],
                    quality_score=item_data.get("quality_score", 0.5),
                    supplier_id=item_data.get("supplier_id"),
                )
                items.append(merch)
            
            # 创建订单对象
            order = Order(
                order_id=order_data["order_id"],
                ordered_sku=ordered_sku,
                customer_id=order_data["customer_id"],
                items=items,
                created_at=datetime.strptime(order_data["created_at"], "%Y-%m-%d").date(),
                delivery_time=order_data["delivery_time"],
                cost=order_data["cost"],
                supplier_order_recorded=order_data.get("supplier_order_recorded", True),
            )
            self.orders[order.order_id] = order
