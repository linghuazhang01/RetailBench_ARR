from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from module.news_manager import NewsManager
from module.record_manager import SupplierPriceRecord
from util.logger import get_logger


def _parse_date(val: Any) -> Optional[date]:
    """Accept YYYY-MM-DD or %m/%d/%y, return date or None."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    for fmt in ("%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(str(val), fmt).date()
        except Exception:
            continue
    return None


class SupplierManager:
    """
    读取 *_suppliers.json（结构：supplier_id -> [daily rows]），
    提供按 SKU + 日期聚合后的供给商价格查询。
    """

    def __init__(
        self,
        store_root: str = "data/simulate_data",
        store_id: str = "12",
        begin_date: date | str | None = None,
        end_date: date | str | None = None,
        news_manager: Optional[NewsManager] = None,
        allowed_sku_ids: Optional[List[str]] = None,
    ) -> None:
        self.store_root = Path(store_root) / str(store_id)

        # 解析日期范围（可选）
        self.begin_date: Optional[date] = _parse_date(begin_date)
        self.end_date: Optional[date] = _parse_date(end_date)
        self.news_manager = news_manager
        self.allowed_sku_ids = set(str(s) for s in allowed_sku_ids) if allowed_sku_ids else None

        # sku -> date_iso -> list[dict]
        self._sku_date_prices: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._load()

    def _in_range(self, dt: date) -> bool:
        if self.begin_date and dt < self.begin_date:
            return False
        if self.end_date and dt > self.end_date:
            return False
        return True

    def _load(self) -> None:
        if not self.store_root.exists():
            return

        for path in self.store_root.rglob("*_suppliers.json"):
            sku_id = path.stem.split("_suppliers")[0]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                get_logger().debug(f"[WARN] 读取供应商文件失败 {path}: {exc}")
                continue

            if not isinstance(payload, dict):
                continue

            for supplier_id, rows in payload.items():
                if not isinstance(rows, list):
                    continue

                for row in rows:
                    if not isinstance(row, dict):
                        continue

                    dt = _parse_date(row.get("date"))
                    if dt is None or not self._in_range(dt):
                        continue

                    price = (
                        row.get("supplier_price")
                        or row.get("price")
                        or row.get("base_cost_price")
                    )

                    if price is None:
                        continue

                    try:
                        price_f = float(price)
                    except Exception:
                        # 非法价格，跳过
                        continue

                    record = {
                        "supplier_id": str(supplier_id),
                        "sku_id": row.get("upc") or row.get("sku_id") or sku_id,
                        "date": dt.isoformat(),
                        "price": price_f,
                        "category": row.get("category") or row.get("CATEGORY"),
                        "transport_days_min": row.get("transport_days_min"),
                        "transport_days_max": row.get("transport_days_max"),
                        "raw": row,
                    }

                    sku_key = str(record["sku_id"])
                    if self.allowed_sku_ids and sku_key not in self.allowed_sku_ids:
                        continue

                    self._sku_date_prices.setdefault(sku_key, {}).setdefault(
                        record["date"], []
                    ).append(record)

        # 保持结果有序（按 supplier_id）
        for sku_map in self._sku_date_prices.values():
            for date_key, entries in sku_map.items():
                entries.sort(key=lambda x: x["supplier_id"])

    def get_sku_date(self, sku_id: str, target_date: date | str) -> List[Dict[str, Any]]:
        """
        获取某日的某个 SKU 的所有供给商价格列表（可能为空）。
        如果启用了新闻模块，会根据新闻的供给影响调整价格：
        - supply_effect > 0（供给增加）→ 价格下降
        - supply_effect < 0（供给减少）→ 价格上升
        """
        dt = target_date if isinstance(target_date, date) else _parse_date(target_date)
        if dt is None:
            return []
        
        date_key = dt.isoformat()

        raw_data = list(self._sku_date_prices.get(str(sku_id), {}).get(date_key, []))

        if raw_data and self.news_manager:
            category = raw_data[0].get("category")
            supply_info = self.news_manager.evaluate_impact_for_sku(
                sku_id=sku_id,
                sku_category=category,
                impact_factors=["supply"],
            )
            supply_effect = float(supply_info.get("total_effect", 0.0) or 0.0)
            
            # 根据供给影响调整价格：
            # supply_effect > 0 表示供给增加 → 价格下降（乘以 < 1 的系数）
            # supply_effect < 0 表示供给减少 → 价格上升（乘以 > 1 的系数）
            # 例如 supply_effect = 0.1 → price_multiplier = 0.9（价格下降 10%）
            #     supply_effect = -0.1 → price_multiplier = 1.1（价格上升 10%）
            price_multiplier = 1.0 - supply_effect
            
            # 深拷贝避免修改原始数据，并调整价格
            adjusted_data = []
            for entry in raw_data:
                adjusted_entry = deepcopy(entry)
                original_price = adjusted_entry.get("price", 0.0)
                adjusted_entry["price"] = original_price * price_multiplier
                adjusted_entry["original_price"] = original_price  # 保留原始价格用于参考
                adjusted_entry["supply_effect"] = supply_effect  # 记录影响系数
                adjusted_data.append(adjusted_entry)
            
            return adjusted_data

        return raw_data

    def step(self, current_date: date, record_manager=None) -> None:
        """写入当日供给商价格记录到 record_manager（若提供）；使用 get_sku_date 统一逻辑。"""
        if record_manager is None:
            return
        target_skus = self.allowed_sku_ids or self._sku_date_prices.keys()
        for sku_id in target_skus:
            entries = self.get_sku_date(sku_id, current_date)
            for entry in entries:
                record_manager.add_supplier_price(
                    SupplierPriceRecord(
                        supplier_id=entry["supplier_id"],
                        sku_id=entry["sku_id"],
                        date_obj=current_date,
                        price=entry["price"],
                    )
                )


    def get_quality_score(
        self,
        supplier_id: str,
        sku_id: str,
        target_date: date | str,
    ) -> Optional[float]:
        """
        获取指定供应商在特定日期对某个 SKU 的质量分数（如果数据中提供）。
        依赖 *_suppliers.json 里的原始字段（raw）。
        """
        dt = target_date if isinstance(target_date, date) else _parse_date(target_date)
        if dt is None:
            return None
        date_key = dt.isoformat()
        entries = self._sku_date_prices.get(str(sku_id), {}).get(date_key, [])
        for entry in entries:
            if entry.get("supplier_id") == str(supplier_id):
                raw = entry.get("raw") or {}
                for key in ("quality_score", "quality", "qualityScore"):
                    if key in raw:
                        try:
                            return float(raw[key])
                        except Exception:
                            continue
        return None

    def get_transport_range(
        self,
        supplier_id: str,
        sku_id: str,
        target_date: date | str,
    ) -> Optional[Tuple[int, int]]:
        """
        获取运输时间范围 (min, max)；若数据缺失则返回 None。
        """
        dt = target_date if isinstance(target_date, date) else _parse_date(target_date)
        if dt is None:
            return None
        date_key = dt.isoformat()
        entries = self._sku_date_prices.get(str(sku_id), {}).get(date_key, [])
        for entry in entries:
            if entry.get("supplier_id") != str(supplier_id):
                continue
            t_min = entry.get("transport_days_min")
            t_max = entry.get("transport_days_max")
            if t_min is None or t_max is None:
                raw = entry.get("raw") or {}
                t_min = raw.get("transport_days_min")
                t_max = raw.get("transport_days_max")
            try:
                return int(t_min), int(t_max)
            except Exception:
                continue
        return None

    def get_available_suppliers(self) -> List[str]:
        """
        Return all supplier_ids present in the loaded supplier files.
        """
        supplier_ids = set()
        for date_map in self._sku_date_prices.values():
            for entries in date_map.values():
                for entry in entries:
                    supplier_ids.add(entry.get("supplier_id"))

        supplier_ids.discard(None)
        return sorted(supplier_ids)
