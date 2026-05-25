from __future__ import annotations
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Optional, List, Tuple
import json

class CustomerManager:
    """
    客户（或时间）管理器，这里主要用于维护“今天”的日期。
    """

    def __init__(self, begin_time: str, end_time: str, data_path: str, store_id: str) -> None:
        self.begin_time = datetime.strptime(begin_time, "%m/%d/%y").date()
        self.end_time = datetime.strptime(end_time, "%m/%d/%y").date()
        root = Path(data_path)
        candidates = [
            root / f"store_{store_id}" / f"store_{store_id}_data.json",
            root / f"{store_id}" / "data.json",
        ]
        self.file_path = next((str(p) for p in candidates if p.exists()), None)
        if self.file_path is None:
            raise FileNotFoundError(f"No customer data file found for store {store_id} under {data_path}")

        with open(self.file_path, "r") as f:
            data = json.load(f)

        self.data = []
        for record in data:
            if record.get("date") is None:
                continue
            try:
                record_date = datetime.strptime(record["date"], "%m/%d/%y").date()
            except Exception:
                continue
            if self.begin_time <= record_date <= self.end_time:
                self.data.append(record)

        self._sorted_dates = sorted(
            datetime.strptime(r["date"], "%m/%d/%y").date() for r in self.data
            if r.get("date") is not None
        )


    def next_date(self, current: date) -> date:
        """返回当前日期之后的数据日期，若无则 +1 天。"""
        next_date = next((d for d in self._sorted_dates if d > current), None)
        return next_date or (current + timedelta(days=1))

    def get_customer_count(self, target_date: Optional[date | str] = None) -> Optional[float]:
        """
        获取指定日期的客户数量。

        Args:
            target_date: datetime.date 或 'MM/DD/YY' / 'YYYY-MM-DD' 字符串。
        """
        if target_date is None:
            return None

        dt = target_date if isinstance(target_date, date) else None
        if dt is None:
            for fmt in ("%m/%d/%y", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(str(target_date), fmt).date()
                    break
                except Exception:
                    continue
        if dt is None:
            return None

        target_strs = {dt.strftime("%m/%d/%y"), dt.isoformat()}
        for record in self.data:
            if record['date'] in target_strs:
                return record.get('custcoun')
        return None

    def get_all_customer_counts(self) -> List[Tuple[date, float]]:
        """
        获取所有可用日期的客户数量数据。

        Returns:
            包含 (date, customer_count) 元组的列表，按日期排序。
            返回所有在数据范围内的有效记录。
        """
        result = []
        for record in self.data:
            date_str = record.get('date')
            custcoun = record.get('custcoun')
            if date_str and custcoun is not None:
                try:
                    dt = datetime.strptime(date_str, "%m/%d/%y").date()
                    result.append((dt, custcoun))
                except Exception:
                    continue
        result.sort(key=lambda x: x[0])  # 按日期排序
        return result

    def get_customer_counts_range(self, start_date: Optional[date] = None,
                                end_date: Optional[date] = None) -> List[Tuple[date, float]]:
        """
        获取指定日期范围内的所有客户数量数据。

        Args:
            start_date: 开始日期，默认使用数据范围的开始日期
            end_date: 结束日期，默认使用数据范围的结束日期

        Returns:
            包含 (date, customer_count) 元组的列表，按日期排序。
            只返回指定日期范围内有数据的记录。
        """
        # 使用默认范围
        if start_date is None:
            start_date = self.begin_time
        if end_date is None:
            end_date = self.end_time

        result = []
        for record in self.data:
            date_str = record.get('date')
            custcoun = record.get('custcoun')
            if date_str and custcoun is not None:
                try:
                    dt = datetime.strptime(date_str, "%m/%d/%y").date()
                    if start_date <= dt <= end_date:
                        result.append((dt, custcoun))
                except Exception:
                    continue
        result.sort(key=lambda x: x[0])  # 按日期排序
        return result
