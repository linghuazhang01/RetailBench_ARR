from __future__ import annotations
import argparse
import json
from datetime import date, timedelta
import math
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib

from util.map_date_to_week import calc_week_from_base
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model.sku_model import fit_logit_demand_model

class SKU:
    """
    单个 SKU，对应具体售卖单元。
    只实现图中给出的 4 个方法：
    - __init__
    - set_price
    - get_sales
    - get_attribute
    """

    def __init__(
        self,
        sku_id: str,
        category: str,
        init_price: float = 0.0,
        model_parameters: Dict[str, Any] = None,  # Changed from Optional[Dict[str, float]] to Optional[str]
        description: Dict[str, Any] = None,
        brand: str = None,
        category_effect: Optional[float] = None,
        promotion_day: int = 7,
    ) -> None:
        self.sku_id = sku_id
        self.price = init_price  # Fixed: was 'price' but should be 'init_price'
        self.sales: int = 0  # 累计销量
        self.category_effect = category_effect or 0.0
        self.category = category
        self.brand = brand

        # Extract model parameters if provided
        if model_parameters:
            self.alpha = model_parameters.get('alpha', 0.0)
            self.beta = model_parameters.get('beta', 0.0)
            self.sigma = model_parameters.get('sigma', 0.0)
            self.beta_s = model_parameters.get('beta_s', 0.0)
            self.beta_c = model_parameters.get('beta_c', 0.0)
        else:
            self.alpha = 0.0
            self.beta = 0.0
            self.sigma = 0.0
            self.beta_s = 0.0
            self.beta_c = 0.0

        self.model_params = model_parameters
        self.attributes = description
        self.promotion_day = promotion_day

    def set_price(self, price: float) -> None:
        """设置当前价格。"""
        self.price = float(price)

    def set_same_category_skus(self, skus: List[SKU]) -> None:
        """设置同品类 SKU 列表。"""
        self.same_category_skus = skus

    def get_attribute_attraction(self, current_date: date, external_attraction_rate: float = 0) -> float:
        """
        attraction = exp( alpha + beta*price + Normal(0, sigma) )

        与 simulate() 完全一致，不除以 sigma。
        """

        if external_attraction_rate is None:
            external_attraction_rate = 0

        week = calc_week_from_base(current_date)

        sin_k = np.sin(2 * np.pi * week / 52)
        cos_k = np.cos(2 * np.pi * week / 52)
        noise = np.random.normal(0.0, self.sigma) if self.sigma > 0 else 0.0
        V = self.alpha + (self.beta + self.beta_s * sin_k + self.beta_c * cos_k)  * self.price + (noise / 5)
        
        attraction = np.exp(V) * (1 + external_attraction_rate)

        if attraction < 0:
            attraction = 0

        # print(
        #     f"Value: {V}, External Attraction Rate: {external_attraction_rate}, Raw Attracgtion: {attraction}, Attraction: {np.exp(attraction)}"
        # )
        
        return attraction if attraction > 0 else 0
    
    def compute_attribute_attraction(self, date, external_attraction_rate: float = 0) -> float:
        self.today_attribute_attraction = self.get_attribute_attraction(date, external_attraction_rate)

        if self.today_attribute_attraction is None:
            import pdb; pdb.set_trace()
        return self.today_attribute_attraction
    
    def compute_attraction(self, currnet_market_skus_ids: List[str]) -> float:

        current_same_skus = [item for item in self.same_category_skus if item.sku_id in currnet_market_skus_ids]

        self.today_attraction = self.today_attribute_attraction

        if self.category_effect:
            for sku in current_same_skus:
                if isinstance(sku, SKU):
                    self.today_attraction += sku.category_effect * sku.today_attribute_attraction

        # print(f"Today Attraction: {self.today_attraction}")

        if self.today_attraction < 0:
            self.today_attraction = 0
        
        return self.today_attraction

    def get_sales(self, currnet_market_skus_ids: List[str], customer_count: int = 100) -> int:
        """返回当前销量（可由外部逻辑修改 self.sales）。
        在原有 logit 模型基础上，叠加同品类 SKU 的影响。
        """
        current_same_skus = [item for item in self.same_category_skus if item.sku_id in currnet_market_skus_ids]

        # 1. 计算同品类 SKU 的总 effect
        total_exp_cateogory_effect = self.today_attraction

        for sku in current_same_skus:
            # 如果列表里直接放的是 SKU 实例
            if isinstance(sku, SKU):
                total_exp_cateogory_effect += sku.today_attraction

        logit_prob = total_exp_cateogory_effect

        # 4. 通过 sigmoid 得到购买概率
        prob = logit_prob / (1 + logit_prob)

        if not customer_count:
            import pdb; pdb.set_trace()

        try:
            expected_sales = int(np.random.binomial(customer_count, prob))
        except Exception as e:
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()
            print(e)

        if total_exp_cateogory_effect == 0:
            raw_sales = 0
        else:
            raw_sales = expected_sales * (self.today_attraction) / total_exp_cateogory_effect

        try:
            if raw_sales < 0:
                self.sales = 0
            else:
                base = math.floor(raw_sales)
                fraction = raw_sales - base

                self.sales = base + (1 if random.random() < fraction else 0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()
            print(e)

        
            
        return self.sales

    
    def get_attributes(self) -> Any:
        """获取 SKU 属性。"""
        return self.attributes

    def __repr__(self) -> str:
        """Return human-readable SKU info."""
        return (
            f"SKU Info:\n"
            f"  SKU ID: {self.sku_id}\n"
            # f"  Price: {self.price}\n"
            # f"  Sales: {self.sales}\n"
            # f"  Alpha: {self.alpha}\n"
            # f"  Beta: {self.beta}\n"
            # f"  Sigma: {self.sigma}\n"
            f"  Attributes: {self.attributes}\n"
            # f"  Today Attraction: {self.today_attraction}"
        )


class Merchandise:
    """
    商品维度（一个商品对应一个 SKU）。
    只实现图中给出的 2 个方法：
    - __init__
    - get_attributes
    """

    def __init__(
        self,
        sku: SKU,
        begin_time: date,
        expired_time: date,
        supplier_id: str,
        merch_id: str,
        quality_score: float,
        buy_price: float = 0.0,
    ) -> None:
        self.sku = sku
        self.begin_time = begin_time
        self.expired_time = expired_time
        self.buy_price = buy_price
        # If merch_id is not provided, use sku_id as the merch_id
        self.merch_id = merch_id if merch_id is not None else sku.sku_id
        self.quality_score = quality_score
        self.supplier_id = supplier_id

    def get_attributes(self) -> Dict[str, Any]:
        """读取商品属性。"""
        return {
            "supplier_id": self.supplier_id,           
            "sku_id": getattr(self.sku, "sku_id", None),
            "sku_name": getattr(self.sku, "sku_id", None),  # Using sku_id as name if no name attribute exists
            "category": getattr(self.sku, "attributes", {}).get("category", None),  # Get category from sku attributes
            "begin_time": self.begin_time,
            "expired_time": self.expired_time,
            "buy_price": getattr(self, "buy_price", None),
            "quality_score": getattr(self, "quality_score", None),
        }

    def judge_expired(self, current_date: date) -> bool:
        """判断商品是否过期。"""
        return current_date > self.expired_time

    def __repr__(self) -> str:
        """Return human-readable merchandise info."""
        return (
            f"Merchandise Info:\n"
            f"  SKU ID: {getattr(self.sku, 'sku_id', None)}\n"
            f"  Buy Price: {self.buy_price}\n"
            f"  Begin Time: {self.begin_time}\n"
            f"  Expire Time: {self.expired_time}\n"
            f"  Merchandise ID: {self.merch_id}"
            f"  Supplier ID: {self.supplier_id}"
        )



def simulate_all_store_skus(
    store_root: str = "data/store",
    price_min: float = 0.5,
    price_max: float = 10.0,
    price_step: float = 0.25,
    customer_count: int = 1000,
    plot_dir: str = "price_sales_plots",
    batch_size: int = 20,
) -> Dict[str, Dict[str, List[int]]]:
    """
    从 data/store/*/sku_model_parameter.json 读取模型参数，
    对每个门店、每个 SKU 在给定价格列表下模拟销量。
    返回结构: {store_id: {sku_id: [sales_for_each_price]}}
    """
    prices = list(np.arange(price_min, price_max + 1e-9, price_step))
    store_sales: Dict[str, Dict[str, List[int]]] = {}
    root = Path(store_root)
    if not root.exists():
        print(f"[WARN] store_root 不存在: {store_root}")
        return store_sales

    for store_dir in root.iterdir():
        if not store_dir.is_dir():
            continue
        param_file = store_dir / "sku_model_parameter.json"
        if not param_file.exists():
            continue

        store_id = store_dir.name
        with param_file.open("r", encoding="utf-8") as f:
            params = json.load(f)

        store_sales[store_id] = {}
        for sku_id, model_params in params.items():
            sku = SKU(
                sku_id=sku_id,
                init_price=prices[0],
                model_parameters=model_params,
                category_effect=0.0,
                category="all",
            )
            sku.set_same_category_skus([])

            sales = []
            for p in prices:
                sku.set_price(p)
                sku.compute_attribute_attraction(
                    date=date.today(),
                )
                sku.compute_attraction(
                    currnet_market_skus_ids=list(params.keys())
                )
                sales.append(sku.get_sales(currnet_market_skus_ids=list(params.keys()),customer_count=customer_count))
            store_sales[store_id][sku_id] = sales

        print(f"[INFO] Store {store_id} 模拟完成，SKU 数: {len(store_sales[store_id])}")
        _plot_price_sales(store_id, store_sales[store_id], prices, plot_dir, batch_size)

    return store_sales


def simulate_and_select_by_category(
    category_root: str = "data/filtered_source_data_by_category",
    output_dir: str = "data/top_sku_by_category",
    top_n: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    对 data/filtered_source_data_by_category 下的每个 SKU 拟合参数，
    在每个类目内按 |beta| 降序、sigma 升序选出前 top_n 个，
    并将对应 csv 与参数写到输出目录。

    返回结构:
        {category: {sku_id: {"alpha": float, "beta": float, "sigma": float}}}
        仅包含已选出的 SKU。
    """
    root = Path(category_root)
    if not root.exists():
        print(f"[WARN] category_root 不存在: {category_root}")
        return {}

    # 允许传入单个类目目录（直接包含 csv），否则默认读取子目录作为类目。
    category_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not category_dirs and any(root.glob("*.csv")):
        category_dirs = [root]

    selected_params: Dict[str, Dict[str, Dict[str, float]]] = {}
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for category_dir in sorted(category_dirs):
        category_name = category_dir.name
        sku_params: Dict[str, Dict[str, float]] = {}

        for csv_path in sorted(category_dir.glob("*.csv")):
            sku_id = csv_path.stem
            try:
                alpha, beta_0, beta_s, beta_c, sigma = fit_logit_demand_model(str(csv_path))

                # 保证自身价格敏感性为负（可根据你的业务需要调整）
                if beta_0 >= 0:
                    continue

                params = {
                    "alpha": alpha,
                    "beta": beta_0,
                    "beta_s": beta_s,
                    "beta_c": beta_c,
                    "sigma": sigma,
                }
                sku_params[sku_id] = params
            except Exception as exc:
                print(f"[ERROR] 拟合 {category_name}/{csv_path.name} 失败: {exc}")
                continue

        if not sku_params:
            print(f"[WARN] 类目 {category_name} 无可用参数，跳过")
            continue


        def sort_key(item):
            _sku_id, _params = item
            score = abs(_params["beta_s"]) + abs(_params["beta_c"])
            return (-score, _params["sigma"])

        sorted_items = sorted(sku_params.items(), key=sort_key)

        # 3. 选出前 top_n 个（如果不足就全选）
        if len(sorted_items) <= top_n:
            selected_items = sorted_items
        else:
            selected_items = sorted_items[:top_n]

        selected = dict(selected_items)
        selected_params[category_name] = selected

        # 4. 打印被选中 SKU 的 beta_s、beta_c
        print(f"[INFO] 类目 {category_name} 选出的 SKU：")
        for sku_id, params in selected.items():
            score = abs(params["beta_s"]) + abs(params["beta_c"])
            print(
                f"  - {sku_id}: "
                f"beta_s = {params['beta_s']:.6f}, "
                f"beta_c = {params['beta_c']:.6f}, "
                f"|beta_s| + |beta_c| = {score:.6f}, "
                f"sigma = {params['sigma']:.6f}"
            )

        dest_dir = out_root / category_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        param_file = dest_dir / "category.json"
        with param_file.open("w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2, ensure_ascii=False)

        for sku_id in selected:
            source_csv = category_dir / f"{sku_id}.csv"
            if not source_csv.exists():
                print(f"[WARN] 未找到 {source_csv}，跳过复制")
                continue
            shutil.copy2(source_csv, dest_dir / source_csv.name)

        print(f"[INFO] {category_name}: 选出 {len(selected)} / {len(sku_params)} 个 SKU")

    # 汇总所有类目的已选参数
    test = {}

    for k, v in selected_params.items():
        for sku_id, params in v.items():
            test[sku_id] = params
    all_param_file = out_root / "sku_model_parameter.json"
    with all_param_file.open("w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    return selected_params


def _plot_price_sales(
    store_id: str,
    sku_sales: Dict[str, List[int]],
    prices: List[float],
    out_dir: str,
    batch_size: int,
) -> None:
    """
    每 batch_size 个 SKU 画一张图，展示 price-sales 曲线。
    """
    if not sku_sales:
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sku_items = list(sku_sales.items())
    total = len(sku_items)
    num_batches = math.ceil(total / batch_size)

    for b in range(num_batches):
        batch = sku_items[b * batch_size : (b + 1) * batch_size]
        cols = min(5, batch_size)
        rows = math.ceil(len(batch) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

        for idx, (sku_id, sales) in enumerate(batch):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.plot(prices, sales, marker="o")
            ax.set_title(sku_id, fontsize=10)
            ax.set_xlabel("price")
            ax.set_ylabel("sales")
            ax.grid(True, linestyle="--", alpha=0.4)

        # 隐藏多余子图
        for idx in range(len(batch), rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")

        plt.tight_layout()
        filename = out_path / f"store_{store_id}_batch_{b+1}.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"[PLOT] {filename} (SKU {b*batch_size+1}-{min((b+1)*batch_size,total)}/{total})")



def main() -> None:
    # simulate_and_select_by_category(
    #     category_root='data/source_data_by_category',
    #     output_dir='data/simulate_data/15',
    #     top_n=5,
    # )

    simulate_all_store_skus(
        store_root='data/still/simulate_data',
        price_min=0.5,
        price_max=9999,
        price_step=0.25,
        customer_count=20000,
        plot_dir='data/plot',
        batch_size=20,
    )


if __name__ == "__main__":
    main()
