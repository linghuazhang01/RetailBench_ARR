from __future__ import annotations
from datetime import date, timedelta
from typing import List, Optional, Dict, Any
import json
import random
from uuid import uuid4

from sku import SKU, Merchandise
from model.review_star import ReviewStarSmoothModel
from module.record_manager import RecordManager, ReviewRecord

WINDOW_DAYS = 60

class ReviewManager:
    """
    评论管理：
    1) 查看某个 SKU 的差评（默认 <= 2 星）
    2) 查看某个 SKU 的平均评分
    3) 查看某个 SKU 在时间段内的评价
    4) 计算评价对销量的影响（依赖 review_star 模型）
    """

    def __init__(
        self,
        record_manager: RecordManager,
        review_model_path: Optional[str] = None,
        review_source_path: Optional[str] = None,
        enabled: bool = True,
        gen_prob = 0.1,
    ) -> None:
        self.record_manager = record_manager
        self.enabled = enabled
        self.model = (
            ReviewStarSmoothModel(review_model_path)
            if enabled and review_model_path
            else None
        )
        self.gen_prob = gen_prob
        self.available_reviews_by_category: Dict[str, List[Dict[str, Any]]] = {}
        if review_source_path:
            try:
                with open(review_source_path, "r", encoding="utf-8") as f:
                    # 支持 jsonl 或 json(list)
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == "[":
                        rows = json.load(f)
                    else:
                        rows = [json.loads(line) for line in f if line.strip()]
                    for row in rows:
                        cat = str(row.get("CATEGORY") or row.get("category") or row.get("COM_CODE") or "").strip()
                        if cat:
                            self.available_reviews_by_category.setdefault(cat, []).append(row)
            except Exception:
                # 如果加载失败，不阻塞后续逻辑
                self.available_reviews_by_category = {}

    def get_negative_reviews(
        self,
        sku_id: str,
        max_rating: int = 2,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> List[ReviewRecord]:
        """
        返回评分 <= max_rating 的差评列表。
        """
        reviews = self.record_manager.read_reviews(
            sku_id=sku_id,
            start_date=start_date,
            end_date=end_date,
        )
        return [r for r in reviews if r.rating <= max_rating]

    def get_average_rating(
        self,
        sku_id: str,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> Optional[float]:
        """
        返回指定 SKU 的平均评分，若没有评论返回 None。
        """
        reviews = self.record_manager.read_reviews(
            sku_id=sku_id,
            start_date=start_date,
            end_date=end_date,
        )
        if not reviews:
            return None
        return sum(r.rating for r in reviews) / len(reviews)

    def get_reviews_in_range(
        self,
        sku_id: str,
        ratings: List[int],
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> List[ReviewRecord]:
        """
        获取某个 SKU 在时间范围内的所有评价。
        """
        reviews = self.record_manager.read_reviews(
            sku_id=sku_id,
            start_date=start_date,
            end_date=end_date,
        )
       
        reviews = [r for r in reviews if r.rating in ratings]

        return reviews

    def add_reviews(
        self,
        sku_id: str,
        category: Optional[str],
        rating: int,
        count: int,
        date_obj: Optional[date] = None,
        dimension: Optional[str] = None,
        merchandise_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
    ) -> List[ReviewRecord]:
        """
        写入指定数量的评价记录。

        Args:
            sku_id: SKU 编号
            rating: 星级
            count: 写入数量
            dimension: 维度 price/quality/other
            date_obj: 可选，评价日期，默认使用 today()
        """
        if count <= 0:
            return []
        date_obj = date_obj or date.today()
        if dimension:
            dimension = dimension.lower()
        else:
            dimension = random.choice(["price", "quality", "other"])

        inserted: List[ReviewRecord] = []
        candidates: List[Dict[str, Any]] = []
        if category and category in self.available_reviews_by_category:
            candidates.extend(self.available_reviews_by_category.get(category, []))

        dim_candidates = [
            r for r in candidates
            if r.get("DIMENSION") and r.get("DIMENSION").lower() == dimension
        ] or candidates

        same_rating = [r for r in dim_candidates if int(r.get("STAR", r.get("rating", rating))) == rating]

        for _ in range(count):
            chosen = None
            pool = same_rating or dim_candidates
            if pool:
                chosen = random.choice(pool)
                comment_body = chosen.get("COMMENTS") or chosen.get("comment")
                comment = f"{comment_body}" if comment_body else f"[{dimension}]"
                picked_dim = chosen.get("DIMENSION", None) or dim
            else:
                raise ValueError(f"Unable to find matched data, category={category}, rating={rating}, dimension={dimension}")
            
            review = ReviewRecord(
                record_id=uuid4().hex,
                upc=sku_id,
                category=category,
                date_obj=date_obj,
                rating=rating,
                comment=comment,
                dimension=picked_dim,
                merchandise_id=merchandise_id,
                supplier_id=supplier_id,
            )
            
            self.record_manager.add_review(review)
            inserted.append(review)

        return inserted

    def maybe_generate_review_for_merchandise(
        self,
        merchandise: Merchandise,
        date_obj: Optional[date] = None,
        add_to_db: bool = True,
    ) -> Optional[ReviewRecord]:
        """
        基于 merchandise.quality_score 决定是否生成一条评论。
        - 质量分越高，生成概率和星级越高
        - 若提供 gen_prob（0~1），直接使用该概率；否则按 quality_score 映射
        - rating 基于 quality_score 映射到 1~5 星，若有 review_star 模型可用于后续需求调整
        """
        if merchandise is None:
            return None
        q = getattr(merchandise, "quality_score", None)

        if random.random() > self.gen_prob:
            return None
        
        rating = ReviewStarSmoothModel.simulate_ratings(q, n=1)[0]

        is_bad_quality = merchandise.quality_score < 3

        sku_id = merchandise.sku.sku_id
        category = merchandise.sku.category

        if rating < 3:
            if is_bad_quality:
                dim = random.choices(
                    population=['price', 'quality', 'other'],
                    weights=[0.1, 0.8, 0.1],
                    k=1
                )[0]
            elif merchandise.buy_price * 3 < merchandise.sku.price:
                dim = random.choices(
                    ['price', 'quality', 'other'],
                    weights=[0.8, 0.1, 0.1],
                    k=1
                )[0]
            else:
                dim = random.choice(['price', 'quality', 'other'])
        else:
            dim = random.choice(['price', 'quality', 'other'])


        inserted = self.add_reviews(
            sku_id=sku_id,
            category=category,
            rating=rating,
            count=1,
            dimension=dim,
            date_obj=date_obj,
            merchandise_id=merchandise.merch_id,
            supplier_id=merchandise.supplier_id,
        )
        
        review = inserted[0] if inserted else None
        if review and not add_to_db:
            return review
        return review

    def compute_sales_impact(
        self,
        sku: SKU,
        current_date: date,
    ) -> Optional[float]:
        """
        计算当前评价对销量的相对提升比例（基于 review_star 模型）。

        返回值：
            - float: 增益系数（例 0.1 表示 +10%）
            - None: 无模型或无评分数据。
        """
        if not self.enabled or self.model is None:
            return 0

        # 滑窗向后寻找最近的评分（按 WINDOW_DAYS 向前滚动，最多尝试 6 个窗口）
        window_days = WINDOW_DAYS
        max_windows = 20
        rating_value: Optional[float] = None
        window_end = current_date
        window_start = window_end - timedelta(days=window_days)

        for _ in range(max_windows):
            rating_value = self.get_average_rating(
                sku_id=sku.sku_id,
                start_date=window_start,
                end_date=window_end,
            )
            if rating_value is not None:
                break
            window_start -= timedelta(days=window_days)

        # 若仍无评论，回退到初始评分；若也没有初始评分，则视为无增益
        init_rating_value = getattr(sku, "initial_rating", None)
        if rating_value is None:
            if init_rating_value is None:
                return 0
            rating_value = init_rating_value


        try:
            return float(self.model.predict(sku.category, rating_value)) - float(self.model.predict(sku.category, init_rating_value))
        except Exception as e:
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()
            print(e)
            
            return 0
