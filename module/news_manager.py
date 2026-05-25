from __future__ import annotations

import base64
import json
import pickle
import random
from copy import deepcopy
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from util.logger import get_logger


class NewsManager:
    """
    精简新闻管理器：
    1) 读取新闻文件并按 mode 分类
    2) 记录今日新闻
    3) step：商业新闻连续出现在今日 3~5 天
    4) 按比例采样新的新闻
    """

    def __init__(
        self,
        news_source_path: Optional[str] = None,
        sample_ratios: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
        daily_news_count: Optional[int] = None,
        allowed_categories: Optional[List[str]] = None,
        allowed_sku_ids: Optional[List[str]] = None,
        current_date: date = None,
        impact_mode_weights: Optional[Dict[str, float]] = None,
        impact_base_scale: float = 0.2,
        impact_abs_cap: Optional[float] = None,
    ) -> None:
        self._random_seed = random_seed  # 保存 seed 以便 checkpoint
        self._rng = random.Random(random_seed)
        self.sample_ratios = self._normalize_ratios(sample_ratios)
        self.daily_news_count = int(daily_news_count or 0)
        self.allowed_categories = {str(c) for c in allowed_categories} if allowed_categories else None
        self.allowed_sku_ids = {str(s) for s in allowed_sku_ids} if allowed_sku_ids else None
        # 不同 mode 的影响权重 & 全局缩放系数：从 config 传入
        self.impact_mode_weights: Dict[str, float] = {
            str(k).lower(): float(v) for k, v in (impact_mode_weights or {}).items()
        }
        self.impact_base_scale: float = float(impact_base_scale)
        self.impact_abs_cap: Optional[float] = (
            float(impact_abs_cap) if impact_abs_cap is not None else None
        )
        if self.impact_abs_cap is not None and self.impact_abs_cap < 0:
            raise ValueError("impact_abs_cap must be non-negative")
        self.news_by_mode: Dict[str, List[Dict[str, Any]]] = {}
        self.today_news: List[Dict[str, Any]] = []
        self._rolling: List[Dict[str, Any]] = []  # 商业新闻滚动展示（含 _remain_days）
        self.logger = get_logger()
        if news_source_path:
            self.load(news_source_path, current_date)

    def load(self, path: str, current_date) -> None:
        p = Path(path)
        if not p.exists():
            self.news_by_mode = {}
            self.logger.debug(f"[NewsLoad] 新闻文件不存在: {path}")
            return
        text = p.read_text(encoding="utf-8")
        items = json.loads(text) if text.lstrip().startswith("[") else [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
        self.news_by_mode.clear()
        loaded_count = 0
        for raw in items:
            item = self._normalize(raw)
            if not self._allow_item(item):
                continue
            self.news_by_mode.setdefault(item["mode"], []).append(item)
            loaded_count += 1
        
        # 统计各 mode 的新闻数量
        mode_stats = {mode: len(news_list) for mode, news_list in self.news_by_mode.items()}
        self.logger.debug(
            f"[NewsLoad] 从 {path} 加载了 {loaded_count} 条新闻, "
            f"按 mode 分布: {mode_stats}, daily_news_count={self.daily_news_count}"
        )
        
        if self.daily_news_count > 0 and self.news_by_mode:
            self.step(record_manager=None, current_date=current_date)

    def step(self, record_manager=None, current_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """
        推进一天：
        - 将昨天的新闻（self.today_news）记录到数据库（如果提供 record_manager）
        - 续播商业新闻（总出现 3~5 天）
        - 采样新新闻加入今日，商业新闻纳入续播队列
        
        注意：current_date 应该是"今天"的日期，self.today_news 是"昨天"的新闻列表
        """
        current_date = current_date or date.today()
        next_today: List[Dict[str, Any]] = []
        next_rolling: List[Dict[str, Any]] = []

        if record_manager is not None:
            from module.record_manager import NewRecord  # 局部导入避免循环
            # 记录昨天的新闻（self.today_news）到数据库，日期为 current_date（今天）
            for news in self.today_news:
                self._ensure_id(news)
                record_id = f"{news['id']}_{current_date.isoformat()}"
                rec = NewRecord(
                    record_id=record_id,
                    news_id=news["id"],
                    title=news.get("title", ""),
                    content=news.get("content", ""),
                    date_obj=current_date,
                )
                record_manager.add_news_record(rec)

        # 续播已滚动的商业新闻
        rolling_count = 0
        for item in self._rolling:
            self._ensure_id(item)
            remain = item.get("_remain_days", 0) - 1
            if remain >= 0:
                item["_remain_days"] = remain
                next_today.append(item)
                next_rolling.append(item)
                rolling_count += 1
                self.logger.debug(
                    f"[NewsStep] 续播新闻: id={item.get('id', 'N/A')[:8]}, "
                    f"title={item.get('title', '')[:50]}, remain_days={remain}, "
                    f"mode={item.get('mode', 'N/A')}, impact_factor={item.get('impact_factor', 'N/A')}"
                )

        ## 每天采样的新闻数量需要一样
        new_count = self.daily_news_count - len(next_rolling)

        # 新采样
        new_items = self.sample_news(new_count) if new_count > 0 else []
        for item in new_items:
            self._ensure_id(item)
            if self._is_business(item):
                copied = deepcopy(item)
                copied["_remain_days"] = self._rng.randint(3, 5)  # 加上今天共 3~5 天
                next_today.append(copied)
                next_rolling.append(copied)
                self.logger.debug(
                    f"[NewsStep] 新采样商业新闻: id={copied.get('id', 'N/A')[:8]}, "
                    f"title={copied.get('title', '')[:50]}, remain_days={copied['_remain_days']}, "
                    f"mode={copied.get('mode', 'N/A')}, impact_factor={copied.get('impact_factor', 'N/A')}, "
                    f"impact_direction={copied.get('impact_direction', 'N/A')}, "
                    f"impact_strength={copied.get('impact_strength', 'N/A')}"
                    f"target_category={copied.get('target_category', 'N/A')}"
                    f"target_sku_upc={copied.get('target_sku_upc', 'N/A')}"
                )
            else:
                next_today.append(deepcopy(item))
                self.logger.debug(
                    f"[NewsStep] 新采样普通新闻: id={item.get('id', 'N/A')[:8]}, "
                    f"title={item.get('title', '')[:50]}, mode={item.get('mode', 'N/A')}"
                )

        self._rolling = next_rolling
        self.today_news = deepcopy(next_today)

        # 汇总今日新闻
        self.logger.debug(
            f"[NewsStep] {current_date} 今日新闻汇总: 总数={len(self.today_news)}, "
            f"续播={rolling_count}, 新采样={len(new_items)}, 滚动队列={len(self._rolling)}"
        )

        return deepcopy(self.today_news)

    def get_today_news(self) -> List[Dict[str, Any]]:
        """获取今日新闻列表。"""
        if self.today_news:
            # 按 mode 和 impact_factor 统计
            mode_count = {}
            factor_count = {}
            business_count = 0
            for news in self.today_news:
                mode = news.get("mode", "unknown")
                mode_count[mode] = mode_count.get(mode, 0) + 1
                factor = news.get("impact_factor", "none")
                factor_count[factor] = factor_count.get(factor, 0) + 1
                if self._is_business(news):
                    business_count += 1
            
            self.logger.debug(
                f"[NewsToday] 今日新闻总数={len(self.today_news)}, "
                f"商业新闻={business_count}, mode分布={mode_count}, factor分布={factor_count}"
            )
        else:
            self.logger.debug("[NewsToday] 今日无新闻")
        
        return deepcopy(self.today_news)

    def evaluate_impact_for_sku(
        self,
        sku_id: str,
        sku_category: Optional[str] = None,
        impact_factors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        汇总今日新闻对指定 SKU 的影响。
        impact_factors: 允许的影响维度（如 ['need','supply']），为空则不过滤。
        返回 {"total_effect": float, "matched_news": List[Dict]}。
        """
        factors = {f.lower() for f in impact_factors} if impact_factors else None
        matched: List[Dict[str, Any]] = []
        total = 0.0
        
        self.logger.debug(
            f"[NewsImpact] 评估 SKU {sku_id} (category={sku_category}) 的新闻影响, "
            f"impact_factors={impact_factors}, 今日新闻总数={len(self.today_news)}"
        )
        
        for news in self.today_news:
            if not self._match_sku(news, sku_id, sku_category):
                continue
            factor = str(news.get("impact_factor") or "").lower()
            if factors and factor not in factors:
                self.logger.debug(
                    f"[NewsImpact] SKU {sku_id} 新闻 {news.get('id', 'N/A')[:8]} 不匹配 factor 过滤: "
                    f"news_factor={factor}, required={factors}"
                )
                continue
            
            strength = self._signed_strength(news)
            mode = str(news.get("mode") or "").lower()
            mode_weight = self.impact_mode_weights.get(mode, 1.0)
            contribution = strength * mode_weight
            total += contribution
            
            matched.append(deepcopy(news))
            
            self.logger.debug(
                f"[NewsImpact] SKU {sku_id} 匹配到新闻: id={news.get('id', 'N/A')[:8]}, "
                f"title={news.get('title', '')[:50]}, mode={mode}, factor={factor}, "
                f"direction={news.get('impact_direction', 'N/A')}, "
                f"strength={news.get('impact_strength', 'N/A')}, "
                f"mode_weight={mode_weight:.3f}, contribution={contribution:.4f}"
            )

        uncapped_effect = total * self.impact_base_scale
        final_effect = self._clamp_effect(uncapped_effect)
        self.logger.debug(
            f"[NewsImpact] SKU {sku_id} 影响汇总: 匹配新闻数={len(matched)}, "
            f"raw_total={total:.4f}, base_scale={self.impact_base_scale}, "
            f"abs_cap={self.impact_abs_cap}, "
            f"final_effect={final_effect:.4f}"
        )

        return {"total_effect": final_effect, "matched_news": matched}

    def sample_news(self, count: int) -> List[Dict[str, Any]]:
        if count <= 0 or not self.news_by_mode:
            return []
        modes = list(self.news_by_mode.keys())
        ratios = self._ratios_for_modes(modes)
        picked: List[Dict[str, Any]] = []
        for _ in range(count):
            mode = self._rng.choices(modes, weights=[ratios[m] for m in modes], k=1)[0]
            pool = self.news_by_mode.get(mode) or []
            if not pool:
                continue
            picked.append(deepcopy(self._rng.choice(pool)))
        return picked

    # -------- 工具 --------
    def _normalize(self, item: Dict[str, Any]) -> Dict[str, Any]:
        news_id = item.get("id") or item.get("record_id") or item.get("recordId")
        mode = str(item.get("mode") or item.get("IMPACT_SCOPE") or item.get("MODE") or "neutral").lower()
        direction = str(item.get("impact_direction") or item.get("IMPACT_DIRECTION") or "").lower()
        return {
            "id": news_id,
            "mode": mode,
            "title": item.get("title") or item.get("TITLE") or "",
            "content": item.get("content") or item.get("CONTENT") or "",
            "impact_direction": direction,
            "impact_strength": item.get("impact_strength") or item.get("IMPACT_STRENGTH"),
            "impact_factor": item.get("impact_factor") or item.get("IMPACT_FACTOR"),
            "impact_categories": item.get("impact_categories") or item.get("IMPACT_CATEGORIES") or [],
            "target_category": item.get("target_category") or item.get("TARGET_CATEGORY"),
            "target_sku_upc": item.get("target_sku_upc") or item.get("TARGET_SKU_UPC") or item.get("SKU_UPC"),
            "raw": item,
        }

    def _is_business(self, item: Dict[str, Any]) -> bool:
        return str(item.get("impact_direction") or "").lower() in {"increase", "decrease"}

    def _match_sku(self, news: Dict[str, Any], sku_id: str, sku_category: Optional[str]) -> bool:
        if news.get("target_sku_upc") and str(news.get("target_sku_upc")) == str(sku_id):
            return True
        if news.get("target_category") and sku_category and str(news.get("target_category")) == str(sku_category):
            return True
        return str(news.get("mode") or "").lower() in ["macro_all"]

    def _signed_strength(self, news: Dict[str, Any]) -> float:
        direction = str(news.get("impact_direction") or "").lower()
        strength = news.get("impact_strength")
        try:
            val = float(strength) if strength is not None else 0.0
        except Exception:
            val = 0.0
        if direction == "increase":
            return val
        if direction == "decrease":
            return -val
        return 0.0

    def _clamp_effect(self, effect: float) -> float:
        if self.impact_abs_cap is None:
            return effect
        return max(-self.impact_abs_cap, min(self.impact_abs_cap, effect))

    def _normalize_ratios(self, ratios: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not ratios:
            return None
        pos = {str(k).lower(): float(v) for k, v in ratios.items() if v and float(v) > 0}
        total = sum(pos.values())
        return {k: v / total for k, v in pos.items()} if total > 0 else None

    def _ratios_for_modes(self, modes: List[str]) -> Dict[str, float]:
        if not self.sample_ratios:
            return {m: 1 for m in modes}
        weights = {m: self.sample_ratios.get(m, 0) for m in modes}
        total = sum(weights.values())
        return {m: (weights[m] / total if total > 0 else 1) for m in modes}

    # -------- 历史查询（依赖 record_manager） --------
    def fetch_news_detail(self, record_manager, news_id: str, current_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        # 首先从内存中的 today_news 查找（可能还没有保存到数据库）
        for news in self.today_news:
            if news.get("id") == news_id or news.get("record_id") == news_id:
                # 从 today_news 中找到，返回详细信息
                # 如果新闻对象没有 date 字段，使用传入的 current_date
                news_date = news.get("date")
                if news_date is None and current_date:
                    news_date = current_date.isoformat() if isinstance(current_date, date) else str(current_date)
                elif news_date and isinstance(news_date, date):
                    news_date = news_date.isoformat()
                
                return {
                    "record_id": news.get("record_id") or f"{news.get('id')}_{news_date or ''}",
                    "id": news.get("id"),
                    "title": news.get("title", ""),
                    "content": news.get("content", ""),
                    "date": news_date or "",
                }
        
        # 如果内存中没找到，再从数据库查找
        if record_manager is None:
            return None
        rows = record_manager.read_news(news_id=news_id, limit=1)
        if not rows:
            rows = record_manager.read_news(record_id=news_id, limit=1)
        if not rows:
            return None
        row = rows[-1]
        return {
            "record_id": row.id,
            "id": row.news_id,
            "title": row.title,
            "content": row.content,
            "date": row.date.isoformat(),
        }

    def fetch_news_history(self, record_manager, start: date, end: date) -> List[Dict[str, Any]]:
        if record_manager is None:
            return []
        rows = record_manager.read_news(start_date=start, end_date=end)
        return [
            {
                "record_id": r.id,
                "id": r.news_id,
                "title": r.title,
                "content": r.content,
                "date": r.date.isoformat(),
            }
            for r in rows
        ]

    def _allow_item(self, item: Dict[str, Any]) -> bool:
        """Filter by allowed categories/SKUs when provided."""
        ## SKU_LEVEL 

        if item['mode'] == 'macro_all':
            return True
        if self.allowed_sku_ids and item.get("target_sku_upc"):
            if str(item["target_sku_upc"]) not in self.allowed_sku_ids:
                return False

        
        if self.allowed_categories and item.get("target_category"):
            if str(item["target_category"]) not in self.allowed_categories:
                return False
        return True

    def _ensure_id(self, item: Dict[str, Any]) -> None:
        if not item.get("id"):
            item["id"] = uuid4().hex
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将新闻管理器状态序列化为字典，用于 checkpoint。
        
        Returns:
            更新后的 checkpoint_data，包含新闻管理器状态
        """
        # 保存 Random 对象的状态（通过 getstate，使用 pickle 序列化）
        rng_state = self._rng.getstate()
        rng_state_serialized = base64.b64encode(pickle.dumps(rng_state)).decode('utf-8')
        
        news_state = {
            "today_news": self.today_news,
            "_rolling": self._rolling,
            "rng_state": rng_state_serialized,  # Random 对象的状态（base64 编码的 pickle）
            "random_seed": self._random_seed,  # 保存原始 seed
        }
        
        checkpoint_data["news_manager"] = news_state
        return checkpoint_data
    
    def recover_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """
        从 checkpoint 数据恢复新闻管理器状态。
        
        Args:
            checkpoint_data: checkpoint 数据字典
        """
        news_state = checkpoint_data.get("news_manager", {})
        
        # 恢复 today_news 和 _rolling
        self.today_news = news_state.get("today_news", [])
        self._rolling = news_state.get("_rolling", [])
        
        # 恢复 Random 对象状态
        rng_state_serialized = news_state.get("rng_state")
        if rng_state_serialized:
            try:
                rng_state = pickle.loads(base64.b64decode(rng_state_serialized))
                self._rng.setstate(rng_state)
            except Exception:
                # 如果反序列化失败，使用保存的 seed 重新初始化
                random_seed = news_state.get("random_seed")
                if random_seed is not None:
                    self._rng = random.Random(random_seed)
                    self._random_seed = random_seed
        else:
            # 如果没有保存的状态，使用保存的 seed 重新初始化
            random_seed = news_state.get("random_seed")
            if random_seed is not None:
                self._rng = random.Random(random_seed)
                self._random_seed = random_seed
