from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set


QUALITY_FIRST_STATUS = "quality_first_supplier_reconsideration_required"


@dataclass(frozen=True)
class SupplierSkuQualityCandidate:
    sku_id: str
    supplier_id: str
    review_count: int
    average_review_rating: float
    review_score: float
    return_count: int
    return_denominator: int
    return_rate: float
    return_quality_score: float
    quality_score: float


class QualityFirstSupplierGuard:
    """Gate place_order when a non-quality-first supplier is chosen."""

    VALID_MODES = {"off", "gate"}

    def __init__(self, env: Any, config: Dict[str, Any]) -> None:
        self.env = env
        self.mode = str(config.get("quality_first_mode", "off") or "off").lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported quality_first_mode: {self.mode}")
        self.review_window_days = int(config.get("quality_first_review_window_days", 60))
        self.return_window_days = int(config.get("quality_first_return_window_days", 30))
        self.review_weight = float(config.get("quality_first_review_weight", 0.5))
        self.return_weight = float(config.get("quality_first_return_weight", 0.5))
        self._quality_informed_sku_ids: Set[str] = set()
        self._quality_informed_date: Optional[Any] = None

    def begin_action_round(self) -> None:
        """Start one model action round without consuming informed-SKU bypass."""
        self._clear_stale_informed_skus()

    def end_action_round(self) -> None:
        """End one model action round without consuming informed-SKU bypass."""
        self._clear_stale_informed_skus()

    def before_place_order(
        self,
        items: List[Dict[str, Any]],
        supplier_id: str,
    ) -> Optional[Dict[str, Any]]:
        if self.mode == "off":
            return None

        sku_ids = [str(item["sku_id"]) for item in items if item.get("sku_id")]
        if not sku_ids:
            return None

        if self._should_bypass_informed_skus(set(sku_ids)):
            return None

        per_sku_candidates = self._build_per_sku_candidates(items)
        if not per_sku_candidates:
            return None

        violations = self._find_quality_violations(per_sku_candidates, supplier_id)
        if not violations:
            return None

        table_rows = self._build_table_rows(per_sku_candidates, supplier_id)
        best_suppliers_by_sku = {
            sku_id: candidates[0].supplier_id
            for sku_id, candidates in per_sku_candidates.items()
            if candidates
        }
        payload = {
            "status": QUALITY_FIRST_STATUS,
            "action_executed": False,
            "action_name": "place_order",
            "attempted_action": {
                "name": "place_order",
                "arguments": self.env._json_safe({"items": items, "supplier_id": supplier_id}),
                "action_executed": False,
            },
            "selected_supplier_id": supplier_id,
            "best_suppliers_by_sku": best_suppliers_by_sku,
            "diagnostic_reason": (
                "selected supplier is not the per-SKU quality-first candidate "
                "for at least one attempted SKU"
            ),
            "quality_score_scope": "Quality Score is computed independently per SKU.",
            "formula": (
                "quality_score = 0.5 * normalized_supplier_review_rating "
                "+ 0.5 * (1 - supplier_return_rate)"
            ),
            "review_window_days": self.review_window_days,
            "return_window_days": self.return_window_days,
            "per_sku_quality_table": table_rows,
            "violations": violations,
            "next_step": (
                "The attempted order was not executed because the selected supplier is not "
                "the per-SKU quality-first choice among suppliers with complete review and "
                "return data for at least one SKU. In the revised decision, choose suppliers "
                "with higher Quality Score for each SKU, preferably the rank-1 supplier in "
                "the per-SKU table, and split the order into separate place_order calls when "
                "different SKUs have different best suppliers"
            ),
        }
        self._quality_informed_sku_ids.update(best_suppliers_by_sku.keys())
        self._quality_informed_date = self.env.current_date
        return self.env._wrap_tool_result(payload, self._format_reconsideration(payload))

    def _clear_stale_informed_skus(self) -> None:
        if self._quality_informed_date != self.env.current_date:
            self._quality_informed_sku_ids.clear()
            self._quality_informed_date = None

    def _should_bypass_informed_skus(self, sku_ids: Set[str]) -> bool:
        self._clear_stale_informed_skus()
        return bool(sku_ids) and sku_ids.issubset(self._quality_informed_sku_ids)

    def _build_per_sku_candidates(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[SupplierSkuQualityCandidate]]:
        per_sku: Dict[str, List[SupplierSkuQualityCandidate]] = {}
        for item in items:
            sku_id = str(item["sku_id"])
            entries = self.env.supplier_manager.get_sku_date(sku_id, self.env.current_date)
            candidates: List[SupplierSkuQualityCandidate] = []
            for entry in entries:
                supplier_id = str(entry.get("supplier_id") or "")
                if not supplier_id:
                    continue
                candidate = self._score_supplier_sku(supplier_id, sku_id)
                if candidate is not None:
                    candidates.append(candidate)
            candidates.sort(
                key=lambda c: (
                    -c.quality_score,
                    -c.average_review_rating,
                    c.return_rate,
                    c.supplier_id,
                )
            )
            if candidates:
                per_sku[sku_id] = candidates
        return per_sku

    def _score_supplier_sku(
        self,
        supplier_id: str,
        sku_id: str,
    ) -> Optional[SupplierSkuQualityCandidate]:
        review_stats = self._supplier_review_stats(supplier_id, sku_id)
        return_stats = self._supplier_return_stats(supplier_id, sku_id)
        if review_stats is None or return_stats is None:
            return None

        avg_rating, review_count = review_stats
        return_rate, return_count, denominator = return_stats
        review_score = self._clamp(avg_rating / 5.0)
        return_quality_score = self._clamp(1.0 - return_rate)
        total_weight = self.review_weight + self.return_weight
        if total_weight <= 0:
            return None
        quality_score = (
            self.review_weight * review_score
            + self.return_weight * return_quality_score
        ) / total_weight
        return SupplierSkuQualityCandidate(
            sku_id=sku_id,
            supplier_id=supplier_id,
            review_count=review_count,
            average_review_rating=avg_rating,
            review_score=review_score,
            return_count=return_count,
            return_denominator=denominator,
            return_rate=return_rate,
            return_quality_score=return_quality_score,
            quality_score=quality_score,
        )

    def _supplier_review_stats(
        self,
        supplier_id: str,
        sku_id: str,
    ) -> Optional[tuple[float, int]]:
        end = self.env.current_date
        start = end - timedelta(days=self.review_window_days)
        ratings: List[float] = []
        reviews = self.env.record_manager.read_reviews(
            sku_id=sku_id,
            start_date=start,
            end_date=end,
        )
        for review in reviews:
            if str(getattr(review, "supplier_id", "") or "") != supplier_id:
                continue
            try:
                ratings.append(float(review.rating))
            except (TypeError, ValueError):
                continue
        if not ratings:
            return None
        return sum(ratings) / len(ratings), len(ratings)

    def _supplier_return_stats(
        self,
        supplier_id: str,
        sku_id: str,
    ) -> Optional[tuple[float, int, int]]:
        end = self.env.current_date
        start = end - timedelta(days=self.return_window_days)
        result = self.env.view_supplier_returns_avg_rate(
            supplier_id=supplier_id,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            sku_ids=[sku_id],
        ).get("result", {})
        if not isinstance(result, dict):
            return None
        per_sku = result.get("per_sku")
        row = per_sku.get(sku_id) if isinstance(per_sku, dict) else None
        if not isinstance(row, dict):
            return None
        denominator = int(row.get("denominator_count") or 0)
        return_count = int(row.get("return_count") or 0)
        return_rate = row.get("return_rate")
        if denominator <= 0 or return_rate is None:
            return None
        try:
            return float(return_rate), return_count, denominator
        except (TypeError, ValueError):
            return None

    def _find_quality_violations(
        self,
        per_sku_candidates: Dict[str, List[SupplierSkuQualityCandidate]],
        selected_supplier_id: str,
    ) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for sku_id, candidates in per_sku_candidates.items():
            if not candidates:
                continue
            best = candidates[0]
            if best.supplier_id == selected_supplier_id:
                continue
            selected_rank = None
            selected_quality_score = None
            for rank, candidate in enumerate(candidates, 1):
                if candidate.supplier_id == selected_supplier_id:
                    selected_rank = rank
                    selected_quality_score = candidate.quality_score
                    break
            violations.append(
                {
                    "sku_id": sku_id,
                    "selected_supplier_id": selected_supplier_id,
                    "selected_rank": selected_rank,
                    "selected_quality_score": selected_quality_score,
                    "best_supplier_id": best.supplier_id,
                    "best_quality_score": best.quality_score,
                    "reason": (
                        "selected supplier has no complete per-SKU review and return data"
                        if selected_rank is None
                        else "selected supplier is not rank 1 for this SKU"
                    ),
                }
            )
        return violations

    def _build_table_rows(
        self,
        per_sku_candidates: Dict[str, List[SupplierSkuQualityCandidate]],
        selected_supplier_id: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for sku_id, candidates in per_sku_candidates.items():
            for rank, candidate in enumerate(candidates, 1):
                rows.append(
                    self._candidate_to_dict(
                        candidate,
                        rank,
                        candidate.supplier_id == selected_supplier_id,
                    )
                )
        return rows

    def _candidate_to_dict(
        self,
        candidate: SupplierSkuQualityCandidate,
        rank: int,
        selected: bool,
    ) -> Dict[str, Any]:
        return {
            "sku_id": candidate.sku_id,
            "rank": rank,
            "supplier_id": candidate.supplier_id,
            "review_count": candidate.review_count,
            "average_review_rating": candidate.average_review_rating,
            "review_score": candidate.review_score,
            "return_count": candidate.return_count,
            "return_denominator": candidate.return_denominator,
            "return_rate": candidate.return_rate,
            "return_quality_score": candidate.return_quality_score,
            "quality_score": candidate.quality_score,
            "selected": selected,
        }

    def _format_reconsideration(self, payload: Dict[str, Any]) -> str:
        lines = [
            "Quality First supplier check: attempted order was not executed.",
            f"- selected_supplier: {payload['selected_supplier_id']}",
            f"- diagnostic_reason: {payload['diagnostic_reason']}",
            f"- formula: {payload['formula']}",
            f"- review_window_days: {payload['review_window_days']}",
            f"- return_window_days: {payload['return_window_days']}",
            "",
            "| SKU | Rank | Supplier | Review Rating | Review Count | Return Rate | Return Count | Return Denominator | Quality Score | Selected |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in payload["per_sku_quality_table"]:
            selected = "yes" if row["selected"] else ""
            lines.append(
                "| {sku} | {rank} | {supplier} | {rating:.3f} | {reviews} | "
                "{returns:.4f} | {return_count} | {denominator} | {quality:.4f} | {selected} |".format(
                    sku=row["sku_id"],
                    rank=int(row["rank"]),
                    supplier=row["supplier_id"],
                    rating=float(row["average_review_rating"]),
                    reviews=int(row["review_count"]),
                    returns=float(row["return_rate"]),
                    return_count=int(row["return_count"]),
                    denominator=int(row["return_denominator"]),
                    quality=float(row["quality_score"]),
                    selected=selected,
                )
            )
        lines.extend(
            [
                "",
                "Required revision: optimize supplier choice by Quality Score for the attempted SKUs.",
                "- For each SKU, prefer the supplier with the highest Quality Score in that SKU's table.",
                "- If different SKUs have different high-quality suppliers, split them into separate place_order calls grouped by supplier.",
                "- A later place_order bypasses this check only when every ordered SKU has already appeared in a Quality Score table for the current date.",
                "- If the later order contains any SKU that has not yet appeared in a Quality Score table, it is checked normally.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))
