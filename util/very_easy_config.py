"""
Very Easy / Ultra Easy Environment Configuration Constants

Runnable configs have been moved to util/default_config.py.
These constants are kept for reference and external consumers (e.g. run_very_easy_experiment.py).
"""

VERY_EASY_CONFIG = {
    # ========== Market Configuration ==========
    "num_categories": 3,              # Only 3 categories (vs 5 in Easy)
    "skus_per_category": 6,           # 6 SKUs per category
    "total_skus": 18,                 # Total 18 SKUs (vs ~48 in Easy)

    # ========== Financial Configuration ==========
    "initial_budget": 25000,          # Initial budget: 25k (vs 10k in Easy)
    "daily_rent": 80,                 # Daily rent: 80 (vs 250 in Easy)
                                     # Rent is only 9.6% of budget over 30 days

    # ========== Complexity Configuration ==========
    "enable_news_events": False,      # No dynamic news events
    "enable_supplier_dynamics": False,# No supplier price/quality dynamics
    "enable_quality_changes": False,  # No supplier quality changes over time

    # ========== Demand Configuration ==========
    "customer_traffic_mean": 60,      # Average 60 customers per day
    "customer_traffic_std": 8,        # Low volatility (CV = 0.13)

    # ========== Time Configuration ==========
    "max_days": 30,                   # 30-day episodes
    "episode_timeout": None,          # No timeout

    # ========== Product Configuration ==========
    "shelf_life_range": (7, 21),      # Shelf life 7-21 days (relatively long)
    "price_range": (0.5, 3.0),        # Price range

    # ========== Supplier Configuration ==========
    "suppliers_per_sku": 3,           # 3 suppliers per SKU (vs 5 in Easy)
    "lead_time_mean": 2,              # Average lead time: 2 days
    "lead_time_std": 0.5,             # Stable lead times
}

ULTRA_EASY_CONFIG = {
    **VERY_EASY_CONFIG,
    "num_categories": 2,
    "skus_per_category": 6,
    "total_skus": 12,
    "enable_supplier_dynamics": False,
    "enable_quality_changes": False,
}
