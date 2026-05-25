from __future__ import annotations

import os


def _history_sql_path(env_var: str, default_path: str) -> str:
    return os.environ.get(env_var, default_path)


def _easy_history_sql_path() -> str:
    return _history_sql_path(
        "RETAILBENCH_EASY_INIT_SQL_PATH",
        "data/still_super_easy/simulate_data/15/records.sql",
    )


def _hard_history_sql_path() -> str:
    return _history_sql_path(
        "RETAILBENCH_HARD_INIT_SQL_PATH",
        "data/dynamic/simulate_data/15/records.sql",
    )


def create_easy_config() -> dict:
    return {
        "debug": True,
        "data_dir": "data/still_super_easy/simulate_data",
        "customer_data_path": "data/still_super_easy/customer_number",
        "init_sql_path": _easy_history_sql_path(),
        "supplier_data_dir": "data/still_super_easy/simulate_data",
        "data_begin_time": "06/06/91",
        "data_end_time": "12/31/95",
        "store_begin_time": "09/07/91",
        "store_id": "15",
        "order_record_dir": "order_records",
        "enable_review": True,
        "review_model_path": "data/still_super_easy/review/review_star_smooth_params.json",
        "review_source_path": "data/still_super_easy/review/all_category_reviews.jsonl",
        "review_ratio": 0.02,
        "enable_new": False,
        "news_source_path": "data/still_super_easy/simulate_data/15/news_merged.jsonl",
        "news_impact_base_scale": 0.8,
        "news_impact_abs_cap": None,
        "news_sample_ratios": {
            "neutral": 0.65,
            "single_category": 0.1,
            "macro_all": 0.05,
            "sku_level": 0.2,
        },
        "news_daily_count": 20,
        "news_random_seed": 42,
        "news_impact_mode_weights": {
            "neutral": 0.0,
            "macro_all": 1.0,
            "single_category": 1.0,
            "sku_level": 1.2,
        },
        "initial_funds": 5000,
        "everyday_rent": 0,
        "inventory_capacity": 10000,
        "initial_inventory": {},
        "global_random_seed": 42,
        "selected_categories": [
            "Bathroom_Tissues",
            "Soft_Drinks",
            "Cigarettes",
        ],
        "config_name": "easy",
        "category_effects": {
            "Bathroom Tissues": -0.1,
            "Beer": -0.1,
            "Bottled Juices": -0.1,
            "Canned Soup": -0.1,
            "Canned Tuna": -0.1,
            "Cereals": -0.1,
            "Cheeses": -0.1,
            "Cigarettes": -0.1,
            "Cookies": -0.1,
            "Crackers": -0.1,
            "Dish Detergent": -0.1,
            "Fabric Softeners": -0.1,
            "Front end candies": -0.1,
            "Frozen Entrees": -0.1,
            "Frozen Juices": -0.1,
            "Oatmeal": -0.1,
            "Paper Towels": -0.1,
            "Snack Crackers": -0.1,
            "Soft Drinks": -0.1,
            "Toothpastes": -0.1,
        },
    }


def create_middle_config() -> dict:
    return {
        "debug": True,
        "data_dir": "data/still_super_easy/simulate_data",
        "customer_data_path": "data/still_super_easy/customer_number",
        "init_sql_path": _easy_history_sql_path(),
        "supplier_data_dir": "data/still_super_easy/simulate_data",
        "data_begin_time": "06/06/91",
        "data_end_time": "12/31/95",
        "store_begin_time": "09/07/91",
        "store_id": "15",
        "order_record_dir": "order_records",
        "enable_review": True,
        "review_model_path": "data/still_super_easy/review/review_star_smooth_params.json",
        "review_source_path": "data/still_super_easy/review/all_category_reviews.jsonl",
        "review_ratio": 0.02,
        "enable_new": False,
        "news_source_path": "data/still_super_easy/simulate_data/15/news_merged.jsonl",
        "news_impact_base_scale": 0.8,
        "news_impact_abs_cap": None,
        "news_sample_ratios": {
            "neutral": 0.65,
            "single_category": 0.1,
            "macro_all": 0.05,
            "sku_level": 0.2,
        },
        "news_daily_count": 20,
        "news_random_seed": 42,
        "news_impact_mode_weights": {
            "neutral": 0.0,
            "macro_all": 1.0,
            "single_category": 1.0,
            "sku_level": 1.2,
        },
        "initial_funds": 25000,
        "everyday_rent": 200,
        "inventory_capacity": 10000,
        "initial_inventory": {},
        "global_random_seed": 42,
        "selected_categories": [
            "Bathroom_Tissues",
            "Bottled_Juices",
            "Canned_Tuna",
            "Cheeses",
            "Crackers",
            "Dish_Detergent",
            "Front_end_candies",
            "Oatmeal",
            "Snack_Crackers",
            "Toothpastes",
        ],
        "config_name": "middle",
        "category_effects": {
            "Bathroom Tissues": -0.1,
            "Beer": -0.1,
            "Bottled Juices": -0.1,
            "Canned Soup": -0.1,
            "Canned Tuna": -0.1,
            "Cereals": -0.1,
            "Cheeses": -0.1,
            "Cigarettes": -0.1,
            "Cookies": -0.1,
            "Crackers": -0.1,
            "Dish Detergent": -0.1,
            "Fabric Softeners": -0.1,
            "Front end candies": -0.1,
            "Frozen Entrees": -0.1,
            "Frozen Juices": -0.1,
            "Oatmeal": -0.1,
            "Paper Towels": -0.1,
            "Snack Crackers": -0.1,
            "Soft Drinks": -0.1,
            "Toothpastes": -0.1,
        },
    }


def create_hard_config() -> dict:
    return {
        "debug": True,
        "data_dir": "data/dynamic/simulate_data",
        "customer_data_path": "data/dynamic/customer_number",
        "init_sql_path": _hard_history_sql_path(),
        "data_begin_time": "06/06/91",
        "data_end_time": "12/31/95",
        "store_begin_time": "09/07/91",
        "store_id": "15",
        "order_record_dir": "order_records",
        "enable_review": True,
        "review_model_path": "data/dynamic/review/review_star_smooth_params.json",
        "review_source_path": "data/dynamic/review/all_category_reviews.jsonl",
        "review_ratio": 0.02,
        "enable_new": True,
        "news_source_path": "data/dynamic/simulate_data/15/news_merged.jsonl",
        "news_impact_base_scale": 0.4,
        "news_impact_abs_cap": None,
        "news_sample_ratios": {
            "neutral": 0.9,
            "single_category": 0.02,
            "macro_all": 0.03,
            "sku_level": 0.05,
        },
        "news_daily_count": 20,
        "news_random_seed": 42,
        "news_impact_mode_weights": {
            "neutral": 0.0,
            "macro_all": 1.0,
            "single_category": 1.0,
            "sku_level": 1.2,
        },
        "initial_funds": 30000,
        "everyday_rent": 600,
        "inventory_capacity": 15000,
        "initial_inventory": {},
        "global_random_seed": 42,
        "selected_categories": [
            "Bathroom_Tissues",
            "Beer",
            "Bottled_Juices",
            "Canned_Soup",
            "Canned_Tuna",
            "Cereals",
            "Cheeses",
            "Cigarettes",
            "Cookies",
            "Crackers",
            "Dish_Detergent",
            "Fabric_Softeners",
            "Front_end_candies",
            "Frozen_Entrees",
            "Frozen_Juices",
            "Oatmeal",
            "Paper_Towels",
            "Snack_Crackers",
            "Soft_Drinks",
            "Toothpastes",
        ],
        "config_name": "hard",
        "category_effects": {
            "Bathroom Tissues": -0.1,
            "Beer": -0.1,
            "Bottled Juices": -0.1,
            "Canned Soup": -0.1,
            "Canned Tuna": -0.1,
            "Cereals": -0.1,
            "Cheeses": -0.1,
            "Cigarettes": -0.1,
            "Cookies": -0.1,
            "Crackers": -0.1,
            "Dish Detergent": -0.1,
            "Fabric Softeners": -0.1,
            "Front end candies": -0.1,
            "Frozen Entrees": -0.1,
            "Frozen Juices": -0.1,
            "Oatmeal": -0.1,
            "Paper Towels": -0.1,
            "Snack Crackers": -0.1,
            "Soft Drinks": -0.1,
            "Toothpastes": -0.1,
        },
    }


def create_hard_v2_config() -> dict:
    config = create_hard_config()
    config["config_name"] = "hard_v2"
    config["review_ratio"] = 0.05
    config["news_impact_base_scale"] = 0.8
    config["news_impact_abs_cap"] = 0.5
    config["category_attraction_aggregation"] = "power"
    config["category_attraction_rho"] = 3.0
    config["enable_shelf_constraint"] = True
    config["shelf_sku_capacity"] = 40
    config["category_effects"] = {}
    return config


def easy_config() -> dict:
    return create_easy_config()


def middle_config() -> dict:
    return create_middle_config()


def hard_config() -> dict:
    return create_hard_config()


def hard_v2_config() -> dict:
    return create_hard_v2_config()


def get_config_by_name(config_name: str) -> dict:
    lowered = (config_name or "").lower()
    if lowered in {"easy", "ultra_easy", "still_middle"}:
        return create_easy_config()
    if lowered in {"middle", "dynamic_middle", "still_hard"}:
        return create_middle_config()
    if lowered in {"hard_v2", "dynamic_hard_v2", "stress_hard"}:
        return create_hard_v2_config()
    return create_hard_config()
