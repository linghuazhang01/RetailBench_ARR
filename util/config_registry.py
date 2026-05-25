from __future__ import annotations

from copy import deepcopy

from util.default_config import (
    create_easy_config,
    create_hard_config,
    create_hard_v2_config,
    create_middle_config,
)

CONFIG_TYPE_ALIASES = {
    "easy": "easy",
    "middle": "middle",
    "hard": "hard",
    "hard_v2": "hard_v2",
    # Backward-compatible aliases for older experiment names.
    "ultra_easy": "easy",
    "Ultra_Easy": "easy",
    "ULTRA_EASY": "easy",
    "still_ultra_easy": "easy",
    "still_middle": "easy",
    "dynamic_middle": "middle",
    "still_hard": "middle",
    "dynamic_hard": "hard",
    "dynamic_hard_v2": "hard_v2",
    "stress_hard": "hard_v2",
}

CONFIG_TYPE_CHOICES = ["easy", "middle", "hard", "hard_v2"]


def normalize_config_type(config_type: str | None) -> str:
    if not config_type:
        return "hard"
    return CONFIG_TYPE_ALIASES.get(config_type, "hard")


def load_config_by_type(config_type: str | None) -> dict:
    normalized = normalize_config_type(config_type)
    if normalized == "easy":
        return deepcopy(create_easy_config())
    if normalized == "middle":
        return deepcopy(create_middle_config())
    if normalized == "hard_v2":
        return deepcopy(create_hard_v2_config())
    return deepcopy(create_hard_config())
