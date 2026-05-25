from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "RetailBench"

logger = logging.getLogger(_LOGGER_NAME)

def get_logger() -> logging.Logger:
    """
    返回全局 logger；可传 debug=True/False 调整等级。
    """
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def configure_logger(
    log_dir: Optional[str | Path] = None,
    filename: str = "environment.log",
    debug: Optional[bool] = None,
) -> logging.Logger:
    active_logger = get_logger()

    if debug is not None:
        set_logger_level(debug)

    target_path = None
    if log_dir is not None:
        target_path = Path(log_dir) / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

    existing_file_handlers = [
        handler for handler in active_logger.handlers if getattr(handler, "_retailbench_file_handler", False)
    ]

    if target_path is None:
        for handler in existing_file_handlers:
            active_logger.removeHandler(handler)
            handler.close()
        return active_logger

    for handler in existing_file_handlers:
        if Path(getattr(handler, "baseFilename", "")) == target_path:
            return active_logger
        active_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(target_path, encoding="utf-8")
    file_handler._retailbench_file_handler = True
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    file_handler.setLevel(active_logger.level or logging.INFO)
    active_logger.addHandler(file_handler)
    return active_logger

def set_logger_level(debug: Optional[bool] = None) -> None:
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    # 同时更新所有 handler 的级别，确保日志级别设置生效
    for handler in logger.handlers:
        handler.setLevel(level)
