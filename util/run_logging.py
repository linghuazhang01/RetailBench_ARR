from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class _TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_ACTIVE_LOG_FILE: Any = None
_ACTIVE_LOG_PATH: Path | None = None


def prepare_run_dir(prefix: str, log_dir: str | None = None, base_dir: str = "logs") -> Path:
    if log_dir:
        run_dir = Path(log_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_run_paths(prefix: str, log_dir: str | None = None, base_dir: str = "logs") -> tuple[Path, Path]:
    run_dir = prepare_run_dir(prefix=prefix, log_dir=log_dir, base_dir=base_dir)
    return run_dir, run_dir / f"{prefix}.json"


def attach_stdio_log(log_path: str | Path) -> Path:
    global _ACTIVE_LOG_FILE, _ACTIVE_LOG_PATH

    resolved_path = Path(log_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if _ACTIVE_LOG_PATH == resolved_path:
        return resolved_path

    if _ACTIVE_LOG_FILE is not None:
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR
        _ACTIVE_LOG_FILE.close()

    _ACTIVE_LOG_FILE = resolved_path.open("a", encoding="utf-8")
    _ACTIVE_LOG_PATH = resolved_path
    sys.stdout = _TeeStream(_ORIGINAL_STDOUT, _ACTIVE_LOG_FILE)
    sys.stderr = _TeeStream(_ORIGINAL_STDERR, _ACTIVE_LOG_FILE)
    return resolved_path


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return target


def initialize_run_artifacts(
    *,
    prefix: str,
    config: dict,
    args_payload: Mapping[str, Any],
    log_dir: str | None = None,
    checkpoint_dir: str | None = None,
    db_path: str | None = None,
    base_dir: str = "logs",
) -> dict[str, Any]:
    run_dir, log_path = prepare_run_paths(prefix=prefix, log_dir=log_dir, base_dir=base_dir)
    attach_stdio_log(run_dir / "agent.log")

    resolved_db_path = Path(db_path) if db_path else run_dir / "db"
    resolved_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else run_dir / "checkpoints"

    config["order_record_dir"] = os.fspath(resolved_db_path)
    config["log_dir"] = os.fspath(run_dir)
    config["tool_log_path"] = os.fspath(run_dir / "tool_calls.jsonl")
    config["env_log_filename"] = "environment.log"

    args_record = dict(args_payload)
    args_record["log_dir"] = os.fspath(run_dir)
    args_record["checkpoint_dir"] = os.fspath(resolved_checkpoint_dir)
    args_record["db_path"] = os.fspath(resolved_db_path)

    write_json(run_dir / "config.json", config)
    write_json(run_dir / "args.json", args_record)

    return {
        "run_dir": run_dir,
        "log_path": log_path,
        "checkpoint_dir": resolved_checkpoint_dir,
        "db_path": resolved_db_path,
        "agent_log_path": run_dir / "agent.log",
    }
