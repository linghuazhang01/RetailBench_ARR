#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_ROOT / "agents"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "batches"
if sys.version_info < (3, 10):
    raise SystemExit("RetailBench batch launcher requires Python 3.10 or newer.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.config_registry import CONFIG_TYPE_CHOICES

AVAILABLE_AGENTS = [
    "run_react.py",
    "run_reflection.py",
    "run_step_reflection.py",
    "run_plan_and_act.py",
]

AGENT_OPTION_SUPPORT = {
    "run_react.py": {"max_turns", "max_days", "max_execution_turns", "quality_first"},
    "run_reflection.py": {"max_turns", "max_days", "max_execution_turns", "quality_first"},
    "run_step_reflection.py": {"max_days", "max_steps", "quality_first"},
    "run_plan_and_act.py": {"max_turns", "max_days", "max_execution_turns", "quality_first"},
}


@dataclass(frozen=True)
class ExperimentTask:
    model_order: int
    model: str
    repeat: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch launcher for RetailBench agent experiments")
    parser.add_argument("--agent", type=str, default="run_reflection.py", choices=AVAILABLE_AGENTS)
    parser.add_argument("--config_type", type=str, default="hard", choices=CONFIG_TYPE_CHOICES)
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Model list to run")
    parser.add_argument("--repeats", type=int, default=1, help="How many runs per model")
    parser.add_argument("--workers", type=int, default=1, help="How many experiments to run concurrently")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory that stores batch outputs")
    parser.add_argument("--batch-name", type=str, default=None, help="Optional batch folder name")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument(
        "--service_tier",
        type=str,
        default=None,
        help='Optional OpenAI/OpenRouter service tier, e.g. "flex"',
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=50,
        help="Daily max_turns for agents that support it; default 50",
    )
    parser.add_argument("--max_days", type=int, default=None)
    parser.add_argument("--max_execution_turns", type=int, default=None)
    parser.add_argument("--max_strategy_turns", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--quality_first_mode", type=str, choices=["off", "gate"], default="off")
    parser.add_argument("--caec_mode", type=str, choices=["off", "gate"], default="off", help="Legacy no-op argument")
    parser.add_argument("--caec_config", type=str, default=None, help="Legacy no-op argument")
    parser.add_argument("--fcr_mode", type=str, choices=["off", "gate"], default="off", help="Legacy no-op argument")
    parser.add_argument("--fcr_window_days", type=int, default=14, help="Legacy no-op argument")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=None,
        help="Checkpoint save interval in days; omit to use agent default (disabled)",
    )
    parser.add_argument("--stagger-seconds", type=float, default=0.0, help="Sleep between task submissions")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def redact_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for idx, value in enumerate(redacted[:-1]):
        if value == "--api_key":
            redacted[idx + 1] = "***"
    return redacted


def build_batch_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_name = args.batch_name or f"{args.config_type}_{Path(args.agent).stem}_{timestamp}"
    batch_dir = Path(args.output_dir) / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def build_tasks(models: list[str], repeats: int) -> list[ExperimentTask]:
    model_order = {model: index for index, model in enumerate(models)}
    tasks = [
        ExperimentTask(
            model_order=model_order[model],
            model=model,
            repeat=repeat,
        )
        for model in models
        for repeat in range(1, repeats + 1)
    ]
    tasks.sort(key=lambda task: (task.repeat, task.model_order))
    return tasks


def build_command(args: argparse.Namespace, agent_run_dir: Path, model: str) -> list[str]:
    cmd = [
        sys.executable,
        str(AGENTS_DIR / args.agent),
        "--model",
        model,
        "--config_type",
        args.config_type,
        "--log_dir",
        str(agent_run_dir),
    ]

    supported = AGENT_OPTION_SUPPORT[args.agent]

    if args.api_key:
        cmd.extend(["--api_key", args.api_key])
    if args.base_url:
        cmd.extend(["--base_url", args.base_url])
    if args.checkpoint_interval is not None:
        cmd.extend(["--checkpoint_interval", str(args.checkpoint_interval)])
    if args.max_turns is not None and "max_turns" in supported:
        cmd.extend(["--max_turns", str(args.max_turns)])
    if args.max_days is not None and "max_days" in supported:
        cmd.extend(["--max_days", str(args.max_days)])
    if args.max_execution_turns is not None and "max_execution_turns" in supported:
        cmd.extend(["--max_execution_turns", str(args.max_execution_turns)])
    if args.max_strategy_turns is not None and "max_strategy_turns" in supported:
        cmd.extend(["--max_strategy_turns", str(args.max_strategy_turns)])
    if args.max_steps is not None and "max_steps" in supported:
        cmd.extend(["--max_steps", str(args.max_steps)])
    if "quality_first" in supported:
        cmd.extend(["--quality_first_mode", args.quality_first_mode])

    return cmd


def run_task(
    task: ExperimentTask,
    args: argparse.Namespace,
    batch_dir: Path,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    model_slug = sanitize_name(task.model)
    run_dir = batch_dir / f"{Path(args.agent).stem}_{model_slug}_run{task.repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = run_dir / "launcher.log"
    cmd = build_command(args, run_dir, task.model)
    started_at = datetime.now()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    if args.service_tier:
        env["OPENAI_SERVICE_TIER"] = args.service_tier

    with launcher_log.open("w", encoding="utf-8") as log_file:
        log_file.write(f"start_time={started_at.isoformat()}\n")
        log_file.write(f"command={json.dumps(redact_command(cmd), ensure_ascii=False)}\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        with print_lock:
            print(
                f"[START] model={task.model} repeat={task.repeat} pid={process.pid} "
                f"run_dir={run_dir}"
            )

        return_code = process.wait()

    elapsed = (datetime.now() - started_at).total_seconds()
    result = {
        "agent": args.agent,
        "config_type": args.config_type,
        "model": task.model,
        "repeat": task.repeat,
        "pid": process.pid,
        "status": "success" if return_code == 0 else "failed",
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 2),
        "run_dir": str(run_dir),
        "launcher_log": str(launcher_log),
        "command": redact_command(cmd),
    }

    with (run_dir / "launcher_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with print_lock:
        print(
            f"[DONE] model={task.model} repeat={task.repeat} pid={process.pid} "
            f"status={result['status']} rc={return_code} elapsed={elapsed:.2f}s"
        )

    return result


def main() -> None:
    args = parse_args()
    batch_dir = build_batch_dir(args)
    tasks = build_tasks(args.models, args.repeats)
    print_lock = threading.Lock()

    launcher_config = {
        "agent": args.agent,
        "config_type": args.config_type,
        "models": args.models,
        "schedule_policy": "repeat_ascending_then_model_order",
        "repeats": args.repeats,
        "workers": args.workers,
        "batch_dir": str(batch_dir),
        "stagger_seconds": args.stagger_seconds,
        "max_turns": args.max_turns,
        "max_days": args.max_days,
        "max_execution_turns": args.max_execution_turns,
        "max_strategy_turns": args.max_strategy_turns,
        "max_steps": args.max_steps,
        "quality_first_mode": args.quality_first_mode,
        "ignored_legacy_caec_mode": args.caec_mode,
        "ignored_legacy_caec_config": args.caec_config,
        "ignored_legacy_fcr_mode": args.fcr_mode,
        "ignored_legacy_fcr_window_days": args.fcr_window_days,
        "checkpoint_interval": args.checkpoint_interval,
        "api_key": "***" if args.api_key else None,
        "base_url": args.base_url,
        "service_tier": args.service_tier,
        "created_at": datetime.now().isoformat(),
    }
    with (batch_dir / "launcher_config.json").open("w", encoding="utf-8") as f:
        json.dump(launcher_config, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("RetailBench Batch Launcher")
    print("=" * 80)
    print(f"agent: {args.agent}")
    print(f"config: {args.config_type}")
    print(f"models: {args.models}")
    print("schedule_policy: repeat ascending, then input model order")
    print(f"repeats: {args.repeats}")
    print(f"workers: {args.workers}")
    print(f"batch_dir: {batch_dir}")
    print(f"total_tasks: {len(tasks)}")
    print("=" * 80)

    results: list[dict[str, Any]] = []
    started_at = datetime.now()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(run_task, task, args, batch_dir, print_lock))
            if args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)

        for future in as_completed(futures):
            results.append(future.result())

    total_elapsed = (datetime.now() - started_at).total_seconds()
    results.sort(key=lambda item: (item["repeat"], args.models.index(item["model"])))

    summary = {
        "batch_dir": str(batch_dir),
        "total_tasks": len(results),
        "success_count": sum(1 for item in results if item["status"] == "success"),
        "failed_count": sum(1 for item in results if item["status"] != "success"),
        "elapsed_seconds": round(total_elapsed, 2),
        "results": results,
    }
    with (batch_dir / "launcher_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Batch Summary")
    print("=" * 80)
    for item in results:
        status = "OK" if item["status"] == "success" else "FAIL"
        print(
            f"{status} model={item['model']} repeat={item['repeat']} pid={item['pid']} "
            f"rc={item['return_code']} run_dir={item['run_dir']}"
        )
    print(f"success={summary['success_count']} failed={summary['failed_count']} elapsed={total_elapsed:.2f}s")
    print(f"summary_file={batch_dir / 'launcher_summary.json'}")


if __name__ == "__main__":
    main()
