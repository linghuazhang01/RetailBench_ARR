#!/usr/bin/env python3
"""Pure ReAct baseline for RetailEnvironment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if sys.version_info < (3, 10):
    raise SystemExit("RetailBench agents require Python 3.10 or newer.")

from openai import OpenAI

from retail_environment import RetailEnvironment
from util.config_registry import CONFIG_TYPE_CHOICES, load_config_by_type
from util.run_logging import initialize_run_artifacts
from util.tool_call_parser import parse_tool_args, parse_tool_calls

from module.stream_chat import get_cached_prompt_tokens, stream_chat
from agents.run_reflection import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_GOAL,
    DEFAULT_MODEL,
    DAILY_OPERATION_CLOSEOUT_INSTRUCTION,
    EXECUTION_SYSTEM_PROMPT,
    USER_ENVIRONMENT_WARNINGS,
    apply_negative_funds_update,
    build_action_reconsideration_tool_response_messages,
    build_goal,
    build_openai_tools,
    begin_quality_first_action_round,
    create_openai_client,
    end_quality_first_action_round,
    is_action_reconsideration_status,
    log_message,
    make_tool_response_message,
    recover_from_checkpoint,
    recover_from_day_checkpoint,
    render_sku_descriptions,
    render_tool_definitions,
    render_system_prompt,
    safe_dump,
    sanitize_messages_for_api,
    save_checkpoint,
    save_turn_calls_to_json,
    tool_result_action_executed,
    tool_result_has_error,
    tool_result_status,
    write_log_json_array,
)


def _build_react_user_message(
    day: int,
    funds_formatted: str,
    shelf_formatted: str,
    sku_desc: str,
) -> Dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"# Day {day} - ReAct Operations\n\n"
            f"## Current Status\n\n{funds_formatted}\n\n"
            f"## Shelf Status\n\n{shelf_formatted}\n\n"
            f"## SKU Catalog\n\n{sku_desc}\n\n"
            "## Instructions\n\n"
            f"{USER_ENVIRONMENT_WARNINGS}\n\n"
            f"{DAILY_OPERATION_CLOSEOUT_INSTRUCTION}"
        ),
    }


def _save_day_token_usage(
    log_dir: Optional[Path],
    day: int,
    current_date: Any,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_prompt_tokens: int = 0,
) -> None:
    if log_dir is None:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        day_token_path = log_dir / f"day_{day}_token_usage.json"
        day_token_data = {
            "day": day,
            "current_date": str(current_date),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "uncached_prompt_tokens": max(prompt_tokens - cached_prompt_tokens, 0),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with day_token_path.open("w", encoding="utf-8") as f:
            json.dump(day_token_data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        print(f"[WARN] Failed to write day {day} token usage to json: {exc}")


def _save_day_checkpoint(
    *,
    checkpoint_dir: Optional[Path],
    checkpoint_interval: int,
    day: int,
    max_days: int,
    global_turn: int,
    execution_turns: int,
    execution_messages: List[Dict[str, Any]],
    env: RetailEnvironment,
    log_path: Path,
    run_log: List[Dict[str, Any]],
) -> None:
    should_save_checkpoint = (
        checkpoint_dir is not None
        and checkpoint_interval > 0
        and (day == max_days or day % checkpoint_interval == 0)
    )
    if not should_save_checkpoint or checkpoint_dir is None:
        return

    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(checkpoint_dir, global_turn, execution_messages, env)
        day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
        day_messages_path = checkpoint_dir / f"day_{day}_messages.json"
        day_env_checkpoint_path = checkpoint_dir / f"day_{day}_env_checkpoint.json"

        with day_messages_path.open("w", encoding="utf-8") as f:
            json.dump(execution_messages, f, ensure_ascii=False, indent=2, default=str)
        env.save_checkpoint(day_env_checkpoint_path)

        checkpoint_metadata = {
            "day": day,
            "current_date": str(env.current_date),
            "global_turn": global_turn,
            "execution_turns": execution_turns,
            "messages_path": str(day_messages_path.relative_to(checkpoint_dir)),
            "env_checkpoint_path": str(day_env_checkpoint_path.relative_to(checkpoint_dir)),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with day_checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Checkpoint] Saved day {day} checkpoint to {checkpoint_dir}")
    except Exception as exc:
        print(f"[WARN] Failed to save day {day} checkpoint: {exc}")
        log_message(
            run_log,
            {
                "role": "error",
                "day": day,
                "phase": "checkpoint",
                "message": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        write_log_json_array(log_path, run_log)


def _infer_day_from_environment(env: RetailEnvironment, fallback_day: int) -> int:
    """Infer simulation day index from the recovered environment date."""
    try:
        current_date = env.current_date
        if isinstance(current_date, datetime):
            current_date = current_date.date()
        store_begin = datetime.strptime(env.config["store_begin_time"], "%Y-%m-%d").date()
        return max((current_date - store_begin).days + 1, fallback_day)
    except Exception:
        return fallback_day


def _force_end_today(
    *,
    env: RetailEnvironment,
    day: int,
    turn: int,
    reason: str,
    run_log: List[Dict[str, Any]],
    log_path: Path,
    consecutive_negative_days: int,
) -> tuple[bool, int]:
    print(f"[Day {day}] Executing end_today: {reason}")
    end_today_result = env.exec_tools("end_today")
    if tool_result_has_error(end_today_result):
        err_msg = f"Forced end_today failed ({reason}): {end_today_result.get('formatted', safe_dump(end_today_result))}"
        print(f"[ERROR] {err_msg}")
        log_message(run_log, {"role": "error", "day": day, "turn": turn, "message": err_msg, "error_type": "end_today_failed"})
        write_log_json_array(log_path, run_log)
        return False, consecutive_negative_days

    consecutive_negative_days, funds = apply_negative_funds_update(
        end_today_result,
        env.funds,
        consecutive_negative_days,
    )
    log_message(
        run_log,
        {
            "role": "tool",
            "day": day,
            "phase": "execution",
            "turn": turn,
            "name": "end_today",
            "content": end_today_result.get("formatted", safe_dump(end_today_result)),
            "raw": end_today_result.get("result", safe_dump(end_today_result)),
            "forced": True,
            "reason": reason,
        },
    )
    if funds < 0 and consecutive_negative_days >= 5:
        print(f"[FAILURE] {consecutive_negative_days} consecutive days with negative funds")
        log_message(run_log, {"role": "system", "day": day, "message": f"Failure: {consecutive_negative_days} consecutive days with negative funds", "funds": funds})
        write_log_json_array(log_path, run_log)
        return False, consecutive_negative_days
    return True, consecutive_negative_days


def run_react_loop(
    goal: str,
    env: RetailEnvironment,
    model: str = DEFAULT_MODEL,
    max_turns: int = 50,
    log_path: Path = Path("logs/run_react_history.json"),
    max_input_tokens: int = 60000,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 0,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    start_turn: int = 0,
    start_day: int = 1,
    log_dir: Optional[Path] = None,
    max_days: int = 30,
    max_execution_turns_per_day: int = 50,
    client: Optional[OpenAI] = None,
) -> None:
    all_tools = build_openai_tools(env)
    if not env.config.get("enable_new", False):
        news_tools = {"view_news_history", "view_today_news", "view_news_detail"}
        all_tools = [tool for tool in all_tools if tool["function"]["name"] not in news_tools]
    allowed_tool_names = {tool["function"]["name"] for tool in all_tools}

    execution_system_prompt = render_system_prompt(
        EXECUTION_SYSTEM_PROMPT,
        quality_first_mode=env.config.get("quality_first_mode", "off"),
        tool_definitions=render_tool_definitions(all_tools),
        daily_rent=env.config.get("everyday_rent", 0),
    )
    if client is None:
        client = create_openai_client()

    run_log: List[Dict[str, Any]] = []
    input_tokens = 0
    global_turn = start_turn
    consecutive_negative_days = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cached_prompt_tokens = 0
    initial_execution_messages = list(initial_messages or [])

    for day in range(start_day, max_days + 1):
        print(f"\n{'=' * 80}\nDay {day} - {env.current_date}\n{'=' * 80}\n")
        log_message(run_log, {"role": "system", "day": day, "current_date": str(env.current_date), "message": f"Day {day} started"})

        day_prompt_tokens = 0
        day_completion_tokens = 0
        day_tokens = 0
        day_cached_prompt_tokens = 0
        execution_phase_complete = False
        execution_turns = 0
        consecutive_no_valid_tool_calls = 0
        execution_messages = list(initial_execution_messages) if day == start_day else []
        daily_turn_limit = max_execution_turns_per_day
        if max_turns and max_turns > 0:
            daily_turn_limit = min(daily_turn_limit, max_turns)

        execution_system_msg = {"role": "system", "content": execution_system_prompt}
        funds_result = env.exec_tools("view_funds_and_date")
        shelf_result = env.exec_tools("view_shelf_status")
        execution_user_msg = _build_react_user_message(
            day,
            funds_result.get("formatted", ""),
            shelf_result.get("formatted", ""),
            render_sku_descriptions(env),
        )
        log_message(run_log, {**execution_system_msg, "day": day, "phase": "execution"})
        log_message(run_log, {**execution_user_msg, "day": day, "phase": "execution"})

        while not execution_phase_complete and execution_turns < daily_turn_limit:
            execution_turns += 1
            global_turn += 1

            if input_tokens > max_input_tokens:
                assistant_idxs = [i for i, msg in enumerate(execution_messages) if msg.get("role") == "assistant"]
                if len(assistant_idxs) >= 3:
                    execution_messages = execution_messages[assistant_idxs[2]:]
                elif len(assistant_idxs) >= 2:
                    execution_messages = execution_messages[assistant_idxs[1]:]

            request_messages = (
                [execution_system_msg]
                + execution_messages
                + [execution_user_msg]
            )
            api_messages = sanitize_messages_for_api(request_messages)

            try:
                full_content, final_content, reasoning_content, usage = stream_chat(
                    client=client,
                    model=model,
                    messages=api_messages,
                )
            except Exception as stream_exc:
                err_msg = f"LLM stream_chat failed on day {day} turn {execution_turns}: {type(stream_exc).__name__}: {stream_exc}"
                print(f"[ERROR] {err_msg}")
                log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "error_type": type(stream_exc).__name__})
                write_log_json_array(log_path, run_log)
                execution_messages.append({"role": "user", "content": f"System error: {err_msg}. Continue."})
                continue

            try:
                parse_source = final_content or full_content or ""
                tool_calls_list, parse_method_tag = parse_tool_calls(
                    parse_source,
                    allowed_tool_names=allowed_tool_names,
                )
            except Exception as parse_exc:
                err_msg = f"Failed to parse tool calls on day {day} turn {execution_turns}: {type(parse_exc).__name__}: {parse_exc}"
                print(f"[ERROR] {err_msg}")
                log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "error_type": type(parse_exc).__name__})
                tool_calls_list, parse_method_tag = [], "none"

            turn_data: Optional[Dict[str, Any]] = None
            if log_dir is not None:
                turn_data = {
                    "day": day,
                    "phase": "execution",
                    "turn": execution_turns,
                    "global_turn": global_turn,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "full_content": full_content,
                    "final_content": final_content,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls_list,
                    "usage": usage,
                    "messages": request_messages,
                    "api_messages": api_messages,
                }
                save_turn_calls_to_json(
                    log_dir,
                    "execute",
                    execution_turns,
                    turn_data,
                    day,
                )

            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                tokens = usage.get("total_tokens", 0)
                input_tokens = prompt_tokens
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += tokens
                cached_prompt_tokens = get_cached_prompt_tokens(usage)
                total_cached_prompt_tokens += cached_prompt_tokens
                day_prompt_tokens += prompt_tokens
                day_completion_tokens += completion_tokens
                day_tokens += tokens
                day_cached_prompt_tokens += cached_prompt_tokens
            log_message(run_log, {"role": "usage", "day": day, "phase": "execution", "turn": execution_turns, "prompt_tokens": usage.get("prompt_tokens") if usage else None, "completion_tokens": usage.get("completion_tokens") if usage else None, "total_tokens": usage.get("total_tokens") if usage else None, "cached_prompt_tokens": get_cached_prompt_tokens(usage)})
            log_message(run_log, {"role": "assistant", "day": day, "phase": "execution", "turn": execution_turns, "content": full_content, "full_content": full_content, "final_content": final_content, "reasoning": reasoning_content, "tool_calls": tool_calls_list})

            assistant_message = {"role": "assistant", "content": full_content}
            tool_response_messages: List[Dict[str, Any]] = []
            tool_response_meta: List[Dict[str, Any]] = []
            suppress_assistant_message = False
            valid_tool_calls_count = 0
            skipped_tool_calls_after_reconsideration: List[Dict[str, Any]] = []

            begin_quality_first_action_round(env)
            for call_index, call in enumerate(tool_calls_list):
                if not isinstance(call, dict) or not call.get("name"):
                    err_msg = f"Invalid tool call: {call}"
                    print(f"[ERROR] {err_msg}")
                    log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "call": str(call)})
                    tool_response_messages.append(make_tool_response_message(f"Error: {err_msg}"))
                    continue

                name, args = call.get("name"), parse_tool_args(call.get("arguments"))
                try:
                    result = env.exec_tools(name, **args)
                    if not isinstance(result, dict) or "formatted" not in result:
                        result = {"formatted": str(result), "result": result}
                except Exception as exc:
                    err_msg = f"Error executing tool {name}: {type(exc).__name__}: {exc}"
                    print(f"[ERROR] {err_msg}")
                    if hasattr(env, "logger"):
                        env.logger.error(err_msg)
                    result = {"formatted": err_msg, "result": {"error": str(exc), "error_type": type(exc).__name__}}

                tool_executed_successfully = not tool_result_has_error(result)
                action_executed = tool_result_action_executed(result)
                result_status = tool_result_status(result)
                if is_action_reconsideration_status(result_status):
                    suppress_assistant_message = True

                log_message(
                    run_log,
                    {
                        "role": "tool",
                        "day": day,
                        "phase": "execution",
                        "turn": execution_turns,
                        "name": name,
                        "args": args,
                        "content": result.get("formatted", safe_dump(result)),
                        "raw": result.get("result", safe_dump(result)),
                        "success": tool_executed_successfully,
                        "action_executed": action_executed,
                        "result_status": result_status,
                    },
                )

                if tool_executed_successfully:
                    valid_tool_calls_count += 1

                if is_action_reconsideration_status(result_status):
                    reconsideration_messages = build_action_reconsideration_tool_response_messages(name, args, result)
                    tool_response_messages.extend(reconsideration_messages)
                    skipped_tool_calls_after_reconsideration = tool_calls_list[call_index + 1 :]
                    if skipped_tool_calls_after_reconsideration:
                        log_message(
                            run_log,
                            {
                                "role": "system",
                                "day": day,
                                "phase": "execution",
                                "turn": execution_turns,
                                "message": "Skipped remaining parsed tool calls after action reconsideration.",
                                "skipped_tool_calls": skipped_tool_calls_after_reconsideration,
                            },
                        )
                    break

                tool_response_messages.append(
                    make_tool_response_message(result.get("formatted", safe_dump(result)))
                )

                if name == "end_today" and tool_executed_successfully:
                    execution_phase_complete = True
                    print(f"[Day {day}] Execution phase complete - end_today called")
                    consecutive_negative_days, funds = apply_negative_funds_update(
                        result,
                        env.funds,
                        consecutive_negative_days,
                    )
                    if funds < 0 and consecutive_negative_days >= 5:
                        print(f"[FAILURE] {consecutive_negative_days} consecutive days with negative funds")
                        log_message(run_log, {"role": "system", "day": day, "message": f"Failure: {consecutive_negative_days} consecutive days with negative funds", "funds": funds})
                        end_quality_first_action_round(env)
                        write_log_json_array(log_path, run_log)
                        return
                    break

            end_quality_first_action_round(env)
            if turn_data is not None and skipped_tool_calls_after_reconsideration:
                turn_data["skipped_tool_calls_after_reconsideration"] = skipped_tool_calls_after_reconsideration
                turn_data["executed_tool_call_count_before_reconsideration"] = (
                    len(tool_calls_list) - len(skipped_tool_calls_after_reconsideration)
                )
                save_turn_calls_to_json(log_dir, "execute", execution_turns, turn_data, day)

            if valid_tool_calls_count > 0:
                consecutive_no_valid_tool_calls = 0
            else:
                consecutive_no_valid_tool_calls += 1
                print(f"[Day {day} Turn {execution_turns}] No valid tool calls ({consecutive_no_valid_tool_calls}/5)")

            if consecutive_no_valid_tool_calls >= 5 and not execution_phase_complete:
                execution_phase_complete, consecutive_negative_days = _force_end_today(
                    env=env,
                    day=day,
                    turn=execution_turns,
                    reason="5 consecutive invalid tool-call turns",
                    run_log=run_log,
                    log_path=log_path,
                    consecutive_negative_days=consecutive_negative_days,
                )
                if not execution_phase_complete:
                    return
                write_log_json_array(log_path, run_log)
                break

            if tool_calls_list and tool_response_messages:
                if not suppress_assistant_message:
                    execution_messages.append(assistant_message)
                execution_messages.extend(tool_response_messages)
                logged_user_content = "\n".join(message.get("content", "") for message in tool_response_messages)
                log_message(
                    run_log,
                    {
                        "role": "user",
                        "day": day,
                        "phase": "execution",
                        "turn": execution_turns,
                        "content": logged_user_content,
                        "messages": tool_response_messages,
                        "parse_method": parse_method_tag,
                        "meta": {"tool_responses": tool_response_meta},
                    },
                )
            else:
                execution_messages.append(assistant_message)
                execution_messages.append({"role": "user", "content": "No valid tool call detected. Continue or call end_today."})

            write_log_json_array(log_path, run_log)

        if not execution_phase_complete:
            execution_phase_complete, consecutive_negative_days = _force_end_today(
                env=env,
                day=day,
                turn=execution_turns,
                reason=f"turn limit {daily_turn_limit} reached",
                run_log=run_log,
                log_path=log_path,
                consecutive_negative_days=consecutive_negative_days,
            )
            if not execution_phase_complete:
                return

        _save_day_token_usage(
            log_dir,
            day,
            env.current_date,
            day_prompt_tokens,
            day_completion_tokens,
            day_tokens,
            day_cached_prompt_tokens,
        )
        _save_day_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            day=day,
            max_days=max_days,
            global_turn=global_turn,
            execution_turns=execution_turns,
            execution_messages=execution_messages,
            env=env,
            log_path=log_path,
            run_log=run_log,
        )
        write_log_json_array(log_path, run_log)

    print("\nSimulation completed.")
    print("\n" + "=" * 80)
    print("Token Usage Statistics:")
    print("=" * 80)
    print(f"Total Prompt Tokens: {total_prompt_tokens:,}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Cached Prompt Tokens: {total_cached_prompt_tokens:,}")
    print("=" * 80 + "\n")

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        token_stats_path = log_dir / "token_statistics.json"
        with token_stats_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                    "total_cached_prompt_tokens": total_cached_prompt_tokens,
                    "total_uncached_prompt_tokens": max(total_prompt_tokens - total_cached_prompt_tokens, 0),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
    log_message(
        run_log,
        {
            "role": "system",
            "phase": "summary",
            "message": "Final statistics after ReAct simulation",
            "token_statistics": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cached_prompt_tokens": total_cached_prompt_tokens,
                "total_uncached_prompt_tokens": max(total_prompt_tokens - total_cached_prompt_tokens, 0),
            },
        },
    )
    write_log_json_array(log_path, run_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retail environment with a pure ReAct loop")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save/load checkpoints")
    parser.add_argument("--recover_turn", type=int, help="Turn number to recover from")
    parser.add_argument("--recover_day", type=int, help="Day number to recover from day checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Save checkpoint every N days and on the final day; 0 disables checkpoint saving")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_turns", type=int, default=50, help="Maximum turns per day")
    parser.add_argument("--db_path", type=str, default=None, help="Database path for order records")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for agent/environment logs and run artifacts")
    parser.add_argument(
        "--config_type",
        type=str,
        choices=CONFIG_TYPE_CHOICES,
        default="hard",
        help="Configuration type, supports easy/middle/hard",
    )
    parser.add_argument("--max_input_tokens", type=int, default=50000, help="Maximum input tokens for context window")
    parser.add_argument("--max_days", type=int, default=180, help="Maximum number of days to simulate")
    parser.add_argument("--max_execution_turns", type=int, default=50, help="Maximum execution turns per day")
    parser.add_argument("--quality_first_mode", type=str, choices=["off", "gate"], default="off", help="Quality-first supplier gate mode")
    parser.add_argument("--caec_mode", type=str, choices=["off", "gate"], default="off", help="Legacy no-op argument")
    parser.add_argument("--caec_config", type=str, default=None, help="Legacy no-op argument")
    parser.add_argument("--fcr_mode", type=str, choices=["off", "gate"], default="off", help="Legacy no-op argument")
    parser.add_argument("--fcr_window_days", type=int, default=14, help="Legacy no-op argument")
    parser.add_argument("--api_key", type=str, default=None, help=f"OpenAI API key (default: {DEFAULT_API_KEY})")
    parser.add_argument("--base_url", type=str, default=None, help=f"OpenAI API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    config = load_config_by_type(args.config_type)
    config["quality_first_mode"] = args.quality_first_mode
    artifacts = initialize_run_artifacts(
        prefix="run_react",
        config=config,
        args_payload={
            "recover_turn": args.recover_turn,
            "recover_day": args.recover_day,
            "checkpoint_interval": args.checkpoint_interval,
            "model": args.model,
            "max_turns": args.max_turns,
            "config_type": args.config_type,
            "max_input_tokens": args.max_input_tokens,
            "max_days": args.max_days,
            "max_execution_turns": args.max_execution_turns,
            "quality_first_mode": args.quality_first_mode,
            "legacy_caec_mode": args.caec_mode,
            "legacy_caec_config": args.caec_config,
            "legacy_fcr_mode": args.fcr_mode,
            "legacy_fcr_window_days": args.fcr_window_days,
            "api_key": "***" if args.api_key else None,
            "base_url": args.base_url,
        },
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        db_path=args.db_path,
    )

    goal = build_goal(DEFAULT_GOAL, config)
    checkpoint_dir = artifacts["checkpoint_dir"]
    client = create_openai_client(api_key=args.api_key, base_url=args.base_url)

    initial_messages = None
    start_turn = 0
    start_day = 1

    if args.recover_day is not None:
        print(f"[Checkpoint] Recovering from day {args.recover_day} checkpoint...")
        env, recovered_messages, _recovered_memory, start_day, start_turn = recover_from_day_checkpoint(
            checkpoint_dir,
            args.recover_day,
            config,
        )
        initial_messages = recovered_messages
    elif args.recover_turn is not None:
        print(f"[Checkpoint] Recovering from turn {args.recover_turn}...")
        env, recovered_messages = recover_from_checkpoint(checkpoint_dir, args.recover_turn, config)
        initial_messages = recovered_messages
        start_turn = args.recover_turn
        start_day = _infer_day_from_environment(env, start_day)
    else:
        env = RetailEnvironment(config)

    run_react_loop(
        goal=goal,
        env=env,
        model=args.model,
        max_turns=args.max_turns,
        log_path=artifacts["log_path"],
        max_input_tokens=args.max_input_tokens,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        initial_messages=initial_messages,
        start_turn=start_turn,
        start_day=start_day,
        log_dir=artifacts["run_dir"],
        max_days=args.max_days,
        max_execution_turns_per_day=args.max_execution_turns,
        client=client,
    )


if __name__ == "__main__":
    main()
