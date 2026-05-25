"""Shared parsing helpers for model-emitted tool calls."""

from __future__ import annotations

import json
import re
import ast
from typing import Any, Collection, Dict, List, Optional, Tuple

_TOOL_CALL_XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FUNCTION_STYLE_TOOL_CALL_PATTERN = re.compile(
    r"(?:<tool_call>\s*)?([A-Za-z_]\w*)\s*\((.*?)\)\s*(?:</tool_call>)?",
    re.DOTALL,
)


def parse_tool_args(raw: Any) -> Dict[str, Any]:
    """Safely parse tool arguments coming from the model."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_tool_call(candidate: Any) -> Optional[Dict[str, Any]]:
    """Normalize multiple tool-call shapes into {'name', 'arguments'}."""
    if not isinstance(candidate, dict):
        return None

    name = candidate.get("name")
    arguments = candidate.get("arguments")

    # Support OpenAI-style payloads:
    # {"type":"function","function":{"name":"...","arguments":"{...}"}}
    function_payload = candidate.get("function")
    if (not name) and isinstance(function_payload, dict):
        name = function_payload.get("name")
        arguments = function_payload.get("arguments")

    if not isinstance(name, str) or not name.strip():
        return None

    return {"name": name.strip(), "arguments": arguments if arguments is not None else {}}


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        return str(value)


def dedupe_end_today_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate repeated end_today actions while preserving order.

    Keep only the first end_today call for each equivalent arguments payload.
    """
    deduped: List[Dict[str, Any]] = []
    seen_end_today_signatures = set()

    for call in tool_calls:
        name = call.get("name")
        if name != "end_today":
            deduped.append(call)
            continue

        args_signature = _canonical_json(parse_tool_args(call.get("arguments")))
        if args_signature in seen_end_today_signatures:
            continue

        seen_end_today_signatures.add(args_signature)
        deduped.append(call)

    return deduped


def _filter_allowed_tool_calls(
    tool_calls: List[Dict[str, Any]],
    allowed_tool_names: Optional[Collection[str]],
) -> List[Dict[str, Any]]:
    if allowed_tool_names is None:
        return tool_calls
    allowed = set(allowed_tool_names)
    return [call for call in tool_calls if call.get("name") in allowed]


def _parse_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    parsed_calls: List[Dict[str, Any]] = []
    for json_str in _TOOL_CALL_XML_PATTERN.findall(text):
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        normalized = _normalize_tool_call(parsed)
        if normalized is not None:
            parsed_calls.append(normalized)
    return parsed_calls


def _parse_standalone_json_tool_calls(text: str) -> List[Dict[str, Any]]:
    parsed_calls: List[Dict[str, Any]] = []
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx : i + 1]
                start_idx = -1
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                normalized = _normalize_tool_call(parsed)
                if normalized is not None:
                    parsed_calls.append(normalized)

    return parsed_calls


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _literal_from_ast(key): _literal_from_ast(value)
            for key, value in zip(node.keys, node.values)
        }
    raise ValueError(f"Unsupported function-style argument: {type(node).__name__}")


def _parse_function_style_tool_calls(text: str) -> List[Dict[str, Any]]:
    parsed_calls: List[Dict[str, Any]] = []
    for match in _FUNCTION_STYLE_TOOL_CALL_PATTERN.finditer(text):
        name = match.group(1)
        raw_args = match.group(2).strip()
        if not raw_args:
            parsed_calls.append({"name": name, "arguments": {}})
            continue

        normalized_args = re.sub(r"=\s*true\b", "=True", raw_args, flags=re.IGNORECASE)
        normalized_args = re.sub(r"=\s*false\b", "=False", normalized_args, flags=re.IGNORECASE)
        normalized_args = re.sub(r"=\s*null\b", "=None", normalized_args, flags=re.IGNORECASE)

        try:
            expr = ast.parse(f"{name}({normalized_args})", mode="eval")
        except SyntaxError:
            continue
        if not isinstance(expr.body, ast.Call):
            continue
        if not isinstance(expr.body.func, ast.Name):
            continue
        arguments: Dict[str, Any] = {}
        unsupported = False
        for kw in expr.body.keywords:
            if kw.arg is None:
                unsupported = True
                break
            try:
                arguments[kw.arg] = _literal_from_ast(kw.value)
            except ValueError:
                unsupported = True
                break
        if unsupported or expr.body.args:
            continue
        parsed_calls.append({"name": expr.body.func.id, "arguments": arguments})
    return parsed_calls


def parse_tool_calls(
    text: str,
    allowed_tool_names: Optional[Collection[str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parse tool calls from text. Returns (tool_calls_list, parse_method_tag).

    Supported formats:
    1) XML blocks: <tool_call>{"name":"...", "arguments": {...}}</tool_call> (tag: "xml")
    2) Standalone JSON objects with tool-call shape (tag: "json")
    """
    xml_calls = _filter_allowed_tool_calls(_parse_xml_tool_calls(text), allowed_tool_names)
    if xml_calls:
        return dedupe_end_today_calls(xml_calls), "xml"

    function_calls = _filter_allowed_tool_calls(
        _parse_function_style_tool_calls(text),
        allowed_tool_names,
    )
    if function_calls:
        return dedupe_end_today_calls(function_calls), "function"

    json_calls = _filter_allowed_tool_calls(
        _parse_standalone_json_tool_calls(text),
        allowed_tool_names,
    )
    if json_calls:
        return dedupe_end_today_calls(json_calls), "json"

    return [], "none"
