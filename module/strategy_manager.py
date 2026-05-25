"""
Strategy Manager Module

This module provides functionality for managing retail strategy, including:
- Strategy storage structure
- Strategy formatting
- Strategy update tools (set_macro_strategy, set_execute_strategy, set_action)
- Tool definitions for OpenAI function calling
"""

import json
from typing import Any, Dict, List


def create_initial_strategy() -> Dict[str, Any]:
    """Create initial strategy structure.
    
    Returns:
        Initial strategy dictionary with three components:
        - macro_strategy: array of strings
        - execute_strategy: object with seven fields, all arrays
        - today_action: array of action objects
    """
    return {
        "macro_strategy": ["Focus on maintaining inventory levels and competitive pricing"],
        "execute_strategy": {
            "focus_skus": [],
            "sku_supplier_mapping": [],
            "news_to_monitor": [],
            "skus_to_reorder": [],
            "price_adjustments": [],
            "sku_to_monitor": [],
            "other": [],
        },
        # today_action: 使用 place_order / modify_sku_price 的参数形式的动作列表
        # 例如: [{"tool": "place_order", "arguments": {...}}, {"tool": "modify_sku_price", "arguments": {...}}]
        "today_action": [],
    }


def format_strategy_dict(strategy_store: Dict[str, Any], prefix: str = "") -> str:
    """格式化策略字典为字符串"""
    if not strategy_store:
        return prefix + "(empty)"
    
    formatted = prefix
    macro_strategy = strategy_store.get('macro_strategy', [])
    if isinstance(macro_strategy, list):
        formatted += f"Macro Strategy (array):\n"
        for i, item in enumerate(macro_strategy):
            formatted += f"  {i+1}. {item}\n"
    else:
        formatted += f"Macro Strategy: {macro_strategy}\n"
    
    formatted += "Execute Strategy:\n"
    exec_strat = strategy_store.get('execute_strategy', {})
    formatted += f"  - Focus SKUs: {exec_strat.get('focus_skus', [])}\n"
    formatted += f"  - SKU-Supplier Mapping: {exec_strat.get('sku_supplier_mapping', [])}\n"
    formatted += f"  - News to Monitor: {exec_strat.get('news_to_monitor', [])}\n"
    formatted += f"  - SKUs to Reorder: {exec_strat.get('skus_to_reorder', [])}\n"
    formatted += f"  - Price Adjustments: {exec_strat.get('price_adjustments', [])}\n"
    formatted += f"  - SKUs to Monitor: {exec_strat.get('sku_to_monitor', [])}\n"
    formatted += f"  - Other: {exec_strat.get('other', [])}\n"
    
    today_action = strategy_store.get("today_action", [])
    try:
        today_action_str = json.dumps(today_action, ensure_ascii=False, indent=2)
    except Exception:
        today_action_str = str(today_action)
    formatted += "Today Action (as tool call list):\n"
    formatted += today_action_str + "\n"
    return formatted


class StrategyManager:
    """Manager for retail strategy operations."""
    
    def __init__(self, initial_strategy: Dict[str, Any] = None):
        """Initialize StrategyManager with a strategy store.
        
        Args:
            initial_strategy: Initial strategy dictionary. If None, uses default initial strategy.
        """
        if initial_strategy is None:
            initial_strategy = create_initial_strategy()
        self.strategy_store = initial_strategy
    
    def set_macro_strategy(self, macro_strategy: List[str]) -> Dict[str, Any]:
        """Set the macro strategy (array of strings).
        
        Args:
            macro_strategy: Array of strings representing macro strategy guidelines
            
        Returns:
            Dictionary with 'result' and 'formatted' keys
        """
        try:
            if not isinstance(macro_strategy, list):
                raise ValueError("macro_strategy must be an array")
            if not all(isinstance(item, str) for item in macro_strategy):
                raise ValueError("All items in macro_strategy must be strings")
            old_macro = self.strategy_store["macro_strategy"].copy()
            self.strategy_store["macro_strategy"] = macro_strategy.copy()
            formatted = f"Set macro_strategy:\nOld: {json.dumps(old_macro, ensure_ascii=False, indent=2)}\nNew: {json.dumps(macro_strategy, ensure_ascii=False, indent=2)}"
            return {
                "result": {
                    "macro_strategy": self.strategy_store["macro_strategy"].copy(),
                    "execute_strategy": self.strategy_store["execute_strategy"].copy(),
                    "today_action": self.strategy_store["today_action"].copy(),
                },
                "formatted": formatted
            }
        except ValueError as e:
            return {
                "result": {"error": f"Invalid macro_strategy format: {str(e)}"},
                "formatted": f"Error: Invalid structure. Expected an array of strings. Details: {str(e)}"
            }
    
    def set_execute_strategy(self, execute_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Set the execute strategy (object with seven fields, all arrays).
        
        Args:
            execute_strategy: Object with fields: focus_skus, sku_supplier_mapping, news_to_monitor, 
                             skus_to_reorder, price_adjustments, sku_to_monitor, other (all arrays)
            
        Returns:
            Dictionary with 'result' and 'formatted' keys
        """
        try:
            if not isinstance(execute_strategy, dict):
                raise ValueError("execute_strategy must be an object")
            
            # 验证字段，所有值都必须是数组
            valid_fields = [
                "focus_skus",
                "sku_supplier_mapping",
                "news_to_monitor",
                "skus_to_reorder",
                "price_adjustments",
                "sku_to_monitor",
                "other",
            ]
            for field in valid_fields:
                if field not in execute_strategy:
                    execute_strategy[field] = self.strategy_store["execute_strategy"].get(field, []).copy()
                elif not isinstance(execute_strategy[field], list):
                    raise ValueError(f"Field '{field}' must be an array, got {type(execute_strategy[field]).__name__}")
            
            old_execute = self.strategy_store["execute_strategy"].copy()
            self.strategy_store["execute_strategy"] = execute_strategy.copy()
            formatted = f"Set execute_strategy:\nOld: {json.dumps(old_execute, ensure_ascii=False, indent=2)}\nNew: {json.dumps(execute_strategy, ensure_ascii=False, indent=2)}"
            return {
                "result": {
                    "macro_strategy": self.strategy_store["macro_strategy"].copy(),
                    "execute_strategy": self.strategy_store["execute_strategy"].copy(),
                    "today_action": self.strategy_store["today_action"].copy(),
                },
                "formatted": formatted
            }
        except ValueError as e:
            return {
                "result": {"error": f"Invalid execute_strategy format: {str(e)}"},
                "formatted": (
                    "Error: Invalid structure. Expected an object with seven fields (all arrays): "
                    "focus_skus, sku_supplier_mapping, news_to_monitor, skus_to_reorder, price_adjustments, sku_to_monitor, other. "
                    f"Details: {str(e)}"
                )
            }
    
    def set_action(self, action: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Set the today action (array of action objects).
        
        Args:
            action: Array of action objects, each with 'tool' and 'arguments' fields
            
        Returns:
            Dictionary with 'result' and 'formatted' keys
        """
        try:
            if not isinstance(action, list):
                raise ValueError("action must be an array")
            
            validated_actions = []
            for idx, act in enumerate(action):
                if not isinstance(act, dict):
                    raise ValueError(f"action[{idx}] must be an object")
                tool_name = act.get("tool")
                arguments = act.get("arguments")
                if tool_name not in ("place_order", "modify_sku_price"):
                    raise ValueError(
                        f"action[{idx}].tool must be 'place_order' or 'modify_sku_price', got '{tool_name}'"
                    )
                if not isinstance(arguments, dict):
                    raise ValueError(f"action[{idx}].arguments must be an object")
                validated_actions.append({"tool": tool_name, "arguments": arguments})
            
            old_action = self.strategy_store["today_action"].copy()
            self.strategy_store["today_action"] = validated_actions
            formatted = (
                "Set today_action (tool call list):\n"
                f"Old: {json.dumps(old_action, ensure_ascii=False, indent=2)}\n"
                f"New: {json.dumps(validated_actions, ensure_ascii=False, indent=2)}"
            )
            return {
                "result": {
                    "macro_strategy": self.strategy_store["macro_strategy"].copy(),
                    "execute_strategy": self.strategy_store["execute_strategy"].copy(),
                    "today_action": self.strategy_store["today_action"].copy(),
                },
                "formatted": formatted
            }
        except ValueError as e:
            return {
                "result": {"error": f"Invalid action format: {str(e)}"},
                "formatted": (
                    "Error: Invalid structure. Expected an array of objects with 'tool' and 'arguments' fields. "
                    f"Details: {str(e)}"
                ),
            }
    
    def get_strategy(self) -> Dict[str, Any]:
        """Get current strategy store.
        
        Returns:
            Copy of current strategy store
        """
        return {
            "macro_strategy": self.strategy_store["macro_strategy"].copy(),
            "execute_strategy": self.strategy_store["execute_strategy"].copy(),
            "today_action": self.strategy_store["today_action"].copy(),
        }


def get_strategy_tool_definitions() -> List[Dict[str, Any]]:
    """Get tool definitions for strategy management tools.
    
    Returns:
        List of tool definition dictionaries for OpenAI function calling
    """
    strategy_state_schema = {
        "type": "object",
        "properties": {
            "macro_strategy": {"type": "array", "items": {"type": "string"}},
            "execute_strategy": {
                "type": "object",
                "properties": {
                    "focus_skus": {"type": "array"},
                    "sku_supplier_mapping": {"type": "array"},
                    "news_to_monitor": {"type": "array"},
                    "skus_to_reorder": {"type": "array"},
                    "price_adjustments": {"type": "array"},
                    "sku_to_monitor": {"type": "array"},
                    "other": {"type": "array"},
                },
                "additionalProperties": True,
            },
            "today_action": {"type": "array"},
            "error": {"type": "string"},
        },
        "additionalProperties": True,
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "set_macro_strategy",
                "description": "Set the macro strategy. Macro strategy is an array of strings representing broad strategic guidelines. Returns result as the full strategy state or {'error': str}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "macro_strategy": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Array of strings representing macro strategy guidelines. Example: [\"Focus on high-margin products\", \"Maintain competitive pricing\"]"
                        }
                    },
                    "required": ["macro_strategy"]
                },
                "output_schema": strategy_state_schema,
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_execute_strategy",
                "description": "Set the execute strategy. Execute strategy is an object with seven fields, all values are arrays: focus_skus, sku_supplier_mapping, news_to_monitor, skus_to_reorder, price_adjustments, sku_to_monitor, other. Returns result as the full strategy state or {'error': str}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "execute_strategy": {
                            "type": "object",
                            "properties": {
                                "focus_skus": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of SKU IDs that need attention"
                                },
                                "sku_supplier_mapping": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "sku_id": {"type": "string"},
                                            "supplier_id": {"type": "string"}
                                        },
                                        "required": ["sku_id", "supplier_id"]
                                    },
                                    "description": "Array of mapping objects: [{{\"sku_id\": \"SKU_001\", \"supplier_id\": \"supplier_A\"}}]"
                                },
                                "news_to_monitor": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of news items to monitor"
                                },
                                "skus_to_reorder": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of SKU IDs that need reordering"
                                },
                                "price_adjustments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "sku_id": {"type": "string"},
                                            "adjustment": {"type": "string"}
                                        },
                                        "required": ["sku_id", "adjustment"]
                                    },
                                    "description": "Array of price adjustment objects: [{{\"sku_id\": \"SKU_001\", \"adjustment\": \"increase by 10%\"}}]"
                                },
                                "sku_to_monitor": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of SKU IDs that should be closely monitored"
                                },
                                "other": {
                                    "type": "array",
                                    "description": "Array of any other strategy notes or metadata objects"
                                }
                            },
                            "description": "Object with seven fields (all arrays): focus_skus, sku_supplier_mapping, news_to_monitor, skus_to_reorder, price_adjustments, sku_to_monitor, other"
                        }
                    },
                    "required": ["execute_strategy"]
                },
                "output_schema": strategy_state_schema,
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_action",
                "description": "Set the today action. Today action is an array of action objects, each with 'tool' ('place_order' or 'modify_sku_price') and 'arguments' fields. Returns result as the full strategy state or {'error': str}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {
                                        "type": "string",
                                        "enum": ["place_order", "modify_sku_price"],
                                        "description": "Tool name: 'place_order' or 'modify_sku_price'"
                                    },
                                    "arguments": {
                                        "type": "object",
                                        "description": "Arguments matching the corresponding tool parameters"
                                    }
                                },
                                "required": ["tool", "arguments"]
                            },
                            "description": "Array of action objects. Example: [{{\"tool\": \"place_order\", \"arguments\": {{\"sku_id\": \"SKU_001\", \"supplier_id\": \"supplier_A\", \"quantity\": 100}}}}]"
                        }
                    },
                    "required": ["action"]
                },
                "output_schema": strategy_state_schema,
            }
        }
    ]
