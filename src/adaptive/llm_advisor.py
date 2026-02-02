"""
LLM Advisor with Ollama + DeepSeek Integration

Uses local LLM with tool calling to make complex trading decisions
based on performance metrics and market conditions.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

import httpx

from src.adaptive.asset_config import AdaptiveAssetConfig, save_asset_config
from src.adaptive.performance_tracker import AssetPerformance, RetuneTrigger

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Definitions
# ============================================================================

TOOLS = [
    {
        "name": "retune_asset",
        "description": "Trigger parameter optimization for an asset. Use when an asset's performance has degraded and needs retuning.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol (e.g., BTCUSDT)"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for triggering retune"
                }
            },
            "required": ["symbol", "reason"]
        }
    },
    {
        "name": "disable_asset",
        "description": "Temporarily disable trading for an asset. Use for severe underperformance or market conditions that don't suit the strategy.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol"
                },
                "duration_hours": {
                    "type": "integer",
                    "description": "How long to disable trading (1-168 hours)"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for disabling"
                }
            },
            "required": ["symbol", "duration_hours", "reason"]
        }
    },
    {
        "name": "adjust_conviction",
        "description": "Adjust the conviction threshold for an asset. Higher threshold = fewer but higher quality signals.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol"
                },
                "new_threshold": {
                    "type": "number",
                    "description": "New conviction threshold (60-85)"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for adjustment"
                }
            },
            "required": ["symbol", "new_threshold", "reason"]
        }
    },
    {
        "name": "adjust_stops",
        "description": "Adjust stop loss and take profit ATR multipliers for an asset.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol"
                },
                "stop_atr_mult": {
                    "type": "number",
                    "description": "New stop loss ATR multiplier (1.5-3.0)"
                },
                "tp_atr_mult": {
                    "type": "number",
                    "description": "New take profit ATR multiplier (2.0-4.0)"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for adjustment"
                }
            },
            "required": ["symbol", "stop_atr_mult", "reason"]
        }
    },
    {
        "name": "adjust_position_size",
        "description": "Adjust position size multiplier for an asset. Reduce for volatile/underperforming assets.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol"
                },
                "multiplier": {
                    "type": "number",
                    "description": "New position size multiplier (0.25-1.5)"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for adjustment"
                }
            },
            "required": ["symbol", "multiplier", "reason"]
        }
    },
    {
        "name": "disable_shorts",
        "description": "Disable short selling for an asset. Use when shorts are consistently losing.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for disabling shorts"
                }
            },
            "required": ["symbol", "reason"]
        }
    },
    {
        "name": "trigger_hot_reload",
        "description": "Reload all models and configs without restart. Use after making configuration changes.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "send_alert",
        "description": "Send an alert to the operator. Use for important notifications.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Alert message"
                },
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "critical"],
                    "description": "Alert severity level"
                }
            },
            "required": ["message", "severity"]
        }
    },
    {
        "name": "adjust_leverage",
        "description": "Adjust portfolio leverage based on market conditions. Reduce in high volatility or drawdown, increase in low volatility trending markets.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_leverage": {
                    "type": "number",
                    "description": "New leverage multiplier (1.0-10.0). Current default is 5.0."
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for leverage adjustment (e.g., 'high volatility', 'drawdown protection', 'favorable trend')"
                }
            },
            "required": ["new_leverage", "reason"]
        }
    },
    {
        "name": "no_action",
        "description": "Take no action. Use when current performance is acceptable.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for no action"
                }
            },
            "required": ["reason"]
        }
    }
]


@dataclass
class AdvisorDecision:
    """Decision made by the LLM advisor."""

    tool_name: str
    parameters: dict[str, Any]
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executed: bool = False
    result: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "executed": self.executed,
            "result": self.result,
        }


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection via environment variables."""

    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    timeout: float = 60.0
    temperature: float = 0.3
    max_tokens: int = 1024


class LLMAdvisor:
    """
    LLM-powered trading advisor using Ollama + DeepSeek.

    Analyzes performance metrics and makes decisions using tool calling.
    """

    SYSTEM_PROMPT = """You are a trading strategy optimizer. Analyze performance data and recommend actions.

AVAILABLE ACTIONS:
- adjust_conviction: Raise threshold for underperforming assets (use 75-85 for poor performers)
- disable_shorts: Disable short selling when shorts consistently lose
- adjust_position_size: Reduce multiplier (0.25-0.5) for risky assets
- no_action: When performance is acceptable

RESPONSE FORMAT (JSON only):
{"action": "ACTION_NAME", "symbol": "SYMBOL", "params": {...}, "reason": "brief reason"}

Examples:
{"action": "adjust_conviction", "symbol": "SOLUSDT", "params": {"new_threshold": 80}, "reason": "win rate 33% below threshold"}
{"action": "no_action", "symbol": "BTCUSDT", "params": {}, "reason": "75% win rate is acceptable"}

Respond with ONE JSON object per asset that needs action. If asset is fine, use no_action."""

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        configs: Optional[dict[str, AdaptiveAssetConfig]] = None,
    ):
        self.config = config or OllamaConfig()
        self.asset_configs = configs or {}
        self._decision_history: list[AdvisorDecision] = []
        self._tool_handlers: dict[str, Callable[..., Awaitable[str]]] = {}

    def register_tool_handler(
        self, tool_name: str, handler: Callable[..., Awaitable[str]]
    ) -> None:
        """Register a handler function for a tool."""
        self._tool_handlers[tool_name] = handler

    async def analyze_performance(
        self,
        performance_data: dict[str, AssetPerformance],
        triggers: list[RetuneTrigger],
    ) -> list[AdvisorDecision]:
        """
        Analyze performance data and return recommended actions.

        Args:
            performance_data: Performance metrics per symbol
            triggers: Active retune triggers

        Returns:
            List of decisions/actions to take
        """
        # Build context for LLM
        context = self._build_context(performance_data, triggers)

        # Query LLM
        response = await self._query_ollama(context)

        # Parse tool calls from response
        decisions = self._parse_tool_calls(response)

        # Store in history
        self._decision_history.extend(decisions)

        return decisions

    def _build_context(
        self,
        performance_data: dict[str, AssetPerformance],
        triggers: list[RetuneTrigger],
    ) -> str:
        """Build context message for LLM."""
        lines = ["Asset Performance Report (Rolling 30 days):", ""]

        for symbol, perf in sorted(performance_data.items()):
            status = "⚠️" if perf.needs_retune else "✓"
            lines.extend([
                f"{symbol} {status}:",
                f"  - Trades: {perf.total_trades} ({perf.winning_trades}W/{perf.losing_trades}L)",
                f"  - Win Rate: {perf.win_rate:.1%}",
                f"  - P&L: {perf.total_pnl_r:.2f}R",
                f"  - Expectancy: {perf.expectancy_r:.2f}R/trade",
                f"  - Max Drawdown: {perf.max_drawdown:.1%}",
                f"  - Consecutive Losses: {perf.current_consecutive_losses}",
            ])

            # Include current config
            if symbol in self.asset_configs:
                cfg = self.asset_configs[symbol]
                lines.extend([
                    f"  Current Config:",
                    f"    - Conviction Threshold: {cfg.conviction_threshold}",
                    f"    - Stop ATR Mult: {cfg.stop_atr_multiplier}",
                    f"    - Position Size Mult: {cfg.position_size_multiplier}",
                    f"    - Shorts Enabled: {cfg.short_enabled}",
                ])
            lines.append("")

        if triggers:
            lines.extend(["", "Active Triggers:"])
            for t in triggers:
                lines.append(f"  - {t.symbol}: {t.trigger_type} - {t.reason}")

        lines.extend([
            "",
            "Based on this data, what actions should be taken?",
            "Call the appropriate tool(s) or call no_action if everything is acceptable.",
        ])

        return "\n".join(lines)

    async def _query_ollama(self, context: str) -> str:
        """Query Ollama API with the context."""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                # Use simple generate endpoint for faster response
                response = await client.post(
                    f"{self.config.base_url}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": f"{self.SYSTEM_PROMPT}\n\n{context}",
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")

        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            return json.dumps({
                "action": "send_alert",
                "symbol": "",
                "params": {"message": f"LLM advisor unavailable: {e}", "severity": "warning"},
                "reason": "API error"
            })

    def _parse_tool_calls(self, response: str) -> list[AdvisorDecision]:
        """Parse tool calls from LLM response."""
        decisions = []

        # First try our simple JSON format: {"action": "...", "symbol": "...", ...}
        try:
            # Find all JSON objects in response (handle nested braces in params)
            json_pattern = r'\{[^{}]*"action"\s*:\s*"[^"]+"\s*,[^{}]*(?:\{[^{}]*\})?[^{}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)

            # Also try to parse the whole response as JSON if it's a single object
            if not matches:
                response_stripped = response.strip()
                if response_stripped.startswith('{') and response_stripped.endswith('}'):
                    matches = [response_stripped]

            for match in matches:
                try:
                    data = json.loads(match)
                    action = data.get("action", "no_action")
                    symbol = data.get("symbol", "")
                    params = data.get("params", {})
                    reason = data.get("reason", "")

                    # Add symbol to params if not already there
                    if symbol and "symbol" not in params:
                        params["symbol"] = symbol

                    decisions.append(AdvisorDecision(
                        tool_name=action,
                        parameters=params,
                        reasoning=reason,
                    ))
                except json.JSONDecodeError:
                    continue

            if decisions:
                return decisions

        except Exception:
            pass

        # Fallback: try old tool_calls format
        try:
            data = json.loads(response)
            tool_calls = data.get("tool_calls", [])
            content = data.get("content", "")

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")

                if isinstance(args_str, str):
                    args = json.loads(args_str)
                else:
                    args = args_str

                decisions.append(AdvisorDecision(
                    tool_name=name,
                    parameters=args,
                    reasoning=content,
                ))

        except json.JSONDecodeError:
            # Try to extract from freeform text
            decisions.extend(self._extract_tool_calls_from_text(response))

        return decisions

    def _extract_tool_calls_from_text(self, text: str) -> list[AdvisorDecision]:
        """Extract tool calls from freeform text response."""
        decisions = []

        # Look for JSON-like patterns
        pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}'
        matches = re.findall(pattern, text)

        for match in matches:
            # Try to find the full JSON object
            try:
                start = text.find(f'"name": "{match}"')
                if start == -1:
                    continue

                # Find enclosing braces
                brace_start = text.rfind('{', 0, start)
                brace_end = text.find('}', start) + 1

                if brace_start != -1 and brace_end > brace_start:
                    json_str = text[brace_start:brace_end]
                    obj = json.loads(json_str)

                    decisions.append(AdvisorDecision(
                        tool_name=obj.get("name", ""),
                        parameters=obj.get("parameters", {}),
                        reasoning=text[:200],  # First 200 chars as reasoning
                    ))
            except (json.JSONDecodeError, ValueError):
                continue

        # If no tool calls found, assume no_action
        if not decisions:
            decisions.append(AdvisorDecision(
                tool_name="no_action",
                parameters={"reason": "Could not parse LLM response"},
                reasoning=text[:500],
            ))

        return decisions

    async def execute_decisions(
        self, decisions: list[AdvisorDecision]
    ) -> list[AdvisorDecision]:
        """
        Execute the decisions by calling registered handlers.

        Returns the decisions with execution results filled in.
        """
        for decision in decisions:
            handler = self._tool_handlers.get(decision.tool_name)

            if handler:
                try:
                    result = await handler(**decision.parameters)
                    decision.executed = True
                    decision.result = result
                except Exception as e:
                    decision.result = f"Error: {e}"
                    logger.error(f"Tool execution error: {e}")
            else:
                decision.result = f"No handler registered for {decision.tool_name}"

        return decisions

    def get_decision_history(self) -> list[AdvisorDecision]:
        """Get all decisions made by the advisor."""
        return self._decision_history.copy()


# ============================================================================
# Default Tool Handlers
# ============================================================================

async def handle_adjust_conviction(
    symbol: str,
    new_threshold: float,
    reason: str,
    configs: dict[str, AdaptiveAssetConfig],
) -> str:
    """Handle conviction threshold adjustment."""
    if symbol not in configs:
        return f"Unknown symbol: {symbol}"

    old_value = configs[symbol].conviction_threshold
    configs[symbol].conviction_threshold = max(60, min(85, new_threshold))
    save_asset_config(configs[symbol])

    return f"Adjusted {symbol} conviction: {old_value} -> {new_threshold}. Reason: {reason}"


async def handle_adjust_stops(
    symbol: str,
    stop_atr_mult: float,
    reason: str,
    configs: dict[str, AdaptiveAssetConfig],
    tp_atr_mult: Optional[float] = None,
) -> str:
    """Handle stop/TP adjustment."""
    if symbol not in configs:
        return f"Unknown symbol: {symbol}"

    cfg = configs[symbol]
    old_stop = cfg.stop_atr_multiplier
    cfg.stop_atr_multiplier = max(1.5, min(3.0, stop_atr_mult))

    if tp_atr_mult is not None:
        old_tp = cfg.take_profit_atr_multiplier
        cfg.take_profit_atr_multiplier = max(2.0, min(4.0, tp_atr_mult))
        save_asset_config(cfg)
        return f"Adjusted {symbol} stops: SL {old_stop}->{stop_atr_mult}, TP {old_tp}->{tp_atr_mult}"

    save_asset_config(cfg)
    return f"Adjusted {symbol} stop: {old_stop} -> {stop_atr_mult}. Reason: {reason}"


async def handle_disable_shorts(
    symbol: str,
    reason: str,
    configs: dict[str, AdaptiveAssetConfig],
) -> str:
    """Handle disabling shorts for an asset."""
    if symbol not in configs:
        return f"Unknown symbol: {symbol}"

    configs[symbol].short_enabled = False
    save_asset_config(configs[symbol])

    return f"Disabled shorts for {symbol}. Reason: {reason}"


async def handle_adjust_position_size(
    symbol: str,
    multiplier: float,
    reason: str,
    configs: dict[str, AdaptiveAssetConfig],
) -> str:
    """Handle position size adjustment."""
    if symbol not in configs:
        return f"Unknown symbol: {symbol}"

    old_value = configs[symbol].position_size_multiplier
    configs[symbol].position_size_multiplier = max(0.25, min(1.5, multiplier))
    save_asset_config(configs[symbol])

    return f"Adjusted {symbol} position size: {old_value} -> {multiplier}. Reason: {reason}"


async def handle_send_alert(message: str, severity: str) -> str:
    """Handle sending an alert."""
    logger.log(
        {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}.get(
            severity, logging.INFO
        ),
        f"[ALERT] {message}",
    )
    return f"Alert sent: [{severity.upper()}] {message}"


async def handle_no_action(reason: str) -> str:
    """Handle no-action decision."""
    logger.info(f"No action taken: {reason}")
    return f"No action: {reason}"
