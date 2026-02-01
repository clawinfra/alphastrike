#!/usr/bin/env python3
"""
AI Log Explainer using DeepSeek R1

Uses local Ollama + DeepSeek to generate human-readable explanations
of trading activity, performance metrics, and system decisions.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import httpx
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment variables for LLM configuration
# OLLAMA_BASE_URL: Base URL for Ollama API (default: http://localhost:11434)
# OLLAMA_MODEL: Model to use (default: qwen2.5:3b)
#   Options: "qwen2.5:3b" (fast, 3s), "deepseek-r1:7b" (slower, 11s, better reasoning)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


async def query_deepseek(prompt: str, max_tokens: int = 500) -> str:
    """Query LLM model via Ollama."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


def create_sample_trading_log() -> dict:
    """Create a sample trading log for explanation."""
    now = datetime.now(timezone.utc)

    return {
        "session_id": str(uuid4())[:8],
        "period": "2026-01-15 to 2026-02-01",
        "assets": {
            "BTCUSDT": {
                "trades": 12,
                "wins": 9,
                "losses": 3,
                "win_rate": 0.75,
                "total_pnl_r": 8.5,
                "total_pnl_usd": 425.00,
                "max_drawdown": 0.018,
                "avg_hold_time": "6.2 hours",
                "best_trade": {"pnl_r": 2.1, "side": "LONG", "duration": "4h"},
                "worst_trade": {"pnl_r": -1.0, "side": "SHORT", "duration": "2h"},
            },
            "ETHUSDT": {
                "trades": 8,
                "wins": 7,
                "losses": 1,
                "win_rate": 0.875,
                "total_pnl_r": 7.2,
                "total_pnl_usd": 288.00,
                "max_drawdown": 0.012,
                "avg_hold_time": "5.8 hours",
                "best_trade": {"pnl_r": 1.8, "side": "LONG", "duration": "8h"},
                "worst_trade": {"pnl_r": -0.8, "side": "LONG", "duration": "1h"},
            },
            "SOLUSDT": {
                "trades": 6,
                "wins": 2,
                "losses": 4,
                "win_rate": 0.333,
                "total_pnl_r": -2.4,
                "total_pnl_usd": -72.00,
                "max_drawdown": 0.045,
                "avg_hold_time": "3.1 hours",
                "best_trade": {"pnl_r": 1.2, "side": "LONG", "duration": "6h"},
                "worst_trade": {"pnl_r": -1.2, "side": "SHORT", "duration": "45m"},
            },
        },
        "system_events": [
            {"time": "2026-01-20 14:30", "event": "Model retrained", "reason": "Scheduled weekly update"},
            {"time": "2026-01-25 09:15", "event": "SOL shorts disabled", "reason": "3 consecutive losses"},
            {"time": "2026-01-28 11:00", "event": "SOL conviction raised to 80", "reason": "Win rate below threshold"},
        ],
        "market_conditions": {
            "btc_trend": "BULL (ADX 28)",
            "eth_correlation": 0.82,
            "volatility_regime": "MEDIUM",
        },
    }


async def explain_trading_log(log: dict) -> str:
    """Generate AI explanation of trading log."""

    prompt = f"""You are an expert quantitative trading analyst. Analyze this trading log and provide a clear, insightful explanation.

TRADING LOG:
Period: {log['period']}
Session: {log['session_id']}

ASSET PERFORMANCE:
"""

    for symbol, data in log['assets'].items():
        prompt += f"""
{symbol}:
- Trades: {data['trades']} ({data['wins']}W/{data['losses']}L)
- Win Rate: {data['win_rate']:.1%}
- P&L: {data['total_pnl_r']:.1f}R (${data['total_pnl_usd']:.2f})
- Max Drawdown: {data['max_drawdown']:.1%}
- Avg Hold: {data['avg_hold_time']}
- Best: {data['best_trade']['pnl_r']:.1f}R ({data['best_trade']['side']}, {data['best_trade']['duration']})
- Worst: {data['worst_trade']['pnl_r']:.1f}R ({data['worst_trade']['side']}, {data['worst_trade']['duration']})
"""

    prompt += f"""
SYSTEM EVENTS:
"""
    for event in log['system_events']:
        prompt += f"- [{event['time']}] {event['event']}: {event['reason']}\n"

    prompt += f"""
MARKET CONDITIONS:
- BTC Trend: {log['market_conditions']['btc_trend']}
- ETH Correlation: {log['market_conditions']['eth_correlation']}
- Volatility: {log['market_conditions']['volatility_regime']}

Provide a concise analysis covering:
1. Overall performance summary (1-2 sentences)
2. What worked well and why
3. What underperformed and root cause
4. System adaptations and their effectiveness
5. Recommendations for next period

Be specific and data-driven. No fluff."""

    return await query_deepseek(prompt, max_tokens=500)  # qwen2.5 doesn't need extra tokens


async def explain_single_trade(trade: dict) -> str:
    """Generate AI explanation of a single trade."""

    prompt = f"""Analyze this crypto trade and explain the outcome:

Trade Details:
- Symbol: {trade['symbol']}
- Side: {trade['side']}
- Entry: ${trade['entry_price']:,.2f}
- Exit: ${trade['exit_price']:,.2f}
- P&L: {trade['pnl_r']:.2f}R (${trade['pnl_usd']:.2f})
- Duration: {trade['duration']}
- Exit Reason: {trade['exit_reason']}
- Conviction Score: {trade['conviction']}/100
- Daily Trend: {trade['daily_trend']}
- 4H Signal: {trade['four_hour_signal']}

In 2-3 sentences, explain:
1. Why this trade was taken
2. What caused the outcome
3. Any lessons for future trades"""

    return await query_deepseek(prompt, max_tokens=200)  # Fast model, no thinking overhead


async def explain_retune_decision(trigger: dict, action: dict) -> str:
    """Generate AI explanation of a parameter adjustment."""

    prompt = f"""Explain this trading system adjustment:

TRIGGER:
- Asset: {trigger['symbol']}
- Type: {trigger['type']}
- Reason: {trigger['reason']}
- Current Value: {trigger['current_value']}
- Threshold: {trigger['threshold']}

ACTION TAKEN:
- Tool: {action['tool']}
- Parameters: {json.dumps(action['params'])}

In 2-3 sentences, explain:
1. Why the adjustment was needed
2. What the change does
3. Expected impact on future performance"""

    return await query_deepseek(prompt, max_tokens=200)  # Fast model, no thinking overhead


async def main():
    """Run AI log explanation demo."""
    print("=" * 60)
    print("AI LOG EXPLAINER")
    print(f"Model: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    print("=" * 60)

    # Test 1: Full trading log explanation
    print("\n[1] Generating Trading Session Summary...")
    print("-" * 40)

    log = create_sample_trading_log()
    explanation = await explain_trading_log(log)
    print(explanation)

    # Test 2: Single trade explanation
    print("\n" + "=" * 60)
    print("[2] Explaining Individual Trade...")
    print("-" * 40)

    trade = {
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entry_price": 44850.00,
        "exit_price": 45720.00,
        "pnl_r": 1.8,
        "pnl_usd": 87.00,
        "duration": "5h 20m",
        "exit_reason": "TAKE_PROFIT",
        "conviction": 74,
        "daily_trend": "BULL",
        "four_hour_signal": "LONG",
    }

    trade_explanation = await explain_single_trade(trade)
    print(trade_explanation)

    # Test 3: Retune decision explanation
    print("\n" + "=" * 60)
    print("[3] Explaining Parameter Adjustment...")
    print("-" * 40)

    trigger = {
        "symbol": "SOLUSDT",
        "type": "PERFORMANCE_DROP",
        "reason": "Win rate 33% below 50% threshold",
        "current_value": 0.33,
        "threshold": 0.50,
    }

    action = {
        "tool": "adjust_conviction",
        "params": {"symbol": "SOLUSDT", "new_threshold": 80},
    }

    retune_explanation = await explain_retune_decision(trigger, action)
    print(retune_explanation)

    print("\n" + "=" * 60)
    print("AI LOG EXPLAINER COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
