#!/usr/bin/env python3
"""
Adaptive Trading System Demo

Demonstrates the self-tuning capabilities:
1. Load per-asset configurations
2. Track performance and detect when retuning is needed
3. Query LLM advisor for decisions
4. Execute config adjustments
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive import (
    AdaptiveAssetConfig,
    load_asset_config,
    load_all_asset_configs,
    PerformanceTracker,
    Trade,
    AssetPerformance,
    LLMAdvisor,
    OllamaConfig,
    HotReloadManager,
    TradingState,
)


def create_sample_trades(symbol: str, win_rate: float, num_trades: int) -> list[Trade]:
    """Generate sample trades for testing."""
    trades = []
    now = datetime.now(timezone.utc)

    for i in range(num_trades):
        is_win = (i / num_trades) < win_rate
        pnl_r = 1.5 if is_win else -1.0

        trades.append(Trade(
            trade_id=str(uuid4())[:8],
            symbol=symbol,
            side="LONG" if i % 3 != 0 else "SHORT",
            entry_price=100.0,
            exit_price=101.5 if is_win else 99.0,
            size=0.1,
            entry_time=now - timedelta(hours=num_trades - i),
            exit_time=now - timedelta(hours=num_trades - i - 1),
            pnl=15.0 if is_win else -10.0,
            pnl_r=pnl_r,
            fees=0.5,
            conviction_score=72.0,
            regime="BULL",
            exit_reason="TAKE_PROFIT" if is_win else "STOP_LOSS",
        ))

    return trades


async def demo_performance_tracking():
    """Demo: Track performance and detect retune triggers."""
    print("=" * 60)
    print("DEMO: Performance Tracking")
    print("=" * 60)

    tracker = PerformanceTracker(
        window_days=30,
        win_rate_threshold=0.5,
        max_drawdown_threshold=0.05,
        max_consecutive_losses=5,
        min_trades_for_evaluation=10,
    )

    # Simulate trades for different assets
    print("\nRecording trades...")

    # BTC: Good performance (75% win rate)
    for trade in create_sample_trades("BTCUSDT", win_rate=0.75, num_trades=20):
        trigger = tracker.record_trade(trade)
        if trigger:
            print(f"  TRIGGER: {trigger.symbol} - {trigger.reason}")

    # ETH: Good performance (80% win rate)
    for trade in create_sample_trades("ETHUSDT", win_rate=0.80, num_trades=15):
        trigger = tracker.record_trade(trade)
        if trigger:
            print(f"  TRIGGER: {trigger.symbol} - {trigger.reason}")

    # SOL: Poor performance (30% win rate - should trigger retune)
    for trade in create_sample_trades("SOLUSDT", win_rate=0.30, num_trades=20):
        trigger = tracker.record_trade(trade)
        if trigger:
            print(f"  TRIGGER: {trigger.symbol} - {trigger.reason}")

    print("\n" + tracker.generate_report())

    return tracker


async def demo_config_loading():
    """Demo: Load per-asset configurations."""
    print("=" * 60)
    print("DEMO: Per-Asset Configuration")
    print("=" * 60)

    configs = load_all_asset_configs()

    for symbol, config in configs.items():
        print(f"\n{symbol}:")
        print(f"  Conviction Threshold: {config.conviction_threshold}")
        print(f"  Stop ATR Mult: {config.stop_atr_multiplier}")
        print(f"  Position Size Mult: {config.position_size_multiplier}")
        print(f"  Shorts Enabled: {config.short_enabled}")
        print(f"  Notes: {config.notes}")

    return configs


async def demo_llm_advisor(
    tracker: PerformanceTracker,
    configs: dict[str, AdaptiveAssetConfig],
):
    """Demo: Query LLM advisor for decisions."""
    print("\n" + "=" * 60)
    print("DEMO: LLM Advisor (Ollama + DeepSeek)")
    print("=" * 60)

    # Check if Ollama is available
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                print("\n[WARN] Ollama not available. Skipping LLM demo.")
                print("       Start Ollama with: ollama serve")
                print("       Pull model with: ollama pull deepseek-r1:8b")
                return
    except Exception:
        print("\n[WARN] Ollama not available. Skipping LLM demo.")
        print("       Start Ollama with: ollama serve")
        print("       Pull model with: ollama pull deepseek-r1:8b")
        return

    advisor = LLMAdvisor(
        config=OllamaConfig(
            model="deepseek-r1:8b",
            temperature=0.3,
        ),
        configs=configs,
    )

    # Register tool handlers
    async def handle_no_action(reason: str) -> str:
        return f"No action: {reason}"

    async def handle_adjust_conviction(symbol: str, new_threshold: float, reason: str) -> str:
        if symbol in configs:
            old = configs[symbol].conviction_threshold
            configs[symbol].conviction_threshold = new_threshold
            return f"Adjusted {symbol} conviction: {old} -> {new_threshold}"
        return f"Unknown symbol: {symbol}"

    advisor.register_tool_handler("no_action", handle_no_action)
    advisor.register_tool_handler("adjust_conviction", handle_adjust_conviction)

    # Get performance data
    performance_data = tracker.get_all_performance()
    triggers = tracker.get_pending_triggers()

    print("\nQuerying LLM for recommendations...")
    decisions = await advisor.analyze_performance(performance_data, triggers)

    print(f"\nLLM Recommendations ({len(decisions)} decisions):")
    for decision in decisions:
        print(f"\n  Tool: {decision.tool_name}")
        print(f"  Parameters: {decision.parameters}")
        print(f"  Reasoning: {decision.reasoning[:200]}...")

    # Execute decisions
    print("\nExecuting decisions...")
    executed = await advisor.execute_decisions(decisions)
    for decision in executed:
        print(f"  {decision.tool_name}: {decision.result}")


async def demo_hot_reload():
    """Demo: Hot reload and state preservation."""
    print("\n" + "=" * 60)
    print("DEMO: Hot Reload Manager")
    print("=" * 60)

    manager = HotReloadManager(
        state_dir=Path("data/demo_state"),
        min_validation_accuracy=0.55,
    )

    # Create sample state
    state = TradingState(
        positions=[
            {"symbol": "BTCUSDT", "side": "LONG", "size": 0.1, "entry_price": 45000},
        ],
        pending_orders=[],
        current_equity=10500.0,
        peak_equity=11000.0,
        model_versions={"ensemble": "v1.0.0"},
    )

    # Save state
    print("\nSaving trading state...")
    await manager.save_state(state)
    print(f"  State saved with {len(state.positions)} positions")

    # Load state
    print("\nLoading trading state...")
    loaded = await manager.load_state()
    if loaded:
        print(f"  Loaded state from {loaded.saved_at}")
        print(f"  Positions: {len(loaded.positions)}")
        print(f"  Equity: ${loaded.current_equity:,.2f}")

    # Get summary
    summary = manager.get_state_summary()
    print(f"\nState Summary: {summary}")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("ALPHASTRIKE ADAPTIVE TRADING SYSTEM DEMO")
    print("=" * 60 + "\n")

    # Demo 1: Load configs
    configs = await demo_config_loading()

    # Demo 2: Track performance
    tracker = await demo_performance_tracking()

    # Demo 3: LLM advisor (if Ollama available)
    await demo_llm_advisor(tracker, configs)

    # Demo 4: Hot reload
    await demo_hot_reload()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
