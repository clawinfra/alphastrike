#!/usr/bin/env python3
"""
LLM-Adaptive Multi-Asset Backtest

Combines:
1. ML Ensemble (XGBoost, LightGBM, RF) for signal prediction
2. LLM Advisor (Ollama) for strategic parameter adjustment

The LLM periodically reviews performance and makes tool calls to:
- adjust_conviction: Tune signal thresholds
- adjust_position_size: Reduce risk for underperformers
- disable_shorts: Turn off shorts for losing assets
- send_alert: Notify operator

Usage:
    python scripts/llm_adaptive_backtest.py --days 90 --leverage 5
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.multi_asset_engine import (
    AssetConfig,
    MultiAssetBacktestConfig,
    MultiAssetBacktestEngine,
)
from src.exchange.adapters.hyperliquid import HyperliquidAdapter
from src.adaptive import (
    LLMAdvisor,
    OllamaConfig,
    PerformanceTracker,
    Trade,
    AssetPerformance,
    DynamicLeverageManager,
    TradingState,
    HotReloadManager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Default diversified portfolio
DEFAULT_ASSETS = [
    AssetConfig("BTC", "crypto_major", 1.0, 1.0),
    AssetConfig("ETH", "crypto_major", 1.0, 1.5),
    AssetConfig("PAXG", "gold_proxy", 0.3, 2.0),
    AssetConfig("SOL", "crypto_l1", 1.2, 2.0),
    AssetConfig("AAVE", "crypto_defi", 1.0, 3.0),
]


class LLMAdaptiveBacktest:
    """
    Backtest engine with LLM decision layer for adaptive parameter tuning.

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ML PREDICTION LAYER                          │
    │  FeaturePipeline → MLEnsemble → SignalProcessor                │
    └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                  EXECUTION ENGINE                               │
    │  Position sizing, SL/TP, Risk management                       │
    └─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              PERFORMANCE TRACKER                                │
    │  Rolling metrics, trigger detection                            │
    └─────────────────────────────────────────────────────────────────┘
                                 │ (every N trades)
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                LLM DECISION LAYER                               │
    │  Ollama + DeepSeek/Qwen for strategic decisions                │
    │  Tools: adjust_conviction, disable_shorts, etc.                │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: MultiAssetBacktestConfig,
        llm_review_interval: int = 20,  # Review every N trades
    ):
        self.config = config
        self.llm_review_interval = llm_review_interval

        # Per-asset parameters (LLM can adjust these)
        self.asset_params = {
            a.symbol: {
                "conviction_threshold": 0.5,  # Signal confidence threshold
                "position_multiplier": 1.0,   # Position size multiplier
                "short_enabled": True,        # Allow shorts
            }
            for a in config.assets
        }

        # Dynamic leverage manager (NEW - hot reload compatible)
        self.leverage_manager = DynamicLeverageManager(
            state_file=Path("data/state/leverage_state.json"),
            base_leverage=config.leverage,
            min_leverage=1.0,
            max_leverage=10.0,
        )

        # Hot reload manager for state persistence
        self.hot_reload = HotReloadManager(
            state_dir=Path("data/state"),
            model_dir=Path("models"),
        )

        # Performance tracker
        self.tracker = PerformanceTracker(
            window_days=30,
            win_rate_threshold=0.40,
            max_drawdown_threshold=0.15,
            max_consecutive_losses=5,
            min_trades_for_evaluation=10,
        )

        # LLM advisor
        self.advisor = LLMAdvisor(
            config=OllamaConfig(
                model="qwen2.5:1.5b",  # Fast local model
                temperature=0.3,
                timeout=30.0,
            )
        )
        self._register_tool_handlers()

        # Trade counter
        self.trade_count = 0
        self.llm_decisions = []
        self.leverage_changes = []  # Track leverage adjustments

    def _resolve_symbol(self, symbol: str) -> str | None:
        """Resolve symbol to asset_params key, handling USDT suffix."""
        if not symbol:
            return None
        # Try without USDT suffix first
        key = symbol.replace("USDT", "")
        if key in self.asset_params:
            return key
        # Try original symbol
        if symbol in self.asset_params:
            return symbol
        return None

    def _register_tool_handlers(self) -> None:
        """Register LLM tool handlers."""

        async def handle_adjust_conviction(symbol: str = "", new_threshold: float = 70, reason: str = "", **kwargs) -> str:
            key = self._resolve_symbol(symbol)
            if not key:
                return f"Unknown symbol: {symbol}"
            old = self.asset_params[key]["conviction_threshold"]
            self.asset_params[key]["conviction_threshold"] = max(0.4, min(0.9, new_threshold / 100))
            return f"Adjusted {key} conviction: {old:.0%} -> {new_threshold/100:.0%}"

        async def handle_adjust_position_size(symbol: str = "", multiplier: float = 0.5, reason: str = "", **kwargs) -> str:
            key = self._resolve_symbol(symbol)
            if not key:
                return f"Unknown symbol: {symbol}"
            old = self.asset_params[key]["position_multiplier"]
            self.asset_params[key]["position_multiplier"] = max(0.1, min(1.5, multiplier))
            return f"Adjusted {key} position size: {old:.1f}x -> {multiplier:.1f}x"

        async def handle_disable_shorts(symbol: str = "", reason: str = "", **kwargs) -> str:
            key = self._resolve_symbol(symbol)
            if not key:
                return f"Unknown symbol: {symbol}"
            self.asset_params[key]["short_enabled"] = False
            return f"Disabled shorts for {key}"

        async def handle_no_action(reason: str = "", symbol: str = "", **kwargs) -> str:
            if symbol:
                return f"No action needed for {symbol}: {reason}"
            return f"No action: {reason}"

        async def handle_send_alert(message: str = "", severity: str = "info", **kwargs) -> str:
            logger.warning(f"[LLM ALERT - {severity.upper()}] {message}")
            return f"Alert sent: {message}"

        async def handle_adjust_leverage(new_leverage: float = 5.0, reason: str = "", **kwargs) -> str:
            """Adjust portfolio leverage dynamically based on market conditions."""
            actual_leverage, msg = self.leverage_manager.set_leverage(new_leverage, reason)
            self.config.leverage = actual_leverage
            self.leverage_changes.append({
                "trade_count": self.trade_count,
                "old_leverage": self.leverage_manager.state.base_leverage,
                "new_leverage": actual_leverage,
                "reason": reason,
            })
            logger.info(f"[LLM] Leverage adjusted: {msg}")
            return msg

        self.advisor.register_tool_handler("adjust_conviction", handle_adjust_conviction)
        self.advisor.register_tool_handler("adjust_position_size", handle_adjust_position_size)
        self.advisor.register_tool_handler("disable_shorts", handle_disable_shorts)
        self.advisor.register_tool_handler("no_action", handle_no_action)
        self.advisor.register_tool_handler("send_alert", handle_send_alert)
        self.advisor.register_tool_handler("adjust_leverage", handle_adjust_leverage)

    def record_trade(self, trade_data: dict) -> None:
        """Record a completed trade and check if LLM review needed."""
        # Create Trade object for tracker
        trade = Trade(
            trade_id=trade_data["id"],
            symbol=trade_data["symbol"],
            side=trade_data["side"],
            entry_price=trade_data["entry_price"],
            exit_price=trade_data["exit_price"],
            size=trade_data["size"],
            entry_time=trade_data["entry_time"],
            exit_time=trade_data["exit_time"],
            pnl=trade_data["pnl"],
            pnl_r=trade_data["pnl"] / (trade_data["entry_price"] * trade_data["size"] * 0.02),  # ~2% risk
            fees=trade_data.get("fees", 0),
            conviction_score=trade_data.get("confidence", 0.5) * 100,
            regime=trade_data.get("regime", "unknown"),
            exit_reason=trade_data.get("exit_reason", "unknown"),
        )

        self.tracker.record_trade(trade)
        self.trade_count += 1

    async def maybe_llm_review(self) -> list:
        """Check if it's time for LLM review and run if needed."""
        if self.trade_count % self.llm_review_interval != 0:
            return []

        if self.trade_count < self.llm_review_interval:
            return []

        # Get performance data
        performance = self.tracker.get_all_performance()
        if not performance:
            return []

        # Filter for symbols with enough trades
        active_performance = {
            s: p for s, p in performance.items()
            if p.total_trades >= 5
        }

        if not active_performance:
            return []

        logger.info(f"LLM Review at trade #{self.trade_count}")

        # Query LLM
        triggers = self.tracker.get_pending_triggers()
        decisions = await self.advisor.analyze_performance(active_performance, triggers)

        # Execute decisions
        executed = await self.advisor.execute_decisions(decisions)

        for d in executed:
            self.llm_decisions.append({
                "trade_count": self.trade_count,
                "tool": d.tool_name,
                "params": d.parameters,
                "result": d.result,
            })
            logger.info(f"  LLM Decision: {d.tool_name} -> {d.result}")

        return executed

    def get_position_multiplier(self, symbol: str) -> float:
        """Get LLM-adjusted position multiplier for symbol."""
        return self.asset_params.get(symbol, {}).get("position_multiplier", 1.0)

    def is_shorts_enabled(self, symbol: str) -> bool:
        """Check if shorts are enabled for symbol."""
        return self.asset_params.get(symbol, {}).get("short_enabled", True)

    def get_conviction_threshold(self, symbol: str) -> float:
        """Get LLM-adjusted conviction threshold for symbol."""
        return self.asset_params.get(symbol, {}).get("conviction_threshold", 0.5)

    def get_current_leverage(self) -> float:
        """Get current dynamic leverage."""
        return self.leverage_manager.get_leverage()

    def update_leverage_conditions(
        self,
        current_volatility: float,
        current_drawdown: float,
        rolling_win_rate: float,
    ) -> tuple[float, bool, str]:
        """
        Update market conditions and auto-adjust leverage if needed.

        This is called periodically during backtest to adapt leverage
        to changing conditions (volatility, drawdown, performance).

        Returns:
            (new_leverage, changed, reason)
        """
        new_leverage, changed, reason = self.leverage_manager.update_conditions(
            current_volatility=current_volatility,
            current_drawdown=current_drawdown,
            rolling_win_rate=rolling_win_rate,
        )

        if changed:
            # Update config for future trades
            self.config.leverage = new_leverage
            self.leverage_changes.append({
                "trade_count": self.trade_count,
                "new_leverage": new_leverage,
                "reason": f"Auto: {reason}",
            })
            logger.info(f"Auto-adjusted leverage to {new_leverage:.1f}x: {reason}")

        return new_leverage, changed, reason

    def save_state(self) -> None:
        """Save full state for hot reload recovery."""
        state = TradingState(
            current_equity=0.0,  # Would be set from actual trading state
            peak_equity=0.0,
            current_leverage=self.leverage_manager.get_leverage(),
            leverage_reason=self.leverage_manager.state.adjustment_reason,
            asset_params=self.asset_params,
        )
        asyncio.create_task(self.hot_reload.save_state(state))

    def get_leverage_status(self) -> dict:
        """Get leverage status for display."""
        return self.leverage_manager.get_status()


async def fetch_candles(
    adapter,
    symbols: list[str],
    days: int,
    interval: str = "1h",
) -> dict[str, list]:
    """Fetch historical candles for all assets."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    candles_by_asset = {}

    for symbol in symbols:
        try:
            unified_symbol = f"{symbol}USDT"
            logger.info(f"Fetching {symbol}...")

            candles = await adapter.rest.get_candles(
                symbol=unified_symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=days * 24,
            )

            candles_by_asset[symbol] = candles
            logger.info(f"  {symbol}: {len(candles)} candles")

        except Exception as e:
            logger.warning(f"  {symbol}: Failed - {e}")

    return candles_by_asset


def initialize_production_components():
    """Initialize production ML components."""
    from src.execution.signal_processor import SignalProcessor
    from src.features.pipeline import FeaturePipeline
    from src.ml.ensemble import MLEnsemble

    logger.info("Initializing production components...")

    feature_pipeline = FeaturePipeline()
    logger.info("  FeaturePipeline initialized")

    ml_ensemble = MLEnsemble(models_dir=Path("models"))
    ml_ensemble.check_and_reload_models()

    health_status = ml_ensemble.get_health_status()
    healthy_count = sum(1 for h in health_status.values() if h)
    logger.info(f"  MLEnsemble: {healthy_count}/4 healthy models")

    signal_processor = SignalProcessor()
    logger.info("  SignalProcessor initialized")

    return feature_pipeline, ml_ensemble, signal_processor


async def main():
    parser = argparse.ArgumentParser(description="LLM-Adaptive Multi-Asset Backtest")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval")
    parser.add_argument("--llm-interval", type=int, default=20, help="LLM review interval (trades)")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM-ADAPTIVE MULTI-ASSET BACKTEST")
    print("=" * 70)
    print(f"Initial Balance: ${args.balance:,.0f}")
    print(f"Leverage: {args.leverage}x")
    print(f"Days: {args.days}")
    print(f"LLM Review Interval: Every {args.llm_interval} trades")
    print()
    print("Architecture:")
    print("  ML Layer: FeaturePipeline + MLEnsemble + SignalProcessor")
    print("  LLM Layer: Ollama (qwen2.5:1.5b) for adaptive parameter tuning")
    print()

    # Initialize production components
    try:
        feature_pipeline, ml_ensemble, signal_processor = initialize_production_components()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return

    # Initialize Hyperliquid adapter
    adapter = HyperliquidAdapter()
    await adapter.initialize()

    try:
        # Fetch candles
        print("Fetching candles from Hyperliquid...")
        print("-" * 40)

        symbols = [a.symbol for a in DEFAULT_ASSETS]
        candles_by_asset = await fetch_candles(adapter, symbols, args.days, args.interval)

        if not candles_by_asset:
            print("ERROR: No candles fetched")
            return

        print()

        # Configure backtest
        config = MultiAssetBacktestConfig(
            assets=DEFAULT_ASSETS,
            initial_balance=args.balance,
            leverage=args.leverage,
            interval=args.interval,
            warmup_candles=100,
            max_portfolio_heat=0.40,
            max_single_position=0.20,
            per_trade_exposure=0.10,
            stop_loss_atr=1.5,
            take_profit_atr=2.5,
        )

        # Create LLM-adaptive backtest wrapper
        llm_backtest = LLMAdaptiveBacktest(
            config=config,
            llm_review_interval=args.llm_interval,
        )

        # Create engine with production components
        engine = MultiAssetBacktestEngine(
            config=config,
            feature_pipeline=feature_pipeline,
            ml_ensemble=ml_ensemble,
            signal_processor=signal_processor,
        )

        # Run backtest
        print("Running backtest with ML + LLM layers...")
        print("-" * 40)
        result = engine.run(candles_by_asset)

        # Record trades to LLM tracker
        for trade in result.trades:
            llm_backtest.record_trade({
                "id": trade.id,
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "size": trade.size,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "pnl": trade.pnl,
                "fees": trade.fees,
                "confidence": trade.confidence,
                "regime": trade.regime,
            })

            # Check for LLM review
            await llm_backtest.maybe_llm_review()

        # Print results
        print()
        print("=" * 70)
        print("RESULTS - ML + LLM ADAPTIVE SYSTEM")
        print("=" * 70)
        print()

        m = result.metrics
        print(f"{'Metric':<25} {'Value':>15}")
        print("-" * 45)
        print(f"{'Annualized Return (CAGR)':<25} {m.cagr:>14.1f}%")
        print(f"{'Max Drawdown':<25} {m.max_drawdown:>14.1f}%")
        print(f"{'Sharpe Ratio':<25} {m.sharpe_ratio:>15.2f}")
        print(f"{'Win Rate':<25} {m.win_rate:>14.1f}%")
        print(f"{'Total Trades':<25} {m.total_trades:>15}")
        print()

        print(f"Initial Balance: ${config.initial_balance:,.0f}")
        print(f"Final Balance:   ${m.final_balance:,.0f}")
        print(f"Total PnL:       ${m.final_balance - config.initial_balance:,.0f}")
        print()

        # LLM decisions
        print("LLM DECISIONS MADE")
        print("-" * 70)
        if llm_backtest.llm_decisions:
            for d in llm_backtest.llm_decisions:
                print(f"  Trade #{d['trade_count']}: {d['tool']} -> {d['result'][:50]}...")
        else:
            print("  No LLM interventions (performance acceptable)")
        print()

        # Final asset parameters
        print("FINAL ASSET PARAMETERS (LLM-Adjusted)")
        print("-" * 70)
        print(f"{'Asset':<10} {'Conviction':>12} {'Size Mult':>12} {'Shorts':>10}")
        print("-" * 70)
        for symbol, params in llm_backtest.asset_params.items():
            shorts = "YES" if params["short_enabled"] else "NO"
            print(f"{symbol:<10} {params['conviction_threshold']:>11.0%} {params['position_multiplier']:>11.1f}x {shorts:>10}")

        print()

        # Dynamic leverage status (NEW)
        print("DYNAMIC LEVERAGE STATUS")
        print("-" * 70)
        leverage_status = llm_backtest.get_leverage_status()
        print(f"  Current Leverage: {leverage_status['current_leverage']:.1f}x")
        print(f"  Base Leverage:    {leverage_status['base_leverage']:.1f}x")
        print(f"  Valid Range:      {leverage_status['range']}")
        print()
        print("  Market Conditions:")
        for k, v in leverage_status['conditions'].items():
            print(f"    {k}: {v}")
        print()

        # Leverage changes history
        if llm_backtest.leverage_changes:
            print("LEVERAGE CHANGES (Dynamic Adjustments)")
            print("-" * 70)
            for change in llm_backtest.leverage_changes:
                print(f"  Trade #{change['trade_count']}: -> {change['new_leverage']:.1f}x ({change['reason']})")
        else:
            print("  No leverage changes (stable conditions)")

        print()
        print("=" * 70)

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
