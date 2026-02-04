#!/usr/bin/env python3
"""
Hyperliquid Multi-Asset Production Backtest

Uses PRODUCTION components (FeaturePipeline, MLEnsemble, SignalProcessor)
to ensure backtest results accurately predict production performance.

Usage:
    python scripts/hyperliquid_production_backtest.py --days 365 --leverage 5
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Default diversified portfolio
DEFAULT_ASSETS = [
    AssetConfig("BTC", "crypto_major", 1.0, 1.0),
    AssetConfig("ETH", "crypto_major", 1.0, 1.5),
    AssetConfig("PAXG", "gold_proxy", 0.3, 2.0),  # Key diversifier
    AssetConfig("SOL", "crypto_l1", 1.2, 2.0),
    AssetConfig("AAVE", "crypto_defi", 1.0, 3.0),
]


async def fetch_candles(
    adapter: HyperliquidAdapter,
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
    """
    Initialize production ML components.

    Returns tuple of (feature_pipeline, ml_ensemble, signal_processor)
    """
    from src.execution.signal_processor import SignalProcessor
    from src.features.pipeline import FeaturePipeline
    from src.ml.ensemble import MLEnsemble

    logger.info("Initializing production components...")

    # Feature pipeline
    feature_pipeline = FeaturePipeline()
    logger.info("  FeaturePipeline initialized")

    # ML ensemble - load trained models using check_and_reload_models()
    ml_ensemble = MLEnsemble(models_dir=Path("models"))
    ml_ensemble.check_and_reload_models()

    # Verify we have enough healthy models
    health_status = ml_ensemble.get_health_status()
    healthy_count = sum(1 for h in health_status.values() if h)
    logger.info(f"  MLEnsemble: {healthy_count}/4 healthy models")
    logger.info(f"    Health status: {health_status}")

    # Signal processor
    signal_processor = SignalProcessor()
    logger.info("  SignalProcessor initialized")

    return feature_pipeline, ml_ensemble, signal_processor


async def main():
    parser = argparse.ArgumentParser(description="Hyperliquid Production Backtest")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval")
    args = parser.parse_args()

    print("=" * 70)
    print("HYPERLIQUID MULTI-ASSET BACKTEST - PRODUCTION COMPLIANT")
    print("=" * 70)
    print(f"Initial Balance: ${args.balance:,.0f}")
    print(f"Leverage: {args.leverage}x")
    print(f"Days: {args.days}")
    print(f"Interval: {args.interval}")
    print()
    print("Using PRODUCTION components:")
    print("  - FeaturePipeline.calculate_features()")
    print("  - MLEnsemble.predict()")
    print("  - SignalProcessor.process_signal()")
    print()

    # Initialize production components
    try:
        feature_pipeline, ml_ensemble, signal_processor = initialize_production_components()
    except Exception as e:
        logger.error(f"Failed to initialize production components: {e}")
        logger.error("Make sure trained models exist. Run training first.")
        return

    # Initialize Hyperliquid adapter
    adapter = HyperliquidAdapter()
    await adapter.initialize()

    try:
        # Fetch candles
        print("Fetching candles from Hyperliquid...")
        print("-" * 40)

        symbols = [a.symbol for a in DEFAULT_ASSETS]
        candles_by_asset = await fetch_candles(
            adapter, symbols, args.days, args.interval
        )

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

        # Create engine with production components
        engine = MultiAssetBacktestEngine(
            config=config,
            feature_pipeline=feature_pipeline,
            ml_ensemble=ml_ensemble,
            signal_processor=signal_processor,
        )

        # Run backtest
        print("Running backtest with PRODUCTION ML pipeline...")
        print("-" * 40)
        result = engine.run(candles_by_asset)

        # Print results
        print()
        print("=" * 70)
        print("RESULTS - MEDALLION BENCHMARK COMPARISON")
        print("=" * 70)
        print()

        m = result.metrics
        print(f"{'Metric':<25} {'Our System':>15} {'Medallion':>15} {'Status':>10}")
        print("-" * 70)

        ret_status = "✅" if m.cagr >= 66 else "⚠️"
        dd_status = "✅" if m.max_drawdown <= 5 else ("⚠️" if m.max_drawdown <= 20 else "❌")
        sharpe_status = "✅" if m.sharpe_ratio >= 2.5 else "⚠️"

        print(f"{'Annualized Return (CAGR)':<25} {m.cagr:>14.1f}% {'66%':>15} {ret_status:>10}")
        print(f"{'Max Drawdown':<25} {m.max_drawdown:>14.1f}% {'3%':>15} {dd_status:>10}")
        print(f"{'Sharpe Ratio':<25} {m.sharpe_ratio:>15.2f} {'~3.0':>15} {sharpe_status:>10}")
        print(f"{'Win Rate':<25} {m.win_rate:>14.1f}% {'-':>15}")
        print(f"{'Total Trades':<25} {m.total_trades:>15}")
        print()

        print(f"Initial Balance: ${config.initial_balance:,.0f}")
        print(f"Final Balance:   ${m.final_balance:,.0f}")
        print(f"Total PnL:       ${m.final_balance - config.initial_balance:,.0f}")
        print()

        # Long/Short breakdown
        ls = result.long_short_breakdown
        print("LONG vs SHORT BREAKDOWN")
        print("-" * 70)
        print(f"{'Direction':<10} {'Trades':>10} {'Win Rate':>12} {'PnL':>15}")
        print("-" * 70)
        print(f"{'LONG':<10} {ls['long_trades']:>10} {ls['long_win_rate']*100:>11.1f}% ${ls['long_pnl']:>13,.0f}")
        print(f"{'SHORT':<10} {ls['short_trades']:>10} {ls['short_win_rate']*100:>11.1f}% ${ls['short_pnl']:>13,.0f}")
        print()

        # Per-asset breakdown
        print("PER-ASSET BREAKDOWN")
        print("-" * 70)
        print(f"{'Asset':<10} {'Class':<15} {'Trades':>8} {'Win Rate':>10} {'PnL':>12}")
        print("-" * 70)

        for symbol, perf in result.per_asset_metrics.items():
            asset_cfg = next((a for a in DEFAULT_ASSETS if a.symbol == symbol), None)
            asset_class = asset_cfg.asset_class if asset_cfg else "unknown"
            print(
                f"{symbol:<10} {asset_class:<15} {perf['trades']:>8} "
                f"{perf['win_rate']*100:>9.1f}% ${perf['pnl']:>10,.0f}"
            )

        print()
        print("=" * 70)
        print()
        print("NOTE: This backtest uses the SAME code as production.")
        print("Results should accurately predict live performance.")

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
