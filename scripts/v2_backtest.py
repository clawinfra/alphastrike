#!/usr/bin/env python3
"""
AlphaStrike V2 Backtest — The Honest Version

This backtest uses:
1. Walk-forward validation (no in-sample cheating)
2. Regression targets (predict return magnitude)
3. Stacking ensemble (diverse base learners)
4. Unified regime detection
5. Kelly position sizing
6. Realistic costs (fees on notional + funding + per-asset slippage)
7. No look-ahead bias (features from closed candles only)

Run:
    python scripts/v2_backtest.py --days 365
    python scripts/v2_backtest.py --days 365 --permutation-test

Expected output: LOWER numbers than V1, but HONEST numbers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import logging
from datetime import datetime

import numpy as np

from src.v2.features import V2FeaturePipeline, FEATURE_NAMES
from src.v2.models import ModelConfig, StackingEnsemble
from src.v2.regime import UnifiedRegimeDetector, Regime
from src.v2.risk import V2RiskManager, RiskLimits, ASSET_SLIPPAGE_BPS
from src.v2.walk_forward import WalkForwardValidator, WalkForwardConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


ASSETS = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "AVAX", "NEAR", "APT",
    "AAVE", "UNI", "LINK", "FET", "DOGE", "PAXG", "SPX",
]


async def fetch_candles(assets: list[str], days: int) -> dict:
    """Fetch candles from Hyperliquid API."""
    from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient

    client = HyperliquidRESTClient()
    await client.initialize()

    all_candles = {}
    for asset in assets:
        symbol = f"{asset}USDT"
        try:
            candles = await client.get_candles(
                symbol=symbol, interval="1h", limit=days * 24 + 200,
            )
            if candles and len(candles) > 200:
                all_candles[symbol] = candles
                logger.info(f"  {symbol}: {len(candles)} candles")
        except Exception as e:
            logger.warning(f"  {symbol}: Failed - {e}")

    await client.close()
    return all_candles


def prepare_data(
    all_candles: dict,
    pipeline: V2FeaturePipeline,
    target_horizon: int = 12,
) -> dict:
    """
    Prepare features and labels for all assets.

    Returns dict of {symbol: {X, y, close, timestamps}}
    """
    min_candles = pipeline.config.min_candles
    prepared = {}

    # Get BTC close prices for correlation feature
    btc_closes = None
    if "BTCUSDT" in all_candles:
        btc_closes = np.array([c.close for c in all_candles["BTCUSDT"]])

    for symbol, candles in all_candles.items():
        n = len(candles)
        if n < min_candles + target_horizon + 50:
            continue

        close = np.array([c.close for c in candles], dtype=np.float64)
        high = np.array([c.high for c in candles], dtype=np.float64)
        low = np.array([c.low for c in candles], dtype=np.float64)
        volume = np.array([c.volume for c in candles], dtype=np.float64)

        features_list = []
        returns_list = []
        close_list = []
        timestamps = []

        for i in range(min_candles, n - target_horizon):
            # Features from CLOSED candles only (no look-ahead)
            feat = pipeline.calculate(
                close=close[:i],
                high=high[:i],
                low=low[:i],
                volume=volume[:i],
                btc_close=btc_closes[:i] if btc_closes is not None and len(btc_closes) >= i else None,
                timestamp=candles[i].timestamp,
            )
            X_row = pipeline.to_array(feat).flatten()
            features_list.append(X_row)

            # Forward return (REGRESSION target — not binary!)
            future_price = close[i + target_horizon]
            current_price = close[i]
            forward_return = (future_price - current_price) / current_price
            returns_list.append(forward_return)

            close_list.append(close[i])
            timestamps.append(candles[i].timestamp)

        if len(features_list) < 500:
            logger.warning(f"{symbol}: Only {len(features_list)} samples, skipping")
            continue

        X = np.array(features_list, dtype=np.float64)
        y = np.array(returns_list, dtype=np.float64)

        # Clean
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, -0.5, 0.5)  # clip extreme returns

        prepared[symbol] = {
            "X": X,
            "y": y,
            "close": np.array(close_list),
            "timestamps": timestamps,
        }

        logger.info(f"  {symbol}: {len(y)} samples, "
                     f"mean return={np.mean(y):.5f}, std={np.std(y):.5f}")

    return prepared


def run_walk_forward_per_asset(
    prepared: dict,
    config: WalkForwardConfig,
    run_permutation: bool = False,
) -> dict:
    """Run walk-forward validation for each asset."""
    results = {}
    validator = WalkForwardValidator(config)

    for symbol, data in prepared.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD: {symbol}")
        logger.info(f"{'='*60}")

        result = validator.run(
            X=data["X"],
            y=data["y"],
            close_prices=data["close"],
            feature_names=FEATURE_NAMES,
            run_permutation_test=run_permutation,
        )
        results[symbol] = result

        logger.info(
            f"  {symbol}: {result.n_windows} windows, "
            f"avg Sharpe={result.avg_sharpe:.2f}, "
            f"avg return={result.avg_return:.2%}, "
            f"p-value={result.p_value:.4f}"
        )

    return results


def print_results(results: dict, config: WalkForwardConfig) -> None:
    """Print formatted walk-forward results."""
    print("\n" + "=" * 80)
    print("ALPHASTRIKE V2 — WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nTrain window: {config.train_window // 24} days")
    print(f"Test window: {config.test_window // 24} days")
    print(f"Step size: {config.step_size // 24} days")

    print(f"\n{'Symbol':<12} {'Windows':>8} {'Avg Ret':>10} {'Avg Sharpe':>11} "
          f"{'Avg DD':>8} {'Win Rate':>9} {'Trades':>8} {'p-value':>9} {'Sig?':>5}")
    print("-" * 90)

    total_returns = []
    total_sharpes = []

    for symbol, result in sorted(results.items()):
        sig = "✓" if result.p_value < 0.05 else "✗"
        print(
            f"{symbol:<12} {result.n_windows:>8} {result.avg_return:>+9.2%} "
            f"{result.avg_sharpe:>11.2f} {result.avg_max_dd:>7.1%} "
            f"{result.avg_win_rate:>8.1%} {result.avg_n_trades:>8.0f} "
            f"{result.p_value:>9.4f} {sig:>5}"
        )
        total_returns.append(result.avg_return)
        total_sharpes.append(result.avg_sharpe)

    print("-" * 90)

    if total_returns:
        avg_ret = np.mean(total_returns)
        avg_sharpe = np.mean(total_sharpes)
        sig_count = sum(1 for r in results.values() if r.p_value < 0.05)

        print(f"{'PORTFOLIO':<12} {'':>8} {avg_ret:>+9.2%} {avg_sharpe:>11.2f} "
              f"{'':>8} {'':>9} {'':>8} {'':>9} {sig_count}/{len(results)}")

    # Print window stability
    print(f"\n{'='*80}")
    print("WINDOW STABILITY")
    print("=" * 80)

    for symbol, result in sorted(results.items()):
        if result.n_windows == 0:
            continue
        print(
            f"\n{symbol}: "
            f"best={result.best_window_return:+.2%}, "
            f"worst={result.worst_window_return:+.2%}, "
            f"profitable={result.pct_profitable_windows:.0%} of windows, "
            f"Sharpe σ={result.sharpe_std:.2f}"
        )

        if result.permutation_p_value is not None:
            print(f"  Permutation test: p={result.permutation_p_value:.4f} "
                  f"({result.permutation_n_runs} runs)")

    # Verdict
    print(f"\n{'='*80}")
    viable = [s for s, r in results.items()
              if r.avg_sharpe > 0.5 and r.p_value < 0.10 and r.pct_profitable_windows > 0.5]

    if viable:
        print(f"POTENTIALLY VIABLE ASSETS ({len(viable)}/{len(results)}): {', '.join(viable)}")
        print("These assets show statistically significant alpha in walk-forward testing.")
    else:
        print("NO ASSETS SHOW RELIABLE ALPHA IN WALK-FORWARD TESTING.")
        print("This means the V1 backtest results were likely overfitted.")
        print("This is valuable information — it prevents you from losing real money.")

    print("=" * 80)

    # Comparison with V1 claims
    print(f"\n{'='*80}")
    print("V1 vs V2 COMPARISON")
    print("=" * 80)
    print(f"  V1 claimed:  67.5% CAGR, 3.67 Sharpe, 3.9% max DD")
    if total_sharpes:
        print(f"  V2 reality:  {avg_ret:+.1%} avg return, {avg_sharpe:.2f} Sharpe (walk-forward)")
        print(f"  Gap:         This IS the real number. V1 was in-sample.")
    print("=" * 80 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="AlphaStrike V2 Walk-Forward Backtest")
    parser.add_argument("--days", type=int, default=365, help="Data period in days")
    parser.add_argument("--train-days", type=int, default=180, help="Training window in days")
    parser.add_argument("--test-days", type=int, default=30, help="Test window in days")
    parser.add_argument("--permutation-test", action="store_true", help="Run permutation testing")
    args = parser.parse_args()

    # Fetch data
    logger.info(f"Fetching {args.days} days of data for {len(ASSETS)} assets...")
    all_candles = await fetch_candles(ASSETS, args.days)
    if not all_candles:
        logger.error("No data fetched")
        return

    # Prepare features and labels
    logger.info("Preparing features and regression targets...")
    pipeline = V2FeaturePipeline()
    prepared = prepare_data(all_candles, pipeline, target_horizon=12)

    if not prepared:
        logger.error("No assets with sufficient data")
        return

    # Walk-forward config
    wf_config = WalkForwardConfig(
        train_window=args.train_days * 24,
        test_window=args.test_days * 24,
        step_size=args.test_days * 24,
    )

    # Run walk-forward
    results = run_walk_forward_per_asset(
        prepared, wf_config, run_permutation=args.permutation_test
    )

    # Print results
    print_results(results, wf_config)


if __name__ == "__main__":
    asyncio.run(main())
