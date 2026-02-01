#!/usr/bin/env python3
"""
Adaptive Optimization Script

Runs full Bayesian optimization with walk-forward validation to find
optimal parameters for each symbol. This replaces trial-and-error tuning
with systematic, sample-efficient optimization.

Usage:
    # Optimize all symbols
    python scripts/adaptive_optimize.py --optimize-all

    # Optimize single symbol
    python scripts/adaptive_optimize.py --symbol BTCUSDT

    # Run with learned parameters
    python scripts/adaptive_optimize.py --backtest BTCUSDT

    # Show current adaptive state
    python scripts/adaptive_optimize.py --status
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import Candle
from src.data.binance_data import fetch_binance_with_cache
from src.features.pipeline import FeaturePipeline
from src.ml.ensemble import MLEnsemble
from src.strategy.conviction_scorer import ConvictionScorer, TimeframeSignals, MarketContext
from src.strategy.mtf_engine import MultiTimeframeEngine
from src.strategy.regime_detector import RegimeDetector
from src.core.config import MarketRegime
from src.adaptive import (
    AdaptiveManager,
    AdaptiveAssetConfig,
    load_asset_config,
    save_asset_config,
    RegimeAwareParams,
    create_symbol_specific_adjustments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Runs backtests with configurable parameters.

    Used by the optimizer to evaluate different parameter combinations.
    """

    def __init__(
        self,
        candles_cache: dict[str, list[Candle]],
        initial_balance: float = 10000.0,
        leverage: int = 10,
    ):
        self.candles_cache = candles_cache
        self.initial_balance = initial_balance
        self.leverage = leverage

        # Initialize shared components
        self.mtf_engine = MultiTimeframeEngine()
        self.conviction_scorer = ConvictionScorer()
        self.feature_pipeline = FeaturePipeline()
        self.regime_detector = RegimeDetector()

        # Try to load ML ensemble
        self.ensemble = None
        try:
            self.ensemble = MLEnsemble(models_dir=Path("models"))
            self.ensemble.check_and_reload_models()
        except Exception:
            pass

    def run_backtest(
        self,
        symbol: str,
        params: dict,
        start_idx: int = 0,
        end_idx: int = -1,
    ) -> dict:
        """
        Run a single backtest with given parameters.

        Args:
            symbol: Trading pair
            params: Parameter dict (conviction_threshold, stop_atr_multiplier, etc.)
            start_idx: Start index in candles (for walk-forward)
            end_idx: End index (-1 for all)

        Returns:
            Dict with sharpe, return, n_trades, win_rate, max_drawdown
        """
        candles = self.candles_cache.get(symbol, [])
        if not candles:
            return {"sharpe": -10, "return": 0, "n_trades": 0}

        if end_idx == -1:
            end_idx = len(candles)

        candles = candles[start_idx:end_idx]
        if len(candles) < 500:
            return {"sharpe": -10, "return": 0, "n_trades": 0}

        # Extract parameters
        conviction_threshold = params.get("conviction_threshold", 70)
        stop_atr_mult = params.get("stop_atr_multiplier", 2.0)
        tp_atr_mult = params.get("take_profit_atr_multiplier", 2.5)
        size_mult = params.get("position_size_multiplier", 1.0)
        short_enabled = params.get("short_enabled", True)
        require_daily = params.get("require_daily_trend_for_short", False)

        # Run simplified backtest
        balance = self.initial_balance
        equity_curve = [balance]
        trades = []
        position = None
        min_window = 400

        for i in range(min_window, len(candles), 4):  # 4H signals
            window = candles[:i + 1]
            current = candles[i]
            price = current.close

            # Check/update position
            if position:
                # Update high/low
                if price > position["highest"]:
                    position["highest"] = price
                if price < position["lowest"]:
                    position["lowest"] = price

                # Check stop loss
                hit_stop = False
                if position["direction"] == "LONG" and price <= position["stop"]:
                    hit_stop = True
                elif position["direction"] == "SHORT" and price >= position["stop"]:
                    hit_stop = True

                # Check trailing stop (after +1.5R)
                stop_dist = abs(position["entry"] - position["stop"])
                if position["direction"] == "LONG":
                    peak_r = (position["highest"] - position["entry"]) / stop_dist if stop_dist > 0 else 0
                    current_r = (price - position["entry"]) / stop_dist if stop_dist > 0 else 0
                else:
                    peak_r = (position["entry"] - position["lowest"]) / stop_dist if stop_dist > 0 else 0
                    current_r = (position["entry"] - price) / stop_dist if stop_dist > 0 else 0

                # Trail after +1.5R peak
                if peak_r >= 1.5:
                    if peak_r >= 3.0:
                        lock_r = 2.0
                    elif peak_r >= 2.0:
                        lock_r = 1.2
                    else:
                        lock_r = 0.8

                    if position["direction"] == "LONG":
                        new_trail = position["entry"] + lock_r * stop_dist
                        if new_trail > position["trail"]:
                            position["trail"] = new_trail
                        if price <= position["trail"]:
                            hit_stop = True
                    else:
                        new_trail = position["entry"] - lock_r * stop_dist
                        if new_trail < position["trail"]:
                            position["trail"] = new_trail
                        if price >= position["trail"]:
                            hit_stop = True

                if hit_stop:
                    # Close position
                    if position["direction"] == "LONG":
                        pnl = (price - position["entry"]) / position["entry"] * position["size"] * self.leverage
                    else:
                        pnl = (position["entry"] - price) / position["entry"] * position["size"] * self.leverage
                    balance += pnl
                    trades.append({
                        "pnl": pnl,
                        "r": current_r,
                        "direction": position["direction"],
                    })
                    position = None

                equity_curve.append(balance + (0 if not position else 0))
                continue

            # Generate signal
            self.mtf_engine.update_candles(window)
            mtf = self.mtf_engine.analyze()

            if not mtf.aligned or mtf.direction == "HOLD":
                equity_curve.append(balance)
                continue

            # SHORT filter
            if mtf.direction == "SHORT":
                if not short_enabled:
                    equity_curve.append(balance)
                    continue
                if require_daily and mtf.daily.direction == "NEUTRAL":
                    equity_curve.append(balance)
                    continue

            # Calculate features
            try:
                features = self.feature_pipeline.calculate_features(window)
            except Exception:
                equity_curve.append(balance)
                continue

            # Get conviction score
            timeframe_signals = TimeframeSignals(
                daily_trend=mtf.daily.direction,
                daily_adx=mtf.daily.strength,
                four_hour_signal=mtf.direction,
                four_hour_confidence=0.65,
                one_hour_momentum=mtf.one_hour.momentum,
                mtf_aligned=mtf.aligned,
            )

            market_context = MarketContext(
                regime=self._classify_regime(features),
                regime_confidence=0.8,
                volume_ratio=features.get("volume_ratio", 1.0),
                atr_ratio=features.get("atr_ratio", 1.0),
                rsi=features.get("rsi", 50.0),
                price_vs_ema50=features.get("ema_long_ratio", 0.0) * 100,
                price_vs_ema200=0.0,
                bb_position=features.get("bb_position", 0.0),
                model_agreement_pct=0.75,
            )

            conviction = self.conviction_scorer.calculate(timeframe_signals, market_context)

            if conviction.score < conviction_threshold:
                equity_curve.append(balance)
                continue

            if conviction.signal not in ("LONG", "SHORT"):
                equity_curve.append(balance)
                continue

            # Open position
            atr = features.get("atr", price * 0.02)
            stop_distance = atr * stop_atr_mult
            position_value = balance * 0.05 * size_mult

            if conviction.signal == "LONG":
                stop = price - stop_distance
                tp = price + stop_distance * tp_atr_mult
            else:
                stop = price + stop_distance
                tp = price - stop_distance * tp_atr_mult

            position = {
                "entry": price,
                "direction": conviction.signal,
                "size": position_value,
                "stop": stop,
                "tp": tp,
                "trail": stop,
                "highest": price,
                "lowest": price,
            }

            equity_curve.append(balance)

        # Close any remaining position
        if position and candles:
            price = candles[-1].close
            stop_dist = abs(position["entry"] - position["stop"])
            if position["direction"] == "LONG":
                current_r = (price - position["entry"]) / stop_dist if stop_dist > 0 else 0
                pnl = (price - position["entry"]) / position["entry"] * position["size"] * self.leverage
            else:
                current_r = (position["entry"] - price) / stop_dist if stop_dist > 0 else 0
                pnl = (position["entry"] - price) / position["entry"] * position["size"] * self.leverage
            balance += pnl
            trades.append({"pnl": pnl, "r": current_r, "direction": position["direction"]})
            equity_curve.append(balance)

        # Calculate metrics
        n_trades = len(trades)
        if n_trades == 0:
            return {"sharpe": -5, "return": 0, "n_trades": 0, "win_rate": 0, "max_drawdown": 0}

        total_return = (balance - self.initial_balance) / self.initial_balance
        win_rate = sum(1 for t in trades if t["pnl"] > 0) / n_trades

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return {
            "sharpe": float(sharpe),
            "return": float(total_return),
            "n_trades": n_trades,
            "win_rate": float(win_rate),
            "max_drawdown": float(max_dd),
        }

    def _classify_regime(self, features: dict) -> str:
        adx = features.get("adx", 0)
        atr_ratio = features.get("atr_ratio", 1.0)
        if atr_ratio > 2.0:
            return "EXTREME_VOLATILITY"
        elif atr_ratio > 1.5:
            return "HIGH_VOLATILITY"
        elif adx > 25:
            return "TRENDING_UP" if features.get("plus_di", 0) > features.get("minus_di", 0) else "TRENDING_DOWN"
        return "RANGING"


async def fetch_all_candles(
    symbols: list[str],
    days: int = 180,
) -> dict[str, list[Candle]]:
    """Fetch candles for all symbols."""
    logger.info(f"Fetching {days} days of data for {len(symbols)} symbols...")

    timeout = aiohttp.ClientTimeout(total=300)
    candles_cache = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for symbol in symbols:
            raw = await fetch_binance_with_cache(session, symbol, "1h", days=days, use_futures=True)

            candles = [
                Candle(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(c["timestamp"] / 1000, tz=timezone.utc),
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c["volume"],
                    interval="1h",
                )
                for c in raw
            ]
            candles_cache[symbol] = candles
            logger.info(f"  {symbol}: {len(candles)} candles")

    return candles_cache


async def run_optimization(
    symbols: list[str],
    days: int = 180,
    n_trials: int = 40,
) -> dict[str, dict]:
    """
    Run Bayesian optimization for all symbols.

    Returns dict of symbol -> optimization results.
    """
    # Check for optuna
    try:
        from src.adaptive.parameter_optimizer import ParameterOptimizer, OPTUNA_AVAILABLE
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not installed. Run: pip install optuna")
            return {}
    except ImportError:
        logger.error("Could not import optimizer. Run: pip install optuna")
        return {}

    # Fetch data
    candles_cache = await fetch_all_candles(symbols, days)

    # Create backtest runner
    runner = BacktestRunner(candles_cache)

    def backtest_func(symbol: str, params: dict) -> dict:
        """Backtest function for optimizer."""
        walk_forward = params.pop("_walk_forward", False)

        if walk_forward:
            # Split into IS and OOS
            candles = candles_cache.get(symbol, [])
            split_idx = int(len(candles) * 0.7)

            is_result = runner.run_backtest(symbol, params, 0, split_idx)
            oos_result = runner.run_backtest(symbol, params, split_idx, -1)

            return {
                "sharpe": is_result["sharpe"],
                "return": is_result["return"],
                "n_trades": is_result["n_trades"],
                "in_sample_sharpe": is_result["sharpe"],
                "in_sample_return": is_result["return"],
                "out_of_sample_sharpe": oos_result["sharpe"],
                "out_of_sample_return": oos_result["return"],
            }
        else:
            return runner.run_backtest(symbol, params)

    # Create optimizer
    optimizer = ParameterOptimizer(
        backtest_func=backtest_func,
        n_trials=n_trials,
        timeout_seconds=300,
        min_sharpe_threshold=0.3,
        min_out_of_sample_ratio=0.6,
    )

    results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Optimizing {symbol}...")
        logger.info(f"{'='*50}")

        # Load current config
        config = load_asset_config(symbol)

        # Run optimization
        result = optimizer.optimize(
            symbol=symbol,
            trigger_reason="manual_optimization",
            current_config=config,
            warm_start=True,
        )

        results[symbol] = {
            "success": result.success,
            "best_params": result.best_params,
            "sharpe": result.best_value,
            "oos_sharpe": result.out_of_sample_sharpe,
            "n_trials": result.n_trials,
        }

        if result.success:
            # Apply and save
            optimizer.apply_result(result, config)
            save_asset_config(config)

            # Save to learned params
            params_path = Path(f"data/learned_params/{symbol.lower()}_params.json")
            params_path.parent.mkdir(parents=True, exist_ok=True)
            with open(params_path, "w") as f:
                json.dump({
                    "config": result.best_params,
                    "symbol": symbol,
                    "optimized_at": datetime.now(timezone.utc).isoformat(),
                    "sharpe": result.best_value,
                    "oos_sharpe": result.out_of_sample_sharpe,
                }, f, indent=2)

            logger.info(f"Saved optimized params for {symbol}")
        else:
            logger.warning(f"Optimization failed for {symbol}: {result.message}")

    return results


def print_status():
    """Print current adaptive system status."""
    print()
    print("=" * 70)
    print("           ADAPTIVE SYSTEM STATUS")
    print("=" * 70)
    print()

    # Load all configs
    config_dir = Path("configs/assets")
    learned_dir = Path("data/learned_params")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    for symbol in symbols:
        config = load_asset_config(symbol)
        learned_path = learned_dir / f"{symbol.lower()}_params.json"

        print(f"{symbol}:")
        print("-" * 40)
        print(f"  Conviction Threshold: {config.conviction_threshold:.0f}")
        print(f"  Stop ATR Multiplier:  {config.stop_atr_multiplier:.2f}x")
        print(f"  Take Profit ATR:      {config.take_profit_atr_multiplier:.2f}x")
        print(f"  Position Size Mult:   {config.position_size_multiplier:.2f}x")
        print(f"  Short Enabled:        {config.short_enabled}")
        print(f"  Require Daily Trend:  {config.require_daily_trend_for_short}")

        if learned_path.exists():
            with open(learned_path) as f:
                learned = json.load(f)
            print(f"  Last Optimized:       {learned.get('optimized_at', 'N/A')[:10]}")
            print(f"  Optimized Sharpe:     {learned.get('sharpe', 'N/A'):.2f}")
            print(f"  OOS Sharpe:           {learned.get('oos_sharpe', 'N/A'):.2f}")
        else:
            print(f"  Last Optimized:       Never")

        print()

    print("=" * 70)


async def run_backtest_with_learned(
    symbol: str,
    days: int = 90,
):
    """Run backtest with learned parameters."""
    candles_cache = await fetch_all_candles([symbol], days)

    runner = BacktestRunner(candles_cache)

    # Load learned params
    params_path = Path(f"data/learned_params/{symbol.lower()}_params.json")
    if params_path.exists():
        with open(params_path) as f:
            learned = json.load(f)
        params = learned.get("config", {})
        print(f"Using learned params: {params}")
    else:
        params = {}
        print("No learned params found, using defaults")

    result = runner.run_backtest(symbol, params)

    print()
    print("=" * 70)
    print(f"         BACKTEST RESULTS: {symbol}")
    print("=" * 70)
    print()
    print(f"  Sharpe Ratio:   {result['sharpe']:.2f}")
    print(f"  Total Return:   {result['return']:.2%}")
    print(f"  Win Rate:       {result['win_rate']:.1%}")
    print(f"  Total Trades:   {result['n_trades']}")
    print(f"  Max Drawdown:   {result['max_drawdown']:.2%}")
    print()
    print("=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Adaptive Parameter Optimization")
    parser.add_argument("--optimize-all", action="store_true", help="Optimize all symbols")
    parser.add_argument("--symbol", type=str, help="Optimize single symbol")
    parser.add_argument("--backtest", type=str, help="Run backtest with learned params")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--days", type=int, default=180, help="Days of historical data")
    parser.add_argument("--trials", type=int, default=40, help="Number of optimization trials")
    args = parser.parse_args()

    if args.status:
        print_status()
        return 0

    if args.backtest:
        await run_backtest_with_learned(args.backtest, args.days)
        return 0

    if args.optimize_all:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        results = await run_optimization(symbols, args.days, args.trials)

        print()
        print("=" * 70)
        print("         OPTIMIZATION SUMMARY")
        print("=" * 70)
        print()
        for symbol, res in results.items():
            status = "✓" if res["success"] else "✗"
            print(f"  {status} {symbol}: Sharpe={res['sharpe']:.2f}, OOS={res['oos_sharpe']:.2f}")
        print()
        print("=" * 70)
        return 0

    if args.symbol:
        results = await run_optimization([args.symbol], args.days, args.trials)
        return 0

    # Default: show status
    print_status()
    print()
    print("Usage:")
    print("  --optimize-all    Optimize all symbols with Bayesian search")
    print("  --symbol BTCUSDT  Optimize single symbol")
    print("  --backtest BTCUSDT Run backtest with learned params")
    print("  --status          Show current adaptive state")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
