"""Backtest command - Run historical backtests."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from src.cli.utils import (
    DEFAULT_ASSETS,
    console,
    print_banner,
    print_error,
    print_success,
    setup_logging,
)


@click.command()
@click.option(
    "--exchange", "-e",
    type=click.Choice(["hyperliquid", "weex"], case_sensitive=False),
    required=True,
    help="Exchange to backtest on",
)
@click.option(
    "--days",
    type=int,
    default=180,
    help="Backtest period in days (default: 180)",
)
@click.option(
    "--initial-balance",
    type=float,
    default=10000.0,
    help="Starting balance (default: 10000)",
)
@click.option(
    "--base-leverage",
    type=float,
    default=5.0,
    help="Base leverage (default: 5.0)",
)
@click.option(
    "--assets",
    type=str,
    default=None,
    help="Comma-separated assets (default: all)",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Save results to file (JSON/CSV)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging verbosity",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output results as JSON",
)
def backtest(
    exchange: str,
    days: int,
    initial_balance: float,
    base_leverage: float,
    assets: str | None,
    output: str | None,
    log_level: str,
    json_output: bool,
) -> None:
    """Run a historical backtest."""
    setup_logging(log_level, silence_third_party=True)

    # Parse assets
    asset_list = [a.strip().upper() for a in assets.split(",")] if assets else None

    if not json_output:
        print_banner(
            f"ALPHASTRIKE BACKTEST - {exchange.upper()}",
            items={
                "Period": f"{days} days",
                "Initial Balance": f"${initial_balance:,.0f}",
                "Leverage": f"{base_leverage}x",
                "Assets": ", ".join(asset_list[:5]) + "..." if asset_list else f"All {len(DEFAULT_ASSETS)} assets",
            },
        )

    if exchange == "hyperliquid":
        try:
            results = asyncio.run(_run_hyperliquid_backtest(
                days=days,
                initial_balance=initial_balance,
                base_leverage=base_leverage,
                assets=asset_list,
                json_output=json_output,
            ))
        except Exception as e:
            print_error(f"Backtest failed: {e}")
            sys.exit(1)
    elif exchange == "weex":
        print_error("WEEX backtest not yet implemented in CLI.")
        sys.exit(1)

    # Output results
    if json_output:
        console.print_json(json.dumps(results, indent=2, default=str))
    elif output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print_success(f"Results saved to {output_path}")


async def _run_hyperliquid_backtest(
    days: int,
    initial_balance: float,
    base_leverage: float,
    assets: list[str] | None,
    json_output: bool,
) -> dict[str, Any]:
    """Run Hyperliquid backtest using medallion_v2 logic."""
    import numpy as np

    from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient
    from src.features.pipeline import FeaturePipeline
    from src.ml.lightgbm_model import LightGBMModel

    logger = logging.getLogger(__name__)

    # Use default assets if none provided
    asset_list = assets or DEFAULT_ASSETS

    # Initialize client
    client = HyperliquidRESTClient(testnet=False)
    await client.initialize()

    try:
        # Load models
        models: dict[str, LightGBMModel] = {}
        models_dir = Path("models")
        for asset in asset_list:
            model_path = models_dir / f"lightgbm_hyperliquid_{asset.lower()}.txt"
            if model_path.exists():
                model = LightGBMModel()
                model.load(model_path)
                models[f"{asset}USDT"] = model
                logger.info(f"Loaded model for {asset}USDT")

        if not models:
            raise ValueError("No models found. Run 'alphastrike train' first.")

        # Fetch candles
        logger.info(f"Fetching candles for {len(asset_list)} assets...")
        all_candles: dict[str, list] = {}
        for asset in asset_list:
            symbol = f"{asset}USDT"
            candles = await client.get_candles(
                symbol=symbol,
                interval="1h",
                limit=days * 24,
            )
            if candles:
                all_candles[symbol] = candles
                logger.info(f"  {symbol}: {len(candles)} candles")

        # Check for empty data
        if not all_candles:
            raise ValueError("No candle data fetched. Check network connection.")

        # Initialize feature pipeline
        pipeline = FeaturePipeline()

        # Run simulation (simplified - full logic is in medallion_v2_backtest.py)
        balance = initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        trades: list[dict[str, Any]] = []
        wins = 0

        # Get minimum candle count across all assets
        min_candles = min(len(c) for c in all_candles.values())

        # Process each candle
        warmup = 150
        for i in range(warmup, min_candles - 1):
            for symbol, candles in all_candles.items():
                if symbol not in models:
                    continue

                # Get features
                candle_slice = candles[max(0, i - 200):i + 1]
                if len(candle_slice) < 150:
                    continue

                features = pipeline.calculate_features(candle_slice)
                if features is None:
                    continue

                # Get prediction
                model = models[symbol]
                # Convert features dict to numpy array for prediction
                feature_names = pipeline.feature_names
                X = np.array([[features.get(name, 0.0) for name in feature_names]])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                prediction = model.predict(X)[0]

                # Simple signal logic (threshold 70+)
                if prediction > 0.70:
                    # Simulate trade
                    position_size = balance * 0.05 * base_leverage
                    entry = candles[i].close
                    exit_price = candles[i + 1].close
                    pnl = position_size * (exit_price - entry) / entry
                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    drawdown = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, drawdown)
                    trades.append({
                        "symbol": symbol,
                        "pnl": pnl,
                        "entry": entry,
                        "exit": exit_price,
                    })
                    if pnl > 0:
                        wins += 1

        # Calculate metrics
        total_return = (balance - initial_balance) / initial_balance
        cagr = ((balance / initial_balance) ** (365 / days) - 1) if days > 0 else 0
        win_rate = wins / len(trades) if trades else 0
        avg_pnl = sum(t["pnl"] for t in trades) / len(trades) if trades else 0

        results: dict[str, Any] = {
            "exchange": "hyperliquid",
            "period_days": days,
            "initial_balance": initial_balance,
            "final_balance": balance,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "total_trades": len(trades),
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
        }

        # Print results
        if not json_output:
            console.print("\n[bold]BACKTEST RESULTS[/bold]")
            console.print(f"  CAGR:         {cagr * 100:>8.1f}%  {'[green]✓[/green]' if cagr > 0.66 else '[red]✗[/red]'}")
            console.print(f"  Max Drawdown: {max_drawdown * 100:>8.1f}%  {'[green]✓[/green]' if max_drawdown < 0.05 else '[red]✗[/red]'}")
            console.print(f"  Win Rate:     {win_rate * 100:>8.1f}%")
            console.print(f"  Total Trades: {len(trades):>8}")
            console.print(f"  Final Balance: ${balance:>10,.2f}")

        return results

    finally:
        await client.close()
