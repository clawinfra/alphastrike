"""Train command - Train ML models."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

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
    help="Exchange to train models for",
)
@click.option(
    "--assets",
    type=str,
    default=None,
    help="Comma-separated assets to train (default: all)",
)
@click.option(
    "--days",
    type=int,
    default=365,
    help="Training data period in days (default: 365)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="models",
    help="Model output directory (default: models/)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing models",
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
    help="Output training metrics as JSON",
)
def train(
    exchange: str,
    assets: str | None,
    days: int,
    output_dir: str,
    force: bool,
    log_level: str,
    json_output: bool,
) -> None:
    """Train ML models for the specified exchange."""
    setup_logging(log_level, silence_third_party=True)

    # Parse assets
    asset_list = [a.strip().upper() for a in assets.split(",")] if assets else DEFAULT_ASSETS

    # Check existing models
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = []
        for asset in asset_list:
            model_path = output_path / f"lightgbm_{exchange}_{asset.lower()}.txt"
            if model_path.exists():
                existing.append(asset)
        if existing:
            console.print(f"[yellow]Models already exist for: {', '.join(existing)}[/yellow]")
            console.print("[yellow]Use --force to overwrite[/yellow]")
            if not click.confirm("Continue with remaining assets?"):
                sys.exit(0)
            asset_list = [a for a in asset_list if a not in existing]

    if not asset_list:
        print_success("No models to train")
        return

    if not json_output:
        print_banner(
            f"ALPHASTRIKE TRAINING - {exchange.upper()}",
            items={
                "Assets": f"{len(asset_list)} models",
                "Data Period": f"{days} days",
                "Output": str(output_path),
            },
        )

    if exchange == "hyperliquid":
        results = asyncio.run(_train_hyperliquid_models(
            assets=asset_list,
            days=days,
            output_dir=output_path,
            json_output=json_output,
        ))
    elif exchange == "weex":
        print_error("WEEX training not yet implemented in CLI.")
        sys.exit(1)

    if json_output:
        console.print_json(json.dumps(results, indent=2))
    else:
        print_success(f"Trained {len(results['models'])} models")


async def _train_hyperliquid_models(
    assets: list[str],
    days: int,
    output_dir: Path,
    json_output: bool,
) -> dict:
    """Train LightGBM models for Hyperliquid assets."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient
    from src.features.pipeline import FeaturePipeline
    from src.ml.lightgbm_model import LightGBMModel

    logger = logging.getLogger(__name__)

    client = HyperliquidRESTClient(testnet=False)
    await client.initialize()

    results = {"models": [], "errors": []}
    pipeline = FeaturePipeline()

    try:
        for asset in assets:
            symbol = f"{asset}USDT"
            logger.info(f"Training model for {symbol}...")

            try:
                # Fetch candles
                candles = await client.get_candles(
                    symbol=symbol,
                    interval="1h",
                    limit=days * 24,
                )

                if not candles or len(candles) < 500:
                    logger.warning(f"Insufficient data for {symbol}: {len(candles) if candles else 0} candles")
                    results["errors"].append({"asset": asset, "error": "insufficient data"})
                    continue

                # Calculate features for all candles
                X_list = []
                y_list = []

                for i in range(200, len(candles) - 1):
                    candle_slice = candles[i - 200:i + 1]
                    features = pipeline.calculate_features(candle_slice)

                    if features is not None:
                        # Target: price direction in next candle
                        current_close = candles[i].close
                        next_close = candles[i + 1].close
                        target = 1 if next_close > current_close else 0

                        X_list.append(features)
                        y_list.append(target)

                if len(X_list) < 100:
                    logger.warning(f"Insufficient features for {symbol}")
                    results["errors"].append({"asset": asset, "error": "insufficient features"})
                    continue

                X = np.array(X_list)
                y = np.array(y_list)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, shuffle=False
                )

                # Train model
                model = LightGBMModel()
                model.train(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = np.mean((y_pred > 0.5) == y_test)

                # Save model
                model_path = output_dir / f"lightgbm_hyperliquid_{asset.lower()}.txt"
                model.save(model_path)

                results["models"].append({
                    "asset": asset,
                    "samples": len(X),
                    "accuracy": float(accuracy),
                    "path": str(model_path),
                })

                if not json_output:
                    status = "[green]✓[/green]" if accuracy > 0.52 else "[yellow]~[/yellow]"
                    console.print(f"  {symbol}: {status} Accuracy={accuracy:.2%}, saved to {model_path}")

            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
                results["errors"].append({"asset": asset, "error": str(e)})

        return results

    finally:
        await client.close()
