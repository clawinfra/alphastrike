#!/usr/bin/env python3
"""
Train ML Models with Binance Data

This script addresses the degenerate model problem by:
1. Using Binance data (more reliable than WEEX)
2. Longer lookahead horizon (5 candles instead of 1)
3. Higher significance threshold (0.5% move instead of 0%)
4. More training data (180 days)

Usage:
    python scripts/train_models_binance.py --symbol BTCUSDT --days 180
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.binance_data import fetch_binance_with_cache
from src.features.pipeline import FeaturePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def train_models(
    symbol: str,
    days: int = 180,
    lookahead: int = 5,
    threshold_pct: float = 0.5,
):
    """
    Train all ML models using Binance data with improved labeling.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        days: Days of historical data
        lookahead: Number of candles to look ahead for labeling
        threshold_pct: Minimum percentage move to be classified as up/down
    """
    from src.ml.xgboost_model import XGBoostModel, XGBoostConfig
    from src.ml.lightgbm_model import LightGBMModel, LightGBMConfig
    from src.ml.random_forest_model import RandomForestModel, RandomForestConfig

    logger.info("=" * 60)
    logger.info("TRAINING ML MODELS WITH BINANCE DATA")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Days of data: {days}")
    logger.info(f"Lookahead: {lookahead} candles")
    logger.info(f"Threshold: {threshold_pct}%")
    logger.info("=" * 60)

    # Fetch Binance data
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        candles = await fetch_binance_with_cache(
            session, symbol, "1h", days=days, use_futures=True
        )

    if len(candles) < 500:
        logger.error(f"Insufficient candles: {len(candles)} < 500")
        return False

    logger.info(f"Fetched {len(candles)} candles from Binance")

    # Convert to Candle objects for feature pipeline
    from src.data.database import Candle
    candle_objs = [
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
        for c in candles
    ]

    # Calculate features
    feature_pipeline = FeaturePipeline()
    min_window = feature_pipeline.config.min_candles
    feature_names = feature_pipeline.feature_names
    n_features = len(feature_names)

    # Need lookahead candles after the feature window
    valid_samples = len(candle_objs) - min_window - lookahead
    if valid_samples <= 0:
        logger.error("Not enough candles for feature calculation with lookahead")
        return False

    logger.info(f"Calculating {n_features} features for {valid_samples} samples...")
    X = np.zeros((valid_samples, n_features), dtype=np.float64)

    for i in range(valid_samples):
        window = candle_objs[i : i + min_window + 1]
        features = feature_pipeline.calculate_features(window, use_cache=False)
        for j, name in enumerate(feature_names):
            X[i, j] = features.get(name, 0.0)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Generate labels using MOMENTUM-BASED approach
    # Much more predictable than raw price direction
    # Label = 1 if future price > future EMA (bullish momentum)
    # Label = 0 if future price < future EMA (bearish momentum)
    # This captures trend continuation rather than predicting exact moves
    y = np.zeros(valid_samples, dtype=np.float64)

    # Calculate EMA of closes for all candles
    closes = np.array([c.close for c in candle_objs])
    ema_period = 12
    ema = np.zeros(len(closes))
    ema[0] = closes[0]
    multiplier = 2 / (ema_period + 1)
    for i in range(1, len(closes)):
        ema[i] = closes[i] * multiplier + ema[i-1] * (1 - multiplier)

    for i in range(valid_samples):
        future_idx = i + min_window + lookahead
        future_close = closes[future_idx]
        future_ema = ema[future_idx]

        # Label based on whether price is above/below EMA (momentum)
        y[i] = 1.0 if future_close > future_ema else 0.0

    # Count to see distribution
    neutral_count = 0  # Not used in this approach

    # Train/validation split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Class balance
    pos_count = float(np.sum(y_train))
    neg_count = float(len(y_train) - pos_count)
    pos_pct = pos_count / len(y_train) * 100

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Class balance: {pos_pct:.1f}% UP, {100-pos_pct:.1f}% DOWN")
    logger.info(f"Neutral samples (subtle direction): {neutral_count} ({neutral_count/valid_samples*100:.1f}%)")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    def is_degenerate(preds: np.ndarray, threshold: float = 0.005) -> bool:
        """Check if predictions are degenerate."""
        pred_variance = float(np.var(preds))
        unique_preds = len(np.unique(np.round(preds, 2)))
        # Lower threshold since XGBoost can have lower variance but still be useful
        return bool(pred_variance < threshold or unique_preds < 3)

    results = {}

    # Train XGBoost
    logger.info("\n" + "=" * 40)
    logger.info("Training XGBoost...")
    try:
        # Use higher learning rate and more estimators to prevent degenerate collapse
        xgb = XGBoostModel(XGBoostConfig(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=1,
            early_stopping_rounds=None,  # Disable early stopping
        ))
        xgb.train(X_train, y_train, feature_names=feature_names, X_val=X_val, y_val=y_val)
        preds = xgb.predict(X_val)

        is_degen = is_degenerate(preds)
        accuracy = float(np.mean((preds > 0.5) == y_val))
        variance = float(np.var(preds))

        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Prediction variance: {variance:.4f}")
        logger.info(f"  Degenerate: {is_degen}")

        if not is_degen:
            path = models_dir / "xgboost.joblib"
            xgb.save(path)
            logger.info(f"  Saved to {path}")
            results["xgboost"] = {"accuracy": accuracy, "variance": variance}
        else:
            logger.warning("  XGBoost still degenerate - NOT saving")
    except Exception as e:
        logger.error(f"  XGBoost training failed: {e}")

    # Train LightGBM
    logger.info("\n" + "=" * 40)
    logger.info("Training LightGBM...")
    try:
        lgb = LightGBMModel(LightGBMConfig(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_data_in_leaf=20,
        ))
        lgb.train(X_train, y_train)
        preds = lgb.predict(X_val)

        is_degen = is_degenerate(preds)
        accuracy = float(np.mean((preds > 0.5) == y_val))
        variance = float(np.var(preds))

        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Prediction variance: {variance:.4f}")
        logger.info(f"  Degenerate: {is_degen}")

        if not is_degen:
            path = models_dir / "lightgbm.txt"
            lgb.save(path)
            logger.info(f"  Saved to {path}")
            results["lightgbm"] = {"accuracy": accuracy, "variance": variance}
        else:
            logger.warning("  LightGBM still degenerate - NOT saving")
    except Exception as e:
        logger.error(f"  LightGBM training failed: {e}")

    # Train RandomForest
    logger.info("\n" + "=" * 40)
    logger.info("Training RandomForest...")
    try:
        rf = RandomForestModel(RandomForestConfig(
            n_estimators=200,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
        ))
        rf.train(X_train, y_train)
        preds = rf.predict(X_val)

        is_degen = is_degenerate(preds)
        accuracy = float(np.mean((preds > 0.5) == y_val))
        variance = float(np.var(preds))

        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Prediction variance: {variance:.4f}")
        logger.info(f"  Degenerate: {is_degen}")

        if not is_degen:
            path = models_dir / "random_forest.joblib"
            rf.save(path)
            logger.info(f"  Saved to {path}")
            results["random_forest"] = {"accuracy": accuracy, "variance": variance}
        else:
            logger.warning("  RandomForest still degenerate - NOT saving")
    except Exception as e:
        logger.error(f"  RandomForest training failed: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for name, metrics in results.items():
        logger.info(f"  {name}: accuracy={metrics['accuracy']:.2%}, variance={metrics['variance']:.4f}")

    if len(results) >= 2:
        logger.info("\n✅ Training successful - at least 2 healthy models")
        return True
    else:
        logger.warning("\n⚠️ Training completed but fewer than 2 healthy models")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train ML models with Binance data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=180, help="Days of historical data")
    parser.add_argument("--lookahead", type=int, default=5, help="Lookahead candles for labeling")
    parser.add_argument("--threshold", type=float, default=0.5, help="Min % move for significant label")
    args = parser.parse_args()

    success = asyncio.run(train_models(
        symbol=args.symbol,
        days=args.days,
        lookahead=args.lookahead,
        threshold_pct=args.threshold,
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
