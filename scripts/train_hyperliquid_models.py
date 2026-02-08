#!/usr/bin/env python3
"""
Train ML Models on Hyperliquid Data

Per ARCHITECTURE.md Section 5.1:
1. Load historical candles from database
2. Calculate 86 canonical features using FeaturePipeline
3. Generate labels (LONG >0.5% up, SHORT >0.5% down)
4. Train XGBoost, LightGBM, LSTM, RandomForest
5. Save models to models/ directory

Usage:
    python scripts/train_hyperliquid_models.py
    python scripts/train_hyperliquid_models.py --symbol BTC --min-samples 1000
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Assets to train models for (Medallion portfolio)
TRAINING_ASSETS = [
    # Crypto Major
    "BTC", "ETH", "BNB", "XRP",
    # L1/L2
    "SOL", "AVAX", "NEAR", "APT",
    # DeFi
    "AAVE", "UNI", "LINK",
    # AI
    "FET",
    # Meme
    "DOGE",
    # Traditional
    "PAXG", "SPX",
]

# Label threshold per architecture (0.5% move = signal)
LABEL_THRESHOLD = 0.005


async def load_candles_from_api(symbol: str, min_samples: int = 1000) -> list:
    """
    Load candles directly from Hyperliquid API.

    Args:
        symbol: Asset symbol (e.g., "BTC")
        min_samples: Minimum number of candles required

    Returns:
        List of UnifiedCandle objects (oldest first)
    """
    from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient
    from datetime import timezone, timedelta

    client = HyperliquidRESTClient()
    await client.initialize()

    try:
        # Fetch with extra candles for warmup
        limit = min_samples + 500
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=limit)

        unified_symbol = f"{symbol}USDT"
        candles = await client.get_candles(
            symbol=unified_symbol,
            interval="1h",
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )

        if len(candles) < min_samples:
            logger.warning(f"{symbol}: Only {len(candles)} candles from API (need {min_samples})")
            return []

        logger.info(f"{symbol}: Loaded {len(candles)} candles from API")
        return candles

    finally:
        await client.close()


async def load_candles_from_db(symbol: str, min_samples: int = 1000) -> list:
    """
    Load candles from database.

    Args:
        symbol: Asset symbol (e.g., "BTC")
        min_samples: Minimum number of candles required

    Returns:
        List of Candle objects (oldest first)
    """
    from src.data.database import Database

    db = Database()
    await db.initialize()

    try:
        # Database uses hyperliquid_{symbol} format
        db_symbol = f"hyperliquid_{symbol.lower()}"

        candles = await db.get_candles(
            symbol=db_symbol,
            limit=min_samples + 500,  # Extra for warmup
            interval="1h",
        )

        if len(candles) < min_samples:
            logger.warning(f"{symbol}: Only {len(candles)} candles in DB (need {min_samples})")
            return []

        # Database returns newest first, reverse for chronological order
        candles = list(reversed(candles))
        logger.info(f"{symbol}: Loaded {len(candles)} candles from database")
        return candles

    finally:
        await db.close()


def calculate_features(candles: list, feature_pipeline) -> tuple[np.ndarray, list[str]]:
    """
    Calculate features for all candles.

    Args:
        candles: List of Candle objects
        feature_pipeline: FeaturePipeline instance

    Returns:
        (X, feature_names) - Feature matrix and names
    """
    min_window = feature_pipeline.config.min_candles
    feature_names = feature_pipeline.feature_names
    n_features = len(feature_names)

    valid_samples = len(candles) - min_window
    if valid_samples <= 0:
        return np.array([]), feature_names

    logger.info(f"Calculating {n_features} features for {valid_samples} samples...")

    X = np.zeros((valid_samples, n_features), dtype=np.float64)

    for i in range(valid_samples):
        window = candles[i : i + min_window + 1]
        features = feature_pipeline.calculate_features(window, use_cache=False)

        for j, name in enumerate(feature_names):
            X[i, j] = features.get(name, 0.0)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, feature_names


def generate_labels(candles: list, min_window: int, threshold: float = LABEL_THRESHOLD) -> np.ndarray:
    """
    Generate trading labels based on future price movement.

    Per architecture:
    - LONG (1) if price goes up > threshold
    - SHORT (0) if price goes down > threshold
    - HOLD (0.5) otherwise (may be excluded)

    Args:
        candles: List of Candle objects
        min_window: Feature window size
        threshold: Price change threshold (default 0.5%)

    Returns:
        Label array
    """
    valid_samples = len(candles) - min_window

    y = np.zeros(valid_samples, dtype=np.float64)

    for i in range(valid_samples - 1):
        current_price = candles[i + min_window].close
        future_price = candles[i + min_window + 1].close

        if current_price > 0:
            price_change = (future_price - current_price) / current_price
        else:
            price_change = 0

        if price_change > threshold:
            y[i] = 1.0  # LONG
        elif price_change < -threshold:
            y[i] = 0.0  # SHORT
        else:
            y[i] = 0.5  # HOLD (will be converted to binary)

    return y


def is_degenerate(preds: np.ndarray, threshold: float = 0.01) -> bool:
    """Check if predictions are degenerate (all same class)."""
    pred_variance = float(np.var(preds))
    unique_preds = len(np.unique(np.round(preds, 2)))
    return bool(pred_variance < threshold or unique_preds < 3)


async def train_models_for_symbol(
    symbol: str,
    min_samples: int = 1000,
    models_dir: Path = Path("models"),
) -> dict:
    """
    Train all ML models for a single symbol.

    Args:
        symbol: Asset symbol
        min_samples: Minimum training samples
        models_dir: Directory to save models

    Returns:
        Training report dict
    """
    from src.features.pipeline import FeaturePipeline
    from src.ml.xgboost_model import XGBoostModel, XGBoostConfig
    from src.ml.lightgbm_model import LightGBMModel, LightGBMConfig
    from src.ml.random_forest_model import RandomForestModel, RandomForestConfig

    logger.info("=" * 60)
    logger.info(f"TRAINING MODELS FOR {symbol}")
    logger.info("=" * 60)

    start_time = time.time()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load candles - try API first, fallback to database
    candles = await load_candles_from_api(symbol, min_samples)
    if not candles:
        logger.info(f"{symbol}: API failed, trying database...")
        candles = await load_candles_from_db(symbol, min_samples)

    if not candles:
        return {"symbol": symbol, "success": False, "error": "Insufficient data from API and DB"}

    # Initialize feature pipeline
    feature_pipeline = FeaturePipeline()
    min_window = feature_pipeline.config.min_candles

    # Calculate features
    X, feature_names = calculate_features(candles, feature_pipeline)
    if X.size == 0:
        return {"symbol": symbol, "success": False, "error": "Feature calculation failed"}

    # Generate labels
    y = generate_labels(candles, min_window)

    # Remove last sample (no future price)
    X = X[:-1]
    y = y[:-1]

    # EXCLUDE HOLD samples (y == 0.5) for cleaner binary classification
    # Only train on clear directional signals (LONG=1, SHORT=0)
    clear_signal_mask = (y != 0.5)
    X = X[clear_signal_mask]
    y_binary = y[clear_signal_mask]

    logger.info(f"Excluded {np.sum(~clear_signal_mask)} HOLD samples, kept {len(X)} clear signals")

    # Train/validation split (80/20, time-ordered)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_binary[:split_idx], y_binary[split_idx:]

    logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

    # Class balance
    pos_count = float(np.sum(y_train))
    neg_count = float(len(y_train) - pos_count)
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info(f"Class balance: pos={pos_count:.0f}, neg={neg_count:.0f}")

    models_trained = []
    validation_accuracy = {}
    model_paths = {}

    # Train XGBoost
    try:
        logger.info("Training XGBoost...")
        xgb = XGBoostModel(
            XGBoostConfig(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                reg_lambda=3.0,
                scale_pos_weight=scale_pos_weight,
                min_child_weight=5,
            )
        )
        xgb.train(X_train, y_train, feature_names=feature_names, X_val=X_val, y_val=y_val)
        preds = xgb.predict(X_val)

        if is_degenerate(preds):
            logger.warning("XGBoost degenerate, retraining with stronger regularization...")
            xgb = XGBoostModel(
                XGBoostConfig(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    reg_alpha=2.0,
                    reg_lambda=5.0,
                )
            )
            xgb.train(X_train, y_train, feature_names=feature_names, X_val=X_val, y_val=y_val)
            preds = xgb.predict(X_val)

        acc = float(np.mean((preds > 0.5) == y_val))
        path = models_dir / f"xgboost_hyperliquid_{symbol.lower()}.joblib"
        xgb.save(path)
        models_trained.append("xgboost")
        validation_accuracy["xgboost"] = acc
        model_paths["xgboost"] = str(path)
        logger.info(f"  XGBoost: {acc:.2%} accuracy")
    except Exception as e:
        logger.error(f"  XGBoost failed: {e}")

    # Train LightGBM
    try:
        logger.info("Training LightGBM...")
        lgb = LightGBMModel(
            LightGBMConfig(
                n_estimators=150,
                num_leaves=15,
                learning_rate=0.05,
                max_depth=4,
                reg_alpha=0.5,
                reg_lambda=1.0,
                is_unbalance=True,
            )
        )
        lgb.train(X_train, y_train, feature_names=feature_names)
        preds = lgb.predict(X_val)

        if is_degenerate(preds):
            logger.warning("LightGBM degenerate, retraining...")
            lgb = LightGBMModel(
                LightGBMConfig(
                    n_estimators=50,
                    num_leaves=8,
                    learning_rate=0.1,
                    max_depth=3,
                    reg_alpha=1.0,
                    reg_lambda=2.0,
                )
            )
            lgb.train(X_train, y_train, feature_names=feature_names)
            preds = lgb.predict(X_val)

        acc = float(np.mean((preds > 0.5) == y_val))
        path = models_dir / f"lightgbm_hyperliquid_{symbol.lower()}.txt"
        lgb.save(path)
        models_trained.append("lightgbm")
        validation_accuracy["lightgbm"] = acc
        model_paths["lightgbm"] = str(path)
        logger.info(f"  LightGBM: {acc:.2%} accuracy")
    except Exception as e:
        logger.error(f"  LightGBM failed: {e}")

    # Train Random Forest
    try:
        logger.info("Training Random Forest...")
        rf = RandomForestModel(RandomForestConfig(n_estimators=100, max_depth=10))
        rf.train(X_train, y_train.astype(np.int64))
        preds = rf.predict(X_val)
        acc = float(np.mean((preds > 0.5) == y_val))
        path = models_dir / f"random_forest_hyperliquid_{symbol.lower()}.joblib"
        rf.save(path)
        models_trained.append("random_forest")
        validation_accuracy["random_forest"] = acc
        model_paths["random_forest"] = str(path)
        logger.info(f"  Random Forest: {acc:.2%} accuracy")
    except Exception as e:
        logger.error(f"  Random Forest failed: {e}")

    training_time = time.time() - start_time

    return {
        "symbol": symbol,
        "success": len(models_trained) > 0,
        "models_trained": models_trained,
        "validation_accuracy": validation_accuracy,
        "model_paths": model_paths,
        "training_time": training_time,
        "samples": len(X_train) + len(X_val),
    }


async def main():
    parser = argparse.ArgumentParser(description="Train Hyperliquid ML Models")
    parser.add_argument("--symbol", type=str, help="Single symbol to train (default: all)")
    parser.add_argument("--min-samples", type=int, default=2000, help="Minimum training samples")
    parser.add_argument("--all-medallion", action="store_true", help="Train all Medallion portfolio assets")
    args = parser.parse_args()

    print("=" * 70)
    print("HYPERLIQUID ML MODEL TRAINING")
    print("=" * 70)
    print()
    print("Per ARCHITECTURE.md Section 5.1:")
    print("  - FeaturePipeline: 86 canonical features")
    print("  - Labels: LONG (>0.5% up), SHORT (>0.5% down)")
    print("  - Models: XGBoost, LightGBM, RandomForest")
    print("  - Split: 80% train, 20% validation (time-ordered)")
    print()

    # Determine symbols to train
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = TRAINING_ASSETS

    print(f"Training models for: {', '.join(symbols)}")
    print("-" * 40)

    results = []
    for symbol in symbols:
        result = await train_models_for_symbol(symbol, args.min_samples)
        results.append(result)

    # Print summary
    print()
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Symbol':<10} {'Models':<15} {'XGB Acc':>10} {'LGB Acc':>10} {'RF Acc':>10} {'Time':>8}")
    print("-" * 70)

    for r in results:
        symbol = r["symbol"]
        if r["success"]:
            models = len(r["models_trained"])
            xgb_acc = r["validation_accuracy"].get("xgboost", 0) * 100
            lgb_acc = r["validation_accuracy"].get("lightgbm", 0) * 100
            rf_acc = r["validation_accuracy"].get("random_forest", 0) * 100
            time_s = r["training_time"]
            print(f"{symbol:<10} {models:<15} {xgb_acc:>9.1f}% {lgb_acc:>9.1f}% {rf_acc:>9.1f}% {time_s:>7.1f}s")
        else:
            print(f"{symbol:<10} FAILED: {r.get('error', 'Unknown')}")

    print()

    # Check if models are ready for backtesting
    successful = sum(1 for r in results if r["success"])
    if successful >= 3:
        print(f"✅ {successful}/{len(results)} assets trained successfully")
        print()
        print("Models saved to: models/")
        print()
        print("Next step: Run production-compliant backtest")
        print("  python scripts/hyperliquid_production_backtest.py --days 90")
    else:
        print(f"⚠️  Only {successful}/{len(results)} assets trained")
        print("   Check error messages above")


if __name__ == "__main__":
    asyncio.run(main())
