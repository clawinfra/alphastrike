#!/usr/bin/env python3
"""
Train ML Models and Run Backtest with Ensemble Predictions

This script:
1. Downloads historical data from WEEX API
2. Saves candles to the SQLite database
3. Trains all 4 ML models (XGBoost, LightGBM, LSTM, RandomForest)
4. Runs a backtest using ensemble predictions

Usage:
    python scripts/train_and_backtest.py --symbol BTCUSDT --interval 1h --limit 1000
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# WEEX API
WEEX_BASE_URL = "https://api-contract.weex.com"


async def fetch_weex_candles(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
) -> list[dict]:
    """Fetch candles from WEEX v2 API."""
    weex_symbol = f"cmt_{symbol.lower()}"
    url = f"{WEEX_BASE_URL}/capi/v2/market/candles"
    params = {
        "symbol": weex_symbol,
        "granularity": interval,
        "limit": str(min(limit, 1000)),
    }

    async with session.get(url, params=params) as response:
        if response.status != 200:
            logger.error(f"Failed to fetch candles: HTTP {response.status}")
            return []

        data = await response.json()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        candles = []
        if isinstance(data, list):
            for item in data:
                candles.append({
                    "timestamp": datetime.fromtimestamp(int(item[0]) / 1000),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                })

        # Sort by timestamp ascending
        candles.sort(key=lambda x: x["timestamp"])
        return candles


async def save_candles_to_db(candles: list[dict], symbol: str, interval: str):
    """Save candles to SQLite database."""
    from src.data.database import Candle, Database

    db = Database()
    await db.initialize()

    db_candles = [
        Candle(
            symbol=f"cmt_{symbol.lower()}",
            timestamp=c["timestamp"],
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
            interval=interval,
        )
        for c in candles
    ]

    saved = await db.save_candles(db_candles)
    logger.info(f"Saved {saved} candles to database")

    await db.close()
    return saved


async def train_models(symbol: str, interval: str = "1h", min_samples: int = 500):
    """Train all ML models using historical data."""
    from src.data.database import Database, Candle
    from src.features.pipeline import FeaturePipeline
    from src.ml.xgboost_model import XGBoostModel, XGBoostConfig
    from src.ml.lightgbm_model import LightGBMModel, LightGBMConfig
    from src.ml.lstm_model import LSTMModel, LSTMConfig
    from src.ml.random_forest_model import RandomForestModel, RandomForestConfig
    from src.ml.trainer import TrainingReport, RetrainingTrigger
    import time

    logger.info("=" * 60)
    logger.info("TRAINING ML MODELS")
    logger.info("=" * 60)

    db = Database()
    await db.initialize()

    start_time = time.time()
    timestamp = datetime.utcnow()
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Fetch candles with correct interval
    weex_symbol = f"cmt_{symbol.lower()}"
    candles = await db.get_candles(weex_symbol, limit=min_samples + 200, interval=interval)

    if len(candles) < min_samples:
        await db.close()
        return TrainingReport(
            success=False,
            models_trained=[],
            validation_accuracy={},
            training_time_seconds=time.time() - start_time,
            timestamp=timestamp,
            trigger=RetrainingTrigger.INITIAL,
            error_message=f"Insufficient candles: {len(candles)} < {min_samples}",
        )

    # Reverse to get oldest first
    candles = list(reversed(candles))
    logger.info(f"Fetched {len(candles)} candles for training")

    # Calculate features
    feature_pipeline = FeaturePipeline()
    min_window = feature_pipeline.config.min_candles
    feature_names = feature_pipeline.feature_names
    n_features = len(feature_names)

    valid_samples = len(candles) - min_window
    if valid_samples <= 0:
        await db.close()
        return TrainingReport(
            success=False, models_trained=[], validation_accuracy={},
            training_time_seconds=time.time() - start_time, timestamp=timestamp,
            trigger=RetrainingTrigger.INITIAL,
            error_message="Not enough candles for feature calculation",
        )

    logger.info(f"Calculating {n_features} features for {valid_samples} samples...")
    X = np.zeros((valid_samples, n_features), dtype=np.float64)

    for i in range(valid_samples):
        window = candles[i:i + min_window + 1]
        features = feature_pipeline.calculate_features(window, use_cache=False)
        for j, name in enumerate(feature_names):
            X[i, j] = features.get(name, 0.0)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Generate labels (LONG=1 if price goes up, else 0)
    # Using 0 threshold for balanced classes instead of 0.5%
    LABEL_THRESHOLD = 0.0
    y = np.zeros(valid_samples, dtype=np.float64)
    for i in range(valid_samples - 1):
        current_price = candles[i + min_window].close
        future_price = candles[i + min_window + 1].close if i + min_window + 1 < len(candles) else current_price
        price_change = (future_price - current_price) / current_price if current_price > 0 else 0
        y[i] = 1.0 if price_change > LABEL_THRESHOLD else 0.0

    # Remove last sample (no future price)
    X = X[:-1]
    y = y[:-1]

    # Train/validation split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

    models_trained = []
    validation_accuracy = {}
    model_paths = {}

    # Calculate class distribution
    pos_count = float(np.sum(y_train))
    neg_count = float(len(y_train) - pos_count)
    logger.info(f"Class balance: pos={pos_count:.0f}, neg={neg_count:.0f}")

    # Train XGBoost with stronger regularization to prevent overfitting
    try:
        logger.info("Training XGBoost...")
        xgb = XGBoostModel(XGBoostConfig(
            n_estimators=100,
            max_depth=4,  # Reduced to prevent overfitting
            learning_rate=0.05,  # Lower learning rate
            subsample=0.7,  # Row subsampling
            colsample_bytree=0.7,  # Column subsampling
            reg_alpha=0.5,  # Stronger L1 regularization
            reg_lambda=2.0,  # Stronger L2 regularization
        ))
        xgb.train(X_train, y_train, feature_names=feature_names, X_val=X_val, y_val=y_val)
        preds = xgb.predict(X_val)
        acc = float(np.mean((preds > 0.5) == y_val))
        path = models_dir / "xgboost.joblib"
        xgb.save(path)
        models_trained.append("xgboost")
        validation_accuracy["xgboost"] = acc
        model_paths["xgboost"] = str(path)
        logger.info(f"XGBoost: {acc:.2%} accuracy")
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")

    # Train LightGBM
    try:
        logger.info("Training LightGBM...")
        lgb = LightGBMModel(LightGBMConfig(n_estimators=100, num_leaves=31, learning_rate=0.1))
        lgb.train(X_train, y_train)
        preds = lgb.predict(X_val)
        acc = float(np.mean((preds > 0.5) == y_val))
        path = models_dir / "lightgbm.txt"
        lgb.save(path)
        models_trained.append("lightgbm")
        validation_accuracy["lightgbm"] = acc
        model_paths["lightgbm"] = str(path)
        logger.info(f"LightGBM: {acc:.2%} accuracy")
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")

    # Skip LSTM training (too slow on CPU, would need GPU)
    # To enable: set TRAIN_LSTM=True
    TRAIN_LSTM = False
    if TRAIN_LSTM:
        try:
            logger.info("Training LSTM...")
            lstm = LSTMModel(LSTMConfig(input_size=n_features, hidden_size=32, num_layers=1, epochs=10, batch_size=64))
            lstm.train(X_train, y_train)
            preds = lstm.predict(X_val)
            # Match prediction length to validation labels
            if len(preds) < len(y_val):
                y_val_trimmed = y_val[-len(preds):]
                acc = float(np.mean((preds > 0.5) == y_val_trimmed))
            else:
                acc = float(np.mean((preds[:len(y_val)] > 0.5) == y_val))
            path = models_dir / "lstm.pt"
            lstm.save(path)
            models_trained.append("lstm")
            validation_accuracy["lstm"] = acc
            model_paths["lstm"] = str(path)
            logger.info(f"LSTM: {acc:.2%} accuracy")
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
    else:
        logger.info("Skipping LSTM training (CPU too slow, enable with TRAIN_LSTM=True)")

    # Train Random Forest
    try:
        logger.info("Training Random Forest...")
        rf = RandomForestModel(RandomForestConfig(n_estimators=100, max_depth=10))
        rf.train(X_train, y_train.astype(np.int64))
        preds = rf.predict(X_val)
        acc = float(np.mean((preds > 0.5) == y_val))
        path = models_dir / "random_forest.joblib"
        rf.save(path)
        models_trained.append("random_forest")
        validation_accuracy["random_forest"] = acc
        model_paths["random_forest"] = str(path)
        logger.info(f"Random Forest: {acc:.2%} accuracy")
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")

    await db.close()

    report = TrainingReport(
        success=len(models_trained) > 0,
        models_trained=models_trained,
        validation_accuracy=validation_accuracy,
        training_time_seconds=time.time() - start_time,
        timestamp=timestamp,
        trigger=RetrainingTrigger.INITIAL,
        samples_used=len(X),
        feature_count=n_features,
        model_paths=model_paths,
    )

    return report


async def _train_models_old(symbol: str, min_samples: int = 500):
    """Old train function - kept for reference."""
    from src.data.database import Database
    from src.ml.trainer import ModelTrainer, RetrainingTrigger, TrainingReport

    db = Database()
    await db.initialize()

    trainer = ModelTrainer(database=db)
    report = await trainer.train_all_models(
        trigger=RetrainingTrigger.INITIAL,
        symbol=f"cmt_{symbol.lower()}",
        min_candles=min_samples,
    )

    await db.close()

    if report.success:
        logger.info("Training completed successfully!")
        logger.info(f"Models trained: {report.models_trained}")
        logger.info(f"Validation accuracy: {report.validation_accuracy}")
        logger.info(f"Training time: {report.training_time_seconds:.2f}s")
        logger.info(f"Samples used: {report.samples_used}")
        logger.info(f"Feature count: {report.feature_count}")
        logger.info(f"Class distribution: {report.class_distribution}")
    else:
        logger.error(f"Training failed: {report.error_message}")

    return report


def run_ensemble_backtest(
    candles: list[dict],
    initial_balance: float = 10000.0,
    position_size_pct: float = 0.10,
    leverage: int = 5,
    slippage_bps: int = 5,
) -> dict:
    """
    Run backtest using trained ML ensemble for signal generation.
    """
    from src.ml.ensemble import MLEnsemble
    from src.features.pipeline import FeaturePipeline
    from src.data.database import Candle

    logger.info("=" * 60)
    logger.info("RUNNING ENSEMBLE BACKTEST")
    logger.info("=" * 60)

    if len(candles) < 100:
        return {"error": "Not enough candles for backtest"}

    # Initialize ensemble and feature pipeline
    ensemble = MLEnsemble(models_dir=Path("models"))
    ensemble.check_and_reload_models()

    health = ensemble.get_health_status()
    healthy_count = sum(health.values())
    logger.info(f"Loaded models: {healthy_count}/4 healthy")
    logger.info(f"Health status: {health}")

    if healthy_count < 2:
        logger.warning("Not enough healthy models, falling back to simple strategy")
        return run_simple_backtest(candles, initial_balance, position_size_pct, leverage, slippage_bps)

    # Initialize feature pipeline
    feature_pipeline = FeaturePipeline()

    # Convert to Candle objects
    db_candles = [
        Candle(
            symbol="cmt_btcusdt",
            timestamp=c["timestamp"],
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
            interval="1h",
        )
        for c in candles
    ]

    # Run backtest
    balance = initial_balance
    equity_curve = [balance]
    trades = []
    open_trade = None

    # Need minimum window for feature calculation
    min_window = feature_pipeline.config.min_candles

    # Stop-loss and take-profit
    stop_loss_pct = 0.02
    take_profit_pct = 0.03

    for i in range(min_window, len(db_candles) - 1):
        current_candle = db_candles[i]
        current_price = current_candle.close

        # Calculate features
        window = db_candles[max(0, i - min_window) : i + 1]
        try:
            features = feature_pipeline.calculate_features(window, use_cache=False)
            # Convert to array format expected by ensemble
            feature_array = np.array([v for v in features.values() if isinstance(v, (int, float))], dtype=np.float64)
            feature_dict = {"array": feature_array}
        except Exception as e:
            logger.debug(f"Feature calculation failed at {i}: {e}")
            equity_curve.append(balance)
            continue

        # Check SL/TP on open trade
        if open_trade:
            if open_trade["side"] == "long":
                pnl_pct = (current_price - open_trade["entry_price"]) / open_trade["entry_price"]
                if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                    exit_price = current_price * (1 - slippage_bps / 10000)
                    pnl = pnl_pct * open_trade["size"] * leverage
                    open_trade["exit_price"] = exit_price
                    open_trade["exit_time"] = current_candle.timestamp
                    open_trade["pnl"] = pnl
                    balance += pnl
                    trades.append(open_trade)
                    open_trade = None
            else:  # short
                pnl_pct = (open_trade["entry_price"] - current_price) / open_trade["entry_price"]
                if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                    exit_price = current_price * (1 + slippage_bps / 10000)
                    pnl = pnl_pct * open_trade["size"] * leverage
                    open_trade["exit_price"] = exit_price
                    open_trade["exit_time"] = current_candle.timestamp
                    open_trade["pnl"] = pnl
                    balance += pnl
                    trades.append(open_trade)
                    open_trade = None

        # Generate ensemble signal
        if not open_trade:
            try:
                signal, confidence, model_outputs, weighted_avg = ensemble.predict(feature_dict)

                position_value = balance * position_size_pct

                if signal == "LONG" and confidence > 0.3:
                    entry_price = current_price * (1 + slippage_bps / 10000)
                    open_trade = {
                        "entry_time": current_candle.timestamp,
                        "side": "long",
                        "entry_price": entry_price,
                        "size": position_value,
                        "signal_confidence": confidence,
                        "weighted_avg": weighted_avg,
                    }
                elif signal == "SHORT" and confidence > 0.3:
                    entry_price = current_price * (1 - slippage_bps / 10000)
                    open_trade = {
                        "entry_time": current_candle.timestamp,
                        "side": "short",
                        "entry_price": entry_price,
                        "size": position_value,
                        "signal_confidence": confidence,
                        "weighted_avg": weighted_avg,
                    }
            except Exception as e:
                logger.debug(f"Prediction failed at {i}: {e}")

        # Track equity
        if open_trade:
            if open_trade["side"] == "long":
                unrealized = (current_price - open_trade["entry_price"]) / open_trade["entry_price"] * open_trade["size"] * leverage
            else:
                unrealized = (open_trade["entry_price"] - current_price) / open_trade["entry_price"] * open_trade["size"] * leverage
            equity_curve.append(balance + unrealized)
        else:
            equity_curve.append(balance)

    # Close any remaining trade
    if open_trade:
        last_price = db_candles[-1].close
        if open_trade["side"] == "long":
            pnl = (last_price - open_trade["entry_price"]) / open_trade["entry_price"] * open_trade["size"] * leverage
        else:
            pnl = (open_trade["entry_price"] - last_price) / open_trade["entry_price"] * open_trade["size"] * leverage
        open_trade["exit_price"] = last_price
        open_trade["exit_time"] = db_candles[-1].timestamp
        open_trade["pnl"] = pnl
        balance += pnl
        trades.append(open_trade)

    # Calculate metrics
    total_return = (balance - initial_balance) / initial_balance * 100
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        "strategy": "ML Ensemble",
        "initial_balance": initial_balance,
        "final_balance": balance,
        "total_return_pct": total_return,
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_dd * 100,
        "leverage": leverage,
        "trades": trades,
    }


def run_simple_backtest(
    candles: list[dict],
    initial_balance: float,
    position_size_pct: float,
    leverage: int,
    slippage_bps: int,
) -> dict:
    """Fallback simple SMA crossover strategy."""
    if len(candles) < 30:
        return {"error": "Not enough candles"}

    balance = initial_balance
    trades = []
    open_trade = None

    sma_period = 20
    stop_loss_pct = 0.02
    take_profit_pct = 0.03

    prices = [c["close"] for c in candles]

    for i in range(sma_period, len(candles)):
        current_price = prices[i]
        sma = sum(prices[i - sma_period + 1 : i + 1]) / sma_period

        if open_trade:
            if open_trade["side"] == "long":
                pnl_pct = (current_price - open_trade["entry_price"]) / open_trade["entry_price"]
            else:
                pnl_pct = (open_trade["entry_price"] - current_price) / open_trade["entry_price"]

            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                pnl = pnl_pct * open_trade["size"] * leverage
                open_trade["pnl"] = pnl
                balance += pnl
                trades.append(open_trade)
                open_trade = None

        if not open_trade:
            position_value = balance * position_size_pct
            if current_price > sma:
                open_trade = {
                    "entry_time": candles[i]["timestamp"],
                    "side": "long",
                    "entry_price": current_price * (1 + slippage_bps / 10000),
                    "size": position_value,
                }
            elif current_price < sma:
                open_trade = {
                    "entry_time": candles[i]["timestamp"],
                    "side": "short",
                    "entry_price": current_price * (1 - slippage_bps / 10000),
                    "size": position_value,
                }

    if open_trade:
        last_price = prices[-1]
        if open_trade["side"] == "long":
            pnl = (last_price - open_trade["entry_price"]) / open_trade["entry_price"] * open_trade["size"] * leverage
        else:
            pnl = (open_trade["entry_price"] - last_price) / open_trade["entry_price"] * open_trade["size"] * leverage
        open_trade["pnl"] = pnl
        balance += pnl
        trades.append(open_trade)

    total_return = (balance - initial_balance) / initial_balance * 100
    winning = [t for t in trades if t.get("pnl", 0) > 0]

    return {
        "strategy": "SMA Crossover (Fallback)",
        "initial_balance": initial_balance,
        "final_balance": balance,
        "total_return_pct": total_return,
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(trades) - len(winning),
        "win_rate_pct": len(winning) / len(trades) * 100 if trades else 0,
        "leverage": leverage,
        "trades": trades,
    }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train ML models and run backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", default="1h", help="Candle interval")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candles")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    args = parser.parse_args()

    print("=" * 60)
    print("ALPHASTRIKE ML TRAINING & BACKTEST")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Candles: {args.limit}")
    print(f"Initial Balance: ${args.balance:,.2f}")
    print(f"Leverage: {args.leverage}x")
    print()

    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Step 1: Download historical data
        print("Step 1: Downloading historical data from WEEX...")
        candles = await fetch_weex_candles(session, args.symbol, args.interval, args.limit)

        if not candles:
            print("ERROR: No candles downloaded")
            return 1

        print(f"  Downloaded {len(candles)} candles")
        print(f"  First: {candles[0]['timestamp']} - ${candles[0]['close']:,.2f}")
        print(f"  Last: {candles[-1]['timestamp']} - ${candles[-1]['close']:,.2f}")
        print()

        # Step 2: Save to database
        print("Step 2: Saving candles to database...")
        saved = await save_candles_to_db(candles, args.symbol, args.interval)
        print(f"  Saved {saved} candles to SQLite database")
        print()

        # Step 3: Train models (unless skipped)
        if not args.skip_training:
            print("Step 3: Training ML models...")
            report = await train_models(args.symbol, interval=args.interval, min_samples=min(500, len(candles) - 100))

            if not report.success:
                print(f"  Training failed: {report.error_message}")
                print("  Continuing with fallback strategy...")
            else:
                print(f"  Trained {len(report.models_trained)} models")
                for model, acc in report.validation_accuracy.items():
                    print(f"    {model}: {acc:.2%} accuracy")
            print()
        else:
            print("Step 3: Skipping training (--skip-training)")
            print()

        # Step 4: Run backtest
        print("Step 4: Running backtest with ensemble predictions...")
        results = run_ensemble_backtest(
            candles,
            initial_balance=args.balance,
            leverage=args.leverage,
        )

        if "error" in results:
            print(f"ERROR: {results['error']}")
            return 1

        # Print results
        print()
        print("=" * 60)
        print(f"BACKTEST RESULTS ({results.get('strategy', 'Unknown')})")
        print("=" * 60)
        print(f"Initial Balance:  ${results['initial_balance']:>12,.2f}")
        print(f"Final Balance:    ${results['final_balance']:>12,.2f}")
        print(f"Total Return:     {results['total_return_pct']:>12.2f}%")
        print()
        print(f"Total Trades:     {results['total_trades']:>12}")
        print(f"Winning Trades:   {results['winning_trades']:>12}")
        print(f"Losing Trades:    {results['losing_trades']:>12}")
        print(f"Win Rate:         {results['win_rate_pct']:>12.1f}%")
        print()
        if "max_drawdown_pct" in results:
            print(f"Max Drawdown:     {results['max_drawdown_pct']:>12.2f}%")
        print()

        # Recent trades
        if results.get("trades"):
            print("Recent Trades:")
            print("-" * 60)
            for trade in results["trades"][-10:]:
                pnl = trade.get("pnl", 0)
                pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                conf = trade.get("signal_confidence", 0)
                conf_str = f" (conf: {conf:.0%})" if conf else ""
                print(f"  {trade.get('entry_time', 'N/A')} | {trade.get('side', 'N/A'):5} | Entry: ${trade.get('entry_price', 0):,.2f} | PnL: {pnl_str}{conf_str}")

        print("=" * 60)

        if results["total_return_pct"] > 0:
            print(f"Result: PROFITABLE (+{results['total_return_pct']:.2f}%)")
        else:
            print(f"Result: LOSS ({results['total_return_pct']:.2f}%)")

        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
