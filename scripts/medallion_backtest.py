#!/usr/bin/env python3
"""
Medallion-Style Multi-Asset Backtest

Implements Jim Simons' Medallion Fund principles:
1. Trade 20+ uncorrelated instruments
2. Small positions (5-8% max)
3. Correlation-adjusted sizing
4. Cross-asset signals (lead-lag, sector rotation)
5. Traditional assets (PAXG, SPX) for diversification

Target metrics:
- CAGR: 66%+
- Max Drawdown: <5%
- Sharpe Ratio: 2.5+

Usage:
    python scripts/medallion_backtest.py --days 180
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    size_usd: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees: float
    conviction: float
    sector: str


@dataclass
class BacktestConfig:
    """Configuration for the Medallion backtest."""
    # Assets to trade (Medallion-style diversified portfolio)
    assets: list[str] = field(default_factory=lambda: [
        # Crypto Major (25%) - liquid majors
        "BTC", "ETH", "BNB", "XRP",
        # L1/L2 (20%) - ecosystem plays
        "SOL", "AVAX", "NEAR", "APT",
        # DeFi (15%) - protocol tokens
        "AAVE", "UNI", "LINK",
        # AI (10%) - narrative driven
        "FET",
        # Meme (10%) - high volatility
        "DOGE",
        # Traditional (20%) - CRITICAL for drawdown reduction
        "PAXG", "SPX",
    ])

    # Time range
    days: int = 180

    # Portfolio settings
    initial_balance: float = 10000.0
    leverage: int = 3  # Reduced from 5x for lower drawdown

    # Position sizing (Medallion style - conservative)
    max_portfolio_heat: float = 0.30  # 30% max correlation-adjusted exposure
    max_single_position: float = 0.06  # 6% max per position (was 8%)
    base_position_size: float = 0.04  # 4% base size (was 5%)

    # Trading costs
    slippage_bps: float = 5.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005

    # Signal thresholds - balanced for trade frequency
    min_conviction: float = 55.0  # Balance between quality and frequency
    confidence_threshold: float = 0.50


@dataclass
class MedallionMetrics:
    """Performance metrics for the backtest."""
    total_return: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_losses: int = 0
    final_balance: float = 0.0

    # Diversification metrics
    avg_positions: float = 0.0
    avg_correlation: float = 0.0
    diversification_benefit: float = 0.0

    # Per-sector performance
    sector_pnl: dict = field(default_factory=dict)


class TechnicalSignalGenerator:
    """
    Technical signal generator for Medallion-style backtesting.

    Uses asset-class-specific technical indicators:
    - Crypto: Momentum-focused (faster signals, wider thresholds)
    - Traditional: Mean-reversion (tighter RSI, slower signals)

    This approach aligns with Medallion's sector-specific modeling philosophy.
    """

    # Traditional assets need different treatment
    TRADITIONAL_ASSETS = {"PAXGUSDT", "SPXUSDT", "EURUSDT", "GOLDUSDT"}

    def __init__(self):
        # Crypto parameters (volatile assets)
        self.crypto_fast = 10
        self.crypto_slow = 30
        self.crypto_rsi = 14

        # Traditional parameters (stable assets)
        self.trad_fast = 20  # Slower
        self.trad_slow = 50  # Much slower
        self.trad_rsi = 21   # Longer RSI

    def calculate_signal(
        self,
        candles: list,
        cross_asset_signal: float = 0.0,
        symbol: str = "",
    ) -> tuple[str, float]:
        """
        Calculate trading signal with asset-class awareness.

        Returns:
            (direction, conviction) where direction is "LONG", "SHORT", or "HOLD"
            and conviction is 0-100.
        """
        is_traditional = symbol in self.TRADITIONAL_ASSETS

        # Select parameters based on asset class
        if is_traditional:
            fast_period = self.trad_fast
            slow_period = self.trad_slow
            rsi_period = self.trad_rsi
        else:
            fast_period = self.crypto_fast
            slow_period = self.crypto_slow
            rsi_period = self.crypto_rsi

        if len(candles) < slow_period + 5:
            return "HOLD", 0.0

        closes = np.array([c.close for c in candles])

        # 1. Moving average crossover
        fast_ma = float(np.mean(closes[-fast_period:]))
        slow_ma = float(np.mean(closes[-slow_period:]))
        ma_signal = (fast_ma - slow_ma) / slow_ma  # Normalized difference

        # 2. RSI
        rsi = self._calculate_rsi(closes, rsi_period)

        # 3. Momentum (different lookbacks for asset classes)
        mom_period = 20 if is_traditional else 10
        if len(closes) > mom_period:
            momentum = (closes[-1] - closes[-mom_period]) / closes[-mom_period]
        else:
            momentum = 0

        # Asset-class specific signal generation
        if is_traditional:
            return self._generate_traditional_signal(ma_signal, rsi, momentum, cross_asset_signal)
        else:
            return self._generate_crypto_signal(ma_signal, rsi, momentum, cross_asset_signal)

    def _generate_crypto_signal(
        self,
        ma_signal: float,
        rsi: float,
        momentum: float,
        cross_asset_signal: float,
    ) -> tuple[str, float]:
        """
        Generate signal for crypto assets (momentum + mean reversion hybrid).

        Strategy: Trade WITH momentum when trend is clear,
        trade AGAINST extremes when RSI is at limits.
        """
        long_score = 0.0
        short_score = 0.0

        # LONG signals
        if ma_signal > 0.008:  # Strong uptrend
            long_score += 35
        elif ma_signal > 0.003 and momentum > 0.02:  # Mild uptrend + momentum
            long_score += 25
        if rsi < 30:  # Very oversold - mean reversion opportunity
            long_score += 30
        elif rsi < 40 and momentum > 0:  # Oversold with positive momentum
            long_score += 20
        if momentum > 0.025:  # Strong momentum
            long_score += 25
        if cross_asset_signal > 0.25:  # Strong cross-asset confirmation
            long_score += 20

        # SHORT signals
        if ma_signal < -0.008:  # Strong downtrend
            short_score += 35
        elif ma_signal < -0.003 and momentum < -0.02:  # Mild downtrend + momentum
            short_score += 25
        if rsi > 70:  # Very overbought - mean reversion opportunity
            short_score += 30
        elif rsi > 60 and momentum < 0:  # Overbought with negative momentum
            short_score += 20
        if momentum < -0.025:  # Strong negative momentum
            short_score += 25
        if cross_asset_signal < -0.25:  # Strong cross-asset confirmation
            short_score += 20

        # Require clear signal advantage
        if long_score >= 55 and long_score > short_score + 20:
            return "LONG", min(100, long_score)
        elif short_score >= 55 and short_score > long_score + 20:
            return "SHORT", min(100, short_score)

        return "HOLD", 0.0

    def _generate_traditional_signal(
        self,
        ma_signal: float,
        rsi: float,
        momentum: float,
        cross_asset_signal: float,
    ) -> tuple[str, float]:
        """
        Generate signal for traditional assets (mean-reversion focused).

        Traditional assets like gold and SPX tend to mean-revert,
        so we use tighter RSI bands and require stronger confirmation.
        """
        long_score = 0.0
        short_score = 0.0

        # LONG signals for traditional (mean reversion)
        # Only go long when significantly oversold
        if rsi < 30:  # Very oversold
            long_score += 35
        if ma_signal < -0.002 and rsi < 40:  # Below MA but oversold = buy
            long_score += 30
        if momentum < -0.01 and rsi < 35:  # Falling but oversold
            long_score += 20
        if cross_asset_signal < -0.1:  # Risk-off = buy gold
            long_score += 15

        # SHORT signals for traditional
        # Only short when significantly overbought
        if rsi > 70:  # Very overbought
            short_score += 35
        if ma_signal > 0.002 and rsi > 60:  # Above MA and overbought = sell
            short_score += 30
        if momentum > 0.01 and rsi > 65:  # Rising but overbought
            short_score += 20
        if cross_asset_signal > 0.1:  # Risk-on = sell gold
            short_score += 15

        # Higher threshold for traditional (more selective)
        if long_score >= 60 and long_score > short_score + 20:
            return "LONG", min(100, long_score)
        elif short_score >= 60 and short_score > long_score + 20:
            return "SHORT", min(100, short_score)

        return "HOLD", 0.0

    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0  # Neutral

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))


class MedallionBacktestEngine:
    """
    Medallion-style multi-asset backtest engine.

    Combines:
    - Technical signals (MA crossover, RSI, momentum)
    - Cross-asset signals (leader-lag, sector rotation)
    - Medallion position sizing
    - Portfolio heat management

    Note: Uses technical signals instead of ML for asset-agnostic backtesting.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.positions: dict[str, Any] = {}  # symbol -> position info

        # Initialize components (lazy import to avoid circular deps)
        self._init_components()

    def _init_components(self):
        """Initialize trading components."""
        from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient
        from src.exchange.adapters.hyperliquid.mappers import HyperliquidMapper
        from src.features.pipeline import FeaturePipeline
        from src.strategy.cross_asset_signals import CrossAssetSignalEngine
        from src.strategy.medallion_sizer import MedallionPositionSizer, Position
        from src.adaptive.portfolio_risk import PortfolioRiskManager

        # Exchange client
        self.client = HyperliquidRESTClient()
        self.mapper = HyperliquidMapper

        # Feature pipeline for ML
        self.feature_pipeline = FeaturePipeline()

        # Load ML models for each asset
        self.ml_models = self._load_ml_models()

        # Technical signal generator (fallback when ML unavailable)
        self.signal_generator = TechnicalSignalGenerator()

        # Cross-asset signals
        self.cross_asset_engine = CrossAssetSignalEngine()

        # Medallion position sizer
        self.position_sizer = MedallionPositionSizer(
            max_portfolio_heat=self.config.max_portfolio_heat,
            max_single_position=self.config.max_single_position,
            base_position_size=self.config.base_position_size,
        )

        # Portfolio risk manager
        symbols = [f"{a}USDT" for a in self.config.assets]
        self.risk_manager = PortfolioRiskManager(
            symbols=symbols,
            max_heat_pct=self.config.max_portfolio_heat,
            max_single_position_pct=self.config.max_single_position,
        )

        # Position class for sizer
        self.Position = Position

    def _load_ml_models(self) -> dict:
        """Load trained ML models for each asset."""
        from pathlib import Path
        from src.ml.xgboost_model import XGBoostModel
        from src.ml.lightgbm_model import LightGBMModel

        models_dir = Path("models")
        models = {}

        for asset in self.config.assets:
            asset_lower = asset.lower()
            symbol = f"{asset}USDT"

            # Try to load XGBoost (primary) and LightGBM (secondary)
            xgb_path = models_dir / f"xgboost_hyperliquid_{asset_lower}.joblib"
            lgb_path = models_dir / f"lightgbm_hyperliquid_{asset_lower}.txt"

            if xgb_path.exists():
                try:
                    xgb = XGBoostModel()
                    xgb.load(xgb_path)
                    models[symbol] = {"xgboost": xgb, "type": "xgboost"}
                    logger.info(f"  Loaded XGBoost for {symbol}")
                except Exception as e:
                    logger.warning(f"  Failed to load XGBoost for {symbol}: {e}")

            if lgb_path.exists() and symbol not in models:
                try:
                    lgb = LightGBMModel()
                    lgb.load(lgb_path)
                    models[symbol] = {"lightgbm": lgb, "type": "lightgbm"}
                    logger.info(f"  Loaded LightGBM for {symbol}")
                except Exception as e:
                    logger.warning(f"  Failed to load LightGBM for {symbol}: {e}")

        logger.info(f"Loaded ML models for {len(models)}/{len(self.config.assets)} assets")
        return models

    def _precalculate_all_features(self, all_candles: dict) -> dict:
        """
        Pre-calculate features for all assets to speed up backtesting.

        Returns:
            Dict of symbol -> list of feature dicts (one per candle from warmup onwards)
        """
        logger.info("Pre-calculating features for all assets...")
        all_features = {}
        min_window = self.feature_pipeline.config.min_candles

        for symbol, candles in all_candles.items():
            if len(candles) < min_window + 10:
                continue

            symbol_features = []
            for i in range(min_window, len(candles)):
                window = candles[max(0, i - min_window):i + 1]
                try:
                    features = self.feature_pipeline.calculate_features(window, use_cache=False)
                    symbol_features.append(features)
                except Exception:
                    symbol_features.append(None)

            all_features[symbol] = symbol_features
            logger.info(f"  {symbol}: {len(symbol_features)} feature vectors")

        return all_features

    def _get_ml_signal_cached(self, symbol: str, features: dict | None) -> tuple[str, float]:
        """
        Get ML signal from pre-calculated features.

        Returns:
            (direction, confidence) where direction is "LONG", "SHORT", or "HOLD"
        """
        if symbol not in self.ml_models or features is None:
            return "HOLD", 0.0

        try:
            # Get feature vector
            feature_names = self.feature_pipeline.feature_names
            X = np.array([[features.get(name, 0.0) for name in feature_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Get prediction
            model_info = self.ml_models[symbol]
            model = model_info.get("xgboost") or model_info.get("lightgbm")

            if model is None:
                return "HOLD", 0.0

            pred = model.predict(X)[0]

            # Convert prediction to signal
            # Use tighter thresholds for higher quality signals
            # pred > 0.60 = LONG, pred < 0.40 = SHORT (only strong signals)
            if pred > 0.60:
                confidence = min(100, (pred - 0.5) * 200)  # Scale 0.5-1.0 to 0-100
                return "LONG", confidence
            elif pred < 0.40:
                confidence = min(100, (0.5 - pred) * 200)  # Scale 0.0-0.5 to 0-100
                return "SHORT", confidence
            else:
                return "HOLD", 0.0

        except Exception as e:
            logger.debug(f"ML prediction failed for {symbol}: {e}")
            return "HOLD", 0.0

    async def fetch_all_candles(self) -> dict[str, list]:
        """Fetch candles for all assets."""
        from src.exchange.models import UnifiedCandle

        logger.info(f"Fetching candles for {len(self.config.assets)} assets...")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.config.days)

        all_candles: dict[str, list[UnifiedCandle]] = {}

        for asset in self.config.assets:
            symbol = f"{asset}USDT"
            coin = asset

            try:
                # Calculate limit based on days (hourly candles)
                limit = self.config.days * 24 + 100  # Extra for warmup
                candles = await self.client.get_candles(
                    symbol=symbol,
                    interval="1h",
                    limit=limit,
                    start_time=start_time,
                    end_time=end_time,
                )
                if candles:
                    all_candles[symbol] = candles
                    logger.info(f"  {symbol}: {len(candles)} candles")
                else:
                    logger.warning(f"  {symbol}: No candles returned")
            except Exception as e:
                logger.warning(f"  {symbol}: Failed to fetch - {e}")

        return all_candles

    async def run(self) -> MedallionMetrics:
        """Run the full backtest."""
        # Fetch data
        all_candles = await self.fetch_all_candles()

        if not all_candles:
            logger.error("No candles fetched, cannot run backtest")
            return MedallionMetrics()

        logger.info(f"\nRunning Medallion backtest with {len(all_candles)} assets...")

        # Pre-calculate features for ML (much faster than calculating per-candle)
        all_features = self._precalculate_all_features(all_candles)

        # Initialize state
        balance = self.config.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        trade_id = 0

        # Track positions
        positions: dict[str, dict] = {}  # symbol -> {side, size, entry_price, entry_time}

        # Find common time range
        min_len = min(len(c) for c in all_candles.values())
        min_window = self.feature_pipeline.config.min_candles
        warmup = min_window  # Use feature pipeline's min candles as warmup

        if min_len < warmup + 20:
            logger.error(f"Insufficient data: {min_len} candles, need {warmup + 20}")
            return MedallionMetrics()

        logger.info(f"Starting simulation with {min_len} candles, warmup={warmup}")

        # Main loop - iterate through time
        returns = []
        position_counts = []

        for i in range(warmup, min_len):
            # Get current candles for all assets
            current_candles = {
                symbol: candles[:i+1]
                for symbol, candles in all_candles.items()
            }

            # Current timestamp
            timestamp = list(all_candles.values())[0][i].timestamp

            # Update risk manager with current prices
            for symbol, candles in current_candles.items():
                if candles:
                    self.risk_manager.update_price(symbol, candles[-1].close, timestamp)

            # Check for exits first
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                candles = current_candles.get(symbol, [])
                if not candles:
                    continue

                current_price = candles[-1].close
                entry_price = pos["entry_price"]
                side = pos["side"]

                # Calculate unrealized PnL
                if side == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # Exit conditions: Balanced risk/reward
                # Stop: -1.5%, TP: +3%, trailing at +2%
                should_exit = False

                # Track peak PnL for trailing stop
                pos["peak_pnl"] = max(pnl_pct, pos.get("peak_pnl", 0))

                if pnl_pct <= -0.015:  # Stop loss at -1.5%
                    should_exit = True
                elif pnl_pct >= 0.03:  # Take profit at +3%
                    should_exit = True
                elif pnl_pct >= 0.02:  # Trailing stop at +2%
                    # Trail at 50% of peak
                    trail_level = pos["peak_pnl"] * 0.5
                    if pnl_pct < trail_level:
                        should_exit = True
                elif (timestamp - pos["entry_time"]).total_seconds() > 18 * 3600:
                    # Time exit at 18h
                    should_exit = True

                if should_exit:
                    # Calculate PnL
                    size_usd = pos["size"]
                    pnl = size_usd * pnl_pct * self.config.leverage
                    fees = size_usd * (self.config.taker_fee * 2)  # Entry + exit
                    net_pnl = pnl - fees

                    balance += net_pnl

                    # Record trade
                    trade_id += 1
                    sector = self.mapper.get_sector(symbol)
                    self.trades.append(BacktestTrade(
                        id=str(trade_id),
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        exit_price=current_price,
                        size_usd=size_usd,
                        entry_time=pos["entry_time"],
                        exit_time=timestamp,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct * 100,
                        fees=fees,
                        conviction=pos.get("conviction", 50),
                        sector=sector,
                    ))

                    del positions[symbol]
                    self.risk_manager.close_position(symbol)

            # Check market regime (BTC trend as proxy)
            btc_candles = current_candles.get("BTCUSDT", [])
            market_regime = "neutral"
            if len(btc_candles) >= 50:
                btc_closes = [c.close for c in btc_candles[-50:]]
                btc_ma20 = sum(btc_closes[-20:]) / 20
                btc_ma50 = sum(btc_closes) / 50
                btc_momentum = (btc_closes[-1] - btc_closes[-20]) / btc_closes[-20]

                if btc_ma20 > btc_ma50 * 1.01 and btc_momentum > 0.02:
                    market_regime = "bullish"
                elif btc_ma20 < btc_ma50 * 0.99 and btc_momentum < -0.02:
                    market_regime = "bearish"

            # Generate signals for assets not in position
            for symbol, candles in current_candles.items():
                if symbol in positions:
                    continue

                if len(candles) < 35:  # Need enough for technical indicators
                    continue

                # Skip traditional assets in some cases based on regime
                is_traditional = symbol in {"PAXGUSDT", "SPXUSDT"}

                try:
                    # Get pre-calculated features for this candle
                    feature_idx = i - min_window  # Index into pre-calculated features
                    symbol_features = all_features.get(symbol, [])
                    features = symbol_features[feature_idx] if 0 <= feature_idx < len(symbol_features) else None

                    # Get cross-asset signal for confirmation
                    cross_signal = self.cross_asset_engine.calculate_signal(
                        symbol, "LONG", current_candles
                    )

                    # Try ML signal first (our edge)
                    direction, ml_conviction = self._get_ml_signal_cached(symbol, features)

                    # Get technical signal
                    tech_direction, tech_conviction = self.signal_generator.calculate_signal(
                        candles, cross_signal.combined_signal, symbol
                    )

                    # Combine ML and technical signals for stronger conviction
                    # If both agree, boost conviction; if they disagree, reduce
                    if direction != "HOLD" and direction == tech_direction:
                        # Both agree - high conviction
                        base_conviction = min(100, ml_conviction + tech_conviction * 0.5)
                    elif direction != "HOLD":
                        # Only ML signal - moderate conviction
                        base_conviction = ml_conviction * 0.7
                    elif tech_direction != "HOLD":
                        # Only technical signal - moderate conviction
                        direction = tech_direction
                        base_conviction = tech_conviction * 0.8
                    else:
                        direction = "HOLD"
                        base_conviction = 0.0

                    if direction == "HOLD":
                        continue

                    # Regime filter for crypto (not traditional)
                    if not is_traditional:
                        if market_regime == "bearish" and direction == "LONG":
                            base_conviction *= 0.7  # Reduce conviction for longs in bear market
                        elif market_regime == "bullish" and direction == "SHORT":
                            base_conviction *= 0.7  # Reduce conviction for shorts in bull market

                    # Add cross-asset bonus to conviction
                    cross_bonus, _ = self.cross_asset_engine.get_conviction_bonus(
                        symbol, direction, current_candles
                    )
                    conviction = min(100, base_conviction + cross_bonus)

                    if conviction < self.config.min_conviction:
                        continue

                    # Get sector bias from rotation
                    sector_bias = self.cross_asset_engine.get_sector_allocation_bias(
                        current_candles
                    )
                    sector = self.mapper.get_sector(symbol)
                    bias = sector_bias.get(sector, 1.0)

                    # Convert current positions for sizer
                    sizer_positions = {
                        sym: self.Position(
                            symbol=sym,
                            side=p["side"],
                            size_usd=p["size"],
                            entry_price=p["entry_price"],
                        )
                        for sym, p in positions.items()
                    }

                    # Calculate position size
                    sizing = self.position_sizer.calculate_position_size(
                        symbol=symbol,
                        direction=direction,
                        conviction=conviction,
                        balance=balance,
                        current_positions=sizer_positions,
                        sector_bias=bias,
                    )

                    if sizing.blocked or sizing.adjusted_size < 100:
                        continue

                    # Open position
                    entry_price = candles[-1].close
                    positions[symbol] = {
                        "side": direction,
                        "size": sizing.adjusted_size,
                        "entry_price": entry_price,
                        "entry_time": timestamp,
                        "conviction": conviction,
                    }

                    self.risk_manager.register_position(
                        symbol, direction, sizing.adjusted_size, entry_price
                    )

                except Exception as e:
                    # Skip on error
                    continue

            # Track equity
            self.equity_curve.append((timestamp, balance))
            position_counts.append(len(positions))

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance
            if dd > max_drawdown:
                max_drawdown = dd

            # Track returns
            if len(self.equity_curve) > 1:
                prev_balance = self.equity_curve[-2][1]
                ret = (balance - prev_balance) / prev_balance
                returns.append(ret)

        # Close any remaining positions at end
        for symbol, pos in list(positions.items()):
            candles = all_candles.get(symbol, [])
            if candles:
                current_price = candles[-1].close
                entry_price = pos["entry_price"]
                side = pos["side"]

                if side == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                size_usd = pos["size"]
                pnl = size_usd * pnl_pct * self.config.leverage
                fees = size_usd * (self.config.taker_fee * 2)
                balance += pnl - fees

        # Calculate metrics
        metrics = self._calculate_metrics(
            initial_balance=self.config.initial_balance,
            final_balance=balance,
            max_drawdown=max_drawdown,
            returns=returns,
            position_counts=position_counts,
        )

        await self.client.close()

        return metrics

    def _calculate_metrics(
        self,
        initial_balance: float,
        final_balance: float,
        max_drawdown: float,
        returns: list[float],
        position_counts: list[int],
    ) -> MedallionMetrics:
        """Calculate performance metrics."""
        metrics = MedallionMetrics()

        # Basic returns
        metrics.total_return = (final_balance - initial_balance) / initial_balance
        metrics.final_balance = final_balance
        metrics.max_drawdown = max_drawdown
        metrics.total_trades = len(self.trades)

        # Annualized returns (assuming hourly data)
        hours = len(returns)
        years = hours / (24 * 365)
        if years > 0 and final_balance > 0:
            metrics.cagr = (final_balance / initial_balance) ** (1 / years) - 1

        # Sharpe ratio (annualized)
        if returns and np.std(returns) > 0:
            hourly_sharpe = np.mean(returns) / np.std(returns)
            metrics.sharpe_ratio = hourly_sharpe * np.sqrt(24 * 365)

        # Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        if negative_returns and np.std(negative_returns) > 0:
            hourly_sortino = np.mean(returns) / np.std(negative_returns)
            metrics.sortino_ratio = hourly_sortino * np.sqrt(24 * 365)

        # Trade statistics
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]

            metrics.win_rate = len(wins) / len(self.trades) if self.trades else 0
            metrics.avg_trade_pnl = float(np.mean([t.pnl for t in self.trades]))
            metrics.avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
            metrics.avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0

            # Profit factor
            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 1
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0

            # Max consecutive losses
            max_consec = 0
            current_consec = 0
            for t in self.trades:
                if t.pnl <= 0:
                    current_consec += 1
                    max_consec = max(max_consec, current_consec)
                else:
                    current_consec = 0
            metrics.max_consecutive_losses = max_consec

            # Sector PnL
            for t in self.trades:
                if t.sector not in metrics.sector_pnl:
                    metrics.sector_pnl[t.sector] = 0
                metrics.sector_pnl[t.sector] += t.pnl

        # Diversification metrics
        if position_counts:
            metrics.avg_positions = float(np.mean(position_counts))

        return metrics


def print_results(config: BacktestConfig, metrics: MedallionMetrics):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("MEDALLION-STYLE MULTI-ASSET BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Assets: {len(config.assets)} ({', '.join(config.assets[:5])}...)")
    print(f"  Period: {config.days} days")
    print(f"  Initial Balance: ${config.initial_balance:,.0f}")
    print(f"  Leverage: {config.leverage}x")
    print(f"  Max Portfolio Heat: {config.max_portfolio_heat:.0%}")
    print(f"  Max Single Position: {config.max_single_position:.0%}")

    print(f"\n{'=' * 70}")
    print("PERFORMANCE METRICS")
    print("=" * 70)

    # Compare to targets
    cagr_status = "✓" if metrics.cagr >= 0.66 else "✗"
    dd_status = "✓" if metrics.max_drawdown <= 0.05 else "✗"
    sharpe_status = "✓" if metrics.sharpe_ratio >= 2.5 else "✗"

    print(f"\n{'Metric':<30} {'Value':>15} {'Target':>15} {'Status':>8}")
    print("-" * 70)
    print(f"{'Annualized Return (CAGR)':<30} {metrics.cagr:>14.1%} {'>66%':>15} {cagr_status:>8}")
    print(f"{'Max Drawdown':<30} {metrics.max_drawdown:>14.1%} {'<5%':>15} {dd_status:>8}")
    print(f"{'Sharpe Ratio':<30} {metrics.sharpe_ratio:>14.2f} {'>2.5':>15} {sharpe_status:>8}")
    print(f"{'Sortino Ratio':<30} {metrics.sortino_ratio:>14.2f}")
    print(f"{'Win Rate':<30} {metrics.win_rate:>14.1%}")
    print(f"{'Profit Factor':<30} {metrics.profit_factor:>14.2f}")
    print(f"{'Total Trades':<30} {metrics.total_trades:>14}")
    print(f"{'Avg Positions':<30} {metrics.avg_positions:>14.1f}")

    print(f"\n{'=' * 70}")
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"\n  Initial Balance: ${config.initial_balance:,.2f}")
    print(f"  Final Balance:   ${metrics.final_balance:,.2f}")
    print(f"  Total Return:    {metrics.total_return:+.1%}")

    if metrics.sector_pnl:
        print(f"\n{'=' * 70}")
        print("SECTOR PERFORMANCE")
        print("=" * 70)
        sorted_sectors = sorted(metrics.sector_pnl.items(), key=lambda x: x[1], reverse=True)
        for sector, pnl in sorted_sectors:
            status = "+" if pnl > 0 else ""
            print(f"  {sector:<20} ${status}{pnl:,.2f}")

    print("\n" + "=" * 70)

    # Overall assessment
    targets_met = sum([
        metrics.cagr >= 0.66,
        metrics.max_drawdown <= 0.05,
        metrics.sharpe_ratio >= 2.5,
    ])

    if targets_met == 3:
        print("🏆 MEDALLION TARGETS MET! Strategy performs at hedge fund level.")
    elif targets_met >= 2:
        print("📈 STRONG PERFORMANCE - 2/3 targets met. Close to Medallion level.")
    elif targets_met >= 1:
        print("📊 DECENT PERFORMANCE - 1/3 targets met. More tuning needed.")
    else:
        print("⚠️  BELOW TARGETS - All targets missed. Significant tuning needed.")

    print("=" * 70 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Medallion-Style Multi-Asset Backtest"
    )
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=3, help="Leverage (default 3x for low drawdown)")
    parser.add_argument("--heat", type=float, default=0.30, help="Max portfolio heat (default 30%%)")
    parser.add_argument("--position", type=float, default=0.06, help="Max single position (default 6%%)")

    args = parser.parse_args()

    config = BacktestConfig(
        days=args.days,
        initial_balance=args.balance,
        leverage=args.leverage,
        max_portfolio_heat=args.heat,
        max_single_position=args.position,
    )

    engine = MedallionBacktestEngine(config)
    metrics = await engine.run()

    print_results(config, metrics)


if __name__ == "__main__":
    asyncio.run(main())
