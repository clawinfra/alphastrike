#!/usr/bin/env python3
"""
Medallion V2 Backtest

LightGBM-based trading strategy with:
- BULLISH regime filter (60%+ confidence)
- Multi-tier ML signal filtering (65-70+ conviction)
- Dynamic leverage (adjusts for volatility and drawdown)

Target metrics:
- CAGR: 66%+
- Max Drawdown: <5%
- Sharpe Ratio: 2.5+

Usage:
    python scripts/medallion_v2_backtest.py --days 180 --base-leverage 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class Position:
    """Track an open position."""
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    size: float
    entry_time: datetime
    entry_candle_idx: int
    strategy: str
    conviction: float
    # Trailing stop fields (Simons enhancement)
    max_pnl_pct: float = 0.0  # Track max favorable excursion
    trailing_stop_active: bool = False  # Activated after +1R


@dataclass
class Trade:
    """Completed trade record."""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    strategy: str
    conviction: float


@dataclass
class BacktestConfig:
    """Configuration for the backtest."""
    assets: list[str] = field(default_factory=lambda: [
        "BTC", "ETH", "BNB", "XRP", "SOL", "AVAX", "NEAR", "APT",
        "AAVE", "UNI", "LINK", "FET", "DOGE", "PAXG", "SPX",
    ])
    days: int = 90
    initial_balance: float = 10000.0

    # Dynamic leverage (reduces during emergencies, otherwise uses base)
    base_leverage: float = 5.0
    min_leverage: float = 1.0
    max_leverage: float = 10.0

    # Position sizing
    max_portfolio_exposure: float = 0.40  # 40% max total
    max_single_position: float = 0.05     # 5% max per position

    # Trading costs
    slippage_bps: float = 5.0
    taker_fee: float = 0.0005

    # Signal thresholds
    min_conviction: float = 50.0
    ml_long_threshold: float = 0.55   # Relaxed from 0.60
    ml_short_threshold: float = 0.45  # Relaxed from 0.40


@dataclass
class Metrics:
    """Performance metrics."""
    total_return: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    final_balance: float = 0.0

    # Strategy breakdown
    ml_trades: int = 0
    ml_pnl: float = 0.0

    # Long vs Short breakdown
    long_trades: int = 0
    long_pnl: float = 0.0
    long_win_rate: float = 0.0
    short_trades: int = 0
    short_pnl: float = 0.0
    short_win_rate: float = 0.0


class AdaptiveSignalTracker:
    """
    Track signal performance and adapt thresholds dynamically.

    Simons principle: "Track signal decay religiously"
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.long_trades: list[float] = []  # PnL of recent LONG trades
        self.short_trades: list[float] = []  # PnL of recent SHORT trades
        self.regime_accuracy: dict[str, list[bool]] = {
            "BULLISH": [], "BEARISH": [], "RANGING": []
        }

    def record_trade(self, direction: str, pnl: float, regime: str) -> None:
        """Record a completed trade for learning."""
        if direction == "LONG":
            self.long_trades.append(pnl)
            if len(self.long_trades) > self.lookback:
                self.long_trades.pop(0)
        else:
            self.short_trades.append(pnl)
            if len(self.short_trades) > self.lookback:
                self.short_trades.pop(0)

        # Track regime prediction accuracy
        was_correct = pnl > 0
        self.regime_accuracy[regime].append(was_correct)
        if len(self.regime_accuracy[regime]) > self.lookback:
            self.regime_accuracy[regime].pop(0)

    def get_direction_confidence(self, direction: str) -> float:
        """
        Get confidence multiplier for a direction based on recent performance.
        Returns 0.5 to 1.5 multiplier.
        """
        trades = self.long_trades if direction == "LONG" else self.short_trades
        if len(trades) < 5:
            return 1.0  # Not enough data

        win_rate = len([t for t in trades if t > 0]) / len(trades)
        avg_pnl = sum(trades) / len(trades)

        # High win rate + positive PnL = boost confidence
        # Low win rate + negative PnL = reduce confidence
        if win_rate > 0.5 and avg_pnl > 0:
            return min(1.5, 1.0 + win_rate - 0.5)
        elif win_rate < 0.4 or avg_pnl < 0:
            return max(0.5, win_rate)
        return 1.0

    def get_conviction_adjustment(self, direction: str) -> float:
        """
        Get conviction threshold adjustment based on recent performance.
        Positive = require higher conviction (signal struggling)
        Negative = allow lower conviction (signal working well)
        """
        trades = self.long_trades if direction == "LONG" else self.short_trades
        if len(trades) < 10:
            return 0.0

        recent_win_rate = len([t for t in trades[-10:] if t > 0]) / 10

        if recent_win_rate > 0.6:
            return -5.0  # Signal working well, lower bar
        elif recent_win_rate < 0.35:
            return 10.0  # Signal struggling, raise bar
        elif recent_win_rate < 0.45:
            return 5.0
        return 0.0


class MedallionV2Engine:
    """
    Medallion V2 backtest engine - ADAPTIVE VERSION.

    Strategy:
    - LightGBM ML predictions with adaptive thresholds
    - Regime-aware but trades BOTH directions with adaptive conviction
    - Signal performance tracking and decay
    - Volatility-adaptive position sizing and stops
    - Dynamic leverage based on market conditions

    Simons Principles:
    - "Many weak signals > one strong signal"
    - "Track signal decay religiously"
    - "Adapt to changing market conditions"
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []

        # Dynamic leverage state
        self.current_leverage = config.base_leverage
        self.leverage_history: list[tuple[datetime, float, str]] = []

        # Position tracking
        self.positions: dict[str, Position] = {}

        # ADAPTIVE: Signal performance tracker
        self.signal_tracker = AdaptiveSignalTracker(lookback=50)

        # ADAPTIVE: Volatility tracking
        self.current_volatility = 0.02  # Default 2%
        self.volatility_regime = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME

        # Components (lazy init)
        self.client = None
        self.feature_pipeline = None
        self.ml_models = {}

    async def _init_components(self):
        """Initialize trading components."""
        from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient
        from src.features.pipeline import FeaturePipeline
        from src.ml.lightgbm_model import LightGBMModel

        self.client = HyperliquidRESTClient()
        await self.client.initialize()

        self.feature_pipeline = FeaturePipeline()

        # Load LightGBM models (proven superior to ensemble for crypto)
        models_dir = Path("models")
        for asset in self.config.assets:
            symbol = f"{asset}USDT"
            lgb_path = models_dir / f"lightgbm_hyperliquid_{asset.lower()}.txt"
            if lgb_path.exists():
                try:
                    model = LightGBMModel()
                    model.load(lgb_path)
                    self.ml_models[symbol] = model
                except Exception as e:
                    logger.warning(f"Failed to load model for {symbol}: {e}")

        logger.info(f"Loaded {len(self.ml_models)} LightGBM models")

    def _get_ml_signal(
        self,
        symbol: str,
        features: dict,
    ) -> tuple[str, float]:
        """Get LightGBM prediction (best performer for crypto)."""
        if symbol not in self.ml_models or not features:
            return "HOLD", 0.0

        try:
            feature_names = self.feature_pipeline.feature_names
            X = np.array([[features.get(name, 0.0) for name in feature_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            pred = self.ml_models[symbol].predict(X)[0]

            # Relaxed thresholds for more signals
            if pred > self.config.ml_long_threshold:
                confidence = min(100, (pred - 0.5) * 250)
                return "LONG", confidence
            elif pred < self.config.ml_short_threshold:
                confidence = min(100, (0.5 - pred) * 250)
                return "SHORT", confidence

            return "HOLD", 0.0
        except Exception:
            return "HOLD", 0.0

    def _detect_market_regime(
        self,
        btc_candles: list,
        lookback: int = 50,
    ) -> tuple[str, float]:
        """
        Detect market regime using BTC as the market leader.

        Returns:
            (regime, strength): regime is BULLISH/BEARISH/RANGING, strength 0-100
        """
        if len(btc_candles) < lookback:
            return "RANGING", 50.0

        closes = np.array([c.close for c in btc_candles[-lookback:]])

        # Calculate multiple trend indicators
        # 1. Price vs moving averages
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes)
        current_price = closes[-1]

        # 2. Trend direction (linear regression slope)
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        slope_pct = slope / closes[0] * 100  # Normalize as percentage

        # 3. Recent momentum (last 10 vs previous 10)
        recent_avg = np.mean(closes[-10:])
        prev_avg = np.mean(closes[-20:-10])
        momentum = (recent_avg - prev_avg) / prev_avg * 100

        # Determine regime
        bullish_signals = 0
        bearish_signals = 0

        if current_price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if current_price > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if sma_20 > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if slope_pct > 0.1:  # Upward trend
            bullish_signals += 2
        elif slope_pct < -0.1:  # Downward trend
            bearish_signals += 2

        if momentum > 1:  # Strong upward momentum
            bullish_signals += 1
        elif momentum < -1:  # Strong downward momentum
            bearish_signals += 1

        # Calculate regime and strength
        total_signals = bullish_signals + bearish_signals
        if bullish_signals >= 4:
            regime = "BULLISH"
            strength = min(100, bullish_signals / total_signals * 100 + 20)
        elif bearish_signals >= 4:
            regime = "BEARISH"
            strength = min(100, bearish_signals / total_signals * 100 + 20)
        else:
            regime = "RANGING"
            strength = 50.0

        return regime, strength

    def _calculate_exposure(self, balance: float) -> float:
        """Calculate current portfolio exposure."""
        return sum(p.size / balance for p in self.positions.values())

    def _update_volatility_regime(self, btc_candles: list) -> None:
        """
        Update volatility regime for adaptive position sizing and stops.

        Simons principle: Adapt to market conditions
        """
        if len(btc_candles) < 20:
            return

        closes = [c.close for c in btc_candles[-20:]]
        highs = [c.high for c in btc_candles[-20:]]
        lows = [c.low for c in btc_candles[-20:]]

        # Calculate ATR-based volatility
        tr_values = []
        for j in range(1, len(closes)):
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j-1]),
                abs(lows[j] - closes[j-1])
            )
            tr_values.append(tr)

        atr = sum(tr_values) / len(tr_values)
        self.current_volatility = atr / closes[-1]

        # Classify volatility regime
        if self.current_volatility < 0.015:
            self.volatility_regime = "LOW"
        elif self.current_volatility < 0.03:
            self.volatility_regime = "NORMAL"
        elif self.current_volatility < 0.05:
            self.volatility_regime = "HIGH"
        else:
            self.volatility_regime = "EXTREME"

    def _get_adaptive_stop_loss(self) -> float:
        """
        Get volatility-adjusted stop loss.

        Low vol: Tighter stops (0.8%)
        Normal: Standard (1%)
        High vol: Wider stops (1.5%)
        Extreme: Very wide (2%) or no trade
        """
        base_stop = 0.01  # 1%

        if self.volatility_regime == "LOW":
            return base_stop * 0.8
        elif self.volatility_regime == "NORMAL":
            return base_stop
        elif self.volatility_regime == "HIGH":
            return base_stop * 1.5
        else:  # EXTREME
            return base_stop * 2.0

    def _get_adaptive_take_profit(self) -> float:
        """
        Get volatility-adjusted take profit.

        Low vol: Smaller targets (3%)
        Normal: Standard (4%)
        High vol: Larger targets (5%)
        """
        base_tp = 0.04  # 4%

        if self.volatility_regime == "LOW":
            return base_tp * 0.75
        elif self.volatility_regime == "NORMAL":
            return base_tp
        elif self.volatility_regime == "HIGH":
            return base_tp * 1.25
        else:  # EXTREME
            return base_tp * 1.5

    def _get_adaptive_position_size(
        self,
        balance: float,
        current_exposure: float,
        conviction: float,
        direction: str,
    ) -> float:
        """
        Adaptive position sizing based on:
        - Conviction level
        - Volatility regime
        - Recent direction performance
        - Current exposure

        Simons principle: Size based on edge, not fixed percentage
        """
        # Base size from conviction
        base_size = min(
            balance * self.config.max_single_position,
            balance * (1 - current_exposure) * 0.5,
        ) * (conviction / 100)

        # Volatility adjustment (from Simons protocol)
        if self.volatility_regime == "LOW":
            vol_multiplier = 1.2  # Can size up in calm markets
        elif self.volatility_regime == "NORMAL":
            vol_multiplier = 1.0
        elif self.volatility_regime == "HIGH":
            vol_multiplier = 0.6  # Reduce in volatile markets
        else:  # EXTREME
            vol_multiplier = 0.3  # Minimal sizing

        # Direction performance adjustment
        direction_confidence = self.signal_tracker.get_direction_confidence(direction)

        return base_size * vol_multiplier * direction_confidence

    def _update_dynamic_leverage(
        self,
        btc_candles: list,
        balance: float,
        peak_balance: float,
        timestamp: datetime,
    ) -> None:
        """
        Update dynamic leverage based on volatility and drawdown.

        Conservative approach: only reduce during emergencies.
        """
        if len(btc_candles) < 20:
            return

        # Calculate current volatility
        closes = [c.close for c in btc_candles[-20:]]
        highs = [c.high for c in btc_candles[-20:]]
        lows = [c.low for c in btc_candles[-20:]]

        tr_values = []
        for j in range(1, len(closes)):
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j-1]),
                abs(lows[j] - closes[j-1])
            )
            tr_values.append(tr)
        atr = sum(tr_values) / len(tr_values)
        current_vol = atr / closes[-1]

        # Calculate current drawdown
        drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0

        # CONSERVATIVE: Only reduce leverage during emergencies
        # Otherwise use base leverage to maximize returns
        new_leverage = self.config.base_leverage

        emergency_reason = None
        if drawdown > 0.10:  # 10% DD = emergency, reduce to protect capital
            new_leverage = self.config.base_leverage * 0.5
            emergency_reason = f"EMERGENCY DD ({drawdown:.1%})"
        elif current_vol > 0.05:  # 5% volatility = extreme, reduce exposure
            new_leverage = self.config.base_leverage * 0.6
            emergency_reason = f"extreme volatility ({current_vol:.1%})"

        # Clamp to valid range
        new_leverage = max(self.config.min_leverage, min(self.config.max_leverage, new_leverage))

        # Only log changes
        if abs(new_leverage - self.current_leverage) / self.current_leverage > 0.10:
            old_lev = self.current_leverage
            self.current_leverage = new_leverage
            if emergency_reason:
                self.leverage_history.append((timestamp, new_leverage, emergency_reason))
                logger.debug(f"Leverage: {old_lev:.1f}x → {new_leverage:.1f}x ({emergency_reason})")

    async def run(self) -> Metrics:
        """Run the full backtest."""
        await self._init_components()

        # Fetch candles
        all_candles = await self._fetch_candles()
        if not all_candles:
            return Metrics()

        logger.info(f"Running Medallion V2 backtest with {len(all_candles)} assets...")

        # Pre-calculate features
        all_features = self._precalculate_features(all_candles)

        # Initialize state
        balance = self.config.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        trade_id = 0

        min_len = min(len(c) for c in all_candles.values())
        warmup = 150

        if min_len < warmup + 50:
            logger.error(f"Insufficient data: {min_len} candles")
            return Metrics()

        logger.info(f"Simulation: {min_len} candles, warmup={warmup}")

        returns = []

        for i in range(warmup, min_len):
            current_candles = {
                symbol: candles[:i+1]
                for symbol, candles in all_candles.items()
            }
            timestamp = list(all_candles.values())[0][i].timestamp
            current_exposure = self._calculate_exposure(balance)

            # Update dynamic leverage every 24 hours after sufficient trade history
            if i % 24 == 0 and len(self.trades) >= 20:
                btc_candles = current_candles.get("BTCUSDT", [])
                self._update_dynamic_leverage(btc_candles, balance, peak_balance, timestamp)

            # === CHECK EXITS ===
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                candles = current_candles.get(symbol, [])
                if not candles:
                    continue

                price = candles[-1].close
                holding = i - pos.entry_candle_idx

                if pos.direction == "LONG":
                    pnl_pct = (price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - price) / pos.entry_price

                # =================================================================
                # REGIME-ADAPTIVE EXIT LOGIC
                # =================================================================
                # - BULLISH: Standard exits (let winners run)
                # - BEARISH: Tighter exits (preserve capital)
                # - Trailing stops adapt to volatility
                # =================================================================

                # Get current regime for adaptive exits
                current_regime, _ = self._detect_market_regime(
                    current_candles.get("BTCUSDT", [])
                )

                # Base stops
                stop_loss = self._get_adaptive_stop_loss()
                take_profit = self._get_adaptive_take_profit()
                R_UNIT = stop_loss

                # BEARISH REGIME: Tighter exits to preserve capital
                if current_regime == "BEARISH":
                    stop_loss *= 0.8   # 20% tighter stop
                    take_profit *= 0.7  # Take profits earlier
                    max_holding = 24   # Exit faster (24h vs 36h)
                else:
                    max_holding = 36   # Standard holding period

                # Update max PnL tracking
                if pnl_pct > pos.max_pnl_pct:
                    pos.max_pnl_pct = pnl_pct

                # Trailing stop activation
                # BULLISH: Activate after +2R (let winners run)
                # BEARISH: Activate after +1R (lock profits quickly)
                trailing_activation = R_UNIT if current_regime == "BEARISH" else 2 * R_UNIT

                if pnl_pct >= trailing_activation and not pos.trailing_stop_active:
                    pos.trailing_stop_active = True

                # Calculate trailing stop level
                if pos.trailing_stop_active:
                    # Trail at 50% of max profit (60% in BEARISH to lock more)
                    trail_pct = 0.6 if current_regime == "BEARISH" else 0.5
                    trailing_stop = pos.max_pnl_pct * trail_pct
                else:
                    trailing_stop = -stop_loss

                # Exit conditions
                should_exit = False
                exit_reason = ""

                if pnl_pct <= trailing_stop:
                    should_exit = True
                    exit_reason = "trailing_stop" if pos.trailing_stop_active else "stop_loss"
                elif pnl_pct >= take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
                elif holding >= max_holding:
                    should_exit = True
                    exit_reason = "time_exit"

                if should_exit:
                    pnl = pos.size * pnl_pct * self.current_leverage
                    fees = pos.size * self.config.taker_fee * 2
                    net_pnl = pnl - fees
                    balance += net_pnl

                    trade_id += 1
                    trade_record = Trade(
                        id=str(trade_id),
                        symbol=symbol,
                        side=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=price,
                        size_usd=pos.size,
                        entry_time=pos.entry_time,
                        exit_time=timestamp,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct * 100,
                        strategy=pos.strategy,
                        conviction=pos.conviction,
                    )
                    self.trades.append(trade_record)

                    # =================================================================
                    # ADAPTIVE LEARNING: Record trade result for signal adjustment
                    # =================================================================
                    entry_regime = getattr(pos, 'entry_regime', 'UNKNOWN')
                    self.signal_tracker.record_trade(
                        direction=pos.direction,
                        pnl=net_pnl,
                        regime=entry_regime,
                    )

                    del self.positions[symbol]

            # === GENERATE NEW SIGNALS ===

            btc_candles = current_candles.get("BTCUSDT", [])

            # =================================================================
            # CONSERVATIVE REGIME AND VOLATILITY DETECTION
            # =================================================================
            # Simons principle: Only trade in clear market conditions

            # Update volatility regime
            self._update_volatility_regime(btc_candles)

            # Detect market regime
            regime, regime_conf = self._detect_market_regime(btc_candles)

            # STRICT FILTERING:
            # 1. Skip RANGING regime - market drifts up, signals are noise
            # 2. Require strong regime confidence (60%+)
            # 3. Skip extreme volatility
            if regime == "RANGING":
                continue

            if regime_conf < 60:
                continue

            if self.volatility_regime == "EXTREME":
                continue

            # Process ML signals
            for symbol, candles in current_candles.items():
                if symbol in self.positions:
                    continue

                if current_exposure >= self.config.max_portfolio_exposure:
                    break

                if len(candles) < 50:
                    continue

                # Get features
                feature_idx = i - warmup
                symbol_features = all_features.get(symbol, [])
                features = symbol_features[feature_idx] if 0 <= feature_idx < len(symbol_features) else None

                # Get ML signal
                ml_direction, ml_conv = self._get_ml_signal(symbol, features)

                # High-selectivity ML strategy - quality over quantity
                direction = "HOLD"
                conviction = 0.0
                strategy = ""

                # Calculate momentum indicators
                mom_12h = 0.0
                mom_24h = 0.0
                volatility = 0.0
                if len(candles) >= 48:
                    recent_returns = []
                    for j in range(1, min(48, len(candles))):
                        ret = (candles[-j].close - candles[-j-1].close) / candles[-j-1].close
                        recent_returns.append(ret)
                    mom_12h = sum(recent_returns[:12]) if len(recent_returns) >= 12 else 0
                    mom_24h = sum(recent_returns[:24]) if len(recent_returns) >= 24 else 0
                    volatility = np.std(recent_returns[:24]) if len(recent_returns) >= 24 else 0

                # =================================================================
                # ADAPTIVE REGIME-BASED STRATEGY
                # =================================================================
                # SIMONS PRINCIPLE: "When conditions are unfavorable, STOP TRADING"
                #
                # PROVEN:
                # - LONG in BULLISH regime = 67.5% CAGR
                # - SHORT consistently loses money (models not trained for it)
                #
                # ADAPTIVE APPROACH:
                # - BULLISH: Trade LONG (proven edge)
                # - BEARISH: Reduce exposure, close positions early, preserve capital
                # - RANGING: Skip trading (market drifts up, signals are noise)
                #
                # This IS the Simons approach - capital preservation in bad conditions
                # =================================================================

                # Adaptive threshold based on signal performance
                long_perf_adj = self.signal_tracker.get_conviction_adjustment("LONG")

                # =================================================================
                # BULLISH REGIME: Trade LONG (Primary Strategy)
                # =================================================================
                if regime == "BULLISH" and ml_direction == "LONG":
                    # Base threshold with performance adjustment
                    long_threshold = 65 + max(0, long_perf_adj)

                    # High volatility: tighten threshold
                    if self.volatility_regime == "HIGH":
                        long_threshold += 5

                    if ml_conv >= long_threshold and mom_12h > 0.005:
                        # Tier 1: Very high conviction + strong momentum
                        if ml_conv >= 70 and mom_12h > 0.01:
                            direction = "LONG"
                            conviction = min(100, ml_conv + 10)
                            strategy = "ml_tier1_long"
                        # Tier 2: High conviction + positive momentum
                        elif ml_conv >= long_threshold and mom_12h > 0.005 and mom_24h > 0.01:
                            direction = "LONG"
                            conviction = ml_conv
                            strategy = "ml_tier2_long"
                        # Tier 3: High conviction + consistent uptrend
                        elif ml_conv >= long_threshold and mom_12h > 0 and mom_24h > 0:
                            direction = "LONG"
                            conviction = ml_conv - 5
                            strategy = "ml_tier3_long"

                # =================================================================
                # BEARISH REGIME: Capital Preservation Mode
                # =================================================================
                # DON'T SHORT - instead:
                # 1. Close existing LONG positions faster (handled in exit logic)
                # 2. Wait for regime to turn BULLISH
                # 3. Only take LONG if ML is EXTREMELY confident (80%+) - reversal play
                # =================================================================
                elif regime == "BEARISH" and ml_direction == "LONG":
                    # Only take LONG in BEARISH if VERY high confidence reversal
                    # This catches major bottoms but is very selective
                    reversal_threshold = 80  # Very high bar

                    # Require ML confidence AND positive momentum divergence
                    # (price falling but momentum starting to turn)
                    momentum_divergence = mom_12h > -0.005 and mom_24h < -0.01

                    if ml_conv >= reversal_threshold and momentum_divergence:
                        direction = "LONG"
                        conviction = ml_conv - 10  # Reduced conviction for counter-trend
                        strategy = "reversal_long_bearish"

                if direction == "HOLD" or conviction < self.config.min_conviction:
                    continue

                # ADAPTIVE position sizing
                size = self._get_adaptive_position_size(
                    balance, current_exposure, conviction, direction
                )

                if size < 50:
                    continue

                price = candles[-1].close
                if direction == "LONG":
                    price *= (1 + self.config.slippage_bps / 10000)
                else:
                    price *= (1 - self.config.slippage_bps / 10000)

                self.positions[symbol] = Position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=price,
                    size=size,
                    entry_time=timestamp,
                    entry_candle_idx=i,
                    strategy=strategy,
                    conviction=conviction,
                )
                # Store regime for learning when trade closes
                self.positions[symbol].entry_regime = regime  # type: ignore
                current_exposure += size / balance

            # Track equity
            self.equity_curve.append((timestamp, balance))

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance
            if dd > max_drawdown:
                max_drawdown = dd

            if len(self.equity_curve) > 1:
                prev = self.equity_curve[-2][1]
                returns.append((balance - prev) / prev)

        # Close remaining positions
        for pos in list(self.positions.values()):
            candles = all_candles.get(pos.symbol, [])
            if candles:
                price = candles[-1].close
                if pos.direction == "LONG":
                    pnl_pct = (price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - price) / pos.entry_price
                pnl = pos.size * pnl_pct * self.current_leverage
                fees = pos.size * self.config.taker_fee * 2
                balance += pnl - fees

        if self.client:
            await self.client.close()

        return self._calculate_metrics(balance, max_drawdown, returns)

    async def _fetch_candles(self) -> dict:
        """Fetch candles for all assets."""
        logger.info(f"Fetching candles for {len(self.config.assets)} assets...")
        all_candles = {}

        for asset in self.config.assets:
            symbol = f"{asset}USDT"
            try:
                limit = self.config.days * 24 + 200
                candles = await self.client.get_candles(
                    symbol=symbol,
                    interval="1h",
                    limit=limit,
                )
                if candles:
                    all_candles[symbol] = candles
                    logger.info(f"  {symbol}: {len(candles)} candles")
            except Exception as e:
                logger.warning(f"  {symbol}: Failed - {e}")

        return all_candles

    def _precalculate_features(self, all_candles: dict) -> dict:
        """Pre-calculate features for all assets."""
        logger.info("Pre-calculating features...")
        all_features = {}
        min_window = self.feature_pipeline.config.min_candles

        for symbol, candles in all_candles.items():
            if len(candles) < min_window + 10:
                continue

            features_list = []
            for i in range(min_window, len(candles)):
                window = candles[max(0, i - min_window):i + 1]
                try:
                    features = self.feature_pipeline.calculate_features(window, use_cache=False)
                    features_list.append(features)
                except Exception:
                    features_list.append(None)

            all_features[symbol] = features_list

        return all_features

    def _calculate_metrics(
        self,
        final_balance: float,
        max_drawdown: float,
        returns: list,
    ) -> Metrics:
        """Calculate performance metrics."""
        m = Metrics()
        m.total_return = (final_balance - self.config.initial_balance) / self.config.initial_balance
        m.final_balance = final_balance
        m.max_drawdown = max_drawdown
        m.total_trades = len(self.trades)

        hours = len(returns)
        years = hours / (24 * 365)
        if years > 0 and final_balance > 0:
            m.cagr = (final_balance / self.config.initial_balance) ** (1 / years) - 1

        if returns and np.std(returns) > 0:
            m.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(24 * 365))

        neg_rets = [r for r in returns if r < 0]
        if neg_rets and np.std(neg_rets) > 0:
            m.sortino_ratio = float(np.mean(returns) / np.std(neg_rets) * np.sqrt(24 * 365))

        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]
            m.win_rate = len(wins) / len(self.trades)

            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 1
            m.profit_factor = total_wins / total_losses if total_losses > 0 else 0

            # All trades are ML trades
            m.ml_trades = len(self.trades)
            m.ml_pnl = sum(t.pnl for t in self.trades)

            # Long vs Short breakdown
            long_trades = [t for t in self.trades if t.side == "LONG"]
            short_trades = [t for t in self.trades if t.side == "SHORT"]

            m.long_trades = len(long_trades)
            m.long_pnl = sum(t.pnl for t in long_trades)
            m.long_win_rate = len([t for t in long_trades if t.pnl > 0]) / len(long_trades) if long_trades else 0.0

            m.short_trades = len(short_trades)
            m.short_pnl = sum(t.pnl for t in short_trades)
            m.short_win_rate = len([t for t in short_trades if t.pnl > 0]) / len(short_trades) if short_trades else 0.0

        return m


def print_results(config: BacktestConfig, metrics: Metrics, trades: list):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("MEDALLION V2 BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Assets: {len(config.assets)}")
    print(f"  Period: {config.days} days")
    print(f"  Initial: ${config.initial_balance:,.0f}")
    print(f"  Leverage: {config.base_leverage:.0f}x base (dynamic: {config.min_leverage:.0f}x-{config.max_leverage:.0f}x)")

    print(f"\n{'=' * 70}")
    print("PERFORMANCE METRICS")
    print("=" * 70)

    cagr_ok = "OK" if metrics.cagr >= 0.66 else "X"
    dd_ok = "OK" if metrics.max_drawdown <= 0.05 else "X"
    sharpe_ok = "OK" if metrics.sharpe_ratio >= 2.5 else "X"

    print(f"\n{'Metric':<25} {'Value':>15} {'Target':>15} {'Status':>8}")
    print("-" * 65)
    print(f"{'CAGR':<25} {metrics.cagr:>14.1%} {'>66%':>15} {cagr_ok:>8}")
    print(f"{'Max Drawdown':<25} {metrics.max_drawdown:>14.1%} {'<5%':>15} {dd_ok:>8}")
    print(f"{'Sharpe Ratio':<25} {metrics.sharpe_ratio:>14.2f} {'>2.5':>15} {sharpe_ok:>8}")
    print(f"{'Sortino Ratio':<25} {metrics.sortino_ratio:>14.2f}")
    print(f"{'Win Rate':<25} {metrics.win_rate:>14.1%}")
    print(f"{'Profit Factor':<25} {metrics.profit_factor:>14.2f}")
    print(f"{'Total Trades':<25} {metrics.total_trades:>14}")

    print(f"\n{'=' * 70}")
    print("STRATEGY BREAKDOWN")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'Trades':>10} {'Win Rate':>12} {'P&L':>15}")
    print("-" * 60)
    print(f"{'ML Signals (Total)':<20} {metrics.ml_trades:>10} {metrics.win_rate:>11.1%} ${metrics.ml_pnl:>+13,.2f}")
    print(f"{'  └─ LONG':<20} {metrics.long_trades:>10} {metrics.long_win_rate:>11.1%} ${metrics.long_pnl:>+13,.2f}")
    print(f"{'  └─ SHORT':<20} {metrics.short_trades:>10} {metrics.short_win_rate:>11.1%} ${metrics.short_pnl:>+13,.2f}")

    print(f"\n{'=' * 70}")
    print("PORTFOLIO")
    print("=" * 70)
    print(f"  Initial: ${config.initial_balance:,.2f}")
    print(f"  Final:   ${metrics.final_balance:,.2f}")
    print(f"  Return:  {metrics.total_return:+.1%}")

    if trades:
        print(f"\n{'=' * 70}")
        print("RECENT TRADES (last 15)")
        print("=" * 70)
        for t in trades[-15:]:
            pnl_str = f"+${t.pnl:,.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):,.2f}"
            print(f"  {t.entry_time.strftime('%m-%d %H:%M')} {t.symbol:<15} {t.side:<6} {t.strategy:<12} {pnl_str:>12}")

    print("\n" + "=" * 70)
    targets = sum([metrics.cagr >= 0.66, metrics.max_drawdown <= 0.05, metrics.sharpe_ratio >= 2.5])
    if targets == 3:
        print("MEDALLION TARGETS ACHIEVED!")
    elif targets >= 2:
        print("STRONG PERFORMANCE - 2/3 targets met")
    elif targets >= 1:
        print("DECENT - 1/3 targets met, more tuning needed")
    else:
        print("BELOW TARGETS - significant tuning needed")
    print("=" * 70 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Medallion V2 Backtest (Architecture-Compliant)")
    parser.add_argument("--days", type=int, default=180, help="Backtest period in days")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--base-leverage", type=float, default=5.0, help="Base leverage for dynamic calc")
    parser.add_argument("--min-leverage", type=float, default=1.0, help="Min leverage (high-risk floor)")
    parser.add_argument("--max-leverage", type=float, default=10.0, help="Max leverage (favorable ceiling)")
    parser.add_argument("--min-conviction", type=float, default=50, help="Min ML conviction threshold")
    args = parser.parse_args()

    config = BacktestConfig(
        days=args.days,
        initial_balance=args.balance,
        base_leverage=args.base_leverage,
        min_leverage=args.min_leverage,
        max_leverage=args.max_leverage,
        min_conviction=args.min_conviction,
    )

    engine = MedallionV2Engine(config)
    metrics = await engine.run()
    print_results(config, metrics, engine.trades)


if __name__ == "__main__":
    asyncio.run(main())
