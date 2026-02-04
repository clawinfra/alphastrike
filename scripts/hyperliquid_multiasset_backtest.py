#!/usr/bin/env python3
"""
Hyperliquid Multi-Asset Backtest - Medallion-Style Diversification v2

IMPROVED VERSION with:
- Lower conviction thresholds for more trades
- Bollinger Band mean reversion signals
- Volume-weighted momentum
- Regime-adaptive position sizing
- Kelly criterion position sizing
- Separate parameters for low-vol assets (PAXG)

Key insight: Crypto assets are 0.7-0.85 correlated.
PAXG (gold-backed) is ~0.1 correlated to BTC - key diversifier.

Target: 66%+ annual returns with <5% max drawdown (Medallion benchmark)
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# MULTI-ASSET CORRELATION MATRIX
# ============================================================================
# The secret sauce: uncorrelated assets reduce portfolio volatility
# Source: Historical correlations from 2023-2024

ASSET_CORRELATIONS = {
    # ==========================================================================
    # CRYPTO MAJORS (highly correlated: 0.75-0.85)
    # ==========================================================================
    ("BTC", "ETH"): 0.85,
    ("BTC", "SOL"): 0.75,
    ("ETH", "SOL"): 0.80,

    # ==========================================================================
    # DEFI (moderate-high correlation to ETH: 0.65-0.75)
    # ==========================================================================
    ("BTC", "AAVE"): 0.65,
    ("ETH", "AAVE"): 0.72,  # DeFi closer to ETH
    ("SOL", "AAVE"): 0.60,

    # ==========================================================================
    # PAXG (Gold-backed) - KEY DIVERSIFIER
    # Very low correlation to crypto - safe haven during risk-off
    # Historical correlation: ~0.05-0.15 to BTC
    # ==========================================================================
    ("BTC", "PAXG"): 0.08,   # Near zero - this is the magic
    ("ETH", "PAXG"): 0.06,
    ("SOL", "PAXG"): 0.05,
    ("AAVE", "PAXG"): 0.10,
}


def get_correlation(asset1: str, asset2: str) -> float:
    """Get correlation between two assets."""
    if asset1 == asset2:
        return 1.0

    key = (asset1, asset2)
    reverse_key = (asset2, asset1)

    if key in ASSET_CORRELATIONS:
        return ASSET_CORRELATIONS[key]
    if reverse_key in ASSET_CORRELATIONS:
        return ASSET_CORRELATIONS[reverse_key]

    # Default: assume low correlation for unknown pairs
    return 0.20


# ============================================================================
# ASSET CLASS DEFINITIONS
# ============================================================================

@dataclass
class AssetConfig:
    """Configuration for each asset class."""
    symbol: str           # Hyperliquid symbol (e.g., "BTC", "GOLD")
    asset_class: str      # "crypto", "commodity", "forex", "index", "stock"
    volatility_mult: float = 1.0  # Adjust position size based on asset vol
    min_move_pct: float = 0.001   # Minimum move to consider significant
    typical_spread_bps: float = 2.0  # Typical bid-ask spread


# Asset universe - DIVERSIFIED CRYPTO PORTFOLIO (Medallion-style)
# Goal: Maximize diversification WITHIN crypto using uncorrelated sectors
#
# Hyperliquid is crypto-only, so we diversify by:
# 1. Asset type (majors, L1s, DeFi, memes)
# 2. PAXG (gold-backed) - ~0.08 correlation to BTC = KEY DIVERSIFIER
# 3. Different volatility profiles
ASSET_UNIVERSE = [
    # ==========================================================================
    # MAJORS (40% weight) - High liquidity, lower volatility
    # ==========================================================================
    AssetConfig("BTC", "crypto_major", volatility_mult=1.0, min_move_pct=0.002, typical_spread_bps=1.0),
    AssetConfig("ETH", "crypto_major", volatility_mult=1.0, min_move_pct=0.003, typical_spread_bps=1.5),

    # ==========================================================================
    # PAXG - GOLD-BACKED (20% weight) - KEY DIVERSIFIER
    # ~0.08 correlation to BTC - this is what reduces portfolio volatility
    # ==========================================================================
    AssetConfig("PAXG", "gold_proxy", volatility_mult=0.3, min_move_pct=0.001, typical_spread_bps=2.0),

    # ==========================================================================
    # LAYER 1s (20% weight) - Higher beta, trend followers
    # ==========================================================================
    AssetConfig("SOL", "crypto_l1", volatility_mult=1.2, min_move_pct=0.005, typical_spread_bps=2.0),

    # ==========================================================================
    # DEFI (20% weight) - Moderate correlation, fundamentals-driven
    # ==========================================================================
    AssetConfig("AAVE", "crypto_defi", volatility_mult=1.0, min_move_pct=0.005, typical_spread_bps=3.0),
]


# ============================================================================
# POSITION AND TRADE TRACKING
# ============================================================================

@dataclass
class Position:
    """Open position."""
    symbol: str
    asset_class: str
    entry_time: datetime
    entry_price: float
    side: Literal["LONG", "SHORT"]
    size_usd: float
    stop_loss: float
    take_profit: float
    conviction: float
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    trail_activated: bool = False


@dataclass
class Trade:
    """Completed trade."""
    symbol: str
    asset_class: str
    entry_time: datetime
    exit_time: datetime
    side: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    conviction: float
    exit_reason: str


@dataclass
class PortfolioState:
    """Current portfolio state."""
    balance: float
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    peak_balance: float = 0.0
    current_drawdown: float = 0.0


# ============================================================================
# MULTI-ASSET BACKTESTER
# ============================================================================

class HyperliquidMultiAssetBacktest:
    """
    Multi-asset backtest using Hyperliquid API.

    Key features:
    1. Fetches real candles from Hyperliquid for all asset classes
    2. Correlation-aware position sizing
    3. Cross-asset signals (e.g., BTC leads alts, GOLD inverse to risk)
    4. Portfolio heat tracking to limit correlated exposure
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_portfolio_heat: float = 0.40,  # Max 40% correlated exposure
        max_single_position: float = 0.20,  # Max 20% per position (concentrated)
        leverage: int = 7,  # Moderate-high leverage
    ):
        self.initial_balance = initial_balance
        self.max_portfolio_heat = max_portfolio_heat
        self.max_single_position = max_single_position
        self.leverage = leverage

        # State
        self.portfolio = PortfolioState(
            balance=initial_balance,
            peak_balance=initial_balance,
        )
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []

        # Candle data storage
        self.candles: dict[str, list[dict]] = {}

        # Performance tracking per asset
        self.asset_performance: dict[str, dict] = {}

    async def fetch_candles(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 180,
    ) -> list[dict]:
        """Fetch historical candles from Hyperliquid."""
        from src.exchange.adapters.hyperliquid import HyperliquidAdapter

        adapter = HyperliquidAdapter()
        await adapter.initialize()

        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)

            logger.info(f"Fetching {symbol} candles from Hyperliquid...")

            # Hyperliquid uses coin names like "BTC" not "BTCUSDT"
            unified_symbol = f"{symbol}USDT"

            candles = await adapter.rest.get_candles(
                symbol=unified_symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=days * 24,  # 24 candles per day for 1h
            )

            # Convert to dict format
            candle_dicts = []
            for c in candles:
                candle_dicts.append({
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                })

            logger.info(f"  {symbol}: {len(candle_dicts)} candles fetched")
            return candle_dicts

        finally:
            await adapter.close()

    async def fetch_all_candles(self, days: int = 180) -> None:
        """Fetch candles for all assets in universe."""
        from src.exchange.adapters.hyperliquid import HyperliquidAdapter

        adapter = HyperliquidAdapter()
        await adapter.initialize()

        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)

            for asset in ASSET_UNIVERSE:
                try:
                    unified_symbol = f"{asset.symbol}USDT"
                    logger.info(f"Fetching {asset.symbol} ({asset.asset_class})...")

                    candles = await adapter.rest.get_candles(
                        symbol=unified_symbol,
                        interval="1h",
                        start_time=start_time,
                        end_time=end_time,
                        limit=days * 24,
                    )

                    candle_dicts = []
                    for c in candles:
                        candle_dicts.append({
                            "timestamp": c.timestamp,
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "volume": c.volume,
                        })

                    self.candles[asset.symbol] = candle_dicts
                    logger.info(f"  {asset.symbol}: {len(candle_dicts)} candles")

                except Exception as e:
                    logger.warning(f"  {asset.symbol}: Failed to fetch - {e}")

        finally:
            await adapter.close()

    def calculate_portfolio_heat(self) -> float:
        """
        Calculate correlation-adjusted portfolio exposure.

        If we're long BTC and ETH (correlation 0.85), the combined risk
        is higher than if we're long BTC and GOLD (correlation 0.10).
        """
        if not self.portfolio.positions:
            return 0.0

        positions = list(self.portfolio.positions.values())
        n = len(positions)

        if n == 1:
            return positions[0].size_usd / self.portfolio.balance

        # Build correlation-weighted exposure
        total_heat = 0.0
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                corr = get_correlation(pos1.symbol, pos2.symbol)

                # Same direction positions add risk
                # Opposite direction positions hedge
                if pos1.side == pos2.side:
                    heat_contrib = corr * (pos1.size_usd / self.portfolio.balance) * (pos2.size_usd / self.portfolio.balance)
                else:
                    heat_contrib = -corr * (pos1.size_usd / self.portfolio.balance) * (pos2.size_usd / self.portfolio.balance)

                total_heat += heat_contrib

        return np.sqrt(max(0, total_heat))

    def calculate_signals(self, symbol: str, candles: list[dict], idx: int) -> dict:
        """
        Calculate trading signals for an asset.

        IMPROVED v2: More signals, lower thresholds, better for frequent trading:
        1. Trend following (EMA crossover + MACD)
        2. Mean reversion (Bollinger Bands + RSI)
        3. Momentum (ROC + volume-weighted)
        4. Volatility regime adaptive
        5. Cross-asset signals (BTC leads, inverse PAXG)
        """
        if idx < 50:  # Need history for indicators
            return {"signal": 0, "conviction": 0}

        # Get recent prices and volume
        closes = [c["close"] for c in candles[idx-50:idx+1]]
        highs = [c["high"] for c in candles[idx-50:idx+1]]
        lows = [c["low"] for c in candles[idx-50:idx+1]]
        volumes = [c["volume"] for c in candles[idx-50:idx+1]]

        # Asset-specific parameters
        asset_cfg = next((a for a in ASSET_UNIVERSE if a.symbol == symbol), None)
        asset_class = asset_cfg.asset_class if asset_cfg else "crypto"
        # Low volatility assets: commodities, forex have different dynamics
        is_low_vol = asset_class in ("commodity", "forex")

        # 1. TREND: EMA crossover with MACD confirmation
        ema_fast = self._ema(closes, 8 if is_low_vol else 12)
        ema_slow = self._ema(closes, 21 if is_low_vol else 26)
        ema_diff = (ema_fast - ema_slow) / ema_slow

        # MACD
        macd_line = ema_fast - ema_slow
        signal_line = self._ema([ema_fast - ema_slow] * 9, 9)  # Simplified
        macd_signal = 1 if macd_line > signal_line else -1

        # Trend signal (continuous, not binary)
        trend_signal = np.clip(ema_diff * 50, -1, 1)  # Scale EMA diff

        # 2. MEAN REVERSION: Bollinger Bands
        bb_period = 15 if is_low_vol else 20
        bb_middle = np.mean(closes[-bb_period:])
        bb_std = np.std(closes[-bb_period:])
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std

        bb_signal = 0
        if closes[-1] < bb_lower:
            bb_signal = 1.0  # Below lower band = buy
        elif closes[-1] > bb_upper:
            bb_signal = -1.0  # Above upper band = sell
        elif closes[-1] < bb_middle:
            bb_signal = 0.3  # Below middle = slight buy
        else:
            bb_signal = -0.3

        # 3. RSI for mean reversion (with adjusted thresholds for low-vol)
        rsi = self._rsi(closes, 14)
        rsi_lower = 35 if is_low_vol else 30
        rsi_upper = 65 if is_low_vol else 70

        rsi_signal = 0
        if rsi < rsi_lower:
            rsi_signal = (rsi_lower - rsi) / rsi_lower  # Stronger signal for more oversold
        elif rsi > rsi_upper:
            rsi_signal = -(rsi - rsi_upper) / (100 - rsi_upper)

        # 4. MOMENTUM: ROC with volume weighting
        roc_5 = (closes[-1] - closes[-5]) / closes[-5]
        roc_10 = (closes[-1] - closes[-10]) / closes[-10]

        # Volume ratio (recent vs average)
        vol_ratio = volumes[-1] / (np.mean(volumes[-20:]) + 1e-10)
        vol_multiplier = min(1.5, max(0.5, vol_ratio))  # Cap at 0.5-1.5x

        # Momentum combines short and medium term
        raw_momentum = roc_5 * 0.6 + roc_10 * 0.4
        momentum_threshold = 0.005 if is_low_vol else 0.015
        momentum_signal = np.clip(raw_momentum / momentum_threshold, -1, 1) * vol_multiplier

        # 5. Volatility regime
        atr = self._atr(highs, lows, closes, 14)
        atr_pct = atr / closes[-1]

        # Adaptive thresholds per asset class
        if is_low_vol:
            vol_high = 0.01
            vol_low = 0.003
        else:
            vol_high = 0.04
            vol_low = 0.015

        vol_regime = "high" if atr_pct > vol_high else ("low" if atr_pct < vol_low else "normal")

        # 6. Cross-asset signals (asset class specific)
        cross_signal = 0

        # Get BTC momentum as leading indicator
        btc_momentum = 0
        if "BTC" in self.candles and idx < len(self.candles["BTC"]):
            btc_candles = self.candles["BTC"]
            btc_closes = [c["close"] for c in btc_candles[max(0,idx-10):idx+1]]
            if len(btc_closes) >= 5:
                btc_momentum = (btc_closes[-1] - btc_closes[-5]) / btc_closes[-5]

        # CRYPTO (except BTC): BTC leads other crypto assets
        if asset_class.startswith("crypto") and symbol != "BTC":
            cross_signal = np.clip(btc_momentum * 20, -0.5, 0.5)

        # PAXG (gold proxy): Inverse correlation to crypto risk
        # When BTC drops (risk-off), go long PAXG as safe haven
        elif asset_class == "gold_proxy":
            cross_signal = -btc_momentum * 15

        # ============================================================
        # ADAPTIVE STRATEGY: Choose based on volatility regime
        # ============================================================
        # High vol = Mean Reversion (prices overextend and snap back)
        # Normal/Low vol = Trend Following (breakouts work better)

        if vol_regime == "high":
            # MEAN REVERSION in high volatility
            combined = bb_signal * 0.5 + rsi_signal * 0.3 + momentum_signal * 0.2
            strategy = "MR"
        else:
            # TREND FOLLOWING in normal/low volatility
            # Go WITH the trend, not against it
            combined = trend_signal * 0.4 + momentum_signal * 0.4 + cross_signal * 0.2
            strategy = "TF"

        # Conviction calculation
        raw_conviction = abs(combined) * 100

        # Strategy-specific boosts
        if strategy == "MR":
            # Boost for extreme readings
            if rsi < 25 or rsi > 75:
                raw_conviction *= 1.3
            if abs(bb_signal) > 0.7:
                raw_conviction *= 1.2
        else:
            # Boost for strong trend agreement
            if trend_signal * momentum_signal > 0:  # Same direction
                raw_conviction *= 1.2
            # Boost when BTC confirms for alts
            if symbol not in ("BTC", "PAXG") and trend_signal * cross_signal > 0:
                raw_conviction *= 1.15

        conviction = min(100, raw_conviction)

        # Signal threshold - higher for trend following (need clearer signal)
        if strategy == "MR":
            signal_threshold = 0.30
        else:
            signal_threshold = 0.40

        signal = 1 if combined > signal_threshold else (-1 if combined < -signal_threshold else 0)

        return {
            "signal": signal,
            "conviction": conviction,
            "atr": atr,
            "vol_regime": vol_regime,
            "trend": trend_signal,
            "bb": bb_signal,
            "momentum": momentum_signal,
            "rsi": rsi,
            "cross": cross_signal,
            "combined": combined,
        }

    def _ema(self, data: list, period: int) -> float:
        """Exponential moving average."""
        if len(data) < period:
            return data[-1]
        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _rsi(self, closes: list, period: int = 14) -> float:
        """Relative Strength Index."""
        if len(closes) < period + 1:
            return 50

        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        """Average True Range."""
        if len(closes) < period + 1:
            return (highs[-1] - lows[-1])

        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)

        return np.mean(trs[-period:])

    def run_backtest(self) -> dict:
        """
        Run the multi-asset backtest.

        Returns performance metrics.
        """
        if not self.candles:
            logger.error("No candles loaded. Call fetch_all_candles() first.")
            return {}

        # Find common time range across all assets
        min_len = min(len(c) for c in self.candles.values())
        logger.info(f"Running backtest on {min_len} candles across {len(self.candles)} assets")

        # Initialize asset performance tracking
        for symbol in self.candles:
            self.asset_performance[symbol] = {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
            }

        # Main backtest loop
        for idx in range(50, min_len):
            current_time = None

            # Check each asset for signals
            for asset in ASSET_UNIVERSE:
                if asset.symbol not in self.candles:
                    continue

                candles = self.candles[asset.symbol]
                if idx >= len(candles):
                    continue

                current_candle = candles[idx]
                current_time = current_candle["timestamp"]
                current_price = current_candle["close"]

                # Update existing positions
                self._update_positions(asset.symbol, current_candle)

                # Check for new signals
                signals = self.calculate_signals(asset.symbol, candles, idx)

                # MODERATE conviction threshold - more trades for returns
                if signals["signal"] != 0 and signals["conviction"] > 30:
                    self._evaluate_entry(
                        asset=asset,
                        price=current_price,
                        signals=signals,
                        candle=current_candle,
                    )

            # Record equity
            if current_time:
                equity = self._calculate_equity()
                self.equity_curve.append((current_time, equity))

                # Update drawdown
                if equity > self.portfolio.peak_balance:
                    self.portfolio.peak_balance = equity
                self.portfolio.current_drawdown = (
                    self.portfolio.peak_balance - equity
                ) / self.portfolio.peak_balance

        return self._calculate_metrics()

    def _update_positions(self, symbol: str, candle: dict) -> None:
        """Update positions and check for exits."""
        if symbol not in self.portfolio.positions:
            return

        pos = self.portfolio.positions[symbol]
        price = candle["close"]
        high = candle["high"]
        low = candle["low"]

        # ============================================================
        # LIQUIDATION CHECK - Prevent >100% losses
        # ============================================================
        # Calculate margin (notional / leverage)
        margin = pos.size_usd / self.leverage

        # Calculate current unrealized PnL
        if pos.side == "LONG":
            unrealized_pnl = pos.size_usd * (price - pos.entry_price) / pos.entry_price
        else:
            unrealized_pnl = pos.size_usd * (pos.entry_price - price) / pos.entry_price

        # Liquidation when loss exceeds 80% of margin (typical exchange behavior)
        if unrealized_pnl < -margin * 0.80:
            logger.warning(f"LIQUIDATION: {symbol} lost {abs(unrealized_pnl):.0f} > 80% of margin {margin:.0f}")
            self._close_position(symbol, price, "liquidation", candle["timestamp"])
            return

        # Update trailing stop tracking
        if pos.side == "LONG":
            pos.highest_price = max(pos.highest_price, high)

            # Check stop loss
            if low <= pos.stop_loss:
                self._close_position(symbol, pos.stop_loss, "stop_loss", candle["timestamp"])
                return

            # Check take profit
            if high >= pos.take_profit:
                self._close_position(symbol, pos.take_profit, "take_profit", candle["timestamp"])
                return

            # Trailing stop after 1R profit
            if not pos.trail_activated:
                r_profit = (pos.highest_price - pos.entry_price) / (pos.entry_price - pos.stop_loss)
                if r_profit >= 1.0:
                    pos.trail_activated = True
                    pos.stop_loss = pos.entry_price  # Move to breakeven
            else:
                # Trail at 50% of max profit
                trail_stop = pos.entry_price + (pos.highest_price - pos.entry_price) * 0.5
                pos.stop_loss = max(pos.stop_loss, trail_stop)

        else:  # SHORT
            pos.lowest_price = min(pos.lowest_price, low)

            if high >= pos.stop_loss:
                self._close_position(symbol, pos.stop_loss, "stop_loss", candle["timestamp"])
                return

            if low <= pos.take_profit:
                self._close_position(symbol, pos.take_profit, "take_profit", candle["timestamp"])
                return

            if not pos.trail_activated:
                r_profit = (pos.entry_price - pos.lowest_price) / (pos.stop_loss - pos.entry_price)
                if r_profit >= 1.0:
                    pos.trail_activated = True
                    pos.stop_loss = pos.entry_price
            else:
                trail_stop = pos.entry_price - (pos.entry_price - pos.lowest_price) * 0.5
                pos.stop_loss = min(pos.stop_loss, trail_stop)

    def _evaluate_entry(
        self,
        asset: AssetConfig,
        price: float,
        signals: dict,
        candle: dict,
    ) -> None:
        """
        Evaluate whether to enter a new position.

        IMPROVED v2:
        - Kelly criterion position sizing
        - More aggressive base sizing
        - Separate logic for low-vol assets (PAXG)
        - Higher max portfolio heat
        """
        symbol = asset.symbol

        # Skip if already in position
        if symbol in self.portfolio.positions:
            return

        # Check portfolio heat
        current_heat = self.calculate_portfolio_heat()
        if current_heat >= self.max_portfolio_heat:
            return

        # AGGRESSIVE drawdown response - keep trading for returns
        dd_multiplier = 1.0
        if self.portfolio.current_drawdown > 0.30:
            dd_multiplier = 0.3  # Reduce at 30% DD
        elif self.portfolio.current_drawdown > 0.20:
            dd_multiplier = 0.5  # 50% size at 20% DD
        elif self.portfolio.current_drawdown > 0.10:
            dd_multiplier = 0.75  # 75% size at 10% DD

        # ============================================================
        # KELLY CRITERION POSITION SIZING
        # ============================================================
        # Kelly formula: f = (bp - q) / b
        # where b = odds (reward/risk), p = win prob, q = 1-p

        # Estimate win probability from conviction
        # Higher conviction = higher estimated win probability
        conviction = signals["conviction"]
        base_win_prob = 0.48  # Slightly below 50% baseline
        win_prob = base_win_prob + (conviction / 100) * 0.15  # Up to 63% at max conviction
        win_prob = min(0.65, win_prob)

        # Risk/reward from ATR-based stops
        atr = signals["atr"]
        risk_pct = (atr * 1.5) / price  # 1.5 ATR stop
        reward_pct = (atr * 2.5) / price  # 2.5 ATR target
        odds = reward_pct / risk_pct  # Should be ~1.67

        # Kelly fraction - MORE CONSERVATIVE
        lose_prob = 1 - win_prob
        kelly_f = (odds * win_prob - lose_prob) / odds
        kelly_f = max(0, min(0.10, kelly_f))  # Cap at 10% max (reduced from 25%)

        # Use quarter-Kelly for safety (Medallion uses very conservative sizing)
        quarter_kelly = kelly_f * 0.25

        # Base size from Kelly
        kelly_size = self.portfolio.balance * quarter_kelly * self.leverage

        # ============================================================
        # CONCENTRATED SIZING for maximum returns (only 3 assets)
        # ============================================================
        base_pct = 0.15  # 15% base position (concentrated portfolio)
        conviction_scale = 0.6 + (conviction / 100) * 0.4  # 0.6 to 1.0
        fixed_size = self.portfolio.balance * base_pct * conviction_scale * self.leverage

        # Use the larger of the two methods (but both are conservative now)
        size = max(kelly_size, fixed_size)

        # ============================================================
        # ASSET-SPECIFIC ADJUSTMENTS
        # ============================================================
        is_low_vol = asset.asset_class == "gold_proxy"

        if is_low_vol:
            # PAXG: Use slightly larger positions (it's less volatile)
            size *= 1.2  # Modest increase for PAXG
        else:
            # For high-vol assets, scale down more aggressively
            size /= (asset.volatility_mult * 1.2)  # Extra 20% reduction

        # Apply drawdown multiplier
        size *= dd_multiplier

        # Correlation adjustment (keep this but make it less severe)
        for existing_symbol in self.portfolio.positions:
            corr = get_correlation(symbol, existing_symbol)
            if corr > 0.6:  # Only reduce for highly correlated (raised from 0.5)
                size *= (1 - corr * 0.3)  # Reduce by up to 30% (reduced from 50%)

        # Minimum size check
        min_size = 30  # $30 minimum for more trades
        if size < min_size:
            return

        # Maximum size cap - CONCENTRATED for max returns
        max_size = self.portfolio.balance * 0.25 * self.leverage  # 25% max per position
        size = min(size, max_size)

        # ============================================================
        # STOP LOSS / TAKE PROFIT - Strategy-adaptive
        # ============================================================
        vol_regime = signals.get("vol_regime", "normal")

        if is_low_vol:
            # Tight for PAXG
            sl_mult = 0.8
            tp_mult = 1.2
        elif vol_regime == "high":
            # Mean reversion: tight stops, quick exits
            sl_mult = 1.0
            tp_mult = 1.5
        else:
            # Trend following: wider stops to ride the trend
            sl_mult = 1.5
            tp_mult = 2.5

        if signals["signal"] == 1:  # LONG
            stop_loss = price - atr * sl_mult
            take_profit = price + atr * tp_mult
            side = "LONG"
        else:  # SHORT
            stop_loss = price + atr * sl_mult
            take_profit = price - atr * tp_mult
            side = "SHORT"

        # Apply slippage
        slippage = price * (asset.typical_spread_bps / 10000)
        entry_price = price + slippage if side == "LONG" else price - slippage

        # Open position
        self.portfolio.positions[symbol] = Position(
            symbol=symbol,
            asset_class=asset.asset_class,
            entry_time=candle["timestamp"],
            entry_price=entry_price,
            side=side,
            size_usd=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            conviction=signals["conviction"],
            highest_price=entry_price,
            lowest_price=entry_price,
        )

        logger.debug(f"Opened {side} {symbol} @ {entry_price:.2f}, size=${size:.0f}, conviction={conviction:.0f}")

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        exit_time: datetime,
    ) -> None:
        """Close a position and record the trade."""
        if symbol not in self.portfolio.positions:
            return

        pos = self.portfolio.positions[symbol]

        # Calculate PnL
        # NOTE: size_usd already includes leverage, so don't multiply again!
        if pos.side == "LONG":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl = pos.size_usd * pnl_pct  # No leverage here - already in size_usd

        # R-multiple
        risk = abs(pos.entry_price - pos.stop_loss) / pos.entry_price
        r_multiple = pnl_pct / risk if risk > 0 else 0

        # Record trade
        trade = Trade(
            symbol=symbol,
            asset_class=pos.asset_class,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            pnl=pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            conviction=pos.conviction,
            exit_reason=reason,
        )
        self.trades.append(trade)

        # Update balance
        self.portfolio.balance += pnl

        # Update asset performance
        self.asset_performance[symbol]["trades"] += 1
        self.asset_performance[symbol]["pnl"] += pnl
        if pnl > 0:
            self.asset_performance[symbol]["wins"] += 1

        # Remove position
        del self.portfolio.positions[symbol]

        logger.debug(f"Closed {pos.side} {symbol} @ {exit_price:.2f}, PnL=${pnl:.2f} ({reason})")

    def _calculate_equity(self) -> float:
        """Calculate current portfolio equity including unrealized PnL."""
        equity = self.portfolio.balance

        for symbol, pos in self.portfolio.positions.items():
            if symbol in self.candles and self.candles[symbol]:
                current_price = self.candles[symbol][-1]["close"]
                if pos.side == "LONG":
                    unrealized = (current_price - pos.entry_price) / pos.entry_price
                else:
                    unrealized = (pos.entry_price - current_price) / pos.entry_price
                # NOTE: size_usd already includes leverage, don't multiply again
                equity += pos.size_usd * unrealized

        return equity

    def _calculate_metrics(self) -> dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {"error": "No trades executed"}

        # Basic stats
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        win_rate = winning_trades / total_trades

        # Long/Short breakdown
        long_trades = [t for t in self.trades if t.side == "LONG"]
        short_trades = [t for t in self.trades if t.side == "SHORT"]
        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)
        long_wins = sum(1 for t in long_trades if t.pnl > 0)
        short_wins = sum(1 for t in short_trades if t.pnl > 0)

        # PnL
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.portfolio.balance - self.initial_balance) / self.initial_balance

        # Annualized return (assuming 180 days)
        days = 180
        annualized_return = ((1 + total_return) ** (365 / days)) - 1

        # Drawdown from equity curve
        peak = self.initial_balance
        max_dd = 0
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                r = (self.equity_curve[i][1] - self.equity_curve[i-1][1]) / self.equity_curve[i-1][1]
                returns.append(r)

            if returns and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)  # Hourly to annual
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Per-asset breakdown
        asset_breakdown = {}
        for symbol, perf in self.asset_performance.items():
            if perf["trades"] > 0:
                asset_breakdown[symbol] = {
                    "trades": perf["trades"],
                    "win_rate": perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0,
                    "pnl": perf["pnl"],
                }

        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.portfolio.balance,
            "total_return_pct": total_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "max_drawdown_pct": max_dd * 100,
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100,
            "sharpe_ratio": sharpe,
            "total_pnl": total_pnl,
            "asset_breakdown": asset_breakdown,
            # Long/Short breakdown
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "long_win_rate": long_wins / len(long_trades) * 100 if long_trades else 0,
            "short_win_rate": short_wins / len(short_trades) * 100 if short_trades else 0,
        }


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Hyperliquid Multi-Asset Backtest")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage (default: 5x for max returns)")
    args = parser.parse_args()

    print("=" * 70)
    print("HYPERLIQUID MULTI-ASSET BACKTEST - MEDALLION STYLE")
    print("=" * 70)
    print(f"Initial Balance: ${args.balance:,.0f}")
    print(f"Leverage: {args.leverage}x")
    print(f"Days: {args.days}")
    print()

    # Initialize backtester
    bt = HyperliquidMultiAssetBacktest(
        initial_balance=args.balance,
        leverage=args.leverage,
    )

    # Fetch candles from Hyperliquid
    print("Fetching candles from Hyperliquid...")
    print("-" * 40)
    await bt.fetch_all_candles(days=args.days)
    print()

    if not bt.candles:
        print("ERROR: No candles fetched. Check Hyperliquid API connection.")
        return

    # Run backtest
    print("Running backtest...")
    print("-" * 40)
    results = bt.run_backtest()

    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    # Print results
    print()
    print("=" * 70)
    print("RESULTS - MEDALLION BENCHMARK COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Our System':>15} {'Medallion':>15} {'Status':>10}")
    print("-" * 70)

    ann_ret = results["annualized_return_pct"]
    max_dd = results["max_drawdown_pct"]
    sharpe = results["sharpe_ratio"]

    # Compare to Medallion benchmarks
    ret_status = "✅" if ann_ret >= 66 else "⚠️"
    dd_status = "✅" if max_dd <= 5 else ("⚠️" if max_dd <= 10 else "❌")
    sharpe_status = "✅" if sharpe >= 2.5 else "⚠️"

    print(f"{'Annualized Return':<25} {ann_ret:>14.1f}% {'66%':>15} {ret_status:>10}")
    print(f"{'Max Drawdown':<25} {max_dd:>14.1f}% {'3%':>15} {dd_status:>10}")
    print(f"{'Sharpe Ratio':<25} {sharpe:>15.2f} {'~3.0':>15} {sharpe_status:>10}")
    print(f"{'Win Rate':<25} {results['win_rate_pct']:>14.1f}% {'-':>15}")
    print(f"{'Total Trades':<25} {results['total_trades']:>15}")
    print()

    print(f"Initial Balance: ${results['initial_balance']:,.0f}")
    print(f"Final Balance:   ${results['final_balance']:,.0f}")
    print(f"Total PnL:       ${results['total_pnl']:,.0f}")
    print()

    # Long/Short breakdown
    print("LONG vs SHORT BREAKDOWN")
    print("-" * 70)
    print(f"{'Direction':<10} {'Trades':>10} {'Win Rate':>12} {'PnL':>15}")
    print("-" * 70)
    print(f"{'LONG':<10} {results['long_trades']:>10} {results['long_win_rate']:>11.1f}% ${results['long_pnl']:>13,.0f}")
    print(f"{'SHORT':<10} {results['short_trades']:>10} {results['short_win_rate']:>11.1f}% ${results['short_pnl']:>13,.0f}")
    print()

    # Per-asset breakdown
    print("PER-ASSET BREAKDOWN")
    print("-" * 70)
    print(f"{'Asset':<10} {'Class':<12} {'Trades':>8} {'Win Rate':>10} {'PnL':>12}")
    print("-" * 70)

    for symbol, perf in results["asset_breakdown"].items():
        asset_cfg = next((a for a in ASSET_UNIVERSE if a.symbol == symbol), None)
        asset_class = asset_cfg.asset_class if asset_cfg else "unknown"
        print(f"{symbol:<10} {asset_class:<12} {perf['trades']:>8} {perf['win_rate']*100:>9.1f}% ${perf['pnl']:>10,.0f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
