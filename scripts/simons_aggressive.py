#!/usr/bin/env python3
"""
Simons Aggressive Mode - Designed to Beat the Benchmark

Key differences from conservative mode:
1. Lower conviction threshold (55 vs 70) - more trades
2. Kelly criterion position sizing - optimal bet size
3. Mean reversion signals in ranging markets
4. Faster timeframe simulation (4H decisions, but more of them)
5. Alternative signals weighted MORE heavily
6. Cross-asset momentum (BTC leads, alts follow)
7. Funding rate as primary signal (not just a boost)

Target: 30%+ annual with <10% drawdown
Benchmark: Medallion 66% annual with <3% drawdown
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiohttp
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import Candle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SimonsConfig:
    """Aggressive Simons-style configuration."""

    # Position sizing (Kelly-inspired) - balanced for returns + protection
    base_position_pct: float = 0.06  # 6% base position
    max_position_pct: float = 0.15  # 15% max position
    min_position_pct: float = 0.02  # 2% min position

    # Conviction thresholds (balanced for quality + quantity)
    min_conviction: int = 60  # 60 = balanced selectivity
    high_conviction: int = 75  # Boost size above this

    # Stop/TP (tighter for more trades)
    stop_atr_mult: float = 1.5  # Was 2.0, tighter
    tp_atr_mult: float = 2.0  # Was 2.5, take profits faster

    # Alternative signal weights (HIGHER = more influence)
    funding_weight: float = 0.5  # Was 0.4
    oi_weight: float = 0.3  # Was 0.35
    crowd_weight: float = 0.2  # Was 0.25

    # Mean reversion in ranging
    enable_mean_reversion: bool = True
    rsi_oversold: float = 25  # Buy signal
    rsi_overbought: float = 75  # Sell signal

    # Cross-asset signals
    enable_cross_asset: bool = True
    btc_lead_hours: int = 4  # BTC leads alts by ~4 hours

    # Trade frequency
    max_trades_per_day: int = 3  # Simons takes many trades
    min_hours_between_trades: int = 4

    # RISK CONTROLS (Simons-level, tuned for balance)
    max_daily_loss_pct: float = 0.03  # Stop trading if down 3% in a day
    max_drawdown_halt_pct: float = 0.15  # Halt if drawdown exceeds 15%
    drawdown_size_reduction: float = 0.6  # Cut position sizes by 40% during drawdown


@dataclass
class Position:
    """Open position."""
    entry_time: datetime
    entry_price: float
    side: Literal["LONG", "SHORT"]
    size: float
    stop_loss: float
    take_profit: float
    conviction: float
    signal_source: str  # What triggered the trade
    highest_price: float = 0.0  # For trailing stop (LONG)
    lowest_price: float = float('inf')  # For trailing stop (SHORT)
    trail_activated: bool = False  # Activate after 1R profit


@dataclass
class Trade:
    """Completed trade."""
    entry_time: datetime
    exit_time: datetime
    side: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    conviction: float
    signal_source: str
    exit_reason: str


class SimonsAggressive:
    """
    Aggressive trading strategy inspired by Jim Simons.

    Core principles:
    1. Many small bets with edge > few large bets
    2. Statistical edge, not prediction
    3. Mean reversion + momentum hybrid
    4. Obsess over transaction costs
    5. Track every signal's performance
    """

    def __init__(
        self,
        config: SimonsConfig,
        initial_balance: float = 10000.0,
        leverage: int = 10,
        slippage_bps: float = 3.0,  # Lower slippage assumption
    ):
        self.config = config
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.slippage_bps = slippage_bps

        # State
        self.position: Position | None = None
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = [initial_balance]

        # Signal tracking (Simons tracks EVERY signal)
        self.signal_performance: dict[str, list[bool]] = {
            "trend_follow": [],
            "mean_reversion": [],
            "funding_signal": [],
            "crowd_contrarian": [],
            "cross_asset": [],
        }

        # Cross-asset state
        self.btc_returns: list[float] = []

        # Daily trade counter
        self.trades_today: int = 0
        self.last_trade_time: datetime | None = None

        # RISK CONTROL STATE (Simons-level)
        self.daily_start_balance: float = initial_balance
        self.daily_pnl: float = 0.0
        self.current_day: datetime | None = None
        self.trading_halted: bool = False
        self.in_drawdown_mode: bool = False
        self.peak_balance: float = initial_balance

        # Risk control statistics
        self.daily_halt_count: int = 0
        self.drawdown_halt_count: int = 0
        self.drawdown_mode_candles: int = 0
        self.total_candles_processed: int = 0

    def calculate_kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion for optimal position sizing.

        f* = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing (1-p)
            b = ratio of avg_win to avg_loss
        """
        if avg_loss == 0 or win_rate <= 0:
            return self.config.min_position_pct

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss

        kelly = (p * b - q) / b

        # Use half-Kelly for safety (Simons uses fractional Kelly)
        half_kelly = kelly * 0.5

        # Clamp to min/max
        return max(self.config.min_position_pct,
                   min(self.config.max_position_pct, half_kelly))

    def get_signal_win_rate(self, signal_type: str) -> float:
        """Get historical win rate for a signal type."""
        history = self.signal_performance.get(signal_type, [])
        if len(history) < 10:
            return 0.55  # Default assumption
        return sum(history) / len(history)

    def calculate_features(self, candles: list[Candle], i: int) -> dict:
        """Calculate all features for signal generation."""
        if i < 50:
            return {}

        window = candles[max(0, i-50):i+1]
        closes = [c.close for c in window]
        highs = [c.high for c in window]
        lows = [c.low for c in window]
        volumes = [c.volume for c in window]

        current_price = closes[-1]

        # EMAs
        ema_10 = self._ema(closes, 10)
        ema_21 = self._ema(closes, 21)
        ema_50 = self._ema(closes, 50) if len(closes) >= 50 else ema_21

        # RSI
        rsi = self._rsi(closes, 14)

        # ATR
        atr = self._atr(highs, lows, closes, 14)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger(closes, 20, 2.0)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        # Volume ratio
        vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        volume_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        # Trend strength (ADX simplified)
        adx = self._adx_simplified(highs, lows, closes, 14)

        # Price momentum
        momentum_4h = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
        momentum_24h = (closes[-1] - closes[-24]) / closes[-24] if len(closes) >= 24 else 0

        # Regime detection
        if adx > 25:
            if ema_10 > ema_21 > ema_50:
                regime = "TRENDING_UP"
            elif ema_10 < ema_21 < ema_50:
                regime = "TRENDING_DOWN"
            else:
                regime = "MIXED"
        else:
            regime = "RANGING"

        return {
            "price": current_price,
            "ema_10": ema_10,
            "ema_21": ema_21,
            "ema_50": ema_50,
            "rsi": rsi,
            "atr": atr,
            "bb_position": bb_position,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "volume_ratio": volume_ratio,
            "adx": adx,
            "momentum_4h": momentum_4h,
            "momentum_24h": momentum_24h,
            "regime": regime,
        }

    def generate_signal(
        self,
        features: dict,
        symbol: str,
        btc_momentum: float = 0,
    ) -> tuple[Literal["LONG", "SHORT", "HOLD"], float, str]:
        """
        Generate trading signal using multiple strategies.

        Returns: (direction, conviction, signal_source)
        """
        if not features:
            return "HOLD", 0, ""

        price = features["price"]
        ema_10 = features["ema_10"]
        ema_21 = features["ema_21"]
        ema_50 = features["ema_50"]
        rsi = features["rsi"]
        regime = features["regime"]
        bb_position = features["bb_position"]
        volume_ratio = features["volume_ratio"]
        adx = features["adx"]

        signals = []  # (direction, conviction, source)

        # Strategy 1: Trend Following (in trending markets)
        # ADX > 25 for trend confirmation
        if regime in ("TRENDING_UP", "TRENDING_DOWN") and adx > 25:
            if ema_10 > ema_21 > ema_50 and price > ema_10:
                conv = 50 + min(30, adx)  # 50-80 based on trend strength
                signals.append(("LONG", conv, "trend_follow"))
            elif ema_10 < ema_21 < ema_50 and price < ema_10:
                conv = 50 + min(30, adx)
                signals.append(("SHORT", conv, "trend_follow"))

        # Strategy 2: Mean Reversion (in ranging markets)
        if self.config.enable_mean_reversion and regime == "RANGING":
            if rsi < self.config.rsi_oversold and bb_position < 0.2:
                conv = 60 + (self.config.rsi_oversold - rsi)  # Higher conv for more oversold
                signals.append(("LONG", min(85, conv), "mean_reversion"))
            elif rsi > self.config.rsi_overbought and bb_position > 0.8:
                conv = 60 + (rsi - self.config.rsi_overbought)
                signals.append(("SHORT", min(85, conv), "mean_reversion"))

        # Strategy 3: Cross-asset momentum (BTC leads alts)
        if self.config.enable_cross_asset and symbol != "BTCUSDT" and btc_momentum != 0:
            if btc_momentum > 0.02:  # BTC up 2%+
                signals.append(("LONG", 55 + min(25, btc_momentum * 500), "cross_asset"))
            elif btc_momentum < -0.02:  # BTC down 2%+
                signals.append(("SHORT", 55 + min(25, abs(btc_momentum) * 500), "cross_asset"))

        # Strategy 4: Volume breakout
        if volume_ratio > 2.0:  # Volume surge
            if price > ema_21 and features["momentum_4h"] > 0.01:
                signals.append(("LONG", 60, "volume_breakout"))
            elif price < ema_21 and features["momentum_4h"] < -0.01:
                signals.append(("SHORT", 60, "volume_breakout"))

        if not signals:
            return "HOLD", 0, ""

        # Combine signals (Simons combines many weak signals)
        long_signals = [s for s in signals if s[0] == "LONG"]
        short_signals = [s for s in signals if s[0] == "SHORT"]

        if long_signals and not short_signals:
            # Average conviction of all LONG signals
            avg_conv = sum(s[1] for s in long_signals) / len(long_signals)
            # Bonus for multiple agreeing signals
            if len(long_signals) >= 2:
                avg_conv += 10
            best_source = max(long_signals, key=lambda s: s[1])[2]
            return "LONG", min(95, avg_conv), best_source

        elif short_signals and not long_signals:
            avg_conv = sum(s[1] for s in short_signals) / len(short_signals)
            if len(short_signals) >= 2:
                avg_conv += 10
            best_source = max(short_signals, key=lambda s: s[1])[2]
            return "SHORT", min(95, avg_conv), best_source

        else:
            # Conflicting signals - no trade
            return "HOLD", 0, ""

    def run(self, candles: list[Candle], symbol: str = "BTCUSDT") -> dict:
        """Run backtest on candles."""
        if len(candles) < 100:
            return {"error": "Not enough candles"}

        logger.info(f"Running Simons Aggressive on {symbol}: {len(candles)} candles")

        # Track BTC momentum for cross-asset signals
        btc_momentum = 0

        for i in range(50, len(candles)):
            candle = candles[i]
            price = candle.close

            # Update BTC momentum (for cross-asset)
            if symbol == "BTCUSDT" and i >= 4:
                btc_momentum = (price - candles[i-4].close) / candles[i-4].close
                self.btc_returns.append(btc_momentum)

            # === SIMONS RISK CONTROLS ===
            # 1. New day reset
            if self.current_day is None or candle.timestamp.date() != self.current_day.date():
                self.current_day = candle.timestamp
                self.daily_start_balance = self.balance
                self.daily_pnl = 0.0
                self.trades_today = 0
                # Reset daily halt, but not drawdown halt
                if not self.in_drawdown_mode:
                    self.trading_halted = False

            # 2. Update peak balance for drawdown calculation
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
                self.in_drawdown_mode = False  # Exit drawdown mode on new high

            # 3. Calculate current drawdown
            current_drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0

            # 4. Check daily loss limit
            self.daily_pnl = self.balance - self.daily_start_balance
            daily_loss_pct = -self.daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                if not self.trading_halted:
                    logger.info(f"RISK: Daily loss limit hit ({daily_loss_pct*100:.1f}%). Halting trading for today.")
                    self.daily_halt_count += 1
                self.trading_halted = True

            # 5. Check max drawdown halt
            if current_drawdown >= self.config.max_drawdown_halt_pct:
                if not self.trading_halted:
                    logger.info(f"RISK: Max drawdown hit ({current_drawdown*100:.1f}%). HALTING ALL TRADING.")
                    self.drawdown_halt_count += 1
                self.trading_halted = True
                self.in_drawdown_mode = True

            # 6. Enter drawdown mode (reduce sizes) at half the halt threshold
            if current_drawdown >= self.config.max_drawdown_halt_pct * 0.5:
                if not self.in_drawdown_mode:
                    logger.info(f"RISK: Entering drawdown mode ({current_drawdown*100:.1f}%). Reducing position sizes.")
                self.in_drawdown_mode = True

            # Track risk stats
            self.total_candles_processed += 1
            if self.in_drawdown_mode:
                self.drawdown_mode_candles += 1

            # Reset daily counter (legacy check, now handled above)
            if self.last_trade_time and candle.timestamp.date() != self.last_trade_time.date():
                self.trades_today = 0

            # Check position exits first
            if self.position:
                self._check_exit(candle)

            # === SKIP TRADING IF HALTED ===
            if self.trading_halted:
                self._update_equity(price)
                continue

            # Generate signals (every 4 hours for 1H candles)
            if i % 4 == 0 and not self.position:
                if self.trades_today >= self.config.max_trades_per_day:
                    continue

                # Check min time between trades
                if self.last_trade_time:
                    hours_since = (candle.timestamp - self.last_trade_time).total_seconds() / 3600
                    if hours_since < self.config.min_hours_between_trades:
                        continue

                features = self.calculate_features(candles, i)
                if not features:
                    continue

                # Get cross-asset momentum (use stored BTC returns for alts)
                cross_momentum = 0
                if symbol != "BTCUSDT" and self.btc_returns:
                    cross_momentum = np.mean(self.btc_returns[-4:]) if len(self.btc_returns) >= 4 else 0

                direction, conviction, source = self.generate_signal(
                    features, symbol, float(cross_momentum)
                )

                if direction != "HOLD" and conviction >= self.config.min_conviction:
                    self._open_position(candle, direction, conviction, source, features)

            # Update equity
            self._update_equity(price)

        # Close any remaining position
        if self.position:
            self._close_position(candles[-1], candles[-1].close, "end_of_test")

        return self._calculate_results(symbol)

    def _open_position(
        self,
        candle: Candle,
        direction: Literal["LONG", "SHORT"],
        conviction: float,
        source: str,
        features: dict,
    ):
        """Open a new position."""
        price = candle.close
        atr = features.get("atr", price * 0.02)

        # Apply slippage
        if direction == "LONG":
            entry_price = price * (1 + self.slippage_bps / 10000)
        else:
            entry_price = price * (1 - self.slippage_bps / 10000)

        # Calculate position size using Kelly-inspired sizing
        win_rate = self.get_signal_win_rate(source)

        # Estimate avg win/loss based on ATR targets
        avg_win = atr * self.config.tp_atr_mult
        avg_loss = atr * self.config.stop_atr_mult

        position_pct = self.calculate_kelly_size(win_rate, avg_win, avg_loss)

        # Conviction bonus
        if conviction >= self.config.high_conviction:
            position_pct *= 1.5

        position_pct = min(position_pct, self.config.max_position_pct)

        # === DRAWDOWN SIZE REDUCTION (Simons-style capital preservation) ===
        if self.in_drawdown_mode:
            position_pct *= self.config.drawdown_size_reduction
            logger.debug(f"RISK: Drawdown mode - position reduced by {(1-self.config.drawdown_size_reduction)*100:.0f}%")

        position_value = self.balance * position_pct

        # Calculate stops
        stop_distance = atr * self.config.stop_atr_mult
        tp_distance = atr * self.config.tp_atr_mult

        if direction == "LONG":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        self.position = Position(
            entry_time=candle.timestamp,
            entry_price=entry_price,
            side=direction,
            size=position_value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            conviction=conviction,
            signal_source=source,
            highest_price=entry_price,  # Initialize for trailing stop
            lowest_price=entry_price,   # Initialize for trailing stop
            trail_activated=False,
        )

        self.trades_today += 1
        self.last_trade_time = candle.timestamp

        logger.debug(
            f"OPEN {direction} @ ${entry_price:.2f} | "
            f"Size: ${position_value:.2f} ({position_pct*100:.1f}%) | "
            f"Conv: {conviction:.0f} | Source: {source}"
        )

    def _check_exit(self, candle: Candle):
        """Check if position should be closed with trailing stop."""
        if not self.position:
            return

        price = candle.close
        pos = self.position

        # Calculate initial risk (1R)
        initial_risk = abs(pos.entry_price - pos.stop_loss)

        if pos.side == "LONG":
            # Track highest price
            if price > pos.highest_price:
                pos.highest_price = price

            # Activate trailing stop after 1R profit
            current_profit = price - pos.entry_price
            if current_profit >= initial_risk and not pos.trail_activated:
                pos.trail_activated = True
                # Move stop to breakeven
                pos.stop_loss = pos.entry_price
                logger.debug(f"TRAIL: Stop moved to breakeven @ ${pos.stop_loss:.2f}")

            # Trail stop at 50% of highest profit
            if pos.trail_activated and pos.highest_price > pos.entry_price:
                max_profit = pos.highest_price - pos.entry_price
                trail_stop = pos.entry_price + (max_profit * 0.5)
                if trail_stop > pos.stop_loss:
                    pos.stop_loss = trail_stop
                    logger.debug(f"TRAIL: Stop trailed to ${pos.stop_loss:.2f}")

            # Check exits
            if price <= pos.stop_loss:
                reason = "trail_stop" if pos.trail_activated else "stop_loss"
                self._close_position(candle, pos.stop_loss, reason)
            elif price >= pos.take_profit:
                self._close_position(candle, pos.take_profit, "take_profit")
        else:
            # Track lowest price
            if price < pos.lowest_price:
                pos.lowest_price = price

            # Activate trailing stop after 1R profit
            current_profit = pos.entry_price - price
            if current_profit >= initial_risk and not pos.trail_activated:
                pos.trail_activated = True
                pos.stop_loss = pos.entry_price
                logger.debug(f"TRAIL: Stop moved to breakeven @ ${pos.stop_loss:.2f}")

            # Trail stop at 50% of lowest profit
            if pos.trail_activated and pos.lowest_price < pos.entry_price:
                max_profit = pos.entry_price - pos.lowest_price
                trail_stop = pos.entry_price - (max_profit * 0.5)
                if trail_stop < pos.stop_loss:
                    pos.stop_loss = trail_stop
                    logger.debug(f"TRAIL: Stop trailed to ${pos.stop_loss:.2f}")

            # Check exits
            if price >= pos.stop_loss:
                reason = "trail_stop" if pos.trail_activated else "stop_loss"
                self._close_position(candle, pos.stop_loss, reason)
            elif price <= pos.take_profit:
                self._close_position(candle, pos.take_profit, "take_profit")

    def _close_position(self, candle: Candle, exit_price: float, reason: str):
        """Close position and record trade."""
        if not self.position:
            return

        pos = self.position

        # Apply slippage
        if pos.side == "LONG":
            actual_exit = exit_price * (1 - self.slippage_bps / 10000)
            pnl_pct = (actual_exit - pos.entry_price) / pos.entry_price
        else:
            actual_exit = exit_price * (1 + self.slippage_bps / 10000)
            pnl_pct = (pos.entry_price - actual_exit) / pos.entry_price

        pnl = pnl_pct * pos.size * self.leverage
        self.balance += pnl

        # R-multiple
        stop_distance = abs(pos.entry_price - pos.stop_loss)
        r_multiple = (pnl_pct * pos.entry_price) / stop_distance if stop_distance > 0 else 0

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=candle.timestamp,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            conviction=pos.conviction,
            signal_source=pos.signal_source,
            exit_reason=reason,
        )
        self.trades.append(trade)

        # Track signal performance
        if pos.signal_source in self.signal_performance:
            self.signal_performance[pos.signal_source].append(pnl > 0)

        logger.debug(
            f"CLOSE {pos.side} @ ${actual_exit:.2f} | "
            f"PnL: ${pnl:+.2f} ({pnl_pct*100:+.2f}%) | "
            f"R: {r_multiple:+.2f} | {reason}"
        )

        self.position = None

    def _update_equity(self, price: float):
        """Update equity curve."""
        equity = self.balance
        if self.position:
            pos = self.position
            if pos.side == "LONG":
                unrealized = (price - pos.entry_price) / pos.entry_price * pos.size * self.leverage
            else:
                unrealized = (pos.entry_price - price) / pos.entry_price * pos.size * self.leverage
            equity += unrealized
        self.equity_curve.append(equity)

    def _calculate_results(self, symbol: str) -> dict:
        """Calculate final results."""
        if not self.trades:
            return {
                "symbol": symbol,
                "total_trades": 0,
                "total_pnl": 0,
                "total_return_pct": 0,
            }

        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.balance - self.initial_balance) / self.initial_balance

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winners) / len(self.trades) if self.trades else 0

        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0

        profit_factor = sum(t.pnl for t in winners) / abs(sum(t.pnl for t in losers)) if losers else float('inf')

        # Max drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Signal performance
        signal_stats = {}
        for signal, outcomes in self.signal_performance.items():
            if outcomes:
                signal_stats[signal] = {
                    "trades": len(outcomes),
                    "win_rate": sum(outcomes) / len(outcomes),
                }

        # Risk control stats
        drawdown_pct_of_time = (self.drawdown_mode_candles / self.total_candles_processed * 100) if self.total_candles_processed > 0 else 0

        return {
            "symbol": symbol,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_pnl": total_pnl,
            "total_return_pct": total_return * 100,
            "total_trades": len(self.trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio": sharpe,
            "signal_performance": signal_stats,
            "risk_controls": {
                "daily_halts": self.daily_halt_count,
                "drawdown_halts": self.drawdown_halt_count,
                "drawdown_mode_pct": drawdown_pct_of_time,
            },
            "trades": self.trades,
        }

    # Helper functions for indicators
    def _ema(self, data: list, period: int) -> float:
        if len(data) < period:
            return data[-1] if data else 0
        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _rsi(self, data: list, period: int = 14) -> float:
        if len(data) < period + 1:
            return 50
        gains, losses = [], []
        for i in range(1, len(data)):
            diff = data[i] - data[i-1]
            gains.append(max(0, diff))
            losses.append(max(0, -diff))
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        if len(highs) < period + 1:
            return (highs[-1] - lows[-1]) if highs else 0
        tr_values = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        return np.mean(tr_values[-period:])

    def _bollinger(self, data: list, period: int = 20, std_dev: float = 2.0) -> tuple:
        if len(data) < period:
            return data[-1], data[-1], data[-1]
        recent = data[-period:]
        middle = np.mean(recent)
        std = np.std(recent)
        return middle + std_dev * std, middle, middle - std_dev * std

    def _adx_simplified(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        """Simplified ADX calculation."""
        if len(highs) < period + 1:
            return 20  # Default neutral

        plus_dm = []
        minus_dm = []
        tr_values = []

        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]

            plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
            minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)

        if not tr_values:
            return 20

        atr = np.mean(tr_values[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        return dx


async def fetch_candles(session: aiohttp.ClientSession, symbol: str, days: int) -> list[Candle]:
    """Fetch candles from Binance."""
    from src.data.binance_data import fetch_binance_with_cache

    raw = await fetch_binance_with_cache(session, symbol, "1h", days=days)

    candles = []
    for c in raw:
        candle = Candle(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(c["timestamp"] / 1000, tz=timezone.utc),
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
            interval="1h",
        )
        candles.append(candle)

    return candles


def print_results(results: dict):
    """Print backtest results."""
    print()
    print("=" * 70)
    print(f"  {results['symbol']} RESULTS")
    print("=" * 70)
    print(f"  Initial Balance:  ${results['initial_balance']:>12,.2f}")
    print(f"  Final Balance:    ${results['final_balance']:>12,.2f}")
    print(f"  Total PnL:        ${results['total_pnl']:>+12,.2f}")
    print(f"  Total Return:     {results['total_return_pct']:>+12.2f}%")
    print()
    print(f"  Total Trades:     {results['total_trades']:>12}")
    print(f"  Win Rate:         {results['win_rate']*100:>12.1f}%")
    print(f"  Profit Factor:    {results['profit_factor']:>12.2f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:>12.2f}")
    print(f"  Max Drawdown:     {results['max_drawdown_pct']:>12.2f}%")
    print()

    # Risk control stats (Simons-level)
    if results.get('risk_controls'):
        rc = results['risk_controls']
        print("  Risk Controls (Simons-Level):")
        print(f"    Daily Halts:      {rc['daily_halts']:>8} times")
        print(f"    Drawdown Halts:   {rc['drawdown_halts']:>8} times")
        print(f"    In DD Mode:       {rc['drawdown_mode_pct']:>7.1f}% of time")
    print()

    # Signal performance
    if results.get('signal_performance'):
        print("  Signal Performance:")
        for signal, stats in results['signal_performance'].items():
            print(f"    {signal}: {stats['trades']} trades, {stats['win_rate']*100:.1f}% win rate")
    print()

    # Recent trades
    if results.get('trades'):
        print("  Recent Trades:")
        for trade in results['trades'][-10:]:
            pnl_str = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
            print(
                f"    {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
                f"{trade.side:5} | {trade.signal_source:15} | "
                f"R:{trade.r_multiple:+.2f} | {pnl_str}"
            )


async def main():
    """Run aggressive Simons backtest."""
    parser = argparse.ArgumentParser(description="Simons Aggressive Backtest")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("          SIMONS AGGRESSIVE MODE")
    print("      Target: Beat the 66% annual benchmark")
    print("=" * 70)
    print()
    print(f"  Period: {args.days} days")
    print(f"  Balance: ${args.balance:,.2f} per symbol")
    print(f"  Leverage: {args.leverage}x")
    print()

    config = SimonsConfig()
    # Top 8 liquid crypto pairs - maximize opportunities while accepting correlated DD
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT"]

    timeout = aiohttp.ClientTimeout(total=300)

    all_results = []
    total_pnl = 0
    total_trades = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # First fetch BTC to get cross-asset momentum
        btc_candles = await fetch_candles(session, "BTCUSDT", args.days)
        # Limit to requested days
        max_candles = args.days * 24
        if len(btc_candles) > max_candles:
            btc_candles = btc_candles[-max_candles:]

        for symbol in symbols:
            print(f"Processing {symbol}...")

            if symbol == "BTCUSDT":
                candles = btc_candles
            else:
                candles = await fetch_candles(session, symbol, args.days)
                # Limit to requested days
                if len(candles) > max_candles:
                    candles = candles[-max_candles:]

            if len(candles) < 100:
                print(f"  {symbol}: Insufficient data")
                continue

            print(f"  {symbol}: {len(candles)} candles")

            # Run backtest
            strategy = SimonsAggressive(
                config=config,
                initial_balance=args.balance,
                leverage=args.leverage,
            )

            # Pass BTC returns for cross-asset signals
            if symbol != "BTCUSDT":
                btc_strategy = SimonsAggressive(config=config, initial_balance=args.balance, leverage=args.leverage)
                btc_strategy.run(btc_candles, "BTCUSDT")
                strategy.btc_returns = btc_strategy.btc_returns

            results = strategy.run(candles, symbol)
            all_results.append(results)

            total_pnl += results.get("total_pnl", 0)
            total_trades += results.get("total_trades", 0)

            print_results(results)

    # Summary
    print()
    print("=" * 70)
    print("          AGGREGATE RESULTS")
    print("=" * 70)
    print()

    total_initial = args.balance * len(symbols)
    total_return = total_pnl / total_initial * 100

    # Calculate period in days and annualize
    period_days = args.days
    annualized_return = total_return * (365 / period_days)

    print(f"  Total PnL:           ${total_pnl:>+12,.2f}")
    print(f"  Period Return:       {total_return:>+12.2f}%")
    print(f"  Annualized Return:   {annualized_return:>+12.2f}%")
    print(f"  Total Trades:        {total_trades:>12}")
    print()

    # Benchmark comparison
    print("  BENCHMARK COMPARISON:")
    print("  " + "-" * 50)
    print(f"  Our Annualized:      {annualized_return:>+12.2f}%")
    print(f"  Medallion Target:    {66:>+12.2f}%")
    print(f"  Gap:                 {annualized_return - 66:>+12.2f}%")
    print()

    if annualized_return >= 66:
        print("  RESULT: MEDALLION BENCHMARK ACHIEVED!")
    elif annualized_return >= 30:
        print("  RESULT: EXCELLENT - Top tier performance")
    elif annualized_return >= 15:
        print("  RESULT: GOOD - Above market average")
    elif annualized_return > 0:
        print("  RESULT: POSITIVE - But room for improvement")
    else:
        print("  RESULT: NEGATIVE - Strategy needs work")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
