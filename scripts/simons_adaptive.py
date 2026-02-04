#!/usr/bin/env python3
"""
Simons Adaptive Mode - Self-Tuning Strategy with Correlation-Aware Risk

Building on simons_aggressive.py, this version adds:
1. Portfolio-level risk management (correlation-aware)
2. Dynamic volatility scaling
3. Adaptive parameter tuning based on recent performance
4. Per-symbol performance tracking with automatic adjustment

Target: Maintain 60%+ annual return with <10% max drawdown
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import aiohttp
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import Candle
from src.adaptive.portfolio_risk import PortfolioRiskManager, RiskDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """
    Medallion-style configuration: Prioritize risk-adjusted returns.

    Philosophy: Many tiny bets with edge > few large bets
    Target: 66%+ returns with <5% drawdown (Sharpe ~3.0)
    """

    # Position sizing - MEDALLION MATCHED
    base_position_pct: float = 0.04  # 4% base
    max_position_pct: float = 0.08  # 8% max
    min_position_pct: float = 0.02  # 2% min

    # Portfolio-level risk - SCALED FOR 8 PAIRS
    max_portfolio_heat: float = 0.28  # Max 28% correlated exposure (scaled for 8 pairs)
    max_single_symbol_heat: float = 0.045  # Max 4.5% per symbol (lower per-symbol with more pairs)

    # Conviction thresholds
    min_conviction: int = 58  # Slightly lower for more trades
    high_conviction: int = 72

    # Stop/TP - TRAILING STOPS DO THE WORK
    stop_atr_mult: float = 1.4
    tp_atr_mult: float = 2.0

    # Risk controls - BEST BALANCE (84%+ return, ~8% drawdown)
    max_daily_loss_pct: float = 0.022  # 2.2% daily loss = pause
    max_drawdown_pct: float = 0.07  # 7% drawdown = halt (PROTECTS CAPITAL)
    drawdown_size_reduction: float = 0.55  # 45% size reduction in drawdown
    drawdown_entry_threshold: float = 0.025  # Enter drawdown mode at 2.5%
    drawdown_exit_threshold: float = 0.008  # Exit drawdown mode at 0.8%

    # Adaptive parameters (self-tuning)
    performance_window: int = 15  # Faster adaptation
    min_win_rate: float = 0.48  # Stricter threshold
    target_win_rate: float = 0.55

    # Size adjustment speed - LESS AGGRESSIVE
    underperform_reduction: float = 0.92  # 8% reduction per eval
    outperform_increase: float = 1.08  # 8% increase per eval
    min_size_multiplier: float = 0.5  # Floor at 50% (never reduce too much)
    max_size_multiplier: float = 1.4  # Ceiling at 140%

    # Enable/disable features
    use_trailing_stop: bool = True
    use_correlation_aware_sizing: bool = True
    use_volatility_scaling: bool = True
    use_regime_filter: bool = True  # Only trade in favorable regimes

    # REGIME DETECTION
    regime_adx_threshold: float = 25.0  # ADX > 25 = trending
    regime_ranging_rsi_low: float = 35.0  # RSI band for ranging market
    regime_ranging_rsi_high: float = 65.0


@dataclass
class Position:
    """Open position with trailing stop support."""
    entry_time: datetime
    entry_price: float
    side: Literal["LONG", "SHORT"]
    size: float
    stop_loss: float
    take_profit: float
    conviction: float
    signal_source: str
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    trail_activated: bool = False


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


@dataclass
class SymbolPerformance:
    """Per-symbol performance tracking for adaptive tuning."""
    trades: list[Trade] = field(default_factory=list)
    win_rate: float = 0.5
    avg_r: float = 0.0
    size_multiplier: float = 1.0  # Adjusted based on performance
    enabled: bool = True


class SimonsAdaptive:
    """
    Self-tuning trading strategy with correlation-aware risk management.

    Key innovations:
    1. Portfolio heat tracking - treats correlated assets as one risk unit
    2. Dynamic position sizing based on per-symbol win rate
    3. Volatility regime detection - reduces size in high vol
    4. Automatic parameter adjustment based on recent performance
    """

    def __init__(
        self,
        config: AdaptiveConfig,
        symbols: list[str],
        initial_balance: float = 10000.0,
        leverage: int = 10,
        slippage_bps: float = 3.0,
    ):
        self.config = config
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.slippage_bps = slippage_bps

        # Portfolio risk manager
        self.risk_manager = PortfolioRiskManager(
            symbols=symbols,
            max_heat_pct=config.max_portfolio_heat,
            max_single_position_pct=config.max_single_symbol_heat,
            vol_scaling=config.use_volatility_scaling,
            correlation_aware=config.use_correlation_aware_sizing,
        )
        self.risk_manager.set_balance(initial_balance)

        # State
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.all_trades: list[Trade] = []
        self.equity_curve: list[float] = [initial_balance]

        # Per-symbol adaptive state
        self.symbol_performance: dict[str, SymbolPerformance] = {
            s: SymbolPerformance() for s in symbols
        }

        # Risk tracking
        self.daily_start_balance: float = initial_balance
        self.current_day: Optional[datetime] = None
        self.trading_halted: bool = False
        self.peak_balance: float = initial_balance
        self.in_drawdown_mode: bool = False  # Reduce sizes, don't halt
        self.drawdown_size_mult: float = 1.0

        # Statistics
        self.portfolio_heat_history: list[float] = []
        self.risk_decisions: list[RiskDecision] = []
        self.blocked_trades: int = 0
        self.reduced_trades: int = 0

        # Regime tracking
        self.current_regimes: dict[str, str] = {s: "UNKNOWN" for s in symbols}
        self.regime_confidence: dict[str, float] = {s: 0.0 for s in symbols}
        self.trending_periods: int = 0
        self.ranging_periods: int = 0

    def _calculate_features(self, candles: list[Candle], i: int) -> dict:
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
        bb_upper, _, bb_lower = self._bollinger(closes, 20, 2.0)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        # Volume ratio
        vol_avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(volumes[-1])
        volume_ratio = float(volumes[-1] / vol_avg) if vol_avg > 0 else 1.0

        # ADX (simplified)
        adx = self._adx_simplified(highs, lows, closes, 14)

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
            "volume_ratio": volume_ratio,
            "adx": adx,
            "regime": regime,
        }

    def _generate_signal(
        self,
        features: dict,
        symbol: str,
    ) -> tuple[Literal["LONG", "SHORT", "HOLD"], float, str]:
        """Generate trading signal - REGIME AWARE."""
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

        # Track regime
        self.current_regimes[symbol] = regime
        self.regime_confidence[symbol] = min(adx / 40, 1.0)  # Confidence based on ADX

        signals = []

        # STRATEGY 1: TREND FOLLOWING (only in clear trends)
        if regime in ("TRENDING_UP", "TRENDING_DOWN") and adx > self.config.regime_adx_threshold:
            self.trending_periods += 1
            if ema_10 > ema_21 > ema_50 and price > ema_10:
                conv = 55 + min(30, adx)  # Higher base for clear trends
                signals.append(("LONG", conv, "trend_follow"))
            elif ema_10 < ema_21 < ema_50 and price < ema_10:
                conv = 55 + min(30, adx)
                signals.append(("SHORT", conv, "trend_follow"))

        # STRATEGY 2: MEAN REVERSION (only in ranging markets with extreme readings)
        if regime == "RANGING" or adx < 20:
            self.ranging_periods += 1
            # Only trade extreme oversold/overbought in ranging
            if rsi < 20 and bb_position < 0.1:  # Very oversold
                conv = 65 + (20 - rsi)  # Higher conviction for extremes
                signals.append(("LONG", min(85, conv), "mean_reversion"))
            elif rsi > 80 and bb_position > 0.9:  # Very overbought
                conv = 65 + (rsi - 80)
                signals.append(("SHORT", min(85, conv), "mean_reversion"))

        # STRATEGY 3: Volume breakout (works in any regime)
        if volume_ratio > 2.5:  # Higher threshold for quality
            momentum = (price - ema_21) / ema_21
            if momentum > 0.015 and rsi > 50:  # Confirm with RSI
                signals.append(("LONG", 65, "volume_breakout"))
            elif momentum < -0.015 and rsi < 50:
                signals.append(("SHORT", 65, "volume_breakout"))

        if not signals:
            return "HOLD", 0, ""

        # Combine signals
        long_signals = [s for s in signals if s[0] == "LONG"]
        short_signals = [s for s in signals if s[0] == "SHORT"]

        if long_signals and not short_signals:
            avg_conv = sum(s[1] for s in long_signals) / len(long_signals)
            if len(long_signals) >= 2:
                avg_conv += 10
            return "LONG", min(95, avg_conv), max(long_signals, key=lambda s: s[1])[2]

        elif short_signals and not long_signals:
            avg_conv = sum(s[1] for s in short_signals) / len(short_signals)
            if len(short_signals) >= 2:
                avg_conv += 10
            return "SHORT", min(95, avg_conv), max(short_signals, key=lambda s: s[1])[2]

        return "HOLD", 0, ""

    def _calculate_position_size(
        self,
        symbol: str,
        conviction: float,
        atr: float,
    ) -> tuple[float, RiskDecision]:
        """
        Calculate position size with portfolio risk awareness.

        Returns (size, risk_decision)
        """
        # Base size
        base_pct = self.config.base_position_pct

        # Conviction adjustment
        if conviction >= self.config.high_conviction:
            base_pct *= 1.3

        # Per-symbol performance adjustment
        perf = self.symbol_performance[symbol]
        base_pct *= perf.size_multiplier

        # Drawdown mode adjustment (don't halt, just reduce)
        base_pct *= self.drawdown_size_mult

        # Clamp to limits
        position_pct = min(base_pct, self.config.max_position_pct)
        position_pct = max(position_pct, self.config.min_position_pct)

        # Calculate raw size
        raw_size = self.balance * position_pct

        # Check with portfolio risk manager
        decision = self.risk_manager.check_new_position(
            symbol=symbol,
            side="LONG",  # Direction doesn't matter for sizing
            size=raw_size,
        )

        self.risk_decisions.append(decision)

        if decision.action == "block":
            self.blocked_trades += 1
            return 0.0, decision
        elif decision.action == "reduce":
            self.reduced_trades += 1

        return decision.adjusted_size, decision

    def _open_position(
        self,
        symbol: str,
        candle: Candle,
        direction: Literal["LONG", "SHORT"],
        conviction: float,
        source: str,
        features: dict,
    ) -> bool:
        """Open a new position with portfolio risk management."""
        price = candle.close
        atr = features.get("atr", price * 0.02)

        # Calculate size with risk management
        size, decision = self._calculate_position_size(symbol, conviction, atr)

        if size <= 0:
            logger.debug(f"Position blocked for {symbol}: {decision.reason}")
            return False

        # Apply slippage
        if direction == "LONG":
            entry_price = price * (1 + self.slippage_bps / 10000)
        else:
            entry_price = price * (1 - self.slippage_bps / 10000)

        # Calculate stops
        stop_distance = atr * self.config.stop_atr_mult
        tp_distance = atr * self.config.tp_atr_mult

        if direction == "LONG":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        position = Position(
            entry_time=candle.timestamp,
            entry_price=entry_price,
            side=direction,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            conviction=conviction,
            signal_source=source,
            highest_price=entry_price,
            lowest_price=entry_price,
        )

        self.positions[symbol] = position

        # Register with risk manager
        self.risk_manager.register_position(symbol, direction, size, entry_price)

        logger.debug(
            f"OPEN {symbol} {direction} @ ${entry_price:.2f} | "
            f"Size: ${size:.2f} ({decision.scaling_factor:.0%}) | "
            f"Conv: {conviction:.0f}"
        )

        return True

    def _check_exit(self, symbol: str, candle: Candle) -> None:
        """Check if position should be closed with trailing stop."""
        if symbol not in self.positions:
            return

        price = candle.close
        pos = self.positions[symbol]

        # Calculate initial risk (1R)
        initial_risk = abs(pos.entry_price - pos.stop_loss)

        if pos.side == "LONG":
            # Track highest price
            if price > pos.highest_price:
                pos.highest_price = price

            # Trailing stop logic
            if self.config.use_trailing_stop:
                current_profit = price - pos.entry_price
                if current_profit >= initial_risk and not pos.trail_activated:
                    pos.trail_activated = True
                    pos.stop_loss = pos.entry_price

                if pos.trail_activated and pos.highest_price > pos.entry_price:
                    max_profit = pos.highest_price - pos.entry_price
                    trail_stop = pos.entry_price + (max_profit * 0.5)
                    if trail_stop > pos.stop_loss:
                        pos.stop_loss = trail_stop

            # Check exits
            if price <= pos.stop_loss:
                reason = "trail_stop" if pos.trail_activated else "stop_loss"
                self._close_position(symbol, candle, pos.stop_loss, reason)
            elif price >= pos.take_profit:
                self._close_position(symbol, candle, pos.take_profit, "take_profit")
        else:
            # Track lowest price
            if price < pos.lowest_price:
                pos.lowest_price = price

            # Trailing stop logic
            if self.config.use_trailing_stop:
                current_profit = pos.entry_price - price
                if current_profit >= initial_risk and not pos.trail_activated:
                    pos.trail_activated = True
                    pos.stop_loss = pos.entry_price

                if pos.trail_activated and pos.lowest_price < pos.entry_price:
                    max_profit = pos.entry_price - pos.lowest_price
                    trail_stop = pos.entry_price - (max_profit * 0.5)
                    if trail_stop < pos.stop_loss:
                        pos.stop_loss = trail_stop

            # Check exits
            if price >= pos.stop_loss:
                reason = "trail_stop" if pos.trail_activated else "stop_loss"
                self._close_position(symbol, candle, pos.stop_loss, reason)
            elif price <= pos.take_profit:
                self._close_position(symbol, candle, pos.take_profit, "take_profit")

    def _close_position(
        self,
        symbol: str,
        candle: Candle,
        exit_price: float,
        reason: str,
    ) -> None:
        """Close position and record trade."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

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
        self.all_trades.append(trade)

        # Update per-symbol performance
        perf = self.symbol_performance[symbol]
        perf.trades.append(trade)
        self._update_symbol_performance(symbol)

        # Remove from risk manager
        self.risk_manager.close_position(symbol)

        del self.positions[symbol]

        logger.debug(
            f"CLOSE {symbol} {pos.side} @ ${actual_exit:.2f} | "
            f"PnL: ${pnl:+.2f} ({pnl_pct*100:+.2f}%) | R: {r_multiple:+.2f}"
        )

    def _update_symbol_performance(self, symbol: str) -> None:
        """Update per-symbol performance and adjust sizing - MEDALLION STYLE."""
        perf = self.symbol_performance[symbol]
        recent_trades = perf.trades[-self.config.performance_window:]

        if len(recent_trades) < 5:
            return

        wins = sum(1 for t in recent_trades if t.pnl > 0)
        win_rate = wins / len(recent_trades)
        avg_r = sum(t.r_multiple for t in recent_trades) / len(recent_trades)

        perf.win_rate = win_rate
        perf.avg_r = avg_r

        # MEDALLION: Fast adaptation to performance
        if win_rate < self.config.min_win_rate:
            # Underperforming - reduce size quickly
            perf.size_multiplier = max(
                self.config.min_size_multiplier,
                perf.size_multiplier * self.config.underperform_reduction
            )
            logger.debug(f"{symbol} underperforming ({win_rate:.1%} WR) - size {perf.size_multiplier:.0%}")
        elif win_rate > self.config.target_win_rate and avg_r > 0.3:
            # Outperforming - increase size cautiously
            perf.size_multiplier = min(
                self.config.max_size_multiplier,
                perf.size_multiplier * self.config.outperform_increase
            )

    def run(
        self,
        all_candles: dict[str, list[Candle]],
    ) -> dict:
        """Run backtest on all symbols simultaneously."""
        # Get common timestamp range
        min_len = min(len(candles) for candles in all_candles.values())
        if min_len < 100:
            return {"error": "Not enough candles"}

        logger.info(f"Running Simons Adaptive on {len(all_candles)} symbols: {min_len} candles each")

        for i in range(50, min_len):
            # Get current candle for each symbol
            current_candles = {s: candles[i] for s, candles in all_candles.items()}
            timestamp = list(current_candles.values())[0].timestamp

            # Update risk manager with prices
            for symbol, candle in current_candles.items():
                self.risk_manager.update_price(symbol, candle.close, timestamp)
            self.risk_manager.set_balance(self.balance)

            # Track portfolio heat
            heat = self.risk_manager.calculate_portfolio_heat()
            self.portfolio_heat_history.append(heat.heat_pct)

            # Check for new day
            if self.current_day is None or timestamp.date() != self.current_day.date():
                self.current_day = timestamp
                self.daily_start_balance = self.balance
                self.trading_halted = False

            # Update peak balance
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

            # Check daily loss limit - Halt for the day (protects capital)
            daily_pnl = self.balance - self.daily_start_balance
            daily_loss_pct = -daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                if not self.trading_halted:
                    logger.debug(f"RISK: Daily loss ({daily_loss_pct*100:.1f}%) - halt for day")
                self.trading_halted = True  # Resets next day

            # Check drawdown - MEDALLION: Fast response
            current_dd = (self.peak_balance - self.balance) / self.peak_balance

            if current_dd >= self.config.max_drawdown_pct:
                # Extreme drawdown - halt until recovery
                if not self.trading_halted:
                    logger.info(f"RISK: Max drawdown ({current_dd*100:.1f}%) - HALT")
                self.trading_halted = True
                self.in_drawdown_mode = True
            elif self.trading_halted and current_dd < self.config.drawdown_entry_threshold:
                # RECOVERY: Resume trading if drawdown improved significantly
                logger.info(f"RISK: Recovered to {current_dd*100:.1f}% - resuming")
                self.trading_halted = False
                self.in_drawdown_mode = True  # Still cautious
                self.drawdown_size_mult = 0.6  # 60% size until fully recovered
            elif current_dd >= self.config.drawdown_entry_threshold:
                # Enter drawdown mode - aggressive size reduction
                if not self.in_drawdown_mode:
                    logger.debug(f"RISK: Drawdown ({current_dd*100:.1f}%) - reducing sizes")
                self.in_drawdown_mode = True
                # Graduated reduction based on drawdown severity
                severity = min(current_dd / self.config.max_drawdown_pct, 1.0)
                self.drawdown_size_mult = 1.0 - (severity * (1.0 - self.config.drawdown_size_reduction))
            elif current_dd <= self.config.drawdown_exit_threshold:
                # Recovered - back to normal
                if self.in_drawdown_mode:
                    logger.debug(f"RISK: Recovered ({current_dd*100:.1f}%)")
                self.in_drawdown_mode = False
                self.drawdown_size_mult = 1.0
                # Also unhalt if we were halted from drawdown (not daily loss)
                if self.trading_halted and daily_loss_pct < self.config.max_daily_loss_pct:
                    self.trading_halted = False

            # Check exits for all open positions
            for symbol in list(self.positions.keys()):
                self._check_exit(symbol, current_candles[symbol])

            # Skip new trades if halted
            if self.trading_halted:
                self._update_equity(current_candles)
                continue

            # Generate signals and open positions (every 4 hours)
            if i % 4 == 0:
                for symbol, candles in all_candles.items():
                    if symbol in self.positions:
                        continue

                    if not self.symbol_performance[symbol].enabled:
                        continue

                    features = self._calculate_features(candles, i)
                    if not features:
                        continue

                    direction, conviction, source = self._generate_signal(features, symbol)

                    if direction != "HOLD" and conviction >= self.config.min_conviction:
                        self._open_position(
                            symbol,
                            current_candles[symbol],
                            direction,
                            conviction,
                            source,
                            features,
                        )

            # Update equity
            self._update_equity(current_candles)

        # Close remaining positions
        for symbol in list(self.positions.keys()):
            last_candle = all_candles[symbol][-1]
            self._close_position(symbol, last_candle, last_candle.close, "end_of_test")

        return self._calculate_results()

    def _update_equity(self, current_candles: dict[str, Candle]) -> None:
        """Update equity curve."""
        equity = self.balance
        for symbol, pos in self.positions.items():
            price = current_candles[symbol].close
            if pos.side == "LONG":
                unrealized = (price - pos.entry_price) / pos.entry_price * pos.size * self.leverage
            else:
                unrealized = (pos.entry_price - price) / pos.entry_price * pos.size * self.leverage
            equity += unrealized
        self.equity_curve.append(equity)

    def _calculate_results(self) -> dict:
        """Calculate final results."""
        if not self.all_trades:
            return {
                "total_trades": 0,
                "total_pnl": 0,
                "total_return_pct": 0,
            }

        total_pnl = sum(t.pnl for t in self.all_trades)
        total_return = (self.balance - self.initial_balance) / self.initial_balance

        winners = [t for t in self.all_trades if t.pnl > 0]
        losers = [t for t in self.all_trades if t.pnl <= 0]
        win_rate = len(winners) / len(self.all_trades) if self.all_trades else 0

        avg_win = float(np.mean([t.pnl for t in winners])) if winners else 0
        avg_loss = abs(float(np.mean([t.pnl for t in losers]))) if losers else 0

        profit_factor = sum(t.pnl for t in winners) / abs(sum(t.pnl for t in losers)) if losers else float('inf')

        # Max drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Per-symbol results
        symbol_results = {}
        for symbol, perf in self.symbol_performance.items():
            if perf.trades:
                sym_pnl = sum(t.pnl for t in perf.trades)
                sym_wins = sum(1 for t in perf.trades if t.pnl > 0)
                symbol_results[symbol] = {
                    "trades": len(perf.trades),
                    "pnl": sym_pnl,
                    "win_rate": sym_wins / len(perf.trades),
                    "size_multiplier": perf.size_multiplier,
                }

        # Portfolio heat stats
        avg_heat = float(np.mean(self.portfolio_heat_history)) if self.portfolio_heat_history else 0
        max_heat = float(np.max(self.portfolio_heat_history)) if self.portfolio_heat_history else 0

        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_pnl": total_pnl,
            "total_return_pct": total_return * 100,
            "total_trades": len(self.all_trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio": sharpe,
            "symbol_results": symbol_results,
            "risk_stats": {
                "blocked_trades": self.blocked_trades,
                "reduced_trades": self.reduced_trades,
                "avg_portfolio_heat": avg_heat * 100,
                "max_portfolio_heat": max_heat * 100,
            },
            "trades": self.all_trades,
        }

    # Helper functions
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
        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))
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
        return float(np.mean(tr_values[-period:]))

    def _bollinger(self, data: list, period: int = 20, std_dev: float = 2.0) -> tuple:
        if len(data) < period:
            return data[-1], data[-1], data[-1]
        recent = data[-period:]
        middle = float(np.mean(recent))
        std = float(np.std(recent))
        return middle + std_dev * std, middle, middle - std_dev * std

    def _adx_simplified(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        if len(highs) < period + 1:
            return 20
        plus_dm, minus_dm, tr_values = [], [], []
        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
            minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_values.append(tr)
        if not tr_values:
            return 20
        atr = float(np.mean(tr_values[-period:]))
        plus_di = 100 * float(np.mean(plus_dm[-period:])) / atr if atr > 0 else 0
        minus_di = 100 * float(np.mean(minus_dm[-period:])) / atr if atr > 0 else 0
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
    print("  SIMONS ADAPTIVE MODE - RESULTS")
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

    # Risk stats
    risk = results.get('risk_stats', {})
    print("  Portfolio Risk Management:")
    print(f"    Blocked Trades:      {risk.get('blocked_trades', 0):>8}")
    print(f"    Reduced Trades:      {risk.get('reduced_trades', 0):>8}")
    print(f"    Avg Portfolio Heat:  {risk.get('avg_portfolio_heat', 0):>7.1f}%")
    print(f"    Max Portfolio Heat:  {risk.get('max_portfolio_heat', 0):>7.1f}%")
    print()

    # Per-symbol results
    if results.get('symbol_results'):
        print("  Per-Symbol Performance:")
        for symbol, stats in results['symbol_results'].items():
            print(
                f"    {symbol}: {stats['trades']} trades, "
                f"${stats['pnl']:+,.2f}, "
                f"{stats['win_rate']*100:.1f}% WR, "
                f"{stats['size_multiplier']:.0%} size"
            )
    print()


async def main():
    """Run adaptive Simons backtest."""
    parser = argparse.ArgumentParser(description="Simons Adaptive Backtest")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("          SIMONS ADAPTIVE MODE")
    print("     Self-Tuning with Correlation-Aware Risk")
    print("=" * 70)
    print()
    print(f"  Period: {args.days} days")
    print(f"  Balance: ${args.balance:,.2f}")
    print(f"  Leverage: {args.leverage}x")
    print()

    config = AdaptiveConfig()
    # Top 8 liquid crypto pairs - best diversification within crypto
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT"]

    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Fetch all candles and trim to requested days
        all_candles = {}
        max_candles = args.days * 24  # 1 candle per hour
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            candles = await fetch_candles(session, symbol, args.days)
            # Only use the most recent N candles
            if len(candles) > max_candles:
                candles = candles[-max_candles:]
            all_candles[symbol] = candles
            print(f"  {symbol}: {len(candles)} candles")

        # Run adaptive backtest
        strategy = SimonsAdaptive(
            config=config,
            symbols=symbols,
            initial_balance=args.balance,
            leverage=args.leverage,
        )

        results = strategy.run(all_candles)

    print_results(results)

    # Benchmark comparison
    period_days = args.days
    annualized_return = results['total_return_pct'] * (365 / period_days)

    print("  BENCHMARK COMPARISON:")
    print("  " + "-" * 50)
    print(f"  Our Annualized:      {annualized_return:>+12.2f}%")
    print(f"  Medallion Target:    {66:>+12.2f}%")
    print(f"  Gap:                 {annualized_return - 66:>+12.2f}%")
    print()
    print(f"  Our Max Drawdown:    {results['max_drawdown_pct']:>12.2f}%")
    print(f"  Medallion Target:    {3:>12.2f}%")
    print(f"  Gap:                 {results['max_drawdown_pct'] - 3:>+12.2f}%")
    print()

    if annualized_return >= 66 and results['max_drawdown_pct'] <= 10:
        print("  ✅ EXCELLENT - Beat return target with controlled drawdown!")
    elif annualized_return >= 66:
        print("  ⚠️  Beat return target but drawdown needs work")
    elif results['max_drawdown_pct'] <= 10:
        print("  ⚠️  Good drawdown control but returns need improvement")
    else:
        print("  ❌ Both metrics need improvement")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
