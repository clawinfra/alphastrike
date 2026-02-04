"""
AlphaStrike Trading Bot - Circuit Breaker System

Implements trading halts and size reductions based on:
1. Consecutive losses (3 losses → 50% size, 5 → stop 24h)
2. Daily loss limits (-1% → reduce, -1.5% → stop)
3. Correlation limits (no double exposure on correlated assets)
4. System health (model health, data feed, API errors)

This is a critical safety layer for capital preservation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class BreakerState(Enum):
    """Circuit breaker state."""

    NORMAL = "normal"  # Full trading allowed
    REDUCED = "reduced"  # Trading allowed with reduced size
    HALTED = "halted"  # No new trades allowed


class BreakerType(Enum):
    """Type of circuit breaker."""

    CONSECUTIVE_LOSS = "consecutive_loss"
    DAILY_LOSS = "daily_loss"
    WEEKLY_LOSS = "weekly_loss"
    CORRELATION = "correlation"
    SYSTEM_HEALTH = "system_health"


@dataclass
class BreakerStatus:
    """Status of a single circuit breaker."""

    breaker_type: BreakerType
    state: BreakerState
    size_multiplier: float  # 0.0 - 1.0
    reason: str
    triggered_at: datetime | None = None
    resume_at: datetime | None = None


@dataclass
class CircuitBreakerResult:
    """Combined result from all circuit breakers."""

    can_trade: bool
    size_multiplier: float  # Combined multiplier (0.0 - 1.0)
    active_breakers: list[BreakerStatus]
    message: str


@dataclass
class TradeRecord:
    """Record of a trade for tracking."""

    timestamp: datetime
    symbol: str
    direction: Literal["LONG", "SHORT"]
    pnl_pct: float  # P&L as percentage of account
    was_profitable: bool


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: datetime
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_pct: float = 0.0
    peak_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0


class CircuitBreaker:
    """
    Central circuit breaker system for trading safety.

    Monitors multiple risk factors and can halt or reduce trading
    when safety thresholds are exceeded.
    """

    # Consecutive loss settings
    CONSECUTIVE_LOSS_REDUCE = 3  # Reduce size after 3 losses
    CONSECUTIVE_LOSS_HALT = 5  # Halt after 5 losses
    CONSECUTIVE_LOSS_COOLDOWN_HOURS = 24

    # Daily loss settings
    DAILY_LOSS_REDUCE = -0.01  # -1% daily loss → reduce
    DAILY_LOSS_HALT = -0.015  # -1.5% daily loss → halt

    # Weekly loss settings
    WEEKLY_LOSS_HALT = -0.03  # -3% weekly loss → halt

    # Size multipliers
    REDUCED_SIZE_MULTIPLIER = 0.50
    RECOVERY_SIZE_MULTIPLIER = 0.75

    # Correlated assets (shouldn't hold same direction simultaneously)
    CORRELATED_PAIRS = [
        {"BTC", "ETH"},  # High correlation
        {"SOL", "ETH"},  # Medium correlation
    ]

    def __init__(self) -> None:
        """Initialize the circuit breaker system."""
        self._trade_history: list[TradeRecord] = []
        self._daily_stats: dict[str, DailyStats] = {}
        self._open_positions: dict[str, Literal["LONG", "SHORT"]] = {}

        self._consecutive_losses = 0
        self._last_halt_time: datetime | None = None
        self._manual_halt = False

        logger.info("CircuitBreaker system initialized")

    def check(self, symbol: str, direction: Literal["LONG", "SHORT"]) -> CircuitBreakerResult:
        """
        Check all circuit breakers before allowing a trade.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            direction: Proposed trade direction

        Returns:
            CircuitBreakerResult indicating if trade is allowed
        """
        active_breakers: list[BreakerStatus] = []

        # Check manual halt
        if self._manual_halt:
            active_breakers.append(
                BreakerStatus(
                    breaker_type=BreakerType.SYSTEM_HEALTH,
                    state=BreakerState.HALTED,
                    size_multiplier=0.0,
                    reason="Manual trading halt active",
                )
            )

        # Check consecutive losses
        consecutive_status = self._check_consecutive_losses()
        if consecutive_status.state != BreakerState.NORMAL:
            active_breakers.append(consecutive_status)

        # Check daily loss
        daily_status = self._check_daily_loss()
        if daily_status.state != BreakerState.NORMAL:
            active_breakers.append(daily_status)

        # Check weekly loss
        weekly_status = self._check_weekly_loss()
        if weekly_status.state != BreakerState.NORMAL:
            active_breakers.append(weekly_status)

        # Check correlation
        correlation_status = self._check_correlation(symbol, direction)
        if correlation_status.state != BreakerState.NORMAL:
            active_breakers.append(correlation_status)

        # Determine combined result
        if any(b.state == BreakerState.HALTED for b in active_breakers):
            return CircuitBreakerResult(
                can_trade=False,
                size_multiplier=0.0,
                active_breakers=active_breakers,
                message=self._build_halt_message(active_breakers),
            )

        # Calculate combined size multiplier
        size_multiplier = 1.0
        for breaker in active_breakers:
            size_multiplier *= breaker.size_multiplier

        return CircuitBreakerResult(
            can_trade=True,
            size_multiplier=size_multiplier,
            active_breakers=active_breakers,
            message=self._build_status_message(active_breakers, size_multiplier),
        )

    def record_trade(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        pnl_pct: float,
    ) -> None:
        """
        Record a completed trade.

        Args:
            symbol: Trading symbol
            direction: Trade direction
            pnl_pct: P&L as percentage of account (e.g., 0.02 for +2%)
        """
        now = datetime.now(UTC)
        was_profitable = pnl_pct > 0

        # Record trade
        record = TradeRecord(
            timestamp=now,
            symbol=symbol,
            direction=direction,
            pnl_pct=pnl_pct,
            was_profitable=was_profitable,
        )
        self._trade_history.append(record)

        # Update consecutive losses
        if was_profitable:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Update daily stats
        self._update_daily_stats(now, pnl_pct, was_profitable)

        # Remove position from open positions
        self._open_positions.pop(symbol, None)

        logger.info(
            f"Trade recorded: {symbol} {direction} {'WIN' if was_profitable else 'LOSS'} "
            f"({pnl_pct:+.2%}), consecutive_losses={self._consecutive_losses}"
        )

    def open_position(self, symbol: str, direction: Literal["LONG", "SHORT"]) -> None:
        """Record that a position has been opened."""
        self._open_positions[symbol] = direction
        logger.debug(f"Position opened: {symbol} {direction}")

    def close_position(self, symbol: str) -> None:
        """Record that a position has been closed."""
        self._open_positions.pop(symbol, None)
        logger.debug(f"Position closed: {symbol}")

    def manual_halt(self, reason: str = "Manual halt") -> None:
        """Manually halt all trading."""
        self._manual_halt = True
        logger.warning(f"Manual trading halt activated: {reason}")

    def resume_trading(self) -> None:
        """Resume trading after manual halt."""
        self._manual_halt = False
        self._last_halt_time = None
        logger.info("Trading resumed")

    def reset_consecutive_losses(self) -> None:
        """Reset consecutive loss counter (e.g., after model retrain)."""
        self._consecutive_losses = 0
        logger.info("Consecutive loss counter reset")

    def get_status_summary(self) -> dict:
        """Get summary of circuit breaker status."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        daily = self._daily_stats.get(today, DailyStats(date=datetime.now(UTC)))

        return {
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl_pct": daily.pnl_pct,
            "daily_trades": daily.trades,
            "daily_win_rate": daily.wins / daily.trades if daily.trades > 0 else 0,
            "open_positions": dict(self._open_positions),
            "manual_halt": self._manual_halt,
        }

    def _check_consecutive_losses(self) -> BreakerStatus:
        """Check consecutive loss breaker."""
        now = datetime.now(UTC)

        # Check if in cooldown period
        if self._last_halt_time:
            cooldown_end = self._last_halt_time + timedelta(hours=self.CONSECUTIVE_LOSS_COOLDOWN_HOURS)
            if now < cooldown_end:
                return BreakerStatus(
                    breaker_type=BreakerType.CONSECUTIVE_LOSS,
                    state=BreakerState.HALTED,
                    size_multiplier=0.0,
                    reason=f"Consecutive loss halt until {cooldown_end.isoformat()}",
                    triggered_at=self._last_halt_time,
                    resume_at=cooldown_end,
                )

        if self._consecutive_losses >= self.CONSECUTIVE_LOSS_HALT:
            self._last_halt_time = now
            return BreakerStatus(
                breaker_type=BreakerType.CONSECUTIVE_LOSS,
                state=BreakerState.HALTED,
                size_multiplier=0.0,
                reason=f"{self._consecutive_losses} consecutive losses - halted for 24h",
                triggered_at=now,
                resume_at=now + timedelta(hours=self.CONSECUTIVE_LOSS_COOLDOWN_HOURS),
            )

        if self._consecutive_losses >= self.CONSECUTIVE_LOSS_REDUCE:
            return BreakerStatus(
                breaker_type=BreakerType.CONSECUTIVE_LOSS,
                state=BreakerState.REDUCED,
                size_multiplier=self.REDUCED_SIZE_MULTIPLIER,
                reason=f"{self._consecutive_losses} consecutive losses - size reduced 50%",
            )

        return BreakerStatus(
            breaker_type=BreakerType.CONSECUTIVE_LOSS,
            state=BreakerState.NORMAL,
            size_multiplier=1.0,
            reason="OK",
        )

    def _check_daily_loss(self) -> BreakerStatus:
        """Check daily loss breaker."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        daily = self._daily_stats.get(today)

        if not daily:
            return BreakerStatus(
                breaker_type=BreakerType.DAILY_LOSS,
                state=BreakerState.NORMAL,
                size_multiplier=1.0,
                reason="OK",
            )

        if daily.pnl_pct <= self.DAILY_LOSS_HALT:
            return BreakerStatus(
                breaker_type=BreakerType.DAILY_LOSS,
                state=BreakerState.HALTED,
                size_multiplier=0.0,
                reason=f"Daily loss {daily.pnl_pct:.2%} exceeds {self.DAILY_LOSS_HALT:.1%} - halted",
            )

        if daily.pnl_pct <= self.DAILY_LOSS_REDUCE:
            return BreakerStatus(
                breaker_type=BreakerType.DAILY_LOSS,
                state=BreakerState.REDUCED,
                size_multiplier=self.REDUCED_SIZE_MULTIPLIER,
                reason=f"Daily loss {daily.pnl_pct:.2%} - size reduced 50%",
            )

        return BreakerStatus(
            breaker_type=BreakerType.DAILY_LOSS,
            state=BreakerState.NORMAL,
            size_multiplier=1.0,
            reason="OK",
        )

    def _check_weekly_loss(self) -> BreakerStatus:
        """Check weekly loss breaker."""
        now = datetime.now(UTC)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        weekly_pnl = 0.0
        for date_str, stats in self._daily_stats.items():
            if stats.date >= week_start:
                weekly_pnl += stats.pnl_pct

        if weekly_pnl <= self.WEEKLY_LOSS_HALT:
            return BreakerStatus(
                breaker_type=BreakerType.WEEKLY_LOSS,
                state=BreakerState.HALTED,
                size_multiplier=0.0,
                reason=f"Weekly loss {weekly_pnl:.2%} exceeds {self.WEEKLY_LOSS_HALT:.1%} - halted",
            )

        return BreakerStatus(
            breaker_type=BreakerType.WEEKLY_LOSS,
            state=BreakerState.NORMAL,
            size_multiplier=1.0,
            reason="OK",
        )

    def _check_correlation(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
    ) -> BreakerStatus:
        """Check correlation breaker."""
        # Extract base asset from symbol (e.g., BTCUSDT → BTC)
        base_asset = symbol.replace("USDT", "").replace("cmt_", "").upper()

        for correlated_set in self.CORRELATED_PAIRS:
            if base_asset in correlated_set:
                # Check if we have a position in any correlated asset
                for pos_symbol, pos_direction in self._open_positions.items():
                    pos_base = pos_symbol.replace("USDT", "").replace("cmt_", "").upper()

                    if pos_base in correlated_set and pos_base != base_asset:
                        # Same direction on correlated assets = blocked
                        if pos_direction == direction:
                            return BreakerStatus(
                                breaker_type=BreakerType.CORRELATION,
                                state=BreakerState.HALTED,
                                size_multiplier=0.0,
                                reason=f"Cannot {direction} {symbol} - already {pos_direction} {pos_symbol} (correlated)",
                            )

        return BreakerStatus(
            breaker_type=BreakerType.CORRELATION,
            state=BreakerState.NORMAL,
            size_multiplier=1.0,
            reason="OK",
        )

    def _update_daily_stats(
        self,
        timestamp: datetime,
        pnl_pct: float,
        was_profitable: bool,
    ) -> None:
        """Update daily statistics."""
        date_str = timestamp.strftime("%Y-%m-%d")

        if date_str not in self._daily_stats:
            self._daily_stats[date_str] = DailyStats(date=timestamp)

        stats = self._daily_stats[date_str]
        stats.trades += 1
        stats.pnl_pct += pnl_pct

        if was_profitable:
            stats.wins += 1
        else:
            stats.losses += 1

        # Update peak and drawdown
        if stats.pnl_pct > stats.peak_pnl_pct:
            stats.peak_pnl_pct = stats.pnl_pct

        stats.drawdown_pct = stats.peak_pnl_pct - stats.pnl_pct

    def _build_halt_message(self, breakers: list[BreakerStatus]) -> str:
        """Build message for halted state."""
        halted = [b for b in breakers if b.state == BreakerState.HALTED]
        reasons = [b.reason for b in halted]
        return f"TRADING HALTED: {'; '.join(reasons)}"

    def _build_status_message(
        self,
        breakers: list[BreakerStatus],
        size_multiplier: float,
    ) -> str:
        """Build status message."""
        if not breakers or all(b.state == BreakerState.NORMAL for b in breakers):
            return "All circuit breakers normal"

        active = [b for b in breakers if b.state != BreakerState.NORMAL]
        reasons = [b.reason for b in active]
        return f"Size reduced to {size_multiplier:.0%}: {'; '.join(reasons)}"
