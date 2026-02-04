"""
Position Tracker for Live Trading

Tracks open positions, calculates P&L, and manages exit conditions
based on the Medallion V2 strategy parameters.

Exit conditions:
- Stop Loss: -1% (leveraged)
- Take Profit: +4% (leveraged)
- Time Exit: 36 hours
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """Represents an open position in the portfolio."""

    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    size: float  # USD notional
    entry_time: datetime
    strategy: str  # ml_tier1, ml_tier2, ml_tier3
    conviction: float
    order_id: str
    leverage: int = 5

    # Updated dynamically
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_price(self, price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = price

        if self.direction == "LONG":
            self.unrealized_pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # SHORT
            self.unrealized_pnl_pct = (self.entry_price - price) / self.entry_price

        # P&L in USD (leveraged)
        self.unrealized_pnl = self.size * self.unrealized_pnl_pct * self.leverage

    def holding_hours(self) -> float:
        """Calculate hours since entry."""
        now = datetime.now(UTC)
        delta = now - self.entry_time
        return delta.total_seconds() / 3600


@dataclass
class CompletedTrade:
    """Represents a completed (closed) trade."""

    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    strategy: str
    conviction: float
    exit_reason: str  # stop_loss, take_profit, time_exit, manual
    order_id: str
    leverage: int = 5

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "strategy": self.strategy,
            "conviction": round(self.conviction, 1),
            "exit_reason": self.exit_reason,
            "order_id": self.order_id,
            "leverage": self.leverage,
        }


@dataclass
class PositionTrackerConfig:
    """Configuration for position tracking and exit conditions."""

    stop_loss_pct: float = 0.01  # 1% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    time_exit_hours: int = 36  # 36 hour time exit
    max_portfolio_exposure: float = 0.40  # 40% max total exposure
    max_single_position: float = 0.05  # 5% max per position


class PositionTracker:
    """
    Tracks all open positions and manages exit conditions.

    Features:
    - Real-time P&L calculation
    - Exit condition checking (SL/TP/time)
    - Portfolio exposure tracking
    - Position reconciliation with exchange
    """

    def __init__(self, config: PositionTrackerConfig | None = None):
        self.config = config or PositionTrackerConfig()
        self.positions: dict[str, LivePosition] = {}
        self.closed_trades: list[CompletedTrade] = []
        self.total_pnl: float = 0.0

    def add_position(self, position: LivePosition) -> None:
        """Add a new position to tracking."""
        if position.symbol in self.positions:
            logger.warning(f"Position already exists for {position.symbol}, replacing")

        self.positions[position.symbol] = position
        logger.info(
            f"Added position: {position.direction} {position.symbol} "
            f"@ {position.entry_price:.2f}, size=${position.size:.2f}"
        )

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update all position prices from market data."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def check_exit_conditions(
        self, position: LivePosition
    ) -> tuple[bool, str]:
        """
        Check if a position should be exited.

        Returns:
            (should_exit, reason)
        """
        # Stop loss check
        if position.unrealized_pnl_pct <= -self.config.stop_loss_pct:
            return True, "stop_loss"

        # Take profit check
        if position.unrealized_pnl_pct >= self.config.take_profit_pct:
            return True, "take_profit"

        # Time exit check
        if position.holding_hours() >= self.config.time_exit_hours:
            return True, "time_exit"

        return False, ""

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
    ) -> CompletedTrade | None:
        """
        Close a position and record the trade.

        Returns:
            CompletedTrade if position existed, None otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot close {symbol}: position not found")
            return None

        position = self.positions[symbol]
        exit_time = datetime.now(UTC)

        # Calculate final P&L
        if position.direction == "LONG":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        pnl = position.size * pnl_pct * position.leverage

        # Create completed trade record
        trade = CompletedTrade(
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy=position.strategy,
            conviction=position.conviction,
            exit_reason=reason,
            order_id=position.order_id,
            leverage=position.leverage,
        )

        # Update tracking
        self.closed_trades.append(trade)
        self.total_pnl += pnl
        del self.positions[symbol]

        logger.info(
            f"Closed position: {position.direction} {symbol} "
            f"@ {exit_price:.2f}, P&L=${pnl:+.2f} ({reason})"
        )

        return trade

    def calculate_exposure(self, balance: float) -> float:
        """Calculate current portfolio exposure as percentage of balance."""
        if balance <= 0:
            return 0.0

        total_notional = sum(p.size for p in self.positions.values())
        return total_notional / balance

    def can_open_position(self, size: float, balance: float) -> tuple[bool, str]:
        """
        Check if a new position can be opened within risk limits.

        Returns:
            (can_open, rejection_reason)
        """
        # Check single position limit
        if size / balance > self.config.max_single_position:
            return False, f"Position size {size/balance:.1%} exceeds max {self.config.max_single_position:.0%}"

        # Check portfolio exposure limit
        current_exposure = self.calculate_exposure(balance)
        new_exposure = current_exposure + (size / balance)

        if new_exposure > self.config.max_portfolio_exposure:
            return False, f"Exposure {new_exposure:.1%} would exceed max {self.config.max_portfolio_exposure:.0%}"

        return True, ""

    def get_positions_to_exit(self) -> list[tuple[str, str]]:
        """
        Get list of positions that should be exited.

        Returns:
            List of (symbol, reason) tuples
        """
        exits = []
        for symbol, position in self.positions.items():
            should_exit, reason = self.check_exit_conditions(position)
            if should_exit:
                exits.append((symbol, reason))
        return exits

    def get_summary(self) -> dict:
        """Get summary of current positions and performance."""
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())

        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]

        return {
            "open_positions": len(self.positions),
            "total_realized_pnl": round(self.total_pnl, 2),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "total_trades": len(self.closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0,
            "positions": {
                symbol: {
                    "direction": p.direction,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl": round(p.unrealized_pnl, 2),
                    "pnl_pct": round(p.unrealized_pnl_pct * 100, 2),
                    "hours_held": round(p.holding_hours(), 1),
                }
                for symbol, p in self.positions.items()
            },
        }
