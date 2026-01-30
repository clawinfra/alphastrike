"""
AlphaStrike Trading Bot - Portfolio State Manager (US-018)

Centralized portfolio state tracking with position management,
exposure calculations, and P&L monitoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents an open trading position.

    Attributes:
        symbol: Trading pair symbol (e.g., "cmt_btcusdt")
        side: Position direction ("LONG" or "SHORT")
        size: Position size in base currency
        entry_price: Average entry price
        entry_time: When the position was opened
        leverage: Position leverage multiplier
        unrealized_pnl: Current unrealized profit/loss
        current_price: Latest market price for the position
    """
    symbol: str
    side: Literal["LONG", "SHORT"]
    size: float
    entry_price: float
    entry_time: datetime
    leverage: int
    unrealized_pnl: float = 0.0
    current_price: float = 0.0

    def __post_init__(self) -> None:
        """Initialize current_price to entry_price if not set."""
        if self.current_price == 0.0:
            self.current_price = self.entry_price

    @property
    def notional_value(self) -> float:
        """Calculate the notional value of the position."""
        return self.size * self.current_price

    @property
    def position_key(self) -> tuple[str, str]:
        """Return unique key for this position (symbol, side)."""
        return (self.symbol, self.side)

    def calculate_unrealized_pnl(self) -> float:
        """
        Calculate unrealized P&L based on current price.

        Returns:
            Unrealized P&L in quote currency
        """
        if self.side == "LONG":
            pnl = (self.current_price - self.entry_price) * self.size
        else:  # SHORT
            pnl = (self.entry_price - self.current_price) * self.size
        return pnl


class PortfolioManager:
    """
    Manages portfolio state including positions, exposure, and P&L tracking.

    Tracks:
        - All open positions indexed by (symbol, side)
        - Peak balance for drawdown calculation
        - Daily start balance for daily P&L
        - Total realized P&L

    Example:
        manager = PortfolioManager(initial_balance=10000.0)
        position = Position(
            symbol="cmt_btcusdt",
            side="LONG",
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.now(),
            leverage=10
        )
        manager.add_position(position)
        exposure = manager.get_total_exposure()
    """

    def __init__(self, initial_balance: float = 10000.0) -> None:
        """
        Initialize PortfolioManager.

        Args:
            initial_balance: Starting account balance in USDT
        """
        self._positions: dict[tuple[str, str], Position] = {}
        self._balance: float = initial_balance
        self._peak_balance: float = initial_balance
        self._daily_start_balance: float = initial_balance
        self._realized_pnl: float = 0.0

        logger.info(
            "PortfolioManager initialized",
            extra={
                "initial_balance": initial_balance,
            }
        )

    @property
    def balance(self) -> float:
        """Current account balance."""
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        """Set balance and update peak if needed."""
        self._balance = value
        if value > self._peak_balance:
            self._peak_balance = value

    @property
    def peak_balance(self) -> float:
        """Peak balance for drawdown calculation."""
        return self._peak_balance

    @property
    def daily_start_balance(self) -> float:
        """Balance at the start of the trading day."""
        return self._daily_start_balance

    def reset_daily_balance(self) -> None:
        """Reset daily start balance (call at start of each trading day)."""
        self._daily_start_balance = self._balance
        logger.info(
            "Daily balance reset",
            extra={"daily_start_balance": self._daily_start_balance}
        )

    def reset_peak_balance(self) -> None:
        """Reset peak balance to current balance."""
        self._peak_balance = self._balance
        logger.info(
            "Peak balance reset",
            extra={"peak_balance": self._peak_balance}
        )

    def add_position(self, position: Position) -> None:
        """
        Add a new position to the portfolio.

        Args:
            position: Position to add

        Raises:
            ValueError: If position with same symbol/side already exists
        """
        key = position.position_key
        if key in self._positions:
            raise ValueError(
                f"Position already exists for {position.symbol} {position.side}. "
                "Use update methods to modify existing positions."
            )

        self._positions[key] = position
        logger.info(
            "Position added",
            extra={
                "symbol": position.symbol,
                "side": position.side,
                "size": position.size,
                "entry_price": position.entry_price,
                "leverage": position.leverage,
            }
        )

    def remove_position(self, symbol: str, side: str) -> None:
        """
        Remove a position from the portfolio.

        Args:
            symbol: Trading pair symbol
            side: Position side ("LONG" or "SHORT")

        Raises:
            KeyError: If position does not exist
        """
        key = (symbol, side)
        if key not in self._positions:
            raise KeyError(f"No position found for {symbol} {side}")

        removed = self._positions.pop(key)
        logger.info(
            "Position removed",
            extra={
                "symbol": symbol,
                "side": side,
                "size": removed.size,
                "unrealized_pnl": removed.unrealized_pnl,
            }
        )

    def update_position_price(self, symbol: str, side: str, price: float) -> None:
        """
        Update the current price and recalculate unrealized P&L for a position.

        Args:
            symbol: Trading pair symbol
            side: Position side ("LONG" or "SHORT")
            price: New market price

        Raises:
            KeyError: If position does not exist
        """
        key = (symbol, side)
        if key not in self._positions:
            raise KeyError(f"No position found for {symbol} {side}")

        position = self._positions[key]
        position.current_price = price
        position.unrealized_pnl = position.calculate_unrealized_pnl()

    def get_position(self, symbol: str, side: str) -> Position | None:
        """
        Get a specific position.

        Args:
            symbol: Trading pair symbol
            side: Position side ("LONG" or "SHORT")

        Returns:
            Position if found, None otherwise
        """
        return self._positions.get((symbol, side))

    def get_all_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of all Position objects
        """
        return list(self._positions.values())

    def get_total_exposure(self) -> float:
        """
        Calculate total portfolio exposure as ratio of notional value to balance.

        Returns:
            Total exposure ratio (e.g., 0.8 = 80% exposure)
        """
        if self._balance <= 0:
            return 0.0

        total_notional = sum(pos.notional_value for pos in self._positions.values())
        return total_notional / self._balance

    def get_exposure_by_symbol(self, symbol: str) -> float:
        """
        Calculate exposure for a specific symbol (both LONG and SHORT combined).

        Args:
            symbol: Trading pair symbol

        Returns:
            Symbol exposure ratio relative to balance
        """
        if self._balance <= 0:
            return 0.0

        symbol_notional = sum(
            pos.notional_value
            for pos in self._positions.values()
            if pos.symbol == symbol
        )
        return symbol_notional / self._balance

    def get_unrealized_pnl(self) -> float:
        """
        Calculate total unrealized P&L across all positions.

        Returns:
            Total unrealized P&L in quote currency
        """
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def get_daily_pnl(self) -> float:
        """
        Calculate daily P&L (realized + unrealized since day start).

        Returns:
            Daily P&L in quote currency
        """
        current_equity = self._balance + self.get_unrealized_pnl()
        return current_equity - self._daily_start_balance

    def get_drawdown(self) -> float:
        """
        Calculate current drawdown from peak balance.

        Returns:
            Drawdown as a ratio (e.g., 0.05 = 5% drawdown)
        """
        if self._peak_balance <= 0:
            return 0.0

        current_equity = self._balance + self.get_unrealized_pnl()
        drawdown = (self._peak_balance - current_equity) / self._peak_balance
        return max(0.0, drawdown)

    def sync_with_exchange(self, exchange_positions: list[Position]) -> None:
        """
        Synchronize portfolio state with exchange positions.

        Replaces all local positions with exchange positions.
        This should be called periodically to ensure consistency.

        Args:
            exchange_positions: List of positions from the exchange
        """
        old_positions = set(self._positions.keys())
        new_positions = {pos.position_key for pos in exchange_positions}

        # Log discrepancies
        removed = old_positions - new_positions
        added = new_positions - old_positions

        if removed:
            logger.warning(
                "Positions removed during sync",
                extra={"removed": [f"{s}:{side}" for s, side in removed]}
            )
        if added:
            logger.info(
                "New positions from exchange",
                extra={"added": [f"{s}:{side}" for s, side in added]}
            )

        # Replace all positions
        self._positions.clear()
        for position in exchange_positions:
            self._positions[position.position_key] = position

        logger.info(
            "Portfolio synced with exchange",
            extra={
                "position_count": len(self._positions),
                "total_exposure": self.get_total_exposure(),
            }
        )

    def add_realized_pnl(self, pnl: float) -> None:
        """
        Add realized P&L to balance.

        Args:
            pnl: Realized profit/loss amount
        """
        self._realized_pnl += pnl
        self.balance = self._balance + pnl
        logger.info(
            "Realized P&L added",
            extra={
                "pnl": pnl,
                "new_balance": self._balance,
                "total_realized": self._realized_pnl,
            }
        )

    def get_position_count(self) -> int:
        """Return the number of open positions."""
        return len(self._positions)

    def has_position(self, symbol: str, side: str | None = None) -> bool:
        """
        Check if a position exists for the given symbol.

        Args:
            symbol: Trading pair symbol
            side: Optional side filter ("LONG" or "SHORT")

        Returns:
            True if position exists
        """
        if side:
            return (symbol, side) in self._positions
        return any(pos.symbol == symbol for pos in self._positions.values())

    def get_summary(self) -> dict[str, float | int]:
        """
        Get portfolio summary statistics.

        Returns:
            Dictionary with portfolio metrics
        """
        return {
            "balance": self._balance,
            "peak_balance": self._peak_balance,
            "daily_start_balance": self._daily_start_balance,
            "position_count": self.get_position_count(),
            "total_exposure": self.get_total_exposure(),
            "unrealized_pnl": self.get_unrealized_pnl(),
            "daily_pnl": self.get_daily_pnl(),
            "drawdown": self.get_drawdown(),
            "realized_pnl": self._realized_pnl,
        }
