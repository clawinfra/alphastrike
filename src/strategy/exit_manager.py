"""
AlphaStrike Trading Bot - Exit Manager (US-022)

Manages position exits with multi-level take profit, trailing stops,
and time-based exit strategies. Adapts to market regime conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from src.risk.portfolio import Position

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    """Reasons for exiting a position."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TAKE_PROFIT_3 = "take_profit_3"
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    SIGNAL_REVERSAL = "signal_reversal"
    MANUAL = "manual"


@dataclass
class ExitPrices:
    """
    Calculated exit price levels for a position.

    Attributes:
        stop_loss: Initial stop-loss price
        tp1_price: First take-profit level (1.5x ATR)
        tp1_size_pct: Percentage of position to close at TP1 (40%)
        tp2_price: Second take-profit level (2.5x ATR)
        tp2_size_pct: Percentage of position to close at TP2 (35%)
        trailing_distance: Distance for trailing stop in price units
    """

    stop_loss: float
    tp1_price: float
    tp1_size_pct: float
    tp2_price: float
    tp2_size_pct: float
    trailing_distance: float


@dataclass
class PositionExitState:
    """
    Tracks the exit state for an active position.

    Used to manage multi-level take profits and trailing stop adjustments.
    """

    symbol: str
    side: Literal["LONG", "SHORT"]
    entry_price: float
    entry_time: datetime
    exit_prices: ExitPrices
    tp1_hit: bool = False
    tp2_hit: bool = False
    trailing_active: bool = False
    trailing_stop_price: float = 0.0
    highest_price: float = 0.0  # For LONG trailing
    lowest_price: float = 0.0  # For SHORT trailing


class ExitManager:
    """
    Manages position exits with multi-level take profit and trailing stops.

    Exit Strategy Architecture:
    - Multi-Level Take Profit:
        - TP1: 40% of position at 1.5x ATR
        - TP2: 35% of position at 2.5x ATR
        - TP3: 25% of position with trailing stop
    - Trailing Stop:
        - Initial: 2.0x ATR from entry
        - Trail distance: 1.5x ATR
        - Break-even + 0.3x ATR buffer after TP1 hit
    - Time-Based Exits:
        - Maximum hold: 24-96 hours (adaptive)
        - Extension for profitable trades
        - Extension for trending markets

    Regime Multipliers:
        - trending: 1.25x wider stops (more room for trends)
        - ranging: 0.8x tighter stops (quick exits in chop)
        - high_volatility: 2.0x wider stops (accommodate swings)
        - exhaustion: 0.7x tighter stops (quick exits before reversal)

    Example:
        exit_manager = ExitManager()
        exit_prices = exit_manager.calculate_exit_prices(
            entry_price=50000.0,
            side="LONG",
            atr=500.0,
            regime="trending_up"
        )
        should_exit, reason = exit_manager.should_exit(
            position=position,
            current_price=51000.0,
            atr=500.0,
            regime="trending_up"
        )
    """

    # Take profit levels in ATR multiples
    TP1_ATR_MULTIPLIER = 1.5
    TP2_ATR_MULTIPLIER = 2.5

    # Take profit size percentages
    TP1_SIZE_PCT = 0.40
    TP2_SIZE_PCT = 0.35
    TP3_SIZE_PCT = 0.25  # Remaining position trails

    # Stop loss and trailing parameters
    INITIAL_STOP_ATR = 2.0
    TRAILING_DISTANCE_ATR = 1.5
    BREAKEVEN_BUFFER_ATR = 0.3

    # Time-based exit parameters (hours)
    BASE_MAX_HOLD_HOURS = 48
    MIN_MAX_HOLD_HOURS = 24
    MAX_MAX_HOLD_HOURS = 96

    # Regime multipliers for stop distances
    REGIME_MULTIPLIERS: dict[str, float] = {
        "trending_up": 1.25,
        "trending_down": 1.25,
        "trending": 1.25,
        "ranging": 0.8,
        "high_volatility": 2.0,
        "extreme_volatility": 2.0,
        "exhaustion": 0.7,
        "trend_exhaustion": 0.7,
    }

    def __init__(self) -> None:
        """Initialize ExitManager."""
        # Track exit states for active positions
        self._exit_states: dict[tuple[str, str], PositionExitState] = {}
        logger.info("ExitManager initialized")

    def calculate_exit_prices(
        self,
        entry_price: float,
        side: str,
        atr: float,
        regime: str,
    ) -> ExitPrices:
        """
        Calculate exit price levels for a new position.

        Args:
            entry_price: Position entry price
            side: Position side ("LONG" or "SHORT")
            atr: Current ATR value
            regime: Current market regime

        Returns:
            ExitPrices with calculated stop and take profit levels
        """
        # Get regime multiplier
        multiplier = self._get_regime_multiplier(regime)

        # Adjust ATR-based distances by regime
        adjusted_atr = atr * multiplier

        if side.upper() == "LONG":
            # LONG: stops below, profits above
            stop_loss = entry_price - (self.INITIAL_STOP_ATR * adjusted_atr)
            tp1_price = entry_price + (self.TP1_ATR_MULTIPLIER * adjusted_atr)
            tp2_price = entry_price + (self.TP2_ATR_MULTIPLIER * adjusted_atr)
        else:
            # SHORT: stops above, profits below
            stop_loss = entry_price + (self.INITIAL_STOP_ATR * adjusted_atr)
            tp1_price = entry_price - (self.TP1_ATR_MULTIPLIER * adjusted_atr)
            tp2_price = entry_price - (self.TP2_ATR_MULTIPLIER * adjusted_atr)

        trailing_distance = self.TRAILING_DISTANCE_ATR * adjusted_atr

        exit_prices = ExitPrices(
            stop_loss=stop_loss,
            tp1_price=tp1_price,
            tp1_size_pct=self.TP1_SIZE_PCT,
            tp2_price=tp2_price,
            tp2_size_pct=self.TP2_SIZE_PCT,
            trailing_distance=trailing_distance,
        )

        logger.debug(
            "Exit prices calculated",
            extra={
                "entry_price": entry_price,
                "side": side,
                "atr": atr,
                "regime": regime,
                "multiplier": multiplier,
                "stop_loss": stop_loss,
                "tp1_price": tp1_price,
                "tp2_price": tp2_price,
                "trailing_distance": trailing_distance,
            },
        )

        return exit_prices

    def register_position(
        self,
        position: Position,
        atr: float,
        regime: str,
    ) -> ExitPrices:
        """
        Register a new position for exit management.

        Creates exit state tracking and calculates initial exit prices.

        Args:
            position: Position to register
            atr: Current ATR value
            regime: Current market regime

        Returns:
            Calculated ExitPrices for the position
        """
        exit_prices = self.calculate_exit_prices(
            entry_price=position.entry_price,
            side=position.side,
            atr=atr,
            regime=regime,
        )

        # Create exit state
        exit_state = PositionExitState(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_prices=exit_prices,
            highest_price=position.entry_price,
            lowest_price=position.entry_price,
        )

        key = (position.symbol, position.side)
        self._exit_states[key] = exit_state

        logger.info(
            "Position registered for exit management",
            extra={
                "symbol": position.symbol,
                "side": position.side,
                "entry_price": position.entry_price,
                "stop_loss": exit_prices.stop_loss,
                "tp1": exit_prices.tp1_price,
                "tp2": exit_prices.tp2_price,
            },
        )

        return exit_prices

    def unregister_position(self, symbol: str, side: str) -> None:
        """
        Remove a position from exit management tracking.

        Args:
            symbol: Position symbol
            side: Position side
        """
        key = (symbol, side)
        if key in self._exit_states:
            del self._exit_states[key]
            logger.info(
                "Position unregistered from exit management",
                extra={"symbol": symbol, "side": side},
            )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        atr: float,
        regime: str,
    ) -> tuple[bool, str]:
        """
        Determine if a position should be exited.

        Checks all exit conditions in order of priority:
        1. Stop loss hit
        2. Trailing stop hit
        3. Take profit levels hit
        4. Time-based exit

        Args:
            position: Position to evaluate
            current_price: Current market price
            atr: Current ATR value
            regime: Current market regime

        Returns:
            Tuple of (should_exit: bool, reason: str)
            Reason is empty string if should_exit is False
        """
        key = (position.symbol, position.side)

        # Get or create exit state
        if key not in self._exit_states:
            self.register_position(position, atr, regime)

        exit_state = self._exit_states[key]

        # Update price extremes for trailing
        if position.side == "LONG":
            exit_state.highest_price = max(exit_state.highest_price, current_price)
        else:
            if exit_state.lowest_price == 0.0:
                exit_state.lowest_price = current_price
            else:
                exit_state.lowest_price = min(exit_state.lowest_price, current_price)

        # Check stop loss
        if self._check_stop_loss(position.side, current_price, exit_state):
            return True, ExitReason.STOP_LOSS.value

        # Check trailing stop
        if exit_state.trailing_active:
            if self._check_trailing_stop(position.side, current_price, exit_state):
                return True, ExitReason.TRAILING_STOP.value

        # Check take profit levels
        tp_hit, tp_reason = self._check_take_profits(
            position.side, current_price, exit_state, atr, regime
        )
        if tp_hit:
            return True, tp_reason

        # Check time-based exit
        if self._check_time_exit(position, current_price, regime):
            return True, ExitReason.TIME_BASED.value

        return False, ""

    def _check_stop_loss(
        self,
        side: str,
        current_price: float,
        exit_state: PositionExitState,
    ) -> bool:
        """Check if stop loss is hit."""
        if side == "LONG":
            return current_price <= exit_state.exit_prices.stop_loss
        else:
            return current_price >= exit_state.exit_prices.stop_loss

    def _check_trailing_stop(
        self,
        side: str,
        current_price: float,
        exit_state: PositionExitState,
    ) -> bool:
        """Check if trailing stop is hit."""
        if not exit_state.trailing_active:
            return False

        if side == "LONG":
            # Calculate trailing stop from highest price
            trail_stop = exit_state.highest_price - exit_state.exit_prices.trailing_distance
            exit_state.trailing_stop_price = max(
                exit_state.trailing_stop_price, trail_stop
            )
            return current_price <= exit_state.trailing_stop_price
        else:
            # Calculate trailing stop from lowest price
            trail_stop = exit_state.lowest_price + exit_state.exit_prices.trailing_distance
            if exit_state.trailing_stop_price == 0.0:
                exit_state.trailing_stop_price = trail_stop
            else:
                exit_state.trailing_stop_price = min(
                    exit_state.trailing_stop_price, trail_stop
                )
            return current_price >= exit_state.trailing_stop_price

    def _check_take_profits(
        self,
        side: str,
        current_price: float,
        exit_state: PositionExitState,
        atr: float,
        regime: str,
    ) -> tuple[bool, str]:
        """
        Check if take profit levels are hit.

        Returns tuple of (is_hit, reason) for the highest TP level hit.
        """
        exit_prices = exit_state.exit_prices

        if side == "LONG":
            # Check TP1
            if not exit_state.tp1_hit and current_price >= exit_prices.tp1_price:
                exit_state.tp1_hit = True
                self._update_stop_to_breakeven(exit_state, atr, regime)
                logger.info(
                    "TP1 hit - moving stop to breakeven",
                    extra={
                        "symbol": exit_state.symbol,
                        "side": exit_state.side,
                        "price": current_price,
                        "tp1_price": exit_prices.tp1_price,
                    },
                )
                return True, ExitReason.TAKE_PROFIT_1.value

            # Check TP2
            if not exit_state.tp2_hit and current_price >= exit_prices.tp2_price:
                exit_state.tp2_hit = True
                exit_state.trailing_active = True
                logger.info(
                    "TP2 hit - activating trailing stop",
                    extra={
                        "symbol": exit_state.symbol,
                        "side": exit_state.side,
                        "price": current_price,
                        "tp2_price": exit_prices.tp2_price,
                    },
                )
                return True, ExitReason.TAKE_PROFIT_2.value

        else:  # SHORT
            # Check TP1
            if not exit_state.tp1_hit and current_price <= exit_prices.tp1_price:
                exit_state.tp1_hit = True
                self._update_stop_to_breakeven(exit_state, atr, regime)
                logger.info(
                    "TP1 hit - moving stop to breakeven",
                    extra={
                        "symbol": exit_state.symbol,
                        "side": exit_state.side,
                        "price": current_price,
                        "tp1_price": exit_prices.tp1_price,
                    },
                )
                return True, ExitReason.TAKE_PROFIT_1.value

            # Check TP2
            if not exit_state.tp2_hit and current_price <= exit_prices.tp2_price:
                exit_state.tp2_hit = True
                exit_state.trailing_active = True
                logger.info(
                    "TP2 hit - activating trailing stop",
                    extra={
                        "symbol": exit_state.symbol,
                        "side": exit_state.side,
                        "price": current_price,
                        "tp2_price": exit_prices.tp2_price,
                    },
                )
                return True, ExitReason.TAKE_PROFIT_2.value

        return False, ""

    def _update_stop_to_breakeven(
        self,
        exit_state: PositionExitState,
        atr: float,
        regime: str,
    ) -> None:
        """
        Update stop loss to breakeven + buffer after TP1 hit.

        Moves the stop loss to entry price plus a small buffer (0.3x ATR)
        to protect profits while allowing for minor retracements.
        """
        multiplier = self._get_regime_multiplier(regime)
        buffer = self.BREAKEVEN_BUFFER_ATR * atr * multiplier

        if exit_state.side == "LONG":
            # Move stop to entry + buffer
            new_stop = exit_state.entry_price + buffer
            exit_state.exit_prices.stop_loss = new_stop
        else:
            # Move stop to entry - buffer
            new_stop = exit_state.entry_price - buffer
            exit_state.exit_prices.stop_loss = new_stop

        logger.debug(
            "Stop moved to breakeven + buffer",
            extra={
                "symbol": exit_state.symbol,
                "side": exit_state.side,
                "new_stop": new_stop,
                "buffer": buffer,
            },
        )

    def _check_time_exit(
        self,
        position: Position,
        current_price: float,
        regime: str,
    ) -> bool:
        """
        Check if time-based exit should be triggered.

        Args:
            position: Position to check
            current_price: Current market price
            regime: Current market regime

        Returns:
            True if maximum hold time exceeded
        """
        # Calculate how long position has been held
        hold_duration = datetime.now() - position.entry_time
        hold_hours = hold_duration.total_seconds() / 3600

        # Calculate if position is profitable
        if position.side == "LONG":
            is_profitable = current_price > position.entry_price
        else:
            is_profitable = current_price < position.entry_price

        # Get adaptive max hold hours
        max_hours = self.get_max_hold_hours(regime, is_profitable)

        if hold_hours >= max_hours:
            logger.info(
                "Time-based exit triggered",
                extra={
                    "symbol": position.symbol,
                    "side": position.side,
                    "hold_hours": hold_hours,
                    "max_hours": max_hours,
                    "is_profitable": is_profitable,
                    "regime": regime,
                },
            )
            return True

        return False

    def get_max_hold_hours(self, regime: str, is_profitable: bool) -> int:
        """
        Get adaptive maximum hold time based on regime and profitability.

        Rules:
        - Base: 48 hours
        - Profitable trades: +24 hours extension
        - Trending markets: +24 hours extension
        - Ranging/Exhaustion: -12 hours (quicker exit)
        - High volatility: no change to base

        Args:
            regime: Current market regime
            is_profitable: Whether position is currently profitable

        Returns:
            Maximum hold hours (24-96 range)
        """
        max_hours = self.BASE_MAX_HOLD_HOURS

        # Extension for profitable trades
        if is_profitable:
            max_hours += 24

        # Regime adjustments
        regime_lower = regime.lower()
        if regime_lower in ("trending_up", "trending_down", "trending"):
            max_hours += 24
        elif regime_lower in ("ranging", "exhaustion", "trend_exhaustion"):
            max_hours -= 12

        # Clamp to valid range
        max_hours = max(self.MIN_MAX_HOLD_HOURS, min(max_hours, self.MAX_MAX_HOLD_HOURS))

        return max_hours

    def _get_regime_multiplier(self, regime: str) -> float:
        """
        Get the stop distance multiplier for the given regime.

        Args:
            regime: Market regime string

        Returns:
            Multiplier for ATR-based distances
        """
        regime_lower = regime.lower()
        return self.REGIME_MULTIPLIERS.get(regime_lower, 1.0)

    def get_exit_state(self, symbol: str, side: str) -> PositionExitState | None:
        """
        Get the current exit state for a position.

        Args:
            symbol: Position symbol
            side: Position side

        Returns:
            PositionExitState if found, None otherwise
        """
        return self._exit_states.get((symbol, side))

    def get_remaining_size_pct(self, symbol: str, side: str) -> float:
        """
        Get the remaining position size percentage after take profits.

        Used to determine how much of the original position remains.

        Args:
            symbol: Position symbol
            side: Position side

        Returns:
            Remaining position size as decimal (0.0 to 1.0)
        """
        exit_state = self._exit_states.get((symbol, side))
        if exit_state is None:
            return 1.0

        remaining = 1.0
        if exit_state.tp1_hit:
            remaining -= self.TP1_SIZE_PCT
        if exit_state.tp2_hit:
            remaining -= self.TP2_SIZE_PCT

        return max(0.0, remaining)

    def get_all_exit_states(self) -> dict[tuple[str, str], PositionExitState]:
        """
        Get all tracked exit states.

        Returns:
            Dictionary of (symbol, side) -> PositionExitState
        """
        return dict(self._exit_states)

    def update_exit_prices(
        self,
        symbol: str,
        side: str,
        atr: float,
        regime: str,
    ) -> ExitPrices | None:
        """
        Recalculate exit prices for an existing position.

        Useful when ATR or regime changes significantly.
        Only updates if position is still tracked and TP1 hasn't been hit.

        Args:
            symbol: Position symbol
            side: Position side
            atr: New ATR value
            regime: Current market regime

        Returns:
            Updated ExitPrices or None if position not found
        """
        key = (symbol, side)
        exit_state = self._exit_states.get(key)

        if exit_state is None:
            return None

        # Don't update if TP1 already hit (breakeven stop in effect)
        if exit_state.tp1_hit:
            logger.debug(
                "Exit prices not updated - TP1 already hit",
                extra={"symbol": symbol, "side": side},
            )
            return exit_state.exit_prices

        # Recalculate exit prices
        new_exit_prices = self.calculate_exit_prices(
            entry_price=exit_state.entry_price,
            side=side,
            atr=atr,
            regime=regime,
        )

        exit_state.exit_prices = new_exit_prices

        logger.debug(
            "Exit prices updated",
            extra={
                "symbol": symbol,
                "side": side,
                "stop_loss": new_exit_prices.stop_loss,
                "tp1": new_exit_prices.tp1_price,
                "tp2": new_exit_prices.tp2_price,
            },
        )

        return new_exit_prices
