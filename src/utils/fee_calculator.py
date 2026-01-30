"""
AlphaStrike Trading Bot - Fee Calculator Module

Utility module for calculating trading fees, funding impacts, and breakeven
analysis for WEEX perpetual futures trading.
"""

from __future__ import annotations


class FeeCalculator:
    """
    Calculator for WEEX trading fees and cost analysis.

    WEEX Fee Structure:
    - Maker fee: 0.02% (limit orders that provide liquidity)
    - Taker fee: 0.05% (market orders that take liquidity)
    - Funding rate: Variable, applied every 8 hours

    Example:
        calculator = FeeCalculator()
        trade_cost = calculator.calculate_taker_fee(10000.0)  # $5.00 for $10k notional
        funding_cost = calculator.calculate_funding_impact(5000.0, 0.0003, 24)  # 24 hours
    """

    # WEEX fee rates
    MAKER_FEE_RATE: float = 0.0002  # 0.02%
    TAKER_FEE_RATE: float = 0.0005  # 0.05%
    FUNDING_INTERVALS_PER_DAY: int = 3  # Every 8 hours

    def calculate_maker_fee(self, notional: float) -> float:
        """
        Calculate maker fee for a limit order.

        Args:
            notional: Order notional value in USDT

        Returns:
            Fee amount in USDT

        Example:
            >>> calc = FeeCalculator()
            >>> calc.calculate_maker_fee(10000.0)
            2.0
        """
        return notional * self.MAKER_FEE_RATE

    def calculate_taker_fee(self, notional: float) -> float:
        """
        Calculate taker fee for a market order.

        Args:
            notional: Order notional value in USDT

        Returns:
            Fee amount in USDT

        Example:
            >>> calc = FeeCalculator()
            >>> calc.calculate_taker_fee(10000.0)
            5.0
        """
        return notional * self.TAKER_FEE_RATE

    def calculate_funding_impact(
        self,
        position_size: float,
        funding_rate: float,
        hours: int,
    ) -> float:
        """
        Calculate cumulative funding payment impact over a period.

        Funding is paid/received every 8 hours on WEEX.
        Positive funding rate: longs pay shorts
        Negative funding rate: shorts pay longs

        Args:
            position_size: Position size in USDT (absolute value)
            funding_rate: Current funding rate (e.g., 0.0003 for 0.03%)
            hours: Holding period in hours

        Returns:
            Total funding payment (positive = cost, negative = receipt)

        Example:
            >>> calc = FeeCalculator()
            >>> calc.calculate_funding_impact(5000.0, 0.0003, 24)
            4.5  # 3 funding periods in 24 hours
        """
        # Calculate number of funding periods in the given hours
        funding_periods = hours / 8.0

        # Total funding impact
        return position_size * funding_rate * funding_periods

    def calculate_fee_drag(
        self,
        trades_per_day: int,
        avg_notional: float,
    ) -> float:
        """
        Calculate daily fee drag from trading activity.

        Assumes a mix of maker and taker orders (conservative: all taker).

        Args:
            trades_per_day: Number of round-trip trades per day
            avg_notional: Average notional value per trade in USDT

        Returns:
            Daily fee cost in USDT

        Example:
            >>> calc = FeeCalculator()
            >>> calc.calculate_fee_drag(10, 1000.0)
            10.0  # 10 trades * $1000 * 0.05% * 2 (round-trip)
        """
        # Round-trip means entry + exit, so 2x the fee
        fee_per_round_trip = avg_notional * self.TAKER_FEE_RATE * 2
        return trades_per_day * fee_per_round_trip

    def calculate_breakeven_move(
        self,
        entry_price: float,
        is_taker: bool,
        funding_rate: float,
    ) -> float:
        """
        Calculate minimum price move needed to break even after fees.

        Accounts for entry fee, exit fee (assumed same type), and one
        funding payment if position is held across a funding period.

        Args:
            entry_price: Entry price in USDT
            is_taker: Whether using taker (market) orders
            funding_rate: Expected funding rate during holding period

        Returns:
            Breakeven price movement as absolute price change

        Example:
            >>> calc = FeeCalculator()
            >>> calc.calculate_breakeven_move(50000.0, True, 0.0001)
            55.0  # Need $55 move on BTC at $50k with taker fees
        """
        fee_rate = self.TAKER_FEE_RATE if is_taker else self.MAKER_FEE_RATE

        # Total cost rate: entry fee + exit fee + funding
        total_cost_rate = (fee_rate * 2) + abs(funding_rate)

        # Breakeven price movement
        return entry_price * total_cost_rate
