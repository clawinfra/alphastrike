"""
Order Execution Simulator

Simulates order fills with realistic slippage and fee models.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class SimulatedFill:
    """Result of a simulated order execution."""

    order_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    size: float
    entry_price: float
    slippage: float
    fees: float
    timestamp: datetime


class ExecutionSimulator:
    """
    Simulates order execution with realistic slippage and fees.

    Slippage model:
    - Base slippage in basis points
    - Volatility adjustment: increases slippage in high volatility
    - Direction: Always against the trader (worse price)
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
    ):
        """
        Initialize execution simulator.

        Args:
            slippage_bps: Base slippage in basis points (default: 5 = 0.05%)
            maker_fee: Maker fee rate (default: 0.02%)
            taker_fee: Taker fee rate (default: 0.05%)
        """
        self.slippage_bps = slippage_bps
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def execute_market_order(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        size: float,
        current_price: float,
        timestamp: datetime,
        atr_ratio: float = 0.0,
    ) -> SimulatedFill:
        """
        Simulate a market order execution.

        Args:
            symbol: Trading pair
            side: Order side (LONG or SHORT)
            size: Order size in base currency
            current_price: Current market price
            timestamp: Execution timestamp
            atr_ratio: ATR as percentage of price (for volatility adjustment)

        Returns:
            SimulatedFill with execution details
        """
        # Calculate slippage
        # Base slippage + volatility adjustment (+1 bp per 0.5% ATR ratio)
        volatility_adjustment = (atr_ratio / 0.005) if atr_ratio > 0 else 0
        total_slippage_bps = self.slippage_bps + volatility_adjustment
        slippage_rate = total_slippage_bps / 10000

        # Apply slippage against trader
        if side == "LONG":
            # Buying: price goes up (worse for buyer)
            fill_price = current_price * (1 + slippage_rate)
        else:
            # Selling: price goes down (worse for seller)
            fill_price = current_price * (1 - slippage_rate)

        # Calculate fees (market orders use taker fee)
        notional = fill_price * size
        fees = notional * self.taker_fee

        # Calculate actual slippage in absolute terms
        slippage = abs(fill_price - current_price) * size

        return SimulatedFill(
            order_id=f"bt_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,
            size=size,
            entry_price=fill_price,
            slippage=slippage,
            fees=fees,
            timestamp=timestamp,
        )

    def simulate_exit(
        self,
        side: Literal["LONG", "SHORT"],
        size: float,
        exit_price: float,
        atr_ratio: float = 0.0,
    ) -> tuple[float, float]:
        """
        Simulate exit execution and return adjusted price and fees.

        Args:
            side: Original position side
            size: Position size
            exit_price: Target exit price
            atr_ratio: ATR ratio for slippage calculation

        Returns:
            Tuple of (adjusted_exit_price, fees)
        """
        # Calculate slippage
        volatility_adjustment = (atr_ratio / 0.005) if atr_ratio > 0 else 0
        total_slippage_bps = self.slippage_bps + volatility_adjustment
        slippage_rate = total_slippage_bps / 10000

        # Apply slippage against trader (opposite direction for exit)
        if side == "LONG":
            # Exiting long: selling, price goes down
            adjusted_price = exit_price * (1 - slippage_rate)
        else:
            # Exiting short: buying back, price goes up
            adjusted_price = exit_price * (1 + slippage_rate)

        # Calculate fees
        notional = adjusted_price * size
        fees = notional * self.taker_fee

        return adjusted_price, fees

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        side: Literal["LONG", "SHORT"],
        entry_fees: float,
        exit_fees: float,
    ) -> float:
        """
        Calculate realized P&L for a closed trade.

        Args:
            entry_price: Entry fill price
            exit_price: Exit fill price
            size: Position size
            side: Position side
            entry_fees: Entry fees
            exit_fees: Exit fees

        Returns:
            Realized P&L (negative for loss)
        """
        if side == "LONG":
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size

        net_pnl = gross_pnl - entry_fees - exit_fees
        return net_pnl
