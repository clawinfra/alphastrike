"""
AlphaStrike Trading Bot - Order Manager with Execution (US-024)

Manages the complete order lifecycle from signal reception to fill tracking.
Handles slippage estimation, order type selection, leverage setting, risk
validation, and order placement with preset SL/TP.

Order Flow:
1. Signal received -> slippage estimation
2. Order type selection based on market conditions
3. Leverage setting (skipped if position already exists)
4. Risk validation before execution
5. Order placement with preset SL/TP
6. Fill tracking and result reporting
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from src.data.rest_client import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    PositionSide,
    RESTClient,
    RESTClientError,
    TimeInForce,
)

logger = logging.getLogger(__name__)


# Type alias for signal direction
SignalDirection = Literal["LONG", "SHORT", "HOLD"]


@dataclass
class SignalResult:
    """Result from ML ensemble prediction for order execution."""

    signal: SignalDirection
    confidence: float
    weighted_avg: float
    urgency: float = 0.5  # 0.0 = low urgency, 1.0 = high urgency
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


@dataclass
class ExecutionResult:
    """Result of order execution attempt."""

    success: bool
    order_id: str | None
    fill_price: float | None
    fill_size: float | None
    slippage: float | None
    error_message: str | None

    @classmethod
    def failure(cls, error_message: str) -> ExecutionResult:
        """Create a failure result with error message."""
        return cls(
            success=False,
            order_id=None,
            fill_price=None,
            fill_size=None,
            slippage=None,
            error_message=error_message,
        )

    @classmethod
    def from_order_result(
        cls,
        order_result: OrderResult,
        estimated_slippage: float | None = None,
    ) -> ExecutionResult:
        """Create execution result from order result."""
        return cls(
            success=True,
            order_id=order_result.order_id,
            fill_price=order_result.avg_fill_price,
            fill_size=order_result.filled_size if order_result.filled_size > 0 else order_result.size,
            slippage=estimated_slippage,
            error_message=None,
        )


class OrderTypeSelection(str, Enum):
    """Order type selection for execution strategy."""

    MARKET = "market"
    LIMIT_MID = "limit_mid"
    SPLIT_TWAP = "split_twap"


class OrderManager:
    """
    Manages order lifecycle from signal to execution.

    Implements intelligent order type selection based on:
    - High urgency -> Market order
    - Wide spread (> 0.05%) -> Limit at mid price
    - Large order (> 2% of book depth) -> Split + TWAP

    Includes slippage estimation, leverage management, risk validation,
    and preset SL/TP order placement.
    """

    # Thresholds for order type selection
    HIGH_URGENCY_THRESHOLD: float = 0.8
    WIDE_SPREAD_THRESHOLD: float = 0.0005  # 0.05%
    LARGE_ORDER_DEPTH_RATIO: float = 0.02  # 2% of book depth

    # TWAP parameters
    TWAP_SLICES: int = 5
    TWAP_INTERVAL_SECONDS: float = 2.0

    # Slippage estimation parameters
    BASE_SLIPPAGE_BPS: float = 5.0  # 5 basis points base slippage
    SIZE_IMPACT_FACTOR: float = 0.1  # Size impact multiplier

    def __init__(
        self,
        rest_client: RESTClient,
        risk_validator: RiskValidator | None = None,
    ) -> None:
        """
        Initialize Order Manager.

        Args:
            rest_client: REST client for exchange API communication.
            risk_validator: Optional risk validator for pre-trade checks.
        """
        self.rest_client = rest_client
        self.risk_validator = risk_validator

        logger.info("OrderManager initialized")

    async def execute_signal(
        self,
        signal: SignalResult,
        symbol: str,
        balance: float,
        position_size: float | None = None,
        leverage: int = 5,
    ) -> ExecutionResult:
        """
        Execute a trading signal through the complete order flow.

        Order Flow:
        1. Validate signal is actionable (not HOLD)
        2. Estimate slippage for the order
        3. Get market data for order type selection
        4. Select optimal order type
        5. Set leverage (skip if position exists)
        6. Validate risk limits
        7. Place order with preset SL/TP
        8. Track fill and return result

        Args:
            signal: Signal result from ML ensemble.
            symbol: Trading symbol (e.g., "cmt_btcusdt").
            balance: Available account balance in USDT.
            position_size: Calculated position size in base currency.
                          If None, uses full balance with leverage.
            leverage: Leverage to use (1-125, capped by config).

        Returns:
            ExecutionResult with order details or error information.
        """
        # Step 1: Validate signal is actionable
        if signal.signal == "HOLD":
            return ExecutionResult.failure("Signal is HOLD - no action required")

        if signal.confidence < 0.75:
            return ExecutionResult.failure(
                f"Signal confidence {signal.confidence:.2f} below minimum 0.75"
            )

        # Calculate order size if not provided
        if position_size is None or position_size <= 0:
            # Default to 10% of balance with leverage
            position_size = (balance * 0.10 * leverage)

        try:
            # Step 2: Get market data for slippage and order type
            ticker = await self.rest_client.get_ticker(symbol)
            orderbook = await self.rest_client.get_orderbook(symbol, limit=20)

            current_price = float(ticker.get("last", 0))
            if current_price <= 0:
                return ExecutionResult.failure("Invalid market price")

            # Calculate order size in contracts
            order_size = position_size / current_price

            # Step 3: Estimate slippage
            estimated_slippage = self.estimate_slippage(symbol, order_size, orderbook)

            # Step 4: Calculate spread and book depth for order type selection
            best_bid = float(orderbook.get("bids", [[0]])[0][0]) if orderbook.get("bids") else 0
            best_ask = float(orderbook.get("asks", [[0]])[0][0]) if orderbook.get("asks") else 0
            spread = (best_ask - best_bid) / current_price if current_price > 0 else 0

            book_depth = self._calculate_book_depth(orderbook, current_price)

            # Step 5: Select order type
            order_type_selection = self.select_order_type(
                spread=spread,
                size=position_size,
                book_depth=book_depth,
                urgency=signal.urgency,
            )

            # Step 6: Check if position exists for leverage setting
            positions = await self.rest_client.get_positions(symbol)
            position_exists = len(positions) > 0

            if not position_exists:
                # Set leverage before order
                await self.rest_client.set_leverage(symbol, leverage)

            # Step 7: Risk validation
            if self.risk_validator is not None:
                validation_result = self.risk_validator.validate_order(
                    symbol=symbol,
                    size=position_size,
                    balance=balance,
                    leverage=leverage,
                )
                if not validation_result.is_valid:
                    return ExecutionResult.failure(
                        f"Risk validation failed: {validation_result.rejection_reason}"
                    )

            # Step 8: Build and place order
            order_request = self._build_order_request(
                signal=signal,
                symbol=symbol,
                size=order_size,
                order_type_selection=order_type_selection,
                current_price=current_price,
                best_bid=best_bid,
                best_ask=best_ask,
            )

            # Step 9: Execute based on order type selection
            if order_type_selection == OrderTypeSelection.SPLIT_TWAP:
                result = await self._execute_twap(
                    order_request=order_request,
                    total_size=order_size,
                    slices=self.TWAP_SLICES,
                )
            else:
                result = await self.place_order_with_protection(order_request)

            # Update result with slippage estimate
            if result.success:
                result.slippage = estimated_slippage

            return result

        except RESTClientError as e:
            logger.error(f"Order execution failed: {e}")
            return ExecutionResult.failure(f"Exchange error: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error during order execution: {e}")
            return ExecutionResult.failure(f"Unexpected error: {str(e)}")

    def estimate_slippage(
        self,
        symbol: str,
        size: float,
        orderbook: dict | None = None,
    ) -> float:
        """
        Estimate expected slippage for an order.

        Uses orderbook depth analysis combined with size impact estimation.
        Slippage is returned as a decimal (e.g., 0.001 = 0.1%).

        Args:
            symbol: Trading symbol.
            size: Order size in base currency.
            orderbook: Orderbook data with bids and asks.

        Returns:
            Estimated slippage as decimal (0.001 = 0.1%).
        """
        # Base slippage in decimal
        base_slippage = self.BASE_SLIPPAGE_BPS / 10000

        if orderbook is None:
            return base_slippage

        # Calculate liquidity available at each price level
        total_liquidity = 0.0
        asks = orderbook.get("asks", [])
        bids = orderbook.get("bids", [])

        for level in asks[:10]:  # Top 10 ask levels
            if len(level) >= 2:
                total_liquidity += float(level[1])

        for level in bids[:10]:  # Top 10 bid levels
            if len(level) >= 2:
                total_liquidity += float(level[1])

        # Size impact: larger orders relative to liquidity have more slippage
        if total_liquidity > 0:
            size_ratio = size / total_liquidity
            size_impact = size_ratio * self.SIZE_IMPACT_FACTOR
        else:
            size_impact = base_slippage  # Double if no liquidity info

        estimated_slippage = base_slippage + size_impact

        logger.debug(
            f"Slippage estimate for {symbol}",
            extra={
                "size": size,
                "total_liquidity": total_liquidity,
                "base_slippage": base_slippage,
                "size_impact": size_impact,
                "estimated_slippage": estimated_slippage,
            },
        )

        return estimated_slippage

    def select_order_type(
        self,
        spread: float,
        size: float,
        book_depth: float,
        urgency: float,
    ) -> OrderTypeSelection:
        """
        Select optimal order type based on market conditions.

        Selection Logic:
        - High urgency (> 0.8) -> Market order
        - Wide spread (> 0.05%) -> Limit at mid price
        - Large order (> 2% of book depth) -> Split + TWAP
        - Default -> Market order

        Args:
            spread: Bid-ask spread as decimal (0.0005 = 0.05%).
            size: Order size in USDT notional.
            book_depth: Total orderbook depth in USDT.
            urgency: Signal urgency (0.0 to 1.0).

        Returns:
            OrderTypeSelection enum indicating recommended order type.
        """
        # High urgency: market order for immediate execution
        if urgency >= self.HIGH_URGENCY_THRESHOLD:
            logger.debug(
                f"Selected MARKET order: high urgency {urgency:.2f}"
            )
            return OrderTypeSelection.MARKET

        # Large order relative to book depth: split into TWAP
        if book_depth > 0:
            depth_ratio = size / book_depth
            if depth_ratio > self.LARGE_ORDER_DEPTH_RATIO:
                logger.debug(
                    f"Selected SPLIT_TWAP: large order {depth_ratio:.2%} of book depth"
                )
                return OrderTypeSelection.SPLIT_TWAP

        # Wide spread: use limit order at mid price
        if spread > self.WIDE_SPREAD_THRESHOLD:
            logger.debug(
                f"Selected LIMIT_MID: wide spread {spread:.4%}"
            )
            return OrderTypeSelection.LIMIT_MID

        # Default: market order
        logger.debug("Selected MARKET order: default")
        return OrderTypeSelection.MARKET

    async def place_order_with_protection(
        self,
        order: OrderRequest,
    ) -> ExecutionResult:
        """
        Place order with built-in SL/TP protection.

        Uses OrderRequest's preset_stop_loss_price and preset_take_profit_price
        fields for atomic order placement with risk protection.

        Args:
            order: OrderRequest with all order parameters.

        Returns:
            ExecutionResult with order details or error.
        """
        try:
            logger.info(
                f"Placing {order.order_type.value} {order.side.value} order",
                extra={
                    "symbol": order.symbol,
                    "size": order.size,
                    "price": order.price,
                    "sl_price": order.preset_stop_loss_price,
                    "tp_price": order.preset_take_profit_price,
                },
            )

            # Place the order
            order_result = await self.rest_client.place_order(order)

            # Track fill status
            if order.order_type == OrderType.LIMIT:
                # For limit orders, wait briefly for fill
                order_result = await self._wait_for_fill(
                    symbol=order.symbol,
                    order_id=order_result.order_id,
                    timeout_seconds=5.0,
                )

            return ExecutionResult.from_order_result(order_result)

        except RESTClientError as e:
            logger.error(f"Order placement failed: {e}")
            return ExecutionResult.failure(f"Order placement failed: {str(e)}")

    def _build_order_request(
        self,
        signal: SignalResult,
        symbol: str,
        size: float,
        order_type_selection: OrderTypeSelection,
        current_price: float,
        best_bid: float,
        best_ask: float,
    ) -> OrderRequest:
        """
        Build OrderRequest based on signal and market conditions.

        Args:
            signal: Signal result with direction and SL/TP prices.
            symbol: Trading symbol.
            size: Order size in base currency.
            order_type_selection: Selected order type strategy.
            current_price: Current market price.
            best_bid: Best bid price.
            best_ask: Best ask price.

        Returns:
            Configured OrderRequest ready for submission.
        """
        # Determine order side and position side
        if signal.signal == "LONG":
            side = OrderSide.BUY
            position_side = PositionSide.LONG
        else:  # SHORT
            side = OrderSide.SELL
            position_side = PositionSide.SHORT

        # Determine order type and price
        if order_type_selection == OrderTypeSelection.MARKET:
            order_type = OrderType.MARKET
            price = None
            time_in_force = TimeInForce.IOC
        else:  # LIMIT_MID or SPLIT_TWAP (individual slices use limit)
            order_type = OrderType.LIMIT
            # Calculate mid price
            if best_bid > 0 and best_ask > 0:
                mid_price = (best_bid + best_ask) / 2
            else:
                mid_price = current_price
            price = mid_price
            time_in_force = TimeInForce.GTC

        # Generate client order ID
        client_order_id = f"as_{uuid.uuid4().hex[:16]}"

        return OrderRequest(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            position_side=position_side,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            preset_stop_loss_price=signal.stop_loss_price,
            preset_take_profit_price=signal.take_profit_price,
        )

    def _calculate_book_depth(
        self,
        orderbook: dict,
        current_price: float,
    ) -> float:
        """
        Calculate total orderbook depth in USDT.

        Args:
            orderbook: Orderbook data with bids and asks.
            current_price: Current market price for value calculation.

        Returns:
            Total book depth in USDT notional value.
        """
        total_depth = 0.0

        asks = orderbook.get("asks", [])
        bids = orderbook.get("bids", [])

        for level in asks[:20]:
            if len(level) >= 2:
                price = float(level[0])
                size = float(level[1])
                total_depth += price * size

        for level in bids[:20]:
            if len(level) >= 2:
                price = float(level[0])
                size = float(level[1])
                total_depth += price * size

        return total_depth

    async def _execute_twap(
        self,
        order_request: OrderRequest,
        total_size: float,
        slices: int,
    ) -> ExecutionResult:
        """
        Execute order using TWAP (Time-Weighted Average Price) strategy.

        Splits the order into smaller slices executed at regular intervals
        to minimize market impact.

        Args:
            order_request: Base order request template.
            total_size: Total order size to execute.
            slices: Number of slices to split order into.

        Returns:
            Aggregated ExecutionResult from all slices.
        """
        slice_size = total_size / slices
        filled_size = 0.0
        total_value = 0.0
        order_ids: list[str] = []
        errors: list[str] = []

        logger.info(
            f"Executing TWAP: {slices} slices of {slice_size:.6f}",
            extra={
                "symbol": order_request.symbol,
                "total_size": total_size,
                "slices": slices,
            },
        )

        for i in range(slices):
            # Create slice order request
            slice_order = OrderRequest(
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=OrderType.MARKET,  # Use market for TWAP slices
                size=slice_size,
                price=None,
                position_side=order_request.position_side,
                time_in_force=TimeInForce.IOC,
                client_order_id=f"as_twap_{uuid.uuid4().hex[:12]}_{i}",
                # Only set SL/TP on last slice
                preset_stop_loss_price=(
                    order_request.preset_stop_loss_price if i == slices - 1 else None
                ),
                preset_take_profit_price=(
                    order_request.preset_take_profit_price if i == slices - 1 else None
                ),
            )

            try:
                result = await self.rest_client.place_order(slice_order)
                order_ids.append(result.order_id)

                if result.filled_size > 0:
                    filled_size += result.filled_size
                    if result.avg_fill_price:
                        total_value += result.filled_size * result.avg_fill_price
                else:
                    # Assume full fill if no fill info
                    filled_size += slice_size

                logger.debug(
                    f"TWAP slice {i + 1}/{slices} completed",
                    extra={
                        "order_id": result.order_id,
                        "filled_size": result.filled_size,
                    },
                )

            except RESTClientError as e:
                errors.append(f"Slice {i + 1}: {str(e)}")
                logger.warning(f"TWAP slice {i + 1} failed: {e}")

            # Wait between slices (except for last one)
            if i < slices - 1:
                await asyncio.sleep(self.TWAP_INTERVAL_SECONDS)

        # Calculate average fill price
        avg_fill_price = total_value / filled_size if filled_size > 0 else None

        if filled_size > 0:
            return ExecutionResult(
                success=True,
                order_id=",".join(order_ids) if order_ids else None,
                fill_price=avg_fill_price,
                fill_size=filled_size,
                slippage=None,  # Will be updated by caller
                error_message="; ".join(errors) if errors else None,
            )
        else:
            return ExecutionResult.failure(
                f"TWAP execution failed: {'; '.join(errors)}"
            )

    async def _wait_for_fill(
        self,
        symbol: str,
        order_id: str,
        timeout_seconds: float = 5.0,
        poll_interval: float = 0.5,
    ) -> OrderResult:
        """
        Wait for order fill with timeout.

        Polls order status until filled or timeout reached.

        Args:
            symbol: Trading symbol.
            order_id: Order ID to track.
            timeout_seconds: Maximum time to wait for fill.
            poll_interval: Time between status checks.

        Returns:
            Updated OrderResult with fill information.
        """
        elapsed = 0.0

        while elapsed < timeout_seconds:
            try:
                order_info = await self.rest_client.get_order(symbol, order_id)

                status = order_info.get("state", "").lower()
                filled_size = float(order_info.get("filledQty", 0))
                avg_price = float(order_info.get("priceAvg", 0)) if order_info.get("priceAvg") else None

                if status in ("filled", "full_fill"):
                    return OrderResult(
                        order_id=order_id,
                        client_order_id=order_info.get("clientOid"),
                        symbol=symbol,
                        side=OrderSide(order_info.get("side", "buy")),
                        order_type=OrderType(order_info.get("orderType", "market")),
                        size=float(order_info.get("size", 0)),
                        price=float(order_info.get("price", 0)) if order_info.get("price") else None,
                        status=status,
                        filled_size=filled_size,
                        avg_fill_price=avg_price,
                        raw_response=order_info,
                    )

                if status in ("canceled", "cancelled", "rejected"):
                    # Order was cancelled or rejected
                    return OrderResult(
                        order_id=order_id,
                        client_order_id=order_info.get("clientOid"),
                        symbol=symbol,
                        side=OrderSide(order_info.get("side", "buy")),
                        order_type=OrderType(order_info.get("orderType", "market")),
                        size=float(order_info.get("size", 0)),
                        price=float(order_info.get("price", 0)) if order_info.get("price") else None,
                        status=status,
                        filled_size=filled_size,
                        avg_fill_price=avg_price,
                        raw_response=order_info,
                    )

            except RESTClientError as e:
                logger.warning(f"Failed to get order status: {e}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout - return partial result
        logger.warning(f"Order {order_id} fill timeout after {timeout_seconds}s")
        return OrderResult(
            order_id=order_id,
            client_order_id=None,
            symbol=symbol,
            side=OrderSide.BUY,  # Placeholder
            order_type=OrderType.LIMIT,
            size=0,
            price=None,
            status="timeout",
            filled_size=0,
            avg_fill_price=None,
            raw_response={},
        )


@dataclass
class RiskValidationResult:
    """Result of risk validation check."""

    is_valid: bool
    rejection_reason: str | None


class RiskValidator:
    """
    Interface for risk validation before order execution.

    This is a placeholder interface. The actual implementation
    should be provided by the risk management module (US-021).
    """

    def validate_order(
        self,
        symbol: str,
        size: float,
        balance: float,
        leverage: int,
    ) -> RiskValidationResult:
        """
        Validate order against risk limits.

        Args:
            symbol: Trading symbol.
            size: Order size in USDT notional.
            balance: Account balance.
            leverage: Leverage being used.

        Returns:
            RiskValidationResult indicating if order is valid.
        """
        # Default implementation - always valid
        # Override with actual risk manager implementation
        return RiskValidationResult(is_valid=True, rejection_reason=None)
