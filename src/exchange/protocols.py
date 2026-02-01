"""
Exchange Protocol Definitions

Defines abstract interfaces that all exchange adapters must implement.
Uses Python's Protocol for structural subtyping (duck typing with type hints).

Design Principles:
- Protocol-based: No inheritance required, just implement the interface
- Async-first: All I/O operations are async
- Unified types: Uses models from models.py
- Testable: Easy to mock for unit tests
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Protocol, runtime_checkable

from src.exchange.models import (
    ExchangeCapabilities,
    OrderSide,
    PositionSide,
    SymbolInfo,
    UnifiedAccountBalance,
    UnifiedCandle,
    UnifiedOrder,
    UnifiedOrderbook,
    UnifiedOrderResult,
    UnifiedPosition,
    UnifiedTicker,
    UnifiedTrade,
)


@runtime_checkable
class AuthenticationScheme(Protocol):
    """
    Protocol for exchange authentication implementations.

    Different exchanges use different authentication schemes:
    - HMAC-SHA256 (WEEX, Binance)
    - Ed25519 (some DEXs)
    - Wallet signing (Hyperliquid, other DEXs)
    """

    def sign_request(
        self,
        method: str,
        path: str,
        body: str,
        timestamp: str,
    ) -> dict[str, str]:
        """
        Generate authentication headers for a request.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API endpoint path (e.g., "/api/v1/order")
            body: Request body (empty string for GET)
            timestamp: Request timestamp in milliseconds

        Returns:
            Dictionary of authentication headers to add to the request
        """
        ...


@runtime_checkable
class ExchangeRESTProtocol(Protocol):
    """
    Protocol defining REST API operations for any exchange.

    All exchange REST adapters must implement these methods.
    Adapters handle translation between unified models and exchange-specific formats.

    Example:
        class MyExchangeREST:
            async def get_ticker(self, symbol: str) -> UnifiedTicker:
                # Translate symbol to exchange format
                # Make API call
                # Parse response into UnifiedTicker
                ...
    """

    # ==================== Lifecycle ====================

    async def initialize(self) -> None:
        """
        Initialize the exchange client.

        Creates HTTP session, validates credentials, etc.
        Must be called before any other methods.
        """
        ...

    async def close(self) -> None:
        """
        Close the exchange client and cleanup resources.

        Should be called when done using the client.
        """
        ...

    @property
    def exchange_name(self) -> str:
        """Return the exchange identifier (e.g., 'weex', 'binance')."""
        ...

    @property
    def capabilities(self) -> ExchangeCapabilities:
        """Return exchange capabilities and limitations."""
        ...

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> UnifiedTicker:
        """
        Get ticker information for a symbol.

        Args:
            symbol: Unified symbol format (e.g., "BTCUSDT")

        Returns:
            UnifiedTicker with current price and 24h stats
        """
        ...

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> UnifiedOrderbook:
        """
        Get orderbook depth for a symbol.

        Args:
            symbol: Unified symbol format
            limit: Number of price levels (default 20)

        Returns:
            UnifiedOrderbook with bids and asks
        """
        ...

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[UnifiedCandle]:
        """
        Get historical candlestick data.

        Args:
            symbol: Unified symbol format
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Maximum number of candles
            start_time: Start of time range (optional)
            end_time: End of time range (optional)

        Returns:
            List of UnifiedCandle sorted by timestamp ascending
        """
        ...

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[UnifiedTrade]:
        """
        Get recent trades for a symbol.

        Args:
            symbol: Unified symbol format
            limit: Maximum number of trades

        Returns:
            List of UnifiedTrade sorted by timestamp descending
        """
        ...

    async def get_funding_rate(self, symbol: str) -> float:
        """
        Get current funding rate for perpetual futures.

        Args:
            symbol: Unified symbol format

        Returns:
            Current funding rate (e.g., 0.0001 = 0.01%)
        """
        ...

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Get trading rules and constraints for a symbol.

        Args:
            symbol: Unified symbol format

        Returns:
            SymbolInfo with precision, limits, and fees
        """
        ...

    async def get_all_symbols(self) -> list[SymbolInfo]:
        """
        Get all available trading symbols.

        Returns:
            List of SymbolInfo for all tradable pairs
        """
        ...

    # ==================== Account Operations ====================

    async def get_account_balance(self) -> UnifiedAccountBalance:
        """
        Get account balance information.

        Returns:
            UnifiedAccountBalance with total, available, and margin
        """
        ...

    async def get_positions(
        self,
        symbol: str | None = None,
    ) -> list[UnifiedPosition]:
        """
        Get all open positions.

        Args:
            symbol: Filter by symbol (optional, returns all if None)

        Returns:
            List of UnifiedPosition (only positions with size > 0)
        """
        ...

    async def get_position(
        self,
        symbol: str,
        side: PositionSide,
    ) -> UnifiedPosition | None:
        """
        Get a specific position by symbol and side.

        Args:
            symbol: Unified symbol format
            side: Position side (LONG or SHORT)

        Returns:
            UnifiedPosition if exists, None otherwise
        """
        ...

    # ==================== Order Operations ====================

    async def place_order(self, order: UnifiedOrder) -> UnifiedOrderResult:
        """
        Place a new order.

        Args:
            order: UnifiedOrder with all order parameters

        Returns:
            UnifiedOrderResult with order ID and status

        Raises:
            InvalidOrderError: If order parameters are invalid
            InsufficientBalanceError: If not enough balance
            OrderError: For other order-related errors
        """
        ...

    async def cancel_order(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Unified symbol format
            order_id: Exchange order ID (optional if client_order_id provided)
            client_order_id: Client order ID (optional if order_id provided)

        Returns:
            True if cancelled, False if not found or already filled

        Raises:
            OrderNotFoundError: If order doesn't exist
        """
        ...

    async def get_order(
        self,
        symbol: str,
        order_id: str,
    ) -> UnifiedOrderResult:
        """
        Get order details by ID.

        Args:
            symbol: Unified symbol format
            order_id: Exchange order ID

        Returns:
            UnifiedOrderResult with current status

        Raises:
            OrderNotFoundError: If order doesn't exist
        """
        ...

    async def get_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[UnifiedOrderResult]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional, returns all if None)

        Returns:
            List of UnifiedOrderResult for active orders
        """
        ...

    # ==================== Leverage Operations ====================

    async def set_leverage(
        self,
        symbol: str,
        leverage: int,
        side: PositionSide | None = None,
    ) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Unified symbol format
            leverage: Leverage multiplier (1-125)
            side: Position side for hedge mode (optional)

        Returns:
            True if successful

        Raises:
            ExchangeError: If leverage setting fails
        """
        ...

    async def get_leverage(self, symbol: str) -> int:
        """
        Get current leverage for a symbol.

        Args:
            symbol: Unified symbol format

        Returns:
            Current leverage multiplier
        """
        ...

    # ==================== Conditional Orders ====================

    async def place_stop_loss(
        self,
        symbol: str,
        side: OrderSide,
        trigger_price: float,
        size: float,
        position_side: PositionSide | None = None,
    ) -> UnifiedOrderResult:
        """
        Place a stop-loss order.

        Args:
            symbol: Unified symbol format
            side: Order side (BUY to close short, SELL to close long)
            trigger_price: Price at which to trigger
            size: Order quantity
            position_side: Position side for hedge mode (optional)

        Returns:
            UnifiedOrderResult with conditional order ID
        """
        ...

    async def place_take_profit(
        self,
        symbol: str,
        side: OrderSide,
        trigger_price: float,
        size: float,
        position_side: PositionSide | None = None,
    ) -> UnifiedOrderResult:
        """
        Place a take-profit order.

        Args:
            symbol: Unified symbol format
            side: Order side (BUY to close short, SELL to close long)
            trigger_price: Price at which to trigger
            size: Order quantity
            position_side: Position side for hedge mode (optional)

        Returns:
            UnifiedOrderResult with conditional order ID
        """
        ...

    async def cancel_conditional_order(
        self,
        symbol: str,
        order_id: str,
    ) -> bool:
        """
        Cancel a stop-loss or take-profit order.

        Args:
            symbol: Unified symbol format
            order_id: Conditional order ID

        Returns:
            True if cancelled successfully
        """
        ...


# Type alias for WebSocket callbacks
CandleCallback = Callable[[UnifiedCandle], None]
TickerCallback = Callable[[UnifiedTicker], None]
OrderbookCallback = Callable[[UnifiedOrderbook], None]
TradeCallback = Callable[[UnifiedTrade], None]


@runtime_checkable
class ExchangeWebSocketProtocol(Protocol):
    """
    Protocol defining WebSocket operations for real-time data.

    Adapters implement this for exchange-specific WebSocket formats.
    """

    async def connect(self) -> None:
        """
        Connect to WebSocket server.

        Establishes connection and starts heartbeat.
        """
        ...

    async def disconnect(self) -> None:
        """
        Disconnect from WebSocket server.

        Closes connection and cleans up resources.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        ...

    # ==================== Subscriptions ====================

    async def subscribe_candles(
        self,
        symbols: list[str],
        interval: str = "1m",
    ) -> None:
        """
        Subscribe to candle/kline updates.

        Args:
            symbols: List of unified symbols
            interval: Candle interval
        """
        ...

    async def subscribe_tickers(
        self,
        symbols: list[str],
    ) -> None:
        """
        Subscribe to ticker updates.

        Args:
            symbols: List of unified symbols
        """
        ...

    async def subscribe_orderbooks(
        self,
        symbols: list[str],
        depth: int = 20,
    ) -> None:
        """
        Subscribe to orderbook updates.

        Args:
            symbols: List of unified symbols
            depth: Number of price levels
        """
        ...

    async def subscribe_trades(
        self,
        symbols: list[str],
    ) -> None:
        """
        Subscribe to trade updates.

        Args:
            symbols: List of unified symbols
        """
        ...

    async def subscribe_user_data(self) -> None:
        """
        Subscribe to user data stream.

        Includes order updates and position changes.
        Requires authentication.
        """
        ...

    async def unsubscribe(
        self,
        channel: str,
        symbol: str,
    ) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel type (candle, ticker, orderbook, trade)
            symbol: Unified symbol
        """
        ...

    # ==================== Callbacks ====================

    def on_candle(self, callback: CandleCallback) -> None:
        """Register callback for candle updates."""
        ...

    def on_ticker(self, callback: TickerCallback) -> None:
        """Register callback for ticker updates."""
        ...

    def on_orderbook(self, callback: OrderbookCallback) -> None:
        """Register callback for orderbook updates."""
        ...

    def on_trade(self, callback: TradeCallback) -> None:
        """Register callback for trade updates."""
        ...

    # ==================== Event Loop ====================

    async def run(self) -> None:
        """
        Run the WebSocket message loop.

        Processes incoming messages and dispatches to callbacks.
        Blocks until disconnect is called.
        """
        ...


class ExchangeAdapter(ABC):
    """
    Base class for exchange adapters.

    Combines REST and WebSocket clients into a single adapter.
    Provides symbol normalization and common functionality.

    Example:
        class WEEXAdapter(ExchangeAdapter):
            @property
            def rest(self) -> ExchangeRESTProtocol:
                return self._rest_client

            @property
            def websocket(self) -> ExchangeWebSocketProtocol:
                return self._ws_client

            def normalize_symbol(self, symbol: str) -> str:
                return f"cmt_{symbol.lower()}"
    """

    @property
    @abstractmethod
    def rest(self) -> ExchangeRESTProtocol:
        """Get the REST API client."""
        ...

    @property
    @abstractmethod
    def websocket(self) -> ExchangeWebSocketProtocol:
        """Get the WebSocket client."""
        ...

    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """
        Convert unified symbol to exchange-specific format.

        Args:
            symbol: Unified format (e.g., "BTCUSDT")

        Returns:
            Exchange-specific format (e.g., "cmt_btcusdt" for WEEX)
        """
        ...

    @abstractmethod
    def denormalize_symbol(self, exchange_symbol: str) -> str:
        """
        Convert exchange-specific symbol to unified format.

        Args:
            exchange_symbol: Exchange format (e.g., "cmt_btcusdt")

        Returns:
            Unified format (e.g., "BTCUSDT")
        """
        ...

    async def initialize(self) -> None:
        """Initialize both REST and WebSocket clients."""
        await self.rest.initialize()

    async def close(self) -> None:
        """Close both REST and WebSocket clients."""
        await self.rest.close()
        if self.websocket.is_connected:
            await self.websocket.disconnect()
