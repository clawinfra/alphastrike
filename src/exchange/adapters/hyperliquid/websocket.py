"""
Hyperliquid WebSocket Client

Implements ExchangeWebSocketProtocol for real-time market data from Hyperliquid DEX.
Supports subscriptions to candles, trades, orderbooks, and user data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import Enum

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.exchange.adapters.hyperliquid.mappers import HyperliquidMapper
from src.exchange.models import UnifiedCandle, UnifiedOrderbook, UnifiedTicker, UnifiedTrade
from src.exchange.protocols import (
    CandleCallback,
    OrderbookCallback,
    TickerCallback,
    TradeCallback,
)

logger = logging.getLogger(__name__)

# Default WebSocket URLs
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"


class ConnectionState(str, Enum):
    """WebSocket connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class HyperliquidWebSocket:
    """
    Hyperliquid WebSocket client implementing ExchangeWebSocketProtocol.

    Handles connection management, auto-reconnection, and message parsing.
    Converts Hyperliquid-specific formats to unified models.

    Usage:
        ws = HyperliquidWebSocket()
        ws.on_candle(my_handler)
        await ws.connect()
        await ws.subscribe_candles(["BTCUSDT"], "1h")
        await ws.run()
    """

    def __init__(
        self,
        ws_url: str | None = None,
        testnet: bool = False,
    ):
        """
        Initialize WebSocket client.

        Args:
            ws_url: WebSocket URL (uses default if None)
            testnet: Whether to use testnet
        """
        self.ws_url = ws_url or (TESTNET_WS_URL if testnet else MAINNET_WS_URL)

        # Connection state
        self._ws: ClientConnection | None = None
        self._state = ConnectionState.DISCONNECTED
        self._subscriptions: set[str] = set()
        self._reconnect_task: asyncio.Task | None = None

        # Reconnection parameters
        self._base_delay = 1.0
        self._max_delay = 60.0
        self._current_delay = self._base_delay
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 100

        # Ping/pong
        self._ping_interval = 20
        self._ping_task: asyncio.Task | None = None
        self._last_pong = time.time()
        self._pong_timeout = 10

        # Callbacks
        self._candle_callbacks: list[CandleCallback] = []
        self._ticker_callbacks: list[TickerCallback] = []
        self._orderbook_callbacks: list[OrderbookCallback] = []
        self._trade_callbacks: list[TradeCallback] = []

        # Subscription tracking for resubscription
        self._candle_subs: dict[str, str] = {}  # symbol -> interval
        self._l2_subs: set[str] = set()  # symbols
        self._trade_subs: set[str] = set()  # symbols
        self._all_mids_subscribed = False

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    # ==================== Connection Management ====================

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning("Already connected or connecting")
            return

        self._state = ConnectionState.CONNECTING
        logger.info(f"Connecting to {self.ws_url}")

        try:
            self._ws = await websockets.connect(
                self.ws_url,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=10,
            )

            self._state = ConnectionState.CONNECTED
            self._current_delay = self._base_delay
            self._reconnect_attempts = 0
            self._last_pong = time.time()

            logger.info("Hyperliquid WebSocket connected")

            # Start heartbeat
            self._ping_task = asyncio.create_task(self._heartbeat_loop())

            # Resubscribe to previous subscriptions
            await self._resubscribe()

        except (WebSocketException, OSError) as e:
            logger.error(f"Connection failed: {e}")
            self._state = ConnectionState.DISCONNECTED
            raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        logger.info("Disconnecting Hyperliquid WebSocket")

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._state = ConnectionState.DISCONNECTED
        logger.info("Hyperliquid WebSocket disconnected")

    async def _reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        self._state = ConnectionState.RECONNECTING

        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            logger.info(
                f"Reconnecting (attempt {self._reconnect_attempts}), "
                f"waiting {self._current_delay:.1f}s"
            )

            await asyncio.sleep(self._current_delay)

            try:
                await self.connect()
                return
            except (WebSocketException, OSError) as e:
                logger.warning(f"Reconnection failed: {e}")
                self._current_delay = min(self._current_delay * 2, self._max_delay)

        logger.error(f"Max reconnection attempts ({self._max_reconnect_attempts}) exceeded")
        self._state = ConnectionState.DISCONNECTED

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while self._state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self._ping_interval)

                if not self._ws:
                    break

                if time.time() - self._last_pong > self._ping_interval + self._pong_timeout:
                    logger.warning("Pong timeout, reconnecting")
                    await self._trigger_reconnect()
                    break

                # Hyperliquid expects ping as JSON
                ping_msg = json.dumps({"method": "ping"})
                await self._ws.send(ping_msg)
                logger.debug("Sent ping")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await self._trigger_reconnect()
                break

    async def _trigger_reconnect(self) -> None:
        """Trigger reconnection."""
        if self._state == ConnectionState.RECONNECTING:
            return

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _resubscribe(self) -> None:
        """Resubscribe to all previous subscriptions after reconnect."""
        logger.info("Resubscribing to channels")

        # Resubscribe to allMids
        if self._all_mids_subscribed:
            await self._send_subscribe({"type": "allMids"})

        # Resubscribe to candles
        for symbol, interval in self._candle_subs.items():
            coin = HyperliquidMapper.to_hyperliquid_coin(symbol)
            await self._send_subscribe({
                "type": "candle",
                "coin": coin,
                "interval": interval,
            })

        # Resubscribe to L2 books
        for symbol in self._l2_subs:
            coin = HyperliquidMapper.to_hyperliquid_coin(symbol)
            await self._send_subscribe({
                "type": "l2Book",
                "coin": coin,
            })

        # Resubscribe to trades
        for symbol in self._trade_subs:
            coin = HyperliquidMapper.to_hyperliquid_coin(symbol)
            await self._send_subscribe({
                "type": "trades",
                "coin": coin,
            })

    async def _send_subscribe(self, subscription: dict) -> None:
        """Send subscription message."""
        if not self._ws:
            raise RuntimeError("Not connected")

        message = {
            "method": "subscribe",
            "subscription": subscription,
        }

        await self._ws.send(json.dumps(message))
        logger.debug(f"Subscribed to {subscription}")

    async def _send_unsubscribe(self, subscription: dict) -> None:
        """Send unsubscription message."""
        if not self._ws:
            return

        message = {
            "method": "unsubscribe",
            "subscription": subscription,
        }

        await self._ws.send(json.dumps(message))
        logger.debug(f"Unsubscribed from {subscription}")

    # ==================== Subscriptions ====================

    async def subscribe_candles(
        self,
        symbols: list[str],
        interval: str = "1h",
    ) -> None:
        """
        Subscribe to candle updates for symbols.

        Args:
            symbols: List of unified symbols (e.g., ["BTCUSDT"])
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        """
        hl_interval = HyperliquidMapper.to_hyperliquid_interval(interval)

        for symbol in symbols:
            self._candle_subs[symbol] = hl_interval
            self._subscriptions.add(f"candle:{symbol}:{interval}")

            if self.is_connected:
                coin = HyperliquidMapper.to_hyperliquid_coin(symbol)
                await self._send_subscribe({
                    "type": "candle",
                    "coin": coin,
                    "interval": hl_interval,
                })

    async def subscribe_tickers(self, symbols: list[str]) -> None:
        """
        Subscribe to ticker updates.

        Note: Hyperliquid uses allMids for price updates.
        """
        self._all_mids_subscribed = True

        for symbol in symbols:
            self._subscriptions.add(f"ticker:{symbol}")

        if self.is_connected:
            await self._send_subscribe({"type": "allMids"})

    async def subscribe_orderbooks(
        self,
        symbols: list[str],
        depth: int = 20,
    ) -> None:
        """Subscribe to orderbook updates for symbols."""
        for symbol in symbols:
            self._l2_subs.add(symbol)
            self._subscriptions.add(f"l2Book:{symbol}")

            if self.is_connected:
                coin = HyperliquidMapper.to_hyperliquid_coin(symbol)
                await self._send_subscribe({
                    "type": "l2Book",
                    "coin": coin,
                })

    async def subscribe_trades(self, symbols: list[str]) -> None:
        """Subscribe to trade updates for symbols."""
        for symbol in symbols:
            self._trade_subs.add(symbol)
            self._subscriptions.add(f"trades:{symbol}")

            if self.is_connected:
                coin = HyperliquidMapper.to_hyperliquid_coin(symbol)
                await self._send_subscribe({
                    "type": "trades",
                    "coin": coin,
                })

    async def subscribe_user_data(self) -> None:
        """
        Subscribe to user data stream.

        Note: Hyperliquid requires user address for user-specific subscriptions.
        This is typically done via REST polling or specific user subscriptions.
        """
        logger.warning(
            "User data subscription not fully supported via WebSocket. "
            "Use REST API for position/order updates."
        )

    async def unsubscribe(
        self,
        channel: str,
        symbol: str,
    ) -> None:
        """Unsubscribe from a channel."""
        coin = HyperliquidMapper.to_hyperliquid_coin(symbol)

        if channel == "candle":
            if symbol in self._candle_subs:
                del self._candle_subs[symbol]
            if self.is_connected:
                await self._send_unsubscribe({"type": "candle", "coin": coin})

        elif channel == "l2Book":
            self._l2_subs.discard(symbol)
            if self.is_connected:
                await self._send_unsubscribe({"type": "l2Book", "coin": coin})

        elif channel == "trades":
            self._trade_subs.discard(symbol)
            if self.is_connected:
                await self._send_unsubscribe({"type": "trades", "coin": coin})

        # Remove from subscriptions set
        matching = [s for s in self._subscriptions if f":{symbol}" in s and s.startswith(channel)]
        for s in matching:
            self._subscriptions.discard(s)

    # ==================== Callbacks ====================

    def on_candle(self, callback: CandleCallback) -> None:
        """Register callback for candle updates."""
        self._candle_callbacks.append(callback)

    def on_ticker(self, callback: TickerCallback) -> None:
        """Register callback for ticker updates."""
        self._ticker_callbacks.append(callback)

    def on_orderbook(self, callback: OrderbookCallback) -> None:
        """Register callback for orderbook updates."""
        self._orderbook_callbacks.append(callback)

    def on_trade(self, callback: TradeCallback) -> None:
        """Register callback for trade updates."""
        self._trade_callbacks.append(callback)

    # ==================== Message Parsing ====================

    def _parse_candle(
        self,
        data: dict,
        coin: str,
        interval: str,
    ) -> UnifiedCandle | None:
        """Parse candle data from WebSocket message."""
        try:
            symbol = HyperliquidMapper.from_hyperliquid_coin(coin)
            return HyperliquidMapper.to_unified_candle(data, symbol, interval)
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse candle: {e}")
            return None

    def _parse_ticker(
        self,
        mids: dict[str, str],
        coin: str,
    ) -> UnifiedTicker | None:
        """Parse ticker data from allMids message."""
        try:
            return HyperliquidMapper.to_unified_ticker(mids, coin)
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse ticker: {e}")
            return None

    def _parse_orderbook(
        self,
        data: dict,
        coin: str,
    ) -> UnifiedOrderbook | None:
        """Parse orderbook data from WebSocket message."""
        try:
            symbol = HyperliquidMapper.from_hyperliquid_coin(coin)
            return HyperliquidMapper.to_unified_orderbook(data, symbol)
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse orderbook: {e}")
            return None

    def _parse_trade(
        self,
        data: dict,
    ) -> UnifiedTrade | None:
        """Parse trade data from WebSocket message."""
        try:
            return HyperliquidMapper.to_unified_trade(data)
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse trade: {e}")
            return None

    async def _handle_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:100]}")
            return

        # Handle pong
        if data.get("method") == "pong" or "pong" in str(data):
            self._last_pong = time.time()
            logger.debug("Received pong")
            return

        # Handle subscription confirmation
        if data.get("method") == "subscribed":
            logger.debug(f"Subscription confirmed: {data.get('subscription', {})}")
            return

        # Handle channel data
        channel = data.get("channel")

        if channel == "allMids":
            # All mid prices update
            mids = data.get("data", {}).get("mids", {})
            for coin in mids:
                ticker = self._parse_ticker(mids, coin)
                if ticker:
                    for callback in self._ticker_callbacks:
                        callback(ticker)

        elif channel == "candle":
            # Candle update
            candle_data = data.get("data", {})
            coin = candle_data.get("s", "")
            interval = candle_data.get("i", "1h")

            candle = self._parse_candle(candle_data, coin, interval)
            if candle:
                for callback in self._candle_callbacks:
                    callback(candle)

        elif channel == "l2Book":
            # Orderbook update
            book_data = data.get("data", {})
            coin = book_data.get("coin", "")

            orderbook = self._parse_orderbook(book_data, coin)
            if orderbook:
                for callback in self._orderbook_callbacks:
                    callback(orderbook)

        elif channel == "trades":
            # Trade updates
            trades_data = data.get("data", [])
            for trade_data in trades_data:
                trade = self._parse_trade(trade_data)
                if trade:
                    for callback in self._trade_callbacks:
                        callback(trade)

    # ==================== Event Loop ====================

    async def run(self) -> None:
        """
        Run the WebSocket client message loop.

        Blocks until disconnected or error.
        """
        if not self.is_connected:
            await self.connect()

        while self._state in (ConnectionState.CONNECTED, ConnectionState.RECONNECTING):
            try:
                if self._ws:
                    raw_message = await self._ws.recv()
                    message = (
                        raw_message.decode("utf-8")
                        if isinstance(raw_message, bytes)
                        else raw_message
                    )
                    await self._handle_message(message)
                else:
                    await asyncio.sleep(0.1)

            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                await self._trigger_reconnect()

            except asyncio.CancelledError:
                logger.info("WebSocket run cancelled")
                break

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await self._trigger_reconnect()
