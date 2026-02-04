"""
WEEX WebSocket Client

Implements ExchangeWebSocketProtocol for real-time market data from WEEX.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from enum import Enum

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.core.config import get_settings
from src.exchange.adapters.weex.mappers import WEEXMapper
from src.exchange.models import UnifiedCandle, UnifiedOrderbook, UnifiedTicker, UnifiedTrade
from src.exchange.protocols import (
    CandleCallback,
    OrderbookCallback,
    TickerCallback,
    TradeCallback,
)

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class WEEXWebSocket:
    """
    WEEX WebSocket client implementing ExchangeWebSocketProtocol.

    Handles connection management, auto-reconnection, and message parsing.
    Converts WEEX-specific formats to unified models.

    Usage:
        ws = WEEXWebSocket()
        ws.on_candle(my_handler)
        await ws.connect()
        await ws.subscribe_candles(["BTCUSDT"])
        await ws.run()
    """

    def __init__(
        self,
        ws_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            ws_url: WebSocket URL (uses config if None)
            api_key: API key for private channels (uses config if None)
            api_secret: API secret for private channels (uses config if None)
            api_passphrase: API passphrase for private channels (uses config if None)
        """
        settings = get_settings()
        self.ws_url = ws_url or settings.exchange.ws_url
        self.api_key = api_key or settings.exchange.api_key
        self.api_secret = api_secret or settings.exchange.api_secret
        self.api_passphrase = api_passphrase or settings.exchange.api_passphrase

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
        self._ping_interval = 25
        self._ping_task: asyncio.Task | None = None
        self._last_pong = time.time()
        self._pong_timeout = 10

        # Callbacks (unified types)
        self._candle_callbacks: list[CandleCallback] = []
        self._ticker_callbacks: list[TickerCallback] = []
        self._orderbook_callbacks: list[OrderbookCallback] = []
        self._trade_callbacks: list[TradeCallback] = []

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    def _generate_signature(self, timestamp: str) -> str:
        """Generate signature for authentication."""
        message = f"{timestamp}GET/users/self/verify"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    # ==================== Connection Management ====================

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning("Already connected or connecting")
            return

        self._state = ConnectionState.CONNECTING
        logger.info(f"Connecting to {self.ws_url}")

        try:
            # WEEX uses /public/v1 for public channels
            public_url = f"{self.ws_url}/public/v1"
            self._ws = await websockets.connect(
                public_url,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=10,
            )

            self._state = ConnectionState.CONNECTED
            self._current_delay = self._base_delay
            self._reconnect_attempts = 0
            self._last_pong = time.time()

            logger.info("WEEX WebSocket connected")

            # Start heartbeat
            self._ping_task = asyncio.create_task(self._heartbeat_loop())

            # Resubscribe to previous subscriptions
            if self._subscriptions:
                await self._resubscribe()

        except (WebSocketException, OSError) as e:
            logger.error(f"Connection failed: {e}")
            self._state = ConnectionState.DISCONNECTED
            raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        logger.info("Disconnecting WEEX WebSocket")

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
        logger.info("WEEX WebSocket disconnected")

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

                ping_msg = json.dumps({"op": "ping"})
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
        logger.info(f"Resubscribing to {len(self._subscriptions)} channels")

        for sub_key in self._subscriptions.copy():
            parts = sub_key.split(":")
            if len(parts) >= 2:
                channel = parts[0]
                symbol = parts[1]
                interval = parts[2] if len(parts) > 2 else None

                if channel == "candle":
                    await self._send_subscribe(channel, symbol, interval or "1m")
                else:
                    await self._send_subscribe(channel, symbol)

    async def _send_subscribe(
        self,
        channel: str,
        symbol: str,
        interval: str | None = None,
    ) -> None:
        """Send subscription message."""
        if not self._ws:
            raise RuntimeError("Not connected")

        weex_symbol = WEEXMapper.to_weex_symbol(symbol)
        channel_name = channel

        if channel == "candle":
            weex_interval = WEEXMapper.to_weex_interval(interval or "1m")
            channel_name = f"candle{weex_interval}"

        message = {
            "op": "subscribe",
            "args": [{"instType": "mc", "channel": channel_name, "instId": weex_symbol}],
        }

        await self._ws.send(json.dumps(message))
        logger.debug(f"Subscribed to {channel}:{symbol}")

    async def _send_unsubscribe(
        self,
        channel: str,
        symbol: str,
        interval: str | None = None,
    ) -> None:
        """Send unsubscription message."""
        if not self._ws:
            return

        weex_symbol = WEEXMapper.to_weex_symbol(symbol)
        channel_name = channel

        if channel == "candle":
            weex_interval = WEEXMapper.to_weex_interval(interval or "1m")
            channel_name = f"candle{weex_interval}"

        message = {
            "op": "unsubscribe",
            "args": [{"instType": "mc", "channel": channel_name, "instId": weex_symbol}],
        }

        await self._ws.send(json.dumps(message))
        logger.debug(f"Unsubscribed from {channel}:{symbol}")

    # ==================== Subscriptions ====================

    async def subscribe_candles(
        self,
        symbols: list[str],
        interval: str = "1m",
    ) -> None:
        """
        Subscribe to candle updates for symbols.

        Args:
            symbols: List of unified symbols (e.g., ["BTCUSDT"])
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        """
        for symbol in symbols:
            sub_key = f"candle:{symbol}:{interval}"
            if sub_key not in self._subscriptions:
                self._subscriptions.add(sub_key)
                if self.is_connected:
                    await self._send_subscribe("candle", symbol, interval)

    async def subscribe_tickers(self, symbols: list[str]) -> None:
        """Subscribe to ticker updates for symbols."""
        for symbol in symbols:
            sub_key = f"ticker:{symbol}"
            if sub_key not in self._subscriptions:
                self._subscriptions.add(sub_key)
                if self.is_connected:
                    await self._send_subscribe("ticker", symbol)

    async def subscribe_orderbooks(
        self,
        symbols: list[str],
        depth: int = 20,
    ) -> None:
        """Subscribe to orderbook updates for symbols."""
        for symbol in symbols:
            sub_key = f"books:{symbol}"
            if sub_key not in self._subscriptions:
                self._subscriptions.add(sub_key)
                if self.is_connected:
                    await self._send_subscribe("books", symbol)

    async def subscribe_trades(self, symbols: list[str]) -> None:
        """Subscribe to trade updates for symbols."""
        for symbol in symbols:
            sub_key = f"trade:{symbol}"
            if sub_key not in self._subscriptions:
                self._subscriptions.add(sub_key)
                if self.is_connected:
                    await self._send_subscribe("trade", symbol)

    async def subscribe_user_data(self) -> None:
        """Subscribe to user data stream (orders, positions)."""
        # WEEX requires authentication for private channels
        if not self._ws:
            raise RuntimeError("Not connected")

        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp)

        login_msg = {
            "op": "login",
            "args": [
                {
                    "apiKey": self.api_key,
                    "passphrase": self.api_passphrase,
                    "timestamp": timestamp,
                    "sign": signature,
                }
            ],
        }

        await self._ws.send(json.dumps(login_msg))
        logger.info("Sent login for user data subscription")

    async def unsubscribe(
        self,
        channel: str,
        symbol: str,
    ) -> None:
        """Unsubscribe from a channel."""
        # Find matching subscription key
        matching_keys = [k for k in self._subscriptions if k.startswith(f"{channel}:{symbol}")]

        for sub_key in matching_keys:
            self._subscriptions.discard(sub_key)
            parts = sub_key.split(":")
            interval = parts[2] if len(parts) > 2 else None

            if self.is_connected:
                await self._send_unsubscribe(channel, symbol, interval)

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
        data: list,
        weex_symbol: str,
        channel: str,
    ) -> UnifiedCandle | None:
        """Parse candle data from WebSocket message."""
        try:
            symbol = WEEXMapper.from_weex_symbol(weex_symbol)
            # Extract interval from channel (e.g., "candle1m" -> "1m")
            interval = channel.replace("candle", "") or "1m"
            interval = WEEXMapper.from_weex_interval(interval)

            return WEEXMapper.to_unified_candle(data, symbol, interval)
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse candle: {e}")
            return None

    def _parse_ticker(
        self,
        data: dict,
        weex_symbol: str,
    ) -> UnifiedTicker | None:
        """Parse ticker data from WebSocket message."""
        try:
            symbol = WEEXMapper.from_weex_symbol(weex_symbol)
            return WEEXMapper.to_unified_ticker(data, symbol)
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse ticker: {e}")
            return None

    def _parse_orderbook(
        self,
        data: dict,
        weex_symbol: str,
    ) -> UnifiedOrderbook | None:
        """Parse orderbook data from WebSocket message."""
        try:
            symbol = WEEXMapper.from_weex_symbol(weex_symbol)
            return WEEXMapper.to_unified_orderbook(data, symbol)
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse orderbook: {e}")
            return None

    def _parse_trade(
        self,
        data: dict,
        weex_symbol: str,
    ) -> UnifiedTrade | None:
        """Parse trade data from WebSocket message."""
        try:
            symbol = WEEXMapper.from_weex_symbol(weex_symbol)
            return WEEXMapper.to_unified_trade(data, symbol)
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

        event = data.get("event")
        op = data.get("op")

        # Handle pong
        if op == "pong" or event == "pong":
            self._last_pong = time.time()
            logger.debug("Received pong")
            return

        # Handle subscription confirmation
        if event == "subscribe":
            logger.debug(f"Subscription confirmed: {data.get('arg', {})}")
            return

        # Handle error
        if event == "error":
            logger.error(f"WebSocket error: {data.get('msg', 'Unknown error')}")
            return

        # Handle data push
        if "data" not in data or "arg" not in data:
            return

        arg = data["arg"]
        channel = arg.get("channel", "")
        weex_symbol = arg.get("instId", "")
        push_data = data["data"]

        # Normalize to list for consistent processing
        items = push_data if isinstance(push_data, list) else [push_data]

        self._dispatch_channel_data(channel, weex_symbol, items)

    def _dispatch_channel_data(
        self,
        channel: str,
        weex_symbol: str,
        items: list,
    ) -> None:
        """Dispatch data to appropriate callbacks based on channel type."""
        if channel.startswith("candle"):
            for item in items:
                candle = self._parse_candle(item, weex_symbol, channel)
                if candle:
                    for callback in self._candle_callbacks:
                        callback(candle)

        elif channel == "ticker":
            for item in items:
                ticker = self._parse_ticker(item, weex_symbol)
                if ticker:
                    for callback in self._ticker_callbacks:
                        callback(ticker)

        elif channel.startswith("books"):
            for item in items:
                orderbook = self._parse_orderbook(item, weex_symbol)
                if orderbook:
                    for callback in self._orderbook_callbacks:
                        callback(orderbook)

        elif channel == "trade":
            for item in items:
                trade = self._parse_trade(item, weex_symbol)
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
