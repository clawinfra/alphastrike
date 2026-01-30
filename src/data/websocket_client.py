"""
AlphaStrike Trading Bot - WEEX WebSocket Client

Real-time market data streaming from WEEX exchange.
Handles candle, ticker, and orderbook subscriptions.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.core.config import get_settings
from src.data.database import Candle

logger = logging.getLogger(__name__)


class ChannelType(str, Enum):
    """WebSocket channel types."""
    CANDLE = "candle"
    TICKER = "ticker"
    ORDERBOOK = "books"
    TRADE = "trade"


class ConnectionState(str, Enum):
    """WebSocket connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


@dataclass
class TickerData:
    """Real-time ticker data."""
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    volume_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderbookUpdate:
    """Orderbook update data."""
    symbol: str
    bids: list[tuple[float, float]]  # (price, size)
    asks: list[tuple[float, float]]  # (price, size)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeData:
    """Trade tick data."""
    symbol: str
    price: float
    size: float
    side: str  # "buy" or "sell"
    timestamp: datetime


# Type aliases for callbacks
CandleCallback = Callable[[Candle], None]
TickerCallback = Callable[[TickerData], None]
OrderbookCallback = Callable[[OrderbookUpdate], None]
TradeCallback = Callable[[TradeData], None]


class WebSocketClient:
    """
    Async WebSocket client for WEEX real-time data.

    Handles connection management, auto-reconnection, and message parsing.

    Usage:
        client = WebSocketClient()
        client.on_candle = my_candle_handler
        await client.connect()
        await client.subscribe_candles(["cmt_btcusdt", "cmt_ethusdt"])
        # Run forever
        await client.run()
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
        self._max_reconnect_attempts = 100  # Give up after many attempts

        # Ping/pong
        self._ping_interval = 25  # seconds
        self._ping_task: asyncio.Task | None = None
        self._last_pong = time.time()
        self._pong_timeout = 10  # seconds

        # Callbacks
        self.on_candle: CandleCallback | None = None
        self.on_ticker: TickerCallback | None = None
        self.on_orderbook: OrderbookCallback | None = None
        self.on_trade: TradeCallback | None = None

        # Message buffer for candles (by symbol)
        self._candle_buffers: dict[str, list[Candle]] = {}

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

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

    async def connect(self) -> None:
        """
        Connect to WebSocket server.

        Establishes connection and starts heartbeat.
        """
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
                ping_interval=None,  # We handle pings manually
                ping_timeout=None,
                close_timeout=10,
            )

            self._state = ConnectionState.CONNECTED
            self._current_delay = self._base_delay
            self._reconnect_attempts = 0
            self._last_pong = time.time()

            logger.info("WebSocket connected")

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
        logger.info("Disconnecting WebSocket")

        # Cancel tasks
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

        # Close connection
        if self._ws:
            await self._ws.close()
            self._ws = None

        self._state = ConnectionState.DISCONNECTED
        logger.info("WebSocket disconnected")

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
                # Exponential backoff
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

                # Check for pong timeout
                if time.time() - self._last_pong > self._ping_interval + self._pong_timeout:
                    logger.warning("Pong timeout, reconnecting")
                    await self._trigger_reconnect()
                    break

                # Send ping
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
            # Parse subscription key: "channel:symbol:interval"
            parts = sub_key.split(":")
            if len(parts) >= 2:
                channel = parts[0]
                symbol = parts[1]
                interval = parts[2] if len(parts) > 2 else None

                if channel == ChannelType.CANDLE.value:
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

        args: list[dict[str, str]] = []

        if channel == ChannelType.CANDLE.value:
            args.append({
                "instType": "mc",
                "channel": f"{channel}{interval}",
                "instId": symbol,
            })
        else:
            args.append({
                "instType": "mc",
                "channel": channel,
                "instId": symbol,
            })

        message = {
            "op": "subscribe",
            "args": args,
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

        args: list[dict[str, str]] = []

        if channel == ChannelType.CANDLE.value:
            args.append({
                "instType": "mc",
                "channel": f"{channel}{interval}",
                "instId": symbol,
            })
        else:
            args.append({
                "instType": "mc",
                "channel": channel,
                "instId": symbol,
            })

        message = {
            "op": "unsubscribe",
            "args": args,
        }

        await self._ws.send(json.dumps(message))
        logger.debug(f"Unsubscribed from {channel}:{symbol}")

    # ==================== Subscription Methods ====================

    async def subscribe_candles(
        self,
        symbols: list[str],
        interval: str = "1m",
    ) -> None:
        """
        Subscribe to candle updates for symbols.

        Args:
            symbols: List of trading pair symbols
            interval: Candle interval (1m, 5m, 15m, 1H, 4H, 1D)
        """
        for symbol in symbols:
            sub_key = f"{ChannelType.CANDLE.value}:{symbol}:{interval}"
            if sub_key not in self._subscriptions:
                self._subscriptions.add(sub_key)
                if self.is_connected:
                    await self._send_subscribe(ChannelType.CANDLE.value, symbol, interval)

    async def subscribe_tickers(self, symbols: list[str]) -> None:
        """
        Subscribe to ticker updates for symbols.

        Args:
            symbols: List of trading pair symbols
        """
        for symbol in symbols:
            sub_key = f"{ChannelType.TICKER.value}:{symbol}"
            if sub_key not in self._subscriptions:
                self._subscriptions.add(sub_key)
                if self.is_connected:
                    await self._send_subscribe(ChannelType.TICKER.value, symbol)

    async def subscribe_orderbooks(self, symbols: list[str]) -> None:
        """
        Subscribe to orderbook updates for symbols.

        Args:
            symbols: List of trading pair symbols
        """
        for symbol in symbols:
            sub_key = f"{ChannelType.ORDERBOOK.value}:{symbol}"
            if sub_key not in self._subscriptions:
                self._subscriptions.add(sub_key)
                if self.is_connected:
                    await self._send_subscribe(ChannelType.ORDERBOOK.value, symbol)

    async def subscribe_trades(self, symbols: list[str]) -> None:
        """
        Subscribe to trade updates for symbols.

        Args:
            symbols: List of trading pair symbols
        """
        for symbol in symbols:
            sub_key = f"{ChannelType.TRADE.value}:{symbol}"
            if sub_key not in self._subscriptions:
                self._subscriptions.add(sub_key)
                if self.is_connected:
                    await self._send_subscribe(ChannelType.TRADE.value, symbol)

    async def unsubscribe(self, channel: ChannelType, symbol: str, interval: str | None = None) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel type
            symbol: Trading pair symbol
            interval: Interval for candle channel
        """
        if channel == ChannelType.CANDLE:
            sub_key = f"{channel.value}:{symbol}:{interval or '1m'}"
        else:
            sub_key = f"{channel.value}:{symbol}"

        if sub_key in self._subscriptions:
            self._subscriptions.remove(sub_key)
            if self.is_connected:
                await self._send_unsubscribe(channel.value, symbol, interval)

    # ==================== Message Handling ====================

    def _parse_candle(self, data: dict, symbol: str) -> Candle | None:
        """Parse candle data from message."""
        try:
            # WEEX format: [timestamp, open, high, low, close, volume, ...]
            if isinstance(data, list) and len(data) >= 6:
                return Candle(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(int(data[0]) / 1000),
                    open=float(data[1]),
                    high=float(data[2]),
                    low=float(data[3]),
                    close=float(data[4]),
                    volume=float(data[5]),
                    interval="1m",  # Will be updated based on channel
                )
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse candle: {e}")
        return None

    def _parse_ticker(self, data: dict, symbol: str) -> TickerData | None:
        """Parse ticker data from message."""
        try:
            return TickerData(
                symbol=symbol,
                last_price=float(data.get("last", 0)),
                bid_price=float(data.get("bidPr", 0)),
                ask_price=float(data.get("askPr", 0)),
                bid_size=float(data.get("bidSz", 0)),
                ask_size=float(data.get("askSz", 0)),
                volume_24h=float(data.get("baseVolume", 0)),
                high_24h=float(data.get("high24h", 0)),
                low_24h=float(data.get("low24h", 0)),
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse ticker: {e}")
        return None

    def _parse_orderbook(self, data: dict, symbol: str) -> OrderbookUpdate | None:
        """Parse orderbook data from message."""
        try:
            bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
            return OrderbookUpdate(
                symbol=symbol,
                bids=bids,
                asks=asks,
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse orderbook: {e}")
        return None

    def _parse_trade(self, data: dict, symbol: str) -> TradeData | None:
        """Parse trade data from message."""
        try:
            return TradeData(
                symbol=symbol,
                price=float(data.get("price", 0)),
                size=float(data.get("size", 0)),
                side=data.get("side", "buy"),
                timestamp=datetime.fromtimestamp(int(data.get("ts", 0)) / 1000),
            )
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
        if data.get("op") == "pong" or data.get("event") == "pong":
            self._last_pong = time.time()
            logger.debug("Received pong")
            return

        # Handle subscription confirmation
        if data.get("event") == "subscribe":
            logger.debug(f"Subscription confirmed: {data.get('arg', {})}")
            return

        # Handle error
        if data.get("event") == "error":
            logger.error(f"WebSocket error: {data.get('msg', 'Unknown error')}")
            return

        # Handle data push
        if "data" in data and "arg" in data:
            arg = data["arg"]
            channel = arg.get("channel", "")
            symbol = arg.get("instId", "")
            push_data = data["data"]

            # Process based on channel type
            if channel.startswith("candle"):
                # Extract interval from channel name (e.g., "candle1m")
                interval = channel.replace("candle", "") or "1m"
                for item in push_data if isinstance(push_data, list) else [push_data]:
                    candle = self._parse_candle(item, symbol)
                    if candle and self.on_candle:
                        # Update interval from channel
                        candle = Candle(
                            symbol=candle.symbol,
                            timestamp=candle.timestamp,
                            open=candle.open,
                            high=candle.high,
                            low=candle.low,
                            close=candle.close,
                            volume=candle.volume,
                            interval=interval,
                        )
                        self.on_candle(candle)

            elif channel == "ticker":
                for item in push_data if isinstance(push_data, list) else [push_data]:
                    ticker = self._parse_ticker(item, symbol)
                    if ticker and self.on_ticker:
                        self.on_ticker(ticker)

            elif channel.startswith("books"):
                for item in push_data if isinstance(push_data, list) else [push_data]:
                    orderbook = self._parse_orderbook(item, symbol)
                    if orderbook and self.on_orderbook:
                        self.on_orderbook(orderbook)

            elif channel == "trade":
                for item in push_data if isinstance(push_data, list) else [push_data]:
                    trade = self._parse_trade(item, symbol)
                    if trade and self.on_trade:
                        self.on_trade(trade)

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
                    # Convert bytes to string if needed
                    message = raw_message.decode("utf-8") if isinstance(raw_message, bytes) else raw_message
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

    async def run_forever(self) -> None:
        """
        Run forever, handling reconnections.

        Use this for production deployment.
        """
        while True:
            try:
                await self.run()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in run_forever: {e}")
                await asyncio.sleep(5)

            if self._state == ConnectionState.DISCONNECTED:
                logger.info("Permanently disconnected, stopping")
                break


# Module-level singleton
_ws_client: WebSocketClient | None = None


async def get_websocket_client() -> WebSocketClient:
    """
    Get the WebSocket client singleton instance.
    """
    global _ws_client
    if _ws_client is None:
        _ws_client = WebSocketClient()
    return _ws_client


async def close_websocket_client() -> None:
    """Close the WebSocket client singleton."""
    global _ws_client
    if _ws_client:
        await _ws_client.disconnect()
        _ws_client = None
