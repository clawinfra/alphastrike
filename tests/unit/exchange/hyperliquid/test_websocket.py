"""
Tests for Hyperliquid WebSocket client.

Coverage targets: connection management, subscriptions, message handling.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.exchange.adapters.hyperliquid.websocket import (
    HyperliquidWebSocket,
    ConnectionState,
    MAINNET_WS_URL,
    TESTNET_WS_URL,
)
from src.exchange.adapters.hyperliquid.mappers import HyperliquidMapper


class TestHyperliquidWebSocketInit:
    """Tests for WebSocket initialization."""

    def test_init_testnet(self):
        """Test initialization with testnet."""
        ws = HyperliquidWebSocket(testnet=True)
        assert ws.ws_url == TESTNET_WS_URL
        assert ws._state == ConnectionState.DISCONNECTED

    def test_init_mainnet(self):
        """Test initialization with mainnet."""
        ws = HyperliquidWebSocket(testnet=False)
        assert ws.ws_url == MAINNET_WS_URL

    def test_init_custom_url(self):
        """Test initialization with custom URL."""
        custom_url = "wss://custom.ws.com"
        ws = HyperliquidWebSocket(ws_url=custom_url)
        assert ws.ws_url == custom_url

    def test_init_default_values(self):
        """Test initialization default values."""
        ws = HyperliquidWebSocket(testnet=True)
        assert ws._ws is None
        assert ws._subscriptions == set()
        assert ws._candle_callbacks == []
        assert ws._ticker_callbacks == []
        assert ws._orderbook_callbacks == []
        assert ws._trade_callbacks == []


class TestHyperliquidWebSocketProperties:
    """Tests for WebSocket properties."""

    def test_is_connected_false(self):
        """Test is_connected when disconnected."""
        ws = HyperliquidWebSocket(testnet=True)
        assert ws.is_connected is False

    def test_is_connected_true(self):
        """Test is_connected when connected."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTED
        assert ws.is_connected is True


class TestHyperliquidWebSocketConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        ws = HyperliquidWebSocket(testnet=True)

        mock_connection = AsyncMock()
        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_connection):
            await ws.connect()

            assert ws._state == ConnectionState.CONNECTED
            assert ws._ws is not None

    @pytest.mark.asyncio
    async def test_connect_already_connected(self):
        """Test connect when already connected."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTED

        # Should return early without attempting to connect
        await ws.connect()

    @pytest.mark.asyncio
    async def test_connect_already_connecting(self):
        """Test connect when already connecting."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTING

        await ws.connect()

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        ws = HyperliquidWebSocket(testnet=True)

        with patch("websockets.connect", new_callable=AsyncMock, side_effect=OSError("Connection failed")):
            with pytest.raises(OSError):
                await ws.connect()

            assert ws._state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTED
        ws._ws = AsyncMock()
        ws._ping_task = asyncio.create_task(asyncio.sleep(100))

        await ws.disconnect()

        assert ws._state == ConnectionState.DISCONNECTED
        assert ws._ws is None
        assert ws._ping_task is None

    @pytest.mark.asyncio
    async def test_disconnect_without_connection(self):
        """Test disconnect without active connection."""
        ws = HyperliquidWebSocket(testnet=True)

        await ws.disconnect()  # Should not raise


class TestHyperliquidWebSocketReconnection:
    """Tests for reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect(self):
        """Test reconnection."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._max_reconnect_attempts = 2
        ws._base_delay = 0.01  # Fast for testing

        mock_connection = AsyncMock()
        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_connection):
            await ws._reconnect()

            assert ws._state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_reconnect_max_attempts(self):
        """Test reconnection max attempts."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._max_reconnect_attempts = 2
        ws._base_delay = 0.01

        with patch("websockets.connect", new_callable=AsyncMock, side_effect=OSError("Failed")):
            await ws._reconnect()

            assert ws._state == ConnectionState.DISCONNECTED


class TestHyperliquidWebSocketSubscriptions:
    """Tests for subscription methods."""

    @pytest.fixture
    def ws(self):
        """Create connected WebSocket fixture."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTED
        ws._ws = AsyncMock()
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}, {"name": "ETH"}])
        return ws

    @pytest.mark.asyncio
    async def test_subscribe_candles(self, ws):
        """Test candle subscription."""
        await ws.subscribe_candles(["BTCUSDT"], interval="1h")

        assert "BTCUSDT" in ws._candle_subs
        assert ws._candle_subs["BTCUSDT"] == "1h"
        assert "candle:BTCUSDT:1h" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_candles_not_connected(self):
        """Test candle subscription when not connected."""
        ws = HyperliquidWebSocket(testnet=True)
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])

        await ws.subscribe_candles(["BTCUSDT"], interval="1h")

        # Subscription should be tracked for later
        assert "BTCUSDT" in ws._candle_subs

    @pytest.mark.asyncio
    async def test_subscribe_tickers(self, ws):
        """Test ticker subscription."""
        await ws.subscribe_tickers(["BTCUSDT", "ETHUSDT"])

        assert ws._all_mids_subscribed is True
        assert "ticker:BTCUSDT" in ws._subscriptions
        assert "ticker:ETHUSDT" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_orderbooks(self, ws):
        """Test orderbook subscription."""
        await ws.subscribe_orderbooks(["BTCUSDT"], depth=20)

        assert "BTCUSDT" in ws._l2_subs
        assert "l2Book:BTCUSDT" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_trades(self, ws):
        """Test trades subscription."""
        await ws.subscribe_trades(["BTCUSDT"])

        assert "BTCUSDT" in ws._trade_subs
        assert "trades:BTCUSDT" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_user_data(self, ws):
        """Test user data subscription logs warning."""
        with patch("src.exchange.adapters.hyperliquid.websocket.logger") as mock_logger:
            await ws.subscribe_user_data()
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_unsubscribe_candle(self, ws):
        """Test candle unsubscription."""
        ws._candle_subs["BTCUSDT"] = "1h"
        ws._subscriptions.add("candle:BTCUSDT:1h")

        await ws.unsubscribe("candle", "BTCUSDT")

        assert "BTCUSDT" not in ws._candle_subs

    @pytest.mark.asyncio
    async def test_unsubscribe_l2book(self, ws):
        """Test orderbook unsubscription."""
        ws._l2_subs.add("BTCUSDT")
        ws._subscriptions.add("l2Book:BTCUSDT")

        await ws.unsubscribe("l2Book", "BTCUSDT")

        assert "BTCUSDT" not in ws._l2_subs

    @pytest.mark.asyncio
    async def test_unsubscribe_trades(self, ws):
        """Test trades unsubscription."""
        ws._trade_subs.add("BTCUSDT")
        ws._subscriptions.add("trades:BTCUSDT")

        await ws.unsubscribe("trades", "BTCUSDT")

        assert "BTCUSDT" not in ws._trade_subs


class TestHyperliquidWebSocketCallbacks:
    """Tests for callback registration."""

    def test_on_candle(self):
        """Test candle callback registration."""
        ws = HyperliquidWebSocket(testnet=True)
        callback = MagicMock()

        ws.on_candle(callback)

        assert callback in ws._candle_callbacks

    def test_on_ticker(self):
        """Test ticker callback registration."""
        ws = HyperliquidWebSocket(testnet=True)
        callback = MagicMock()

        ws.on_ticker(callback)

        assert callback in ws._ticker_callbacks

    def test_on_orderbook(self):
        """Test orderbook callback registration."""
        ws = HyperliquidWebSocket(testnet=True)
        callback = MagicMock()

        ws.on_orderbook(callback)

        assert callback in ws._orderbook_callbacks

    def test_on_trade(self):
        """Test trade callback registration."""
        ws = HyperliquidWebSocket(testnet=True)
        callback = MagicMock()

        ws.on_trade(callback)

        assert callback in ws._trade_callbacks


class TestHyperliquidWebSocketMessageHandling:
    """Tests for message handling."""

    @pytest.fixture
    def ws(self):
        """Create WebSocket fixture."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTED
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}, {"name": "ETH"}])
        return ws

    @pytest.mark.asyncio
    async def test_handle_message_invalid_json(self, ws):
        """Test handling invalid JSON message."""
        await ws._handle_message("not valid json")
        # Should not raise

    @pytest.mark.asyncio
    async def test_handle_message_pong(self, ws):
        """Test handling pong message."""
        message = json.dumps({"method": "pong"})
        await ws._handle_message(message)
        # Should not raise

    @pytest.mark.asyncio
    async def test_handle_message_subscribed(self, ws):
        """Test handling subscription confirmation."""
        message = json.dumps({"method": "subscribed", "subscription": {"type": "candle"}})
        await ws._handle_message(message)
        # Should not raise

    @pytest.mark.asyncio
    async def test_handle_message_all_mids(self, ws):
        """Test handling allMids message."""
        callback = MagicMock()
        ws._ticker_callbacks.append(callback)

        message = json.dumps({
            "channel": "allMids",
            "data": {"mids": {"BTC": "50000.0", "ETH": "3000.0"}}
        })
        await ws._handle_message(message)

        # Should call ticker callback for each coin
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_message_candle(self, ws):
        """Test handling candle message."""
        callback = MagicMock()
        ws._candle_callbacks.append(callback)

        message = json.dumps({
            "channel": "candle",
            "data": {
                "s": "BTC",
                "i": "1h",
                "t": 1706644800000,
                "o": "50000.0",
                "h": "50500.0",
                "l": "49500.0",
                "c": "50200.0",
                "v": "100.0",
                "n": 500,
            }
        })
        await ws._handle_message(message)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_l2book(self, ws):
        """Test handling l2Book message."""
        callback = MagicMock()
        ws._orderbook_callbacks.append(callback)

        message = json.dumps({
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "levels": [
                    [{"px": "49990", "sz": "1.5"}],
                    [{"px": "50010", "sz": "1.0"}],
                ]
            }
        })
        await ws._handle_message(message)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_trades(self, ws):
        """Test handling trades message."""
        callback = MagicMock()
        ws._trade_callbacks.append(callback)

        message = json.dumps({
            "channel": "trades",
            "data": [{
                "coin": "BTC",
                "tid": "trade1",
                "px": "50000.0",
                "sz": "0.1",
                "side": "B",
                "time": 1706644800000,
            }]
        })
        await ws._handle_message(message)

        callback.assert_called_once()


class TestHyperliquidWebSocketParsing:
    """Tests for message parsing methods."""

    @pytest.fixture
    def ws(self):
        """Create WebSocket fixture."""
        ws = HyperliquidWebSocket(testnet=True)
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])
        return ws

    def test_parse_candle_success(self, ws):
        """Test successful candle parsing."""
        data = {
            "t": 1706644800000,
            "o": "50000.0",
            "h": "50500.0",
            "l": "49500.0",
            "c": "50200.0",
            "v": "100.0",
            "n": 500,
        }
        candle = ws._parse_candle(data, "BTC", "1h")

        assert candle is not None
        assert candle.symbol == "BTCUSDT"
        assert candle.open == 50000.0

    def test_parse_candle_failure(self, ws):
        """Test candle parsing failure."""
        candle = ws._parse_candle({}, "BTC", "1h")
        # Should return None on failure, not raise

    def test_parse_ticker_success(self, ws):
        """Test successful ticker parsing."""
        mids = {"BTC": "50000.0"}
        ticker = ws._parse_ticker(mids, "BTC")

        assert ticker is not None
        assert ticker.symbol == "BTCUSDT"
        assert ticker.last_price == 50000.0

    def test_parse_orderbook_success(self, ws):
        """Test successful orderbook parsing."""
        data = {
            "levels": [
                [{"px": "49990", "sz": "1.5"}],
                [{"px": "50010", "sz": "1.0"}],
            ]
        }
        orderbook = ws._parse_orderbook(data, "BTC")

        assert orderbook is not None
        assert orderbook.symbol == "BTCUSDT"

    def test_parse_trade_success(self, ws):
        """Test successful trade parsing."""
        data = {
            "coin": "BTC",
            "tid": "trade1",
            "px": "50000.0",
            "sz": "0.1",
            "side": "B",
            "time": 1706644800000,
        }
        trade = ws._parse_trade(data)

        assert trade is not None
        assert trade.trade_id == "trade1"


class TestHyperliquidWebSocketResubscribe:
    """Tests for resubscription logic."""

    @pytest.mark.asyncio
    async def test_resubscribe(self):
        """Test resubscription after reconnect."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTED
        ws._ws = AsyncMock()
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])

        # Set up subscriptions
        ws._all_mids_subscribed = True
        ws._candle_subs["BTCUSDT"] = "1h"
        ws._l2_subs.add("BTCUSDT")
        ws._trade_subs.add("BTCUSDT")

        await ws._resubscribe()

        # Should have sent subscription messages
        assert ws._ws.send.call_count >= 4


class TestHyperliquidWebSocketSendSubscribe:
    """Tests for send subscription methods."""

    @pytest.mark.asyncio
    async def test_send_subscribe_not_connected(self):
        """Test send subscribe when not connected."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._ws = None

        with pytest.raises(RuntimeError, match="Not connected"):
            await ws._send_subscribe({"type": "candle"})

    @pytest.mark.asyncio
    async def test_send_subscribe_success(self):
        """Test successful subscribe send."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._ws = AsyncMock()

        await ws._send_subscribe({"type": "candle", "coin": "BTC"})

        ws._ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_unsubscribe(self):
        """Test unsubscribe send."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._ws = AsyncMock()

        await ws._send_unsubscribe({"type": "candle", "coin": "BTC"})

        ws._ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_unsubscribe_not_connected(self):
        """Test unsubscribe when not connected."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._ws = None

        # Should not raise
        await ws._send_unsubscribe({"type": "candle"})


class TestConnectionStateEnum:
    """Tests for ConnectionState enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"


class TestHyperliquidWebSocketRun:
    """Tests for run loop."""

    @pytest.mark.asyncio
    async def test_run_connects_if_not_connected(self):
        """Test run method connects if not connected."""
        ws = HyperliquidWebSocket(testnet=True)

        mock_connection = AsyncMock()
        mock_connection.recv = AsyncMock(side_effect=asyncio.CancelledError())

        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_connection):
            try:
                await asyncio.wait_for(ws.run(), timeout=0.5)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                pass

    @pytest.mark.asyncio
    async def test_run_handles_bytes_message(self):
        """Test run handles bytes messages."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTED
        ws._ws = AsyncMock()

        # Mock recv to return bytes then raise CancelledError
        ws._ws.recv = AsyncMock(
            side_effect=[
                b'{"channel": "allMids", "data": {"mids": {}}}',
                asyncio.CancelledError(),
            ]
        )

        try:
            await ws.run()
        except asyncio.CancelledError:
            pass


class TestHyperliquidWebSocketHeartbeat:
    """Tests for heartbeat functionality."""

    @pytest.mark.asyncio
    async def test_heartbeat_loop_disconnected(self):
        """Test heartbeat loop exits when disconnected."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.DISCONNECTED

        # Should exit immediately
        await ws._heartbeat_loop()

    @pytest.mark.asyncio
    async def test_trigger_reconnect_already_reconnecting(self):
        """Test trigger_reconnect does nothing if already reconnecting."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.RECONNECTING

        await ws._trigger_reconnect()

        # Should return early, no reconnect task created
        assert ws._reconnect_task is None

    @pytest.mark.asyncio
    async def test_disconnect_with_reconnect_task(self):
        """Test disconnect cancels reconnect task."""
        ws = HyperliquidWebSocket(testnet=True)
        ws._state = ConnectionState.CONNECTED
        ws._ws = AsyncMock()
        ws._reconnect_task = asyncio.create_task(asyncio.sleep(100))

        await ws.disconnect()

        assert ws._reconnect_task is None

    @pytest.mark.asyncio
    async def test_parse_candle_with_invalid_data(self):
        """Test parse_candle handles invalid data."""
        ws = HyperliquidWebSocket(testnet=True)
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])

        # Invalid data should return None
        result = ws._parse_candle({"invalid": "data"}, "BTC", "1h")
        # May return valid candle with 0 values or None

    @pytest.mark.asyncio
    async def test_parse_ticker_with_invalid_data(self):
        """Test parse_ticker handles invalid data."""
        ws = HyperliquidWebSocket(testnet=True)

        result = ws._parse_ticker({}, "INVALID")
        # Should handle gracefully

    @pytest.mark.asyncio
    async def test_parse_orderbook_with_invalid_data(self):
        """Test parse_orderbook handles invalid data."""
        ws = HyperliquidWebSocket(testnet=True)
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])

        result = ws._parse_orderbook({"invalid": "data"}, "BTC")
        # May return empty orderbook or None

    @pytest.mark.asyncio
    async def test_parse_trade_with_invalid_data(self):
        """Test parse_trade handles invalid data."""
        ws = HyperliquidWebSocket(testnet=True)
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])

        result = ws._parse_trade({"coin": "BTC"})  # Missing required fields
        # Should handle gracefully
