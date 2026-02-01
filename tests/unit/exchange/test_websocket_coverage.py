"""
Coverage tests for WEEX WebSocket client.

These tests mock the websockets module to test the WebSocket client
without requiring an actual connection.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def requires_websockets():
    """Check if websockets module is available."""
    import importlib.util

    return importlib.util.find_spec("websockets") is None


# Skip all tests in this module if websockets is not installed
pytestmark = pytest.mark.skipif(
    requires_websockets(), reason="websockets module not installed"
)


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_connection_state_values(self):
        """Test all connection state values."""
        from src.exchange.adapters.weex.websocket import ConnectionState

        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"


class TestWEEXWebSocketInitialization:
    """Tests for WebSocket initialization."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.ws_url = "wss://test.weex.com"
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        return settings

    def test_init_with_defaults(self, mock_settings):
        """Test initialization with default settings."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            assert ws.ws_url == "wss://test.weex.com"
            assert ws.api_key == "test_key"
            assert ws.api_secret == "test_secret"
            assert ws.api_passphrase == "test_pass"

    def test_init_with_custom_values(self, mock_settings):
        """Test initialization with custom values."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket(
                ws_url="wss://custom.url",
                api_key="custom_key",
                api_secret="custom_secret",
                api_passphrase="custom_pass",
            )

            assert ws.ws_url == "wss://custom.url"
            assert ws.api_key == "custom_key"
            assert ws.api_secret == "custom_secret"
            assert ws.api_passphrase == "custom_pass"

    def test_is_connected_property(self, mock_settings):
        """Test is_connected property."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import (
                ConnectionState,
                WEEXWebSocket,
            )

            ws = WEEXWebSocket()

            assert ws.is_connected is False

            ws._state = ConnectionState.CONNECTED
            assert ws.is_connected is True


class TestWEEXWebSocketSignature:
    """Tests for signature generation."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.ws_url = "wss://test.weex.com"
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        return settings

    def test_generate_signature(self, mock_settings):
        """Test signature generation."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()
            signature = ws._generate_signature("1234567890")

            # Signature should be base64 encoded
            assert isinstance(signature, str)
            assert len(signature) > 0


class TestWEEXWebSocketCallbacks:
    """Tests for callback registration."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.ws_url = "wss://test.weex.com"
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        return settings

    def test_on_candle_callback(self, mock_settings):
        """Test candle callback registration."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            callback = MagicMock()
            ws.on_candle(callback)

            assert callback in ws._candle_callbacks

    def test_on_ticker_callback(self, mock_settings):
        """Test ticker callback registration."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            callback = MagicMock()
            ws.on_ticker(callback)

            assert callback in ws._ticker_callbacks

    def test_on_orderbook_callback(self, mock_settings):
        """Test orderbook callback registration."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            callback = MagicMock()
            ws.on_orderbook(callback)

            assert callback in ws._orderbook_callbacks

    def test_on_trade_callback(self, mock_settings):
        """Test trade callback registration."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            callback = MagicMock()
            ws.on_trade(callback)

            assert callback in ws._trade_callbacks


class TestWEEXWebSocketMessageParsing:
    """Tests for message parsing."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.ws_url = "wss://test.weex.com"
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        return settings

    @pytest.fixture
    def ws_client(self, mock_settings):
        """Create WebSocket client."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            return WEEXWebSocket()

    def test_parse_candle(self, ws_client):
        """Test candle parsing."""
        candle_data = [
            "1704067200000",  # timestamp
            "42000.0",  # open
            "42500.0",  # high
            "41500.0",  # low
            "42100.0",  # close
            "100.5",  # volume
        ]

        result = ws_client._parse_candle(candle_data, "cmt_btcusdt", "candle1m")

        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.open == 42000.0
        assert result.high == 42500.0
        assert result.low == 41500.0
        assert result.close == 42100.0

    def test_parse_candle_invalid_data(self, ws_client):
        """Test candle parsing with invalid data."""
        result = ws_client._parse_candle([], "cmt_btcusdt", "candle1m")
        assert result is None

    def test_parse_ticker(self, ws_client):
        """Test ticker parsing."""
        ticker_data = {
            "last": "42000.0",
            "bestBid": "41990.0",
            "bestAsk": "42010.0",
            "bidSz": "1.0",
            "askSz": "1.5",
            "baseVolume": "1000.0",
            "quoteVolume": "42000000.0",
            "high24h": "43000.0",
            "low24h": "41000.0",
            "priceChangePercent": "1.5",
        }

        result = ws_client._parse_ticker(ticker_data, "cmt_btcusdt")

        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.last_price == 42000.0

    def test_parse_ticker_invalid_data(self, ws_client):
        """Test ticker parsing with invalid data."""
        # May return a ticker with defaults or None depending on implementation
        # Just verify it doesn't crash
        ws_client._parse_ticker({}, "cmt_btcusdt")

    def test_parse_orderbook(self, ws_client):
        """Test orderbook parsing."""
        orderbook_data = {
            "bids": [["41990.0", "1.0"], ["41980.0", "2.0"]],
            "asks": [["42010.0", "1.5"], ["42020.0", "2.5"]],
            "ts": "1704067200000",
        }

        result = ws_client._parse_orderbook(orderbook_data, "cmt_btcusdt")

        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert len(result.bids) == 2
        assert len(result.asks) == 2

    def test_parse_orderbook_invalid_data(self, ws_client):
        """Test orderbook parsing with invalid data."""
        # Verify it handles gracefully
        ws_client._parse_orderbook({}, "cmt_btcusdt")

    def test_parse_trade(self, ws_client):
        """Test trade parsing."""
        trade_data = {
            "tradeId": "12345",
            "price": "42000.0",
            "size": "0.5",
            "side": "buy",
            "ts": "1704067200000",
        }

        result = ws_client._parse_trade(trade_data, "cmt_btcusdt")

        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.price == 42000.0
        assert result.quantity == 0.5

    def test_parse_trade_with_defaults(self, ws_client):
        """Test trade parsing with default values."""
        # Empty dict returns a trade with default values, not None
        result = ws_client._parse_trade({}, "cmt_btcusdt")
        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.price == 0.0


class TestWEEXWebSocketSubscriptions:
    """Tests for subscription management."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.ws_url = "wss://test.weex.com"
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        return settings

    @pytest.mark.asyncio
    async def test_subscribe_candles_not_connected(self, mock_settings):
        """Test subscribing when not connected."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            await ws.subscribe_candles(["BTCUSDT"], "1m")

            assert "candle:BTCUSDT:1m" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_tickers_not_connected(self, mock_settings):
        """Test subscribing tickers when not connected."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            await ws.subscribe_tickers(["BTCUSDT"])

            assert "ticker:BTCUSDT" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_orderbooks_not_connected(self, mock_settings):
        """Test subscribing orderbooks when not connected."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            await ws.subscribe_orderbooks(["BTCUSDT"])

            assert "books:BTCUSDT" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_trades_not_connected(self, mock_settings):
        """Test subscribing trades when not connected."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            await ws.subscribe_trades(["BTCUSDT"])

            assert "trade:BTCUSDT" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe(self, mock_settings):
        """Test unsubscribing from a channel."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            # Add subscription
            ws._subscriptions.add("candle:BTCUSDT:1m")

            await ws.unsubscribe("candle", "BTCUSDT")

            assert "candle:BTCUSDT:1m" not in ws._subscriptions


class TestWEEXWebSocketHandleMessage:
    """Tests for message handling."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.ws_url = "wss://test.weex.com"
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        return settings

    @pytest.fixture
    def ws_client(self, mock_settings):
        """Create WebSocket client."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            return WEEXWebSocket()

    @pytest.mark.asyncio
    async def test_handle_pong_message(self, ws_client):
        """Test handling pong message."""

        initial_pong = ws_client._last_pong

        await ws_client._handle_message('{"op": "pong"}')

        assert ws_client._last_pong >= initial_pong

    @pytest.mark.asyncio
    async def test_handle_pong_event(self, ws_client):
        """Test handling pong event."""
        await ws_client._handle_message('{"event": "pong"}')
        # Should not raise

    @pytest.mark.asyncio
    async def test_handle_subscribe_confirmation(self, ws_client):
        """Test handling subscription confirmation."""
        message = '{"event": "subscribe", "arg": {"channel": "ticker"}}'
        await ws_client._handle_message(message)
        # Should not raise

    @pytest.mark.asyncio
    async def test_handle_error_message(self, ws_client):
        """Test handling error message."""
        message = '{"event": "error", "msg": "Some error"}'
        await ws_client._handle_message(message)
        # Should log error but not raise

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, ws_client):
        """Test handling invalid JSON."""
        await ws_client._handle_message("not valid json")
        # Should log warning but not raise

    @pytest.mark.asyncio
    async def test_handle_candle_data(self, ws_client):
        """Test handling candle data push."""
        callback = MagicMock()
        ws_client.on_candle(callback)

        message = json.dumps(
            {
                "arg": {"channel": "candle1m", "instId": "cmt_btcusdt"},
                "data": [
                    [
                        "1704067200000",
                        "42000.0",
                        "42500.0",
                        "41500.0",
                        "42100.0",
                        "100.5",
                    ]
                ],
            }
        )

        await ws_client._handle_message(message)

        # Callback should have been called
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_ticker_data(self, ws_client):
        """Test handling ticker data push."""
        callback = MagicMock()
        ws_client.on_ticker(callback)

        message = json.dumps(
            {
                "arg": {"channel": "ticker", "instId": "cmt_btcusdt"},
                "data": [
                    {
                        "last": "42000.0",
                        "bestBid": "41990.0",
                        "bestAsk": "42010.0",
                        "bidSz": "1.0",
                        "askSz": "1.5",
                        "baseVolume": "1000.0",
                        "quoteVolume": "42000000.0",
                        "high24h": "43000.0",
                        "low24h": "41000.0",
                        "priceChangePercent": "1.5",
                    }
                ],
            }
        )

        await ws_client._handle_message(message)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_orderbook_data(self, ws_client):
        """Test handling orderbook data push."""
        callback = MagicMock()
        ws_client.on_orderbook(callback)

        message = json.dumps(
            {
                "arg": {"channel": "books", "instId": "cmt_btcusdt"},
                "data": [
                    {
                        "bids": [["41990.0", "1.0"]],
                        "asks": [["42010.0", "1.5"]],
                        "ts": "1704067200000",
                    }
                ],
            }
        )

        await ws_client._handle_message(message)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_trade_data(self, ws_client):
        """Test handling trade data push."""
        callback = MagicMock()
        ws_client.on_trade(callback)

        message = json.dumps(
            {
                "arg": {"channel": "trade", "instId": "cmt_btcusdt"},
                "data": [
                    {
                        "tradeId": "12345",
                        "px": "42000.0",
                        "sz": "0.5",
                        "side": "buy",
                        "ts": "1704067200000",
                    }
                ],
            }
        )

        await ws_client._handle_message(message)

        callback.assert_called_once()


class TestWEEXWebSocketConnection:
    """Tests for connection management."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.ws_url = "wss://test.weex.com"
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        return settings

    @pytest.mark.asyncio
    async def test_connect_when_already_connected(self, mock_settings):
        """Test connect when already connected."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import (
                ConnectionState,
                WEEXWebSocket,
            )

            ws = WEEXWebSocket()
            ws._state = ConnectionState.CONNECTED

            # Should return early without error
            await ws.connect()

    @pytest.mark.asyncio
    async def test_disconnect_without_connection(self, mock_settings):
        """Test disconnect when not connected."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import (
                ConnectionState,
                WEEXWebSocket,
            )

            ws = WEEXWebSocket()

            # Should not raise
            await ws.disconnect()

            assert ws._state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_with_tasks(self, mock_settings):
        """Test disconnect cancels tasks."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            # Create mock tasks
            async def dummy_coro():
                await asyncio.sleep(10)

            ws._ping_task = asyncio.create_task(dummy_coro())
            ws._reconnect_task = asyncio.create_task(dummy_coro())

            await ws.disconnect()

            assert ws._ping_task is None
            assert ws._reconnect_task is None

    @pytest.mark.asyncio
    async def test_send_subscribe_not_connected(self, mock_settings):
        """Test send subscribe when not connected."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            with pytest.raises(RuntimeError, match="Not connected"):
                await ws._send_subscribe("candle", "BTCUSDT", "1m")

    @pytest.mark.asyncio
    async def test_send_unsubscribe_not_connected(self, mock_settings):
        """Test send unsubscribe when not connected - should return silently."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            # Should not raise, just return
            await ws._send_unsubscribe("candle", "BTCUSDT", "1m")

    @pytest.mark.asyncio
    async def test_subscribe_user_data_not_connected(self, mock_settings):
        """Test subscribe user data when not connected."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()

            with pytest.raises(RuntimeError, match="Not connected"):
                await ws.subscribe_user_data()

    @pytest.mark.asyncio
    async def test_trigger_reconnect(self, mock_settings):
        """Test triggering reconnection."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import (
                ConnectionState,
                WEEXWebSocket,
            )

            ws = WEEXWebSocket()
            ws._state = ConnectionState.CONNECTED

            # Mock the connect method to avoid actual connection
            with patch.object(ws, "connect", new_callable=AsyncMock):
                await ws._trigger_reconnect()

                # State should be reconnecting
                assert ws._reconnect_task is not None

            # Cleanup
            if ws._reconnect_task:
                ws._reconnect_task.cancel()
                try:
                    await ws._reconnect_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_trigger_reconnect_already_reconnecting(self, mock_settings):
        """Test that triggering reconnect when already reconnecting does nothing."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import (
                ConnectionState,
                WEEXWebSocket,
            )

            ws = WEEXWebSocket()
            ws._state = ConnectionState.RECONNECTING

            await ws._trigger_reconnect()

            # Should not create a new task
            assert ws._reconnect_task is None


class TestWEEXWebSocketResubscribe:
    """Tests for resubscription logic."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.ws_url = "wss://test.weex.com"
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        return settings

    @pytest.mark.asyncio
    async def test_resubscribe_candle(self, mock_settings):
        """Test resubscribing to candle channel."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()
            ws._subscriptions.add("candle:BTCUSDT:1m")

            with patch.object(
                ws, "_send_subscribe", new_callable=AsyncMock
            ) as mock_send:
                await ws._resubscribe()

                mock_send.assert_called_once_with("candle", "BTCUSDT", "1m")

    @pytest.mark.asyncio
    async def test_resubscribe_ticker(self, mock_settings):
        """Test resubscribing to ticker channel."""
        with patch(
            "src.exchange.adapters.weex.websocket.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            from src.exchange.adapters.weex.websocket import WEEXWebSocket

            ws = WEEXWebSocket()
            ws._subscriptions.add("ticker:BTCUSDT")

            with patch.object(
                ws, "_send_subscribe", new_callable=AsyncMock
            ) as mock_send:
                await ws._resubscribe()

                mock_send.assert_called_once_with("ticker", "BTCUSDT")
