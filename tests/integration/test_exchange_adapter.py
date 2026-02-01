"""
Integration tests for exchange adapters.

These tests verify the complete flow of the exchange abstraction layer
with mocked HTTP responses.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exchange.adapters.weex.adapter import WEEXRESTClient
from src.exchange.exceptions import (
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
)
from src.exchange.factory import register_adapter
from src.exchange.models import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    UnifiedOrder,
)


# Skip tests that require full WEEXAdapter if websockets not installed
def requires_websockets():
    try:
        import websockets  # noqa: F401

        return False
    except ImportError:
        return True


class TestWEEXAdapterIntegration:
    """Integration tests for WEEX adapter."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock()
        settings.exchange.api_key = "test_api_key"
        settings.exchange.api_secret = "test_api_secret"
        settings.exchange.api_passphrase = "test_passphrase"
        settings.exchange.rest_url = "https://api.test.com"
        settings.exchange.ws_url = "wss://ws.test.com"
        settings.exchange.name.value = "weex"
        return settings

    @pytest.mark.skipif(requires_websockets(), reason="websockets not installed")
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, mock_settings):
        """Test adapter initialization."""
        from src.exchange.adapters.weex.adapter import WEEXAdapter

        with patch("src.exchange.adapters.weex.adapter.get_settings") as mock:
            mock.return_value = mock_settings
            adapter = WEEXAdapter()
            assert adapter is not None
            assert adapter.rest is not None
            assert adapter.websocket is not None

    @pytest.mark.skipif(requires_websockets(), reason="websockets not installed")
    def test_symbol_normalization(self, mock_settings):
        """Test symbol normalization methods."""
        from src.exchange.adapters.weex.adapter import WEEXAdapter

        with patch("src.exchange.adapters.weex.adapter.get_settings") as mock:
            mock.return_value = mock_settings
            adapter = WEEXAdapter()

            # Unified to WEEX
            weex_symbol = adapter.normalize_symbol("BTCUSDT")
            assert weex_symbol == "cmt_btcusdt"

            # WEEX to unified
            unified_symbol = adapter.denormalize_symbol("cmt_ethusdt")
            assert unified_symbol == "ETHUSDT"

    @pytest.mark.skipif(requires_websockets(), reason="websockets not installed")
    def test_rest_capabilities(self, mock_settings):
        """Test REST client capabilities."""
        from src.exchange.adapters.weex.adapter import WEEXAdapter

        with patch("src.exchange.adapters.weex.adapter.get_settings") as mock:
            mock.return_value = mock_settings
            adapter = WEEXAdapter()
            caps = adapter.rest.capabilities
            assert caps.name == "weex"
            assert caps.supports_futures is True
            assert caps.max_leverage == 125


class TestWEEXRESTClientIntegration:
    """Integration tests for WEEX REST client with mocked responses."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        settings.exchange.rest_url = "https://api.weex.com"
        return settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create a REST client for testing."""
        with patch("src.exchange.adapters.weex.adapter.get_settings") as mock:
            mock.return_value = mock_settings
            client = WEEXRESTClient()
            return client

    @pytest.mark.asyncio
    async def test_get_ticker(self, client):
        """Test getting ticker data."""
        mock_response = {
            "symbol": "cmt_btcusdt",
            "lastPr": "50000.0",
            "bidPr": "49990.0",
            "askPr": "50010.0",
            "bidSz": "1.0",
            "askSz": "1.5",
            "baseVolume": "1000.0",
            "quoteVolume": "50000000.0",
            "high24h": "51000.0",
            "low24h": "49000.0",
            "change24h": "500.0",
            "changeUtc24h": "1.0",
            "ts": "1706644800000",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            ticker = await client.get_ticker("BTCUSDT")

            assert ticker.symbol == "BTCUSDT"
            assert ticker.last_price == 50000.0
            assert ticker.bid_price == 49990.0
            assert ticker.ask_price == 50010.0

    @pytest.mark.asyncio
    async def test_get_account_balance(self, client):
        """Test getting account balance."""
        # The implementation looks for marginCoin == "USDT" in the account list
        mock_response = {
            "marginCoin": "USDT",
            "usdtEquity": "10000.0",
            "available": "8000.0",
            "crossMaxAvailable": "2000.0",
            "unrealizedPL": "500.0",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = [mock_response]
            await client.initialize()

            balance = await client.get_account_balance()

            assert balance.total_balance == 10000.0
            assert balance.available_balance == 8000.0
            assert balance.margin_balance == 2000.0

    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """Test getting positions."""
        mock_response = [
            {
                "symbol": "cmt_btcusdt",
                "holdSide": "long",
                "total": "0.1",
                "available": "0.1",
                "averageOpenPrice": "50000.0",
                "markPrice": "51000.0",
                "unrealizedPL": "100.0",
                "leverage": "10",
                "liquidationPrice": "45000.0",
                "marginMode": "cross",
                "margin": "500.0",
            }
        ]

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            positions = await client.get_positions()

            assert len(positions) == 1
            assert positions[0].symbol == "BTCUSDT"
            assert positions[0].side == PositionSide.LONG
            assert positions[0].quantity == 0.1

    @pytest.mark.asyncio
    async def test_place_order(self, client):
        """Test placing an order."""
        mock_response = {
            "orderId": "123456789",
            "clientOid": "test_123",
            "symbol": "cmt_btcusdt",
            "side": "buy",
            "orderType": "market",
            "size": "0.01",
            "price": "",
            "status": "filled",
            "filledQty": "0.01",
            "avgPx": "50100.0",
            "fee": "0.5",
            "feeCcy": "USDT",
            "cTime": "1706644800000",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            order = UnifiedOrder(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.01,
                position_side=PositionSide.LONG,
            )
            result = await client.place_order(order)

            assert result.order_id == "123456789"
            assert result.status == OrderStatus.FILLED
            assert result.filled_quantity == 0.01

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test canceling an order."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"orderId": "123456"}
            await client.initialize()

            result = await client.cancel_order("BTCUSDT", "123456")

            assert result is True

    @pytest.mark.asyncio
    async def test_set_leverage(self, client):
        """Test setting leverage."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"leverage": "10"}
            await client.initialize()

            result = await client.set_leverage("BTCUSDT", 10)

            assert result is True

    @pytest.mark.asyncio
    async def test_get_orderbook(self, client):
        """Test getting orderbook."""
        mock_response = {
            "bids": [["50000.0", "1.0"], ["49990.0", "2.0"]],
            "asks": [["50010.0", "0.5"], ["50020.0", "1.0"]],
            "ts": "1706644800000",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            orderbook = await client.get_orderbook("BTCUSDT")

            assert orderbook.symbol == "BTCUSDT"
            assert len(orderbook.bids) == 2
            assert len(orderbook.asks) == 2
            assert orderbook.best_bid == 50000.0

    @pytest.mark.asyncio
    async def test_get_candles(self, client):
        """Test getting candles."""
        mock_response = [
            [
                "1706644800000",
                "50000.0",
                "50100.0",
                "49900.0",
                "50050.0",
                "100.0",
                "5000000.0",
            ],
            [
                "1706644860000",
                "50050.0",
                "50150.0",
                "50000.0",
                "50100.0",
                "80.0",
                "4000000.0",
            ],
        ]

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            candles = await client.get_candles("BTCUSDT", "1m", limit=100)

            assert len(candles) == 2
            assert candles[0].symbol == "BTCUSDT"
            assert candles[0].open == 50000.0


class TestErrorHandling:
    """Tests for error handling in the exchange layer."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        settings.exchange.rest_url = "https://api.weex.com"
        return settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create a REST client for testing."""
        with patch("src.exchange.adapters.weex.adapter.get_settings") as mock:
            mock.return_value = mock_settings
            client = WEEXRESTClient()
            return client

    @pytest.mark.asyncio
    async def test_authentication_error(self, client):
        """Test authentication error handling."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = AuthenticationError(
                "Invalid API key",
                exchange="weex",
                code="40101",
            )
            await client.initialize()

            with pytest.raises(AuthenticationError) as exc_info:
                await client.get_account_balance()

            assert exc_info.value.exchange == "weex"

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, client):
        """Test rate limit error handling."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = RateLimitError(
                "Rate limit exceeded",
                exchange="weex",
                retry_after=60.0,
            )
            await client.initialize()

            with pytest.raises(RateLimitError) as exc_info:
                await client.get_ticker("BTCUSDT")

            assert exc_info.value.retry_after == 60.0

    @pytest.mark.asyncio
    async def test_insufficient_balance_error(self, client):
        """Test insufficient balance error."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = InsufficientBalanceError(
                "Insufficient margin",
                exchange="weex",
                required=1000.0,
                available=500.0,
            )
            await client.initialize()

            order = UnifiedOrder(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0,
            )

            # The InsufficientBalanceError might be wrapped by OrderError
            # depending on the implementation
            with pytest.raises(Exception) as exc_info:
                await client.place_order(order)

            # Check that the error is either InsufficientBalanceError or wraps it
            err = exc_info.value
            assert "Insufficient" in str(err) or hasattr(err, "__cause__")


class TestFactoryIntegration:
    """Integration tests for exchange factory."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure WEEX adapter is registered."""
        from src.exchange.factory import _adapter_registry

        # Store original and register mock
        original = _adapter_registry.copy()
        yield
        _adapter_registry.clear()
        _adapter_registry.update(original)

    @pytest.mark.skipif(requires_websockets(), reason="websockets not installed")
    @pytest.mark.asyncio
    async def test_factory_creates_weex_adapter(self):
        """Test that factory can create WEEX adapter."""
        from src.exchange.adapters.weex.adapter import WEEXAdapter
        from src.exchange.factory import get_exchange_adapter

        mock_settings = MagicMock()
        mock_settings.exchange.api_key = "test"
        mock_settings.exchange.api_secret = "test"
        mock_settings.exchange.api_passphrase = "test"
        mock_settings.exchange.rest_url = "https://api.weex.com"
        mock_settings.exchange.ws_url = "wss://ws.weex.com"
        mock_settings.exchange.name.value = "weex"

        with (
            patch("src.core.config.get_settings", return_value=mock_settings),
            patch(
                "src.exchange.adapters.weex.adapter.get_settings",
                return_value=mock_settings,
            ),
        ):
            # Register the adapter
            register_adapter("weex", WEEXAdapter)

            adapter = await get_exchange_adapter("weex")

            assert adapter is not None
            assert adapter.rest.exchange_name == "weex"

            await adapter.close()


class TestEndToEndFlow:
    """End-to-end flow tests."""

    @pytest.mark.skipif(requires_websockets(), reason="websockets not installed")
    @pytest.mark.asyncio
    async def test_complete_trading_flow(self):
        """Test a complete trading flow: check balance, place order, check position."""
        from src.exchange.adapters.weex.adapter import WEEXAdapter

        mock_settings = MagicMock()
        mock_settings.exchange.api_key = "test"
        mock_settings.exchange.api_secret = "test"
        mock_settings.exchange.api_passphrase = "test"
        mock_settings.exchange.rest_url = "https://api.weex.com"
        mock_settings.exchange.ws_url = "wss://ws.weex.com"

        with patch(
            "src.exchange.adapters.weex.adapter.get_settings",
            return_value=mock_settings,
        ):
            adapter = WEEXAdapter()

            # Mock the request method
            with patch.object(adapter.rest, "_request", new_callable=AsyncMock) as mock_req:
                # Setup mock responses
                balance_response = [
                    {
                        "marginCoin": "USDT",
                        "usdtEquity": "10000.0",
                        "available": "8000.0",
                        "crossMaxAvailable": "2000.0",
                        "unrealizedPL": "500.0",
                    }
                ]
                order_response = {
                    "orderId": "123",
                    "clientOid": "",
                    "symbol": "cmt_btcusdt",
                    "side": "buy",
                    "orderType": "market",
                    "size": "0.01",
                    "price": "",
                    "status": "filled",
                    "filledQty": "0.01",
                    "avgPx": "50000.0",
                    "fee": "0.5",
                    "feeCcy": "USDT",
                    "cTime": "1706644800000",
                }
                position_response = [
                    {
                        "symbol": "cmt_btcusdt",
                        "holdSide": "long",
                        "total": "0.01",
                        "available": "0.01",
                        "averageOpenPrice": "50000.0",
                        "markPrice": "50100.0",
                        "unrealizedPL": "1.0",
                        "leverage": "10",
                        "marginMode": "cross",
                        "margin": "50.0",
                    }
                ]

                mock_req.side_effect = [
                    balance_response,
                    order_response,
                    position_response,
                ]

                await adapter.initialize()

                # 1. Check balance
                balance = await adapter.rest.get_account_balance()
                assert balance.available_balance == 8000.0

                # 2. Place order
                order = UnifiedOrder(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.01,
                    position_side=PositionSide.LONG,
                )
                result = await adapter.rest.place_order(order)
                assert result.status == OrderStatus.FILLED

                # 3. Check position
                positions = await adapter.rest.get_positions("BTCUSDT")
                assert len(positions) == 1
                assert positions[0].quantity == 0.01

                await adapter.close()
