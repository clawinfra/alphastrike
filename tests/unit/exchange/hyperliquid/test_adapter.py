"""
Tests for Hyperliquid REST adapter.

Coverage targets: REST client methods, error handling, credential management.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

from src.exchange.adapters.hyperliquid.adapter import (
    HyperliquidRESTClient,
    HyperliquidAdapter,
    MAINNET_URL,
    TESTNET_URL,
)
from src.exchange.adapters.hyperliquid.websocket import ConnectionState
from src.exchange.adapters.hyperliquid.mappers import HyperliquidMapper
from src.exchange.exceptions import (
    AuthenticationError,
    ExchangeError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderNotFoundError,
    RateLimitError,
)
from src.exchange.models import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    UnifiedOrder,
)


class TestHyperliquidRESTClientInit:
    """Tests for REST client initialization."""

    def test_init_with_testnet(self):
        """Test initialization with testnet."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=None,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=True)

            assert client.testnet is True
            assert client.base_url == TESTNET_URL

    def test_init_with_mainnet(self):
        """Test initialization with mainnet."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=None,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=False)

            assert client.testnet is False
            assert client.base_url == MAINNET_URL

    def test_init_with_private_key(self):
        """Test initialization with explicit private key."""
        pk = "0x" + "a" * 64
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "b" * 64,  # Different key in settings
                    wallet_address="0x1234",
                )
            )
            client = HyperliquidRESTClient(private_key=pk, testnet=True)

            # Should use provided key, not settings
            assert client._signer.private_key == pk
            # wallet_address should be derived from pk, not loaded from settings
            assert client._signer.wallet_address != "0x1234"

    def test_init_loads_from_settings(self):
        """Test initialization loads credentials from settings."""
        pk = "0x" + "a" * 64
        addr = "0x1234567890abcdef1234567890abcdef12345678"
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=pk,
                    wallet_address=addr,
                )
            )
            client = HyperliquidRESTClient(testnet=True)

            assert client._signer.private_key == pk
            assert client._signer.wallet_address == addr

    def test_init_with_custom_url(self):
        """Test initialization with custom base URL."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=None,
                    wallet_address=None,
                )
            )
            custom_url = "https://custom.api.com"
            client = HyperliquidRESTClient(base_url=custom_url, testnet=True)

            assert client.base_url == custom_url


class TestHyperliquidRESTClientProperties:
    """Tests for REST client properties."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            return HyperliquidRESTClient(testnet=True)

    def test_exchange_name(self, client):
        """Test exchange_name property."""
        assert client.exchange_name == "hyperliquid"

    def test_capabilities(self, client):
        """Test capabilities property."""
        caps = client.capabilities
        assert caps.name == "hyperliquid"
        assert caps.supports_futures is True

    def test_wallet_address(self, client):
        """Test wallet_address property."""
        assert client.wallet_address is not None
        assert client.wallet_address.startswith("0x")


class TestHyperliquidRESTClientLifecycle:
    """Tests for REST client lifecycle methods."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            return HyperliquidRESTClient(testnet=True)

    @pytest.mark.asyncio
    async def test_initialize(self, client):
        """Test initialize creates session and loads metadata."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps({
            "universe": [{"name": "BTC"}, {"name": "ETH"}]
        }))

        mock_session = MagicMock()
        mock_session.request = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(),
            )
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client.initialize()
            assert client._session is not None

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test close cleans up session."""
        client._session = MagicMock()
        client._session.close = AsyncMock()

        await client.close()

        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_without_session(self, client):
        """Test close without initialized session."""
        client._session = None
        await client.close()  # Should not raise


class TestHyperliquidRESTClientErrorHandling:
    """Tests for REST client error handling."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            return HyperliquidRESTClient(testnet=True)

    def test_handle_error_insufficient_balance(self, client):
        """Test insufficient balance error handling."""
        with pytest.raises(InsufficientBalanceError):
            client._handle_error("Insufficient balance for order")

    def test_handle_error_not_found(self, client):
        """Test order not found error handling."""
        with pytest.raises(OrderNotFoundError):
            client._handle_error("Order not found")

        with pytest.raises(OrderNotFoundError):
            client._handle_error("unknown order id")

    def test_handle_error_invalid_order(self, client):
        """Test invalid order error handling."""
        with pytest.raises(InvalidOrderError):
            client._handle_error("Invalid order parameters")

    def test_handle_error_generic(self, client):
        """Test generic error handling."""
        with pytest.raises(ExchangeError):
            client._handle_error("Some random error")


class TestHyperliquidRESTClientMarketData:
    """Tests for market data methods."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=True)
            HyperliquidMapper.set_asset_meta([{"name": "BTC"}, {"name": "ETH"}])
            return client

    @pytest.mark.asyncio
    async def test_get_ticker(self, client):
        """Test get_ticker."""
        mock_response = {"BTC": "50000.0", "ETH": "3000.0"}

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            ticker = await client.get_ticker("BTCUSDT")

            assert ticker.symbol == "BTCUSDT"
            assert ticker.last_price == 50000.0

    @pytest.mark.asyncio
    async def test_get_all_tickers(self, client):
        """Test get_all_tickers."""
        mock_response = {"BTC": "50000.0", "ETH": "3000.0"}

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            tickers = await client.get_all_tickers()

            assert "BTCUSDT" in tickers
            assert "ETHUSDT" in tickers

    @pytest.mark.asyncio
    async def test_get_orderbook(self, client):
        """Test get_orderbook."""
        mock_response = {
            "levels": [
                [{"px": "49990", "sz": "1.5"}],
                [{"px": "50010", "sz": "1.0"}],
            ]
        }

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            orderbook = await client.get_orderbook("BTCUSDT")

            assert orderbook.symbol == "BTCUSDT"
            assert len(orderbook.bids) == 1
            assert len(orderbook.asks) == 1

    @pytest.mark.asyncio
    async def test_get_candles(self, client):
        """Test get_candles."""
        mock_response = [{
            "t": 1706644800000,
            "o": "50000.0",
            "h": "50500.0",
            "l": "49500.0",
            "c": "50200.0",
            "v": "1000.5",
            "n": 5000,
        }]

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            candles = await client.get_candles("BTCUSDT", interval="1h", limit=10)

            assert len(candles) == 1
            assert candles[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_recent_trades(self, client):
        """Test get_recent_trades returns empty (not supported)."""
        trades = await client.get_recent_trades("BTCUSDT")
        assert trades == []

    @pytest.mark.asyncio
    async def test_get_funding_rate(self, client):
        """Test get_funding_rate returns 0 (placeholder)."""
        rate = await client.get_funding_rate("BTCUSDT")
        assert rate == 0.0


class TestHyperliquidRESTClientAccount:
    """Tests for account methods."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=True)
            HyperliquidMapper.set_asset_meta([{"name": "BTC"}])
            return client

    @pytest.mark.asyncio
    async def test_get_account_balance(self, client):
        """Test get_account_balance."""
        spot_response = {
            "balances": [{"coin": "USDC", "total": "10000.0", "hold": "2000.0"}]
        }
        perps_response = {
            "marginSummary": {"totalNtlPos": "5000.0"}
        }

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [spot_response, perps_response]

            balance = await client.get_account_balance()

            assert balance.total_balance == 10000.0
            assert balance.available_balance == 8000.0

    @pytest.mark.asyncio
    async def test_get_account_balance_no_wallet(self, client):
        """Test get_account_balance without wallet address."""
        client._signer._wallet_address = None

        with pytest.raises(AuthenticationError):
            await client.get_account_balance()

    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """Test get_positions."""
        mock_response = {
            "assetPositions": [{
                "position": {
                    "coin": "BTC",
                    "szi": "0.5",
                    "entryPx": "50000.0",
                    "unrealizedPnl": "500.0",
                    "leverage": {"type": "cross", "value": 10},
                }
            }]
        }

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            positions = await client.get_positions()

            assert len(positions) == 1
            assert positions[0].symbol == "BTCUSDT"
            assert positions[0].quantity == 0.5

    @pytest.mark.asyncio
    async def test_get_positions_filter_by_symbol(self, client):
        """Test get_positions with symbol filter."""
        mock_response = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "0.5", "entryPx": "50000.0", "leverage": {"value": 1}}},
                {"position": {"coin": "ETH", "szi": "1.0", "entryPx": "3000.0", "leverage": {"value": 1}}},
            ]
        }

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            positions = await client.get_positions(symbol="BTCUSDT")

            assert len(positions) == 1
            assert positions[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_position(self, client):
        """Test get_position."""
        mock_response = {
            "assetPositions": [{
                "position": {
                    "coin": "BTC",
                    "szi": "0.5",
                    "entryPx": "50000.0",
                    "leverage": {"type": "cross", "value": 10},
                }
            }]
        }

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            position = await client.get_position("BTCUSDT", PositionSide.LONG)

            assert position is not None
            assert position.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_position_not_found(self, client):
        """Test get_position when position doesn't exist."""
        mock_response = {"assetPositions": []}

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            position = await client.get_position("BTCUSDT", PositionSide.LONG)

            assert position is None


class TestHyperliquidRESTClientOrders:
    """Tests for order methods."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=True)
            HyperliquidMapper.set_asset_meta([{"name": "BTC"}])
            return client

    @pytest.mark.asyncio
    async def test_place_order(self, client):
        """Test place_order."""
        mock_response = {
            "response": {
                "data": {
                    "statuses": [{"resting": {"oid": 12345}}]
                }
            }
        }

        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
        )

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                with patch.object(client, "_exchange_request", new_callable=AsyncMock) as mock_req:
                    mock_req.return_value = mock_response

                    result = await client.place_order(order)

                    assert result.order_id == "12345"
                    assert result.status == OrderStatus.NEW

    @pytest.mark.asyncio
    async def test_place_order_no_wallet(self, client):
        """Test place_order without wallet."""
        client._signer._wallet_address = None

        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
        )

        with pytest.raises(AuthenticationError):
            await client.place_order(order)

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test cancel_order."""
        mock_response = {
            "response": {
                "data": {
                    "statuses": ["success"]
                }
            }
        }

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                with patch.object(client, "_exchange_request", new_callable=AsyncMock) as mock_req:
                    mock_req.return_value = mock_response

                    result = await client.cancel_order("BTCUSDT", order_id="12345")

                    assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_no_id(self, client):
        """Test cancel_order without order ID."""
        with pytest.raises(InvalidOrderError):
            await client.cancel_order("BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_order(self, client):
        """Test get_order."""
        mock_response = {
            "status": "ok",
            "order": {
                "order": {
                    "oid": 12345,
                    "side": "B",
                    "orderType": "Limit",
                    "origSz": "0.1",
                    "sz": "0.05",
                    "limitPx": "50000.0",
                    "timestamp": 1706644800000,
                },
                "status": "open",
            }
        }

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            order = await client.get_order("BTCUSDT", "12345")

            assert order.order_id == "12345"
            assert order.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, client):
        """Test get_order when order not found."""
        mock_response = {"status": "unknownOid"}

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            with pytest.raises(OrderNotFoundError):
                await client.get_order("BTCUSDT", "99999")

    @pytest.mark.asyncio
    async def test_get_open_orders(self, client):
        """Test get_open_orders."""
        mock_response = [{
            "coin": "BTC",
            "oid": 12345,
            "side": "B",
            "origSz": "0.1",
            "limitPx": "50000.0",
            "timestamp": 1706644800000,
        }]

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            orders = await client.get_open_orders()

            assert len(orders) == 1
            assert orders[0].order_id == "12345"


class TestHyperliquidRESTClientConditionalOrders:
    """Tests for conditional order methods."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=True)
            HyperliquidMapper.set_asset_meta([{"name": "BTC"}])
            return client

    @pytest.mark.asyncio
    async def test_place_stop_loss(self, client):
        """Test place_stop_loss."""
        mock_response = {
            "response": {"data": {"statuses": [{"resting": {"oid": 123}}]}}
        }

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                with patch.object(client, "_exchange_request", new_callable=AsyncMock) as mock_req:
                    mock_req.return_value = mock_response

                    result = await client.place_stop_loss(
                        symbol="BTCUSDT",
                        side=OrderSide.SELL,
                        trigger_price=48000.0,
                        size=0.1,
                    )

                    assert result.order_id == "123"

    @pytest.mark.asyncio
    async def test_place_take_profit(self, client):
        """Test place_take_profit."""
        mock_response = {
            "response": {"data": {"statuses": [{"resting": {"oid": 456}}]}}
        }

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                with patch.object(client, "_exchange_request", new_callable=AsyncMock) as mock_req:
                    mock_req.return_value = mock_response

                    result = await client.place_take_profit(
                        symbol="BTCUSDT",
                        side=OrderSide.SELL,
                        trigger_price=55000.0,
                        size=0.1,
                    )

                    assert result.order_id == "456"


class TestHyperliquidRESTClientLeverage:
    """Tests for leverage methods."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=True)
            HyperliquidMapper.set_asset_meta([{"name": "BTC"}])
            return client

    @pytest.mark.asyncio
    async def test_set_leverage(self, client):
        """Test set_leverage."""
        mock_response = {"status": "ok"}

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                with patch.object(client, "_exchange_request", new_callable=AsyncMock) as mock_req:
                    mock_req.return_value = mock_response

                    result = await client.set_leverage("BTCUSDT", 10)

                    assert result is True

    @pytest.mark.asyncio
    async def test_set_leverage_no_wallet(self, client):
        """Test set_leverage without wallet."""
        client._signer._wallet_address = None

        with pytest.raises(AuthenticationError):
            await client.set_leverage("BTCUSDT", 10)

    @pytest.mark.asyncio
    async def test_get_leverage(self, client):
        """Test get_leverage."""
        mock_response = {
            "assetPositions": [{
                "position": {
                    "coin": "BTC",
                    "szi": "0.1",
                    "entryPx": "50000.0",
                    "leverage": {"value": 10},
                }
            }]
        }

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            leverage = await client.get_leverage("BTCUSDT")

            assert leverage == 10

    @pytest.mark.asyncio
    async def test_get_leverage_no_position(self, client):
        """Test get_leverage without position returns default."""
        mock_response = {"assetPositions": []}

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            leverage = await client.get_leverage("BTCUSDT")

            assert leverage == 1


class TestHyperliquidRESTClientTrades:
    """Tests for trade history methods."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=True)
            HyperliquidMapper.set_asset_meta([{"name": "BTC"}])
            return client

    @pytest.mark.asyncio
    async def test_get_user_trades(self, client):
        """Test get_user_trades."""
        mock_response = [{
            "coin": "BTC",
            "tid": "trade1",
            "px": "50000.0",
            "sz": "0.1",
            "side": "B",
            "time": 1706644800000,
        }]

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            trades = await client.get_user_trades()

            assert len(trades) == 1
            assert trades[0].trade_id == "trade1"

    @pytest.mark.asyncio
    async def test_get_user_trades_with_time_range(self, client):
        """Test get_user_trades with time range."""
        mock_response = []
        start_time = datetime(2024, 1, 1, tzinfo=UTC)
        end_time = datetime(2024, 1, 31, tzinfo=UTC)

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            trades = await client.get_user_trades(
                start_time=start_time,
                end_time=end_time,
            )

            # Should use userFillsByTime
            call_args = mock_req.call_args[0][0]
            assert call_args["type"] == "userFillsByTime"


class TestHyperliquidRESTClientSymbols:
    """Tests for symbol info methods."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            client = HyperliquidRESTClient(testnet=True)
            HyperliquidMapper.set_asset_meta([
                {"name": "BTC", "szDecimals": 4, "maxLeverage": 100},
                {"name": "ETH", "szDecimals": 3, "maxLeverage": 50},
            ])
            return client

    @pytest.mark.asyncio
    async def test_get_symbol_info(self, client):
        """Test get_symbol_info."""
        # Ensure metadata is loaded for this test
        HyperliquidMapper.set_asset_meta([
            {"name": "BTC", "szDecimals": 4, "maxLeverage": 100},
            {"name": "ETH", "szDecimals": 3, "maxLeverage": 50},
        ])
        client._metadata_loaded = True

        info = await client.get_symbol_info("BTCUSDT")

        assert info.symbol == "BTCUSDT"
        assert info.base_asset == "BTC"
        assert info.max_leverage == 100

    @pytest.mark.asyncio
    async def test_get_symbol_info_cached(self, client):
        """Test get_symbol_info uses cache."""
        HyperliquidMapper.set_asset_meta([
            {"name": "BTC", "szDecimals": 4, "maxLeverage": 100},
        ])
        client._metadata_loaded = True

        # First call
        info1 = await client.get_symbol_info("BTCUSDT")
        # Second call should use cache
        info2 = await client.get_symbol_info("BTCUSDT")

        assert info1.symbol == info2.symbol

    @pytest.mark.asyncio
    async def test_get_symbol_info_unknown(self, client):
        """Test get_symbol_info for unknown symbol."""
        HyperliquidMapper._coin_to_index.clear()
        HyperliquidMapper._asset_meta.clear()
        client._metadata_loaded = True

        with pytest.raises(ExchangeError, match="not found"):
            await client.get_symbol_info("UNKNOWNUSDT")

    @pytest.mark.asyncio
    async def test_get_all_symbols(self, client):
        """Test get_all_symbols."""
        HyperliquidMapper.set_asset_meta([
            {"name": "BTC", "szDecimals": 4, "maxLeverage": 100},
            {"name": "ETH", "szDecimals": 3, "maxLeverage": 50},
        ])
        client._metadata_loaded = True

        symbols = await client.get_all_symbols()

        assert len(symbols) == 2
        assert any(s.symbol == "BTCUSDT" for s in symbols)
        assert any(s.symbol == "ETHUSDT" for s in symbols)


class TestHyperliquidRESTClientMetadata:
    """Tests for metadata loading."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            return HyperliquidRESTClient(testnet=True)

    @pytest.mark.asyncio
    async def test_load_metadata_failure(self, client):
        """Test metadata loading failure is handled gracefully."""
        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = Exception("API Error")

            await client._load_metadata()

            # Should not raise, just log warning
            assert client._metadata_loaded is False

    @pytest.mark.asyncio
    async def test_get_candles_empty_response(self, client):
        """Test get_candles with empty response."""
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = []

            candles = await client.get_candles("BTCUSDT", limit=10)

            assert candles == []

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, client):
        """Test cancel_order failure."""
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                with patch.object(client, "_exchange_request", new_callable=AsyncMock) as mock_req:
                    mock_req.side_effect = ExchangeError("Cancel failed")

                    result = await client.cancel_order("BTCUSDT", order_id="12345")

                    assert result is False

    @pytest.mark.asyncio
    async def test_set_leverage_failure(self, client):
        """Test set_leverage failure."""
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}])

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                with patch.object(client, "_exchange_request", new_callable=AsyncMock) as mock_req:
                    mock_req.side_effect = ExchangeError("Leverage update failed")

                    result = await client.set_leverage("BTCUSDT", 10)

                    assert result is False


class TestHyperliquidRESTClientRequest:
    """Tests for request handling."""

    @pytest.fixture
    def client(self):
        """Create client fixture."""
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key="0x" + "a" * 64,
                    wallet_address=None,
                )
            )
            return HyperliquidRESTClient(testnet=True)

    @pytest.mark.asyncio
    async def test_request_auth_error(self, client):
        """Test authentication error."""
        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")

        class MockContextManager:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                pass

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=MockContextManager())
        client._session = mock_session

        with pytest.raises(AuthenticationError):
            await client._request("POST", "/info", data={})

    @pytest.mark.asyncio
    async def test_request_api_error_response(self, client):
        """Test API error in response body."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"status": "err", "response": "Order rejected"}')

        class MockContextManager:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                pass

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=MockContextManager())
        client._session = mock_session

        with pytest.raises(ExchangeError, match="Order rejected"):
            await client._request("POST", "/info", data={})


class TestHyperliquidAdapter:
    """Tests for the main adapter class."""

    def test_init(self):
        """Test adapter initialization."""
        pk = "0x" + "a" * 64
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=pk,
                    wallet_address=None,
                )
            )
            adapter = HyperliquidAdapter(testnet=True)

            assert adapter.rest is not None
            assert adapter.websocket is not None

    def test_normalize_symbol(self):
        """Test symbol normalization."""
        pk = "0x" + "a" * 64
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=pk,
                    wallet_address=None,
                )
            )
            adapter = HyperliquidAdapter(testnet=True)

            assert adapter.normalize_symbol("BTCUSDT") == "BTC"

    def test_denormalize_symbol(self):
        """Test symbol denormalization."""
        pk = "0x" + "a" * 64
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=pk,
                    wallet_address=None,
                )
            )
            adapter = HyperliquidAdapter(testnet=True)

            assert adapter.denormalize_symbol("BTC") == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test adapter initialization."""
        pk = "0x" + "a" * 64
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=pk,
                    wallet_address=None,
                )
            )
            adapter = HyperliquidAdapter(testnet=True)

            with patch.object(adapter._rest, "initialize", new_callable=AsyncMock):
                await adapter.initialize()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test adapter close."""
        pk = "0x" + "a" * 64
        with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                exchange=MagicMock(
                    wallet_private_key=pk,
                    wallet_address=None,
                )
            )
            adapter = HyperliquidAdapter(testnet=True)

            with patch.object(adapter._rest, "close", new_callable=AsyncMock):
                # Mock the is_connected property
                adapter._websocket._state = ConnectionState.DISCONNECTED
                await adapter.close()
