"""
Additional coverage tests for WEEX adapter.

These tests focus on edge cases, error handling, and untested paths
to improve overall coverage.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exchange.adapters.weex.adapter import WEEXRESTClient
from src.exchange.adapters.weex.mappers import WEEXMapper, _utc_now
from src.exchange.exceptions import (
    AuthenticationError,
    ExchangeError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderNotFoundError,
    RateLimitError,
)
from src.exchange.models import (
    MarginMode,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
    UnifiedOrder,
)


class TestWEEXMapperEdgeCases:
    """Tests for edge cases in WEEX mapper."""

    def test_from_weex_order_side_uppercase(self):
        """Test handling uppercase side."""
        assert WEEXMapper.from_weex_order_side("BUY") == OrderSide.BUY
        assert WEEXMapper.from_weex_order_side("SELL") == OrderSide.SELL

    def test_to_weex_order_type_stop_orders(self):
        """Test stop order type mappings."""
        assert WEEXMapper.to_weex_order_type(OrderType.STOP_MARKET) == "market"
        assert WEEXMapper.to_weex_order_type(OrderType.STOP_LIMIT) == "limit"
        assert WEEXMapper.to_weex_order_type(OrderType.TAKE_PROFIT_MARKET) == "market"
        assert WEEXMapper.to_weex_order_type(OrderType.TAKE_PROFIT_LIMIT) == "limit"

    def test_to_weex_position_side_both(self):
        """Test BOTH position side."""
        assert WEEXMapper.to_weex_position_side(PositionSide.BOTH) == "both"

    def test_from_weex_position_side_uppercase(self):
        """Test uppercase position side."""
        assert WEEXMapper.from_weex_position_side("LONG") == PositionSide.LONG
        assert WEEXMapper.from_weex_position_side("SHORT") == PositionSide.SHORT

    def test_from_weex_order_status_all_statuses(self):
        """Test all status mappings."""
        assert WEEXMapper.from_weex_order_status("submitted") == OrderStatus.NEW
        assert WEEXMapper.from_weex_order_status("pending") == OrderStatus.PENDING
        assert WEEXMapper.from_weex_order_status("rejected") == OrderStatus.REJECTED
        assert WEEXMapper.from_weex_order_status("expired") == OrderStatus.EXPIRED
        assert WEEXMapper.from_weex_order_status("unknown") == OrderStatus.NEW

    def test_from_unified_order_with_all_options(self):
        """Test order with all options set."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            position_side=PositionSide.SHORT,
            client_order_id="test123",
            reduce_only=True,
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
            time_in_force=TimeInForce.IOC,
        )
        result = WEEXMapper.from_unified_order(order)

        assert result["symbol"] == "cmt_btcusdt"
        assert result["side"] == "sell"
        assert result["orderType"] == "limit"
        assert result["size"] == "1.0"
        assert result["price"] == "50000.0"
        assert result["holdSide"] == "short"
        assert result["clientOid"] == "test123"
        assert result["reduceOnly"] == "true"
        assert result["presetStopLossPrice"] == "48000.0"
        assert result["presetTakeProfitPrice"] == "55000.0"
        assert result["timeInForceValue"] == "ioc"

    def test_to_unified_order_result_minimal(self):
        """Test order result with minimal fields."""
        response = {"orderId": "123"}
        result = WEEXMapper.to_unified_order_result(response)

        assert result.order_id == "123"
        assert result.symbol == ""
        assert result.side == OrderSide.BUY
        assert result.order_type == OrderType.MARKET

    def test_to_unified_order_result_with_original_order(self):
        """Test order result using original order for defaults."""
        original = UnifiedOrder(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=2.0,
            price=3000.0,
        )
        response = {"orderId": "456", "filledQty": "1.5"}
        result = WEEXMapper.to_unified_order_result(response, original)

        assert result.order_id == "456"
        assert result.symbol == "ETHUSDT"
        assert result.side == OrderSide.SELL
        assert result.order_type == OrderType.LIMIT
        assert result.quantity == 2.0
        assert result.price == 3000.0

    def test_to_unified_position_isolated_mode(self):
        """Test position with isolated margin mode."""
        pos_data = {
            "symbol": "cmt_btcusdt",
            "holdSide": "long",
            "total": "0.5",
            "averageOpenPrice": "40000.0",
            "markPrice": "41000.0",
            "unrealizedPL": "500.0",
            "leverage": "20",
            "marginMode": "isolated",
            "margin": "1000.0",
        }
        result = WEEXMapper.to_unified_position(pos_data)

        assert result.margin_mode == MarginMode.ISOLATED
        assert result.leverage == 20

    def test_to_unified_ticker_with_alternative_fields(self):
        """Test ticker with alternative field names."""
        ticker_data = {
            "last": "50000.0",
            "bestBid": "49990.0",
            "bestAsk": "50010.0",
            "bidSz": "1.0",
            "askSz": "1.5",
            "volume24h": "1000.0",
            "quoteVolume": "50000000.0",
            "high24h": "51000.0",
            "low24h": "49000.0",
            "open24h": "49500.0",
        }
        result = WEEXMapper.to_unified_ticker(ticker_data, "BTCUSDT")

        assert result.last_price == 50000.0
        assert result.bid_price == 49990.0
        assert result.ask_price == 50010.0

    def test_to_unified_candle_dict_format(self):
        """Test candle from dict format."""
        candle_data = {
            "ts": "1706644800000",
            "open": "50000.0",
            "high": "50100.0",
            "low": "49900.0",
            "close": "50050.0",
            "baseVolume": "100.0",
            "quoteVolume": "5000000.0",
            "trades": "500",
        }
        result = WEEXMapper.to_unified_candle(candle_data, "BTCUSDT", "5m")

        assert result.symbol == "BTCUSDT"
        assert result.open == 50000.0
        assert result.interval == "5m"
        assert result.trades == 500

    def test_to_unified_trade(self):
        """Test trade mapping."""
        trade_data = {
            "tradeId": "trade123",
            "price": "50000.0",
            "size": "0.1",
            "side": "buy",
            "ts": "1706644800000",
            "isMaker": True,
        }
        result = WEEXMapper.to_unified_trade(trade_data, "BTCUSDT")

        assert result.trade_id == "trade123"
        assert result.price == 50000.0
        assert result.quantity == 0.1
        assert result.side == OrderSide.BUY
        assert result.is_maker is True

    def test_to_symbol_info(self):
        """Test symbol info mapping."""
        contract_data = {
            "symbol": "cmt_btcusdt",
            "baseCoin": "BTC",
            "quoteCoin": "USDT",
            "pricePlace": "2",
            "volumePlace": "4",
            "minTradeNum": "0.001",
            "maxTradeNum": "1000",
            "minTradeAmount": "5",
            "priceEndStep": "0.5",
            "sizeMultiplier": "0.001",
            "minLever": "1",
            "maxLever": "100",
            "makerFeeRate": "0.0001",
            "takerFeeRate": "0.0005",
        }
        result = WEEXMapper.to_symbol_info(contract_data)

        assert result.symbol == "BTCUSDT"
        assert result.base_asset == "BTC"
        assert result.quote_asset == "USDT"
        assert result.price_precision == 2
        assert result.quantity_precision == 4
        assert result.max_leverage == 100

    def test_interval_mapping(self):
        """Test interval conversions."""
        assert WEEXMapper.to_weex_interval("1h") == "1H"
        assert WEEXMapper.to_weex_interval("4h") == "4H"
        assert WEEXMapper.to_weex_interval("1d") == "1D"
        assert WEEXMapper.to_weex_interval("1w") == "1W"
        assert WEEXMapper.to_weex_interval("unknown") == "1m"

        assert WEEXMapper.from_weex_interval("1H") == "1h"
        assert WEEXMapper.from_weex_interval("4H") == "4h"
        assert WEEXMapper.from_weex_interval("1D") == "1d"
        assert WEEXMapper.from_weex_interval("1W") == "1w"
        assert WEEXMapper.from_weex_interval("unknown") == "1m"

    def test_utc_now_returns_timezone_aware(self):
        """Test that _utc_now returns timezone-aware datetime."""
        dt = _utc_now()
        assert dt.tzinfo is not None


class TestWEEXRESTClientErrorHandling:
    """Tests for REST client error handling."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        settings.exchange.rest_url = "https://api.weex.com"
        settings.exchange.rate_limit_requests = 10
        return settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create a REST client for testing."""
        with patch("src.exchange.adapters.weex.adapter.get_settings") as mock:
            mock.return_value = mock_settings
            client = WEEXRESTClient()
            return client

    @pytest.mark.asyncio
    async def test_handle_error_code_auth_errors(self, client):
        """Test authentication error code handling."""
        with pytest.raises(AuthenticationError):
            client._handle_error_code("40001", "Invalid API key")

        with pytest.raises(AuthenticationError):
            client._handle_error_code("40002", "Invalid signature")

        with pytest.raises(AuthenticationError):
            client._handle_error_code("40003", "Invalid timestamp")

    @pytest.mark.asyncio
    async def test_handle_error_code_rate_limit(self, client):
        """Test rate limit error code handling."""
        with pytest.raises(RateLimitError):
            client._handle_error_code("40007", "Rate limit exceeded")

        with pytest.raises(RateLimitError):
            client._handle_error_code("40008", "Too many requests")

    @pytest.mark.asyncio
    async def test_handle_error_code_insufficient_balance(self, client):
        """Test insufficient balance error code handling."""
        with pytest.raises(InsufficientBalanceError):
            client._handle_error_code("40009", "Insufficient margin")

        with pytest.raises(InsufficientBalanceError):
            client._handle_error_code("40010", "Insufficient balance")

    @pytest.mark.asyncio
    async def test_handle_error_code_invalid_order(self, client):
        """Test invalid order error code handling."""
        with pytest.raises(InvalidOrderError):
            client._handle_error_code("40011", "Invalid quantity")

        with pytest.raises(InvalidOrderError):
            client._handle_error_code("40012", "Invalid price")

        with pytest.raises(InvalidOrderError):
            client._handle_error_code("40013", "Order validation failed")

    @pytest.mark.asyncio
    async def test_handle_error_code_order_not_found(self, client):
        """Test order not found error code handling."""
        with pytest.raises(OrderNotFoundError):
            client._handle_error_code("40014", "Order not found")

    @pytest.mark.asyncio
    async def test_handle_error_code_generic(self, client):
        """Test generic error code handling."""
        with pytest.raises(ExchangeError):
            client._handle_error_code("99999", "Unknown error")

    @pytest.mark.asyncio
    async def test_get_funding_rate(self, client):
        """Test getting funding rate."""
        mock_response = {
            "symbol": "cmt_btcusdt",
            "fundingRate": "0.0001",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            rate = await client.get_funding_rate("BTCUSDT")

            assert rate == 0.0001

    @pytest.mark.asyncio
    async def test_get_recent_trades(self, client):
        """Test getting recent trades."""
        mock_response = [
            {
                "tradeId": "1",
                "price": "50000.0",
                "size": "0.1",
                "side": "buy",
                "ts": "1706644800000",
            },
            {
                "tradeId": "2",
                "price": "50010.0",
                "size": "0.2",
                "side": "sell",
                "ts": "1706644801000",
            },
        ]

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            trades = await client.get_recent_trades("BTCUSDT", limit=10)

            assert len(trades) == 2
            assert trades[0].trade_id == "1"
            assert trades[1].side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_get_all_symbols(self, client):
        """Test getting all symbols."""
        mock_response = [
            {
                "symbol": "cmt_btcusdt",
                "baseCoin": "BTC",
                "quoteCoin": "USDT",
                "pricePlace": "2",
                "volumePlace": "4",
            },
            {
                "symbol": "cmt_ethusdt",
                "baseCoin": "ETH",
                "quoteCoin": "USDT",
                "pricePlace": "2",
                "volumePlace": "4",
            },
        ]

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            symbols = await client.get_all_symbols()

            assert len(symbols) == 2
            assert symbols[0].symbol == "BTCUSDT"
            assert symbols[1].symbol == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_get_symbol_info(self, client):
        """Test getting symbol info."""
        # Response is a list of contracts
        mock_response = [
            {
                "symbol": "cmt_btcusdt",
                "baseCoin": "BTC",
                "quoteCoin": "USDT",
                "pricePlace": "2",
                "volumePlace": "4",
            },
        ]

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            info = await client.get_symbol_info("BTCUSDT")

            assert info.symbol == "BTCUSDT"
            assert info.base_asset == "BTC"

    @pytest.mark.asyncio
    async def test_get_order(self, client):
        """Test getting a single order."""
        mock_response = {
            "orderId": "123",
            "symbol": "cmt_btcusdt",
            "side": "buy",
            "orderType": "limit",
            "size": "0.1",
            "price": "50000.0",
            "status": "filled",
            "filledQty": "0.1",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            order = await client.get_order("BTCUSDT", "123")

            assert order.order_id == "123"
            assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_open_orders(self, client):
        """Test getting open orders."""
        mock_response = [
            {
                "orderId": "123",
                "symbol": "cmt_btcusdt",
                "side": "buy",
                "orderType": "limit",
                "size": "0.1",
                "price": "50000.0",
                "status": "open",
            },
        ]

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            orders = await client.get_open_orders("BTCUSDT")

            assert len(orders) == 1
            assert orders[0].order_id == "123"

    @pytest.mark.asyncio
    async def test_get_open_orders_all_symbols(self, client):
        """Test getting open orders for all symbols."""
        mock_response = [
            {
                "orderId": "123",
                "symbol": "cmt_btcusdt",
                "side": "buy",
                "orderType": "limit",
                "size": "0.1",
                "status": "open",
            },
            {
                "orderId": "456",
                "symbol": "cmt_ethusdt",
                "side": "sell",
                "orderType": "market",
                "size": "1.0",
                "status": "open",
            },
        ]

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            orders = await client.get_open_orders()

            assert len(orders) == 2

    @pytest.mark.asyncio
    async def test_place_stop_loss(self, client):
        """Test placing stop loss order."""
        mock_response = {
            "orderId": "sl123",
            "symbol": "cmt_btcusdt",
            "planType": "loss_plan",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            result = await client.place_stop_loss(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                trigger_price=48000.0,
                size=0.1,
            )

            assert result.order_id == "sl123"

    @pytest.mark.asyncio
    async def test_place_take_profit(self, client):
        """Test placing take profit order."""
        mock_response = {
            "orderId": "tp123",
            "symbol": "cmt_btcusdt",
            "planType": "profit_plan",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            result = await client.place_take_profit(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                trigger_price=55000.0,
                size=0.1,
            )

            assert result.order_id == "tp123"

    @pytest.mark.asyncio
    async def test_cancel_conditional_order(self, client):
        """Test canceling conditional order."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"orderId": "plan123"}
            await client.initialize()

            result = await client.cancel_conditional_order("BTCUSDT", "plan123")

            assert result is True

    @pytest.mark.asyncio
    async def test_get_leverage(self, client):
        """Test getting leverage."""
        mock_response = {
            "symbol": "cmt_btcusdt",
            "crossMarginLeverage": "10",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.initialize()

            leverage = await client.get_leverage("BTCUSDT")

            assert leverage == 10

    @pytest.mark.asyncio
    async def test_client_properties(self, client):
        """Test client properties."""
        assert client.exchange_name == "weex"
        caps = client.capabilities
        assert caps.name == "weex"
        assert caps.supports_futures is True
        assert caps.max_leverage == 125

    @pytest.mark.asyncio
    async def test_close_without_session(self, client):
        """Test closing client without initialized session."""
        # Should not raise
        await client.close()

    @pytest.mark.asyncio
    async def test_close_with_session(self, client):
        """Test closing client with initialized session."""
        await client.initialize()
        await client.close()
        assert client._session is None


class TestWEEXRESTClientRequestHandling:
    """Tests for request handling edge cases."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.exchange.api_key = "test_key"
        settings.exchange.api_secret = "test_secret"
        settings.exchange.api_passphrase = "test_pass"
        settings.exchange.rest_url = "https://api.weex.com"
        settings.exchange.rate_limit_requests = 10
        return settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create a REST client for testing."""
        with patch("src.exchange.adapters.weex.adapter.get_settings") as mock:
            mock.return_value = mock_settings
            client = WEEXRESTClient()
            return client

    @pytest.mark.asyncio
    async def test_request_without_initialization(self, client):
        """Test that request auto-initializes."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value=json.dumps({"code": "00000", "data": {"test": True}})
        )
        mock_response.headers = {}

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Don't call initialize, let _request do it
            result = await client._request("GET", "/test")
            assert result == {"test": True}
