"""
Unit tests for WEEX mappers.
"""

from src.exchange.adapters.weex.mappers import WEEXMapper
from src.exchange.models import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
    UnifiedOrder,
)


class TestSymbolMapping:
    """Tests for symbol normalization."""

    def test_to_weex_symbol(self):
        """Test converting unified symbol to WEEX format."""
        assert WEEXMapper.to_weex_symbol("BTCUSDT") == "cmt_btcusdt"
        assert WEEXMapper.to_weex_symbol("ETHUSDT") == "cmt_ethusdt"
        assert WEEXMapper.to_weex_symbol("SOLUSDT") == "cmt_solusdt"

    def test_to_weex_symbol_case_handling(self):
        """Test case handling in symbol conversion."""
        assert WEEXMapper.to_weex_symbol("btcusdt") == "cmt_btcusdt"
        assert WEEXMapper.to_weex_symbol("BtcUsDt") == "cmt_btcusdt"

    def test_from_weex_symbol(self):
        """Test converting WEEX symbol to unified format."""
        assert WEEXMapper.from_weex_symbol("cmt_btcusdt") == "BTCUSDT"
        assert WEEXMapper.from_weex_symbol("cmt_ethusdt") == "ETHUSDT"

    def test_from_weex_symbol_without_prefix(self):
        """Test handling symbols without cmt_ prefix."""
        assert WEEXMapper.from_weex_symbol("btcusdt") == "BTCUSDT"

    def test_symbol_roundtrip(self):
        """Test that symbol conversion is reversible."""
        original = "BTCUSDT"
        weex = WEEXMapper.to_weex_symbol(original)
        unified = WEEXMapper.from_weex_symbol(weex)
        assert unified == original


class TestOrderSideMapping:
    """Tests for order side mapping."""

    def test_to_weex_order_side_buy(self):
        """Test mapping BUY side."""
        assert WEEXMapper.to_weex_order_side(OrderSide.BUY) == "buy"

    def test_to_weex_order_side_sell(self):
        """Test mapping SELL side."""
        assert WEEXMapper.to_weex_order_side(OrderSide.SELL) == "sell"

    def test_from_weex_order_side_buy(self):
        """Test mapping buy from WEEX."""
        assert WEEXMapper.from_weex_order_side("buy") == OrderSide.BUY

    def test_from_weex_order_side_sell(self):
        """Test mapping sell from WEEX."""
        assert WEEXMapper.from_weex_order_side("sell") == OrderSide.SELL


class TestOrderTypeMapping:
    """Tests for order type mapping."""

    def test_to_weex_order_type_market(self):
        """Test mapping MARKET order type."""
        assert WEEXMapper.to_weex_order_type(OrderType.MARKET) == "market"

    def test_to_weex_order_type_limit(self):
        """Test mapping LIMIT order type."""
        assert WEEXMapper.to_weex_order_type(OrderType.LIMIT) == "limit"

    def test_from_weex_order_type(self):
        """Test mapping order types from WEEX."""
        assert WEEXMapper.from_weex_order_type("market") == OrderType.MARKET
        assert WEEXMapper.from_weex_order_type("limit") == OrderType.LIMIT


class TestPositionSideMapping:
    """Tests for position side mapping."""

    def test_to_weex_position_side(self):
        """Test mapping position sides to WEEX."""
        assert WEEXMapper.to_weex_position_side(PositionSide.LONG) == "long"
        assert WEEXMapper.to_weex_position_side(PositionSide.SHORT) == "short"

    def test_from_weex_position_side(self):
        """Test mapping position sides from WEEX."""
        assert WEEXMapper.from_weex_position_side("long") == PositionSide.LONG
        assert WEEXMapper.from_weex_position_side("short") == PositionSide.SHORT


class TestOrderStatusMapping:
    """Tests for order status mapping."""

    def test_from_weex_status_new(self):
        """Test mapping NEW status."""
        assert WEEXMapper.from_weex_order_status("new") == OrderStatus.NEW
        assert WEEXMapper.from_weex_order_status("open") == OrderStatus.NEW

    def test_from_weex_status_filled(self):
        """Test mapping filled status."""
        assert WEEXMapper.from_weex_order_status("filled") == OrderStatus.FILLED
        assert WEEXMapper.from_weex_order_status("full-fill") == OrderStatus.FILLED

    def test_from_weex_status_cancelled(self):
        """Test mapping cancelled status."""
        assert WEEXMapper.from_weex_order_status("cancelled") == OrderStatus.CANCELED
        assert WEEXMapper.from_weex_order_status("canceled") == OrderStatus.CANCELED

    def test_from_weex_status_partial(self):
        """Test mapping partial fill status."""
        assert (
            WEEXMapper.from_weex_order_status("partially_filled")
            == OrderStatus.PARTIALLY_FILLED
        )
        assert (
            WEEXMapper.from_weex_order_status("partial-fill")
            == OrderStatus.PARTIALLY_FILLED
        )


class TestTimeInForceMapping:
    """Tests for time in force mapping."""

    def test_to_weex_time_in_force(self):
        """Test mapping time in force to WEEX."""
        assert WEEXMapper.to_weex_time_in_force(TimeInForce.GTC) == "normal"
        assert WEEXMapper.to_weex_time_in_force(TimeInForce.IOC) == "ioc"
        assert WEEXMapper.to_weex_time_in_force(TimeInForce.FOK) == "fok"
        assert WEEXMapper.to_weex_time_in_force(TimeInForce.POST_ONLY) == "post_only"


class TestOrderRequestMapping:
    """Tests for order request mapping."""

    def test_from_unified_order_market(self):
        """Test mapping market order to WEEX format."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            position_side=PositionSide.LONG,
        )
        weex_order = WEEXMapper.from_unified_order(order)

        assert weex_order["symbol"] == "cmt_btcusdt"
        assert weex_order["side"] == "buy"
        assert weex_order["orderType"] == "market"
        assert weex_order["size"] == "0.01"

    def test_from_unified_order_limit(self):
        """Test mapping limit order to WEEX format."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.05,
            price=50000.0,
            position_side=PositionSide.SHORT,
        )
        weex_order = WEEXMapper.from_unified_order(order)

        assert weex_order["symbol"] == "cmt_btcusdt"
        assert weex_order["side"] == "sell"
        assert weex_order["orderType"] == "limit"
        assert weex_order["size"] == "0.05"
        assert weex_order["price"] == "50000.0"

    def test_from_unified_order_with_client_id(self):
        """Test mapping order with client order ID."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            client_order_id="my_order_123",
        )
        weex_order = WEEXMapper.from_unified_order(order)

        assert weex_order["clientOid"] == "my_order_123"


class TestOrderResultMapping:
    """Tests for order result mapping."""

    def test_to_unified_order_result(self):
        """Test mapping WEEX order response to unified format."""
        weex_response = {
            "orderId": "123456789",
            "clientOid": "client_123",
            "symbol": "cmt_btcusdt",
            "side": "buy",
            "orderType": "market",
            "size": "0.01",
            "price": "50000.0",
            "status": "filled",
            "filledQty": "0.01",
            "avgPx": "50100.0",
            "fee": "0.5",
            "feeCcy": "USDT",
        }

        result = WEEXMapper.to_unified_order_result(weex_response)

        assert result.order_id == "123456789"
        assert result.client_order_id == "client_123"
        assert result.symbol == "BTCUSDT"
        assert result.side == OrderSide.BUY
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 0.01
        assert result.average_price == 50100.0


class TestPositionMapping:
    """Tests for position mapping."""

    def test_to_unified_position(self):
        """Test mapping WEEX position to unified format."""
        weex_position = {
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

        position = WEEXMapper.to_unified_position(weex_position)

        assert position.symbol == "BTCUSDT"
        assert position.side == PositionSide.LONG
        assert position.quantity == 0.1
        assert position.entry_price == 50000.0
        assert position.mark_price == 51000.0
        assert position.unrealized_pnl == 100.0
        assert position.leverage == 10


class TestBalanceMapping:
    """Tests for balance mapping."""

    def test_to_unified_account_balance(self):
        """Test mapping WEEX balance to unified format."""
        weex_balance = {
            "usdtEquity": "10000.0",
            "available": "8000.0",
            "crossMaxAvailable": "2000.0",
            "unrealizedPL": "500.0",
        }

        balance = WEEXMapper.to_unified_account_balance(weex_balance)

        assert balance.total_balance == 10000.0
        assert balance.available_balance == 8000.0
        assert balance.margin_balance == 2000.0
        assert balance.unrealized_pnl == 500.0


class TestTickerMapping:
    """Tests for ticker mapping."""

    def test_to_unified_ticker(self):
        """Test mapping WEEX ticker to unified format."""
        weex_ticker = {
            "last": "50000.0",
            "bestBid": "49990.0",
            "bestAsk": "50010.0",
            "bidSz": "1.0",
            "askSz": "1.5",
            "baseVolume": "1000.0",
            "quoteVolume": "50000000.0",
            "high24h": "51000.0",
            "low24h": "49000.0",
            "priceChange": "500.0",
            "priceChangePercent": "1.0",
        }

        ticker = WEEXMapper.to_unified_ticker(weex_ticker, "BTCUSDT")

        assert ticker.symbol == "BTCUSDT"
        assert ticker.last_price == 50000.0
        assert ticker.bid_price == 49990.0
        assert ticker.ask_price == 50010.0


class TestOrderbookMapping:
    """Tests for orderbook mapping."""

    def test_to_unified_orderbook(self):
        """Test mapping WEEX orderbook to unified format."""
        weex_orderbook = {
            "bids": [["50000.0", "1.0"], ["49990.0", "2.0"]],
            "asks": [["50010.0", "0.5"], ["50020.0", "1.0"]],
            "ts": "1706644800000",
        }

        # Note: argument order is (orderbook_data, symbol)
        orderbook = WEEXMapper.to_unified_orderbook(weex_orderbook, "BTCUSDT")

        assert orderbook.symbol == "BTCUSDT"
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.bids[0] == (50000.0, 1.0)
        assert orderbook.asks[0] == (50010.0, 0.5)


class TestCandleMapping:
    """Tests for candle mapping."""

    def test_to_unified_candle_list_format(self):
        """Test mapping WEEX candle from list format."""
        weex_candle = [
            "1706644800000",  # timestamp
            "50000.0",  # open
            "50100.0",  # high
            "49900.0",  # low
            "50050.0",  # close
            "100.0",  # volume
            "5000000.0",  # quote volume
        ]

        # Note: argument order is (candle_data, symbol, interval)
        candle = WEEXMapper.to_unified_candle(weex_candle, "BTCUSDT", "1m")

        assert candle.symbol == "BTCUSDT"
        assert candle.open == 50000.0
        assert candle.high == 50100.0
        assert candle.low == 49900.0
        assert candle.close == 50050.0
        assert candle.volume == 100.0
        assert candle.interval == "1m"
