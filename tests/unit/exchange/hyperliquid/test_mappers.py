"""
Tests for Hyperliquid data mappers.

Coverage targets: symbol translation, order/position mapping, balance parsing.
"""

import pytest
from datetime import datetime, UTC

from src.exchange.adapters.hyperliquid.mappers import (
    HyperliquidMapper,
    float_to_wire,
    _utc_now,
)
from src.exchange.models import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
    MarginMode,
    UnifiedOrder,
)


class TestFloatToWire:
    """Tests for float_to_wire conversion."""

    def test_integer_value(self):
        """Test integer values remove trailing zeros."""
        assert float_to_wire(75000.0) == "75000"
        assert float_to_wire(100.0) == "100"

    def test_decimal_value(self):
        """Test decimal values preserve precision."""
        assert float_to_wire(0.001) == "0.001"
        assert float_to_wire(50000.5) == "50000.5"

    def test_small_decimal(self):
        """Test small decimal values."""
        assert float_to_wire(0.00000001) == "0.00000001"

    def test_large_decimal(self):
        """Test large values."""
        assert float_to_wire(1000000.123456) == "1000000.123456"


class TestUtcNow:
    """Tests for _utc_now helper."""

    def test_returns_timezone_aware(self):
        """Test that _utc_now returns timezone-aware datetime."""
        dt = _utc_now()
        assert dt.tzinfo is not None
        assert dt.tzinfo == UTC


class TestSymbolTranslation:
    """Tests for symbol translation methods."""

    def test_to_hyperliquid_coin_usdt_suffix(self):
        """Test stripping USDT suffix."""
        assert HyperliquidMapper.to_hyperliquid_coin("BTCUSDT") == "BTC"
        assert HyperliquidMapper.to_hyperliquid_coin("ETHUSDT") == "ETH"
        assert HyperliquidMapper.to_hyperliquid_coin("SOLUSDT") == "SOL"

    def test_to_hyperliquid_coin_usd_suffix(self):
        """Test stripping USD suffix."""
        assert HyperliquidMapper.to_hyperliquid_coin("BTCUSD") == "BTC"
        assert HyperliquidMapper.to_hyperliquid_coin("ETHUSD") == "ETH"

    def test_to_hyperliquid_coin_no_suffix(self):
        """Test coin without suffix."""
        assert HyperliquidMapper.to_hyperliquid_coin("BTC") == "BTC"

    def test_to_hyperliquid_coin_from_cache(self):
        """Test symbol resolution from cache."""
        # Set up cache
        HyperliquidMapper.SYMBOL_TO_COIN["TESTUSDT"] = "TEST"
        assert HyperliquidMapper.to_hyperliquid_coin("TESTUSDT") == "TEST"
        # Clean up
        del HyperliquidMapper.SYMBOL_TO_COIN["TESTUSDT"]

    def test_from_hyperliquid_coin_basic(self):
        """Test adding USDT suffix."""
        assert HyperliquidMapper.from_hyperliquid_coin("BTC") == "BTCUSDT"
        assert HyperliquidMapper.from_hyperliquid_coin("ETH") == "ETHUSDT"

    def test_from_hyperliquid_coin_spot(self):
        """Test spot pair with @ format."""
        assert HyperliquidMapper.from_hyperliquid_coin("@107") == "SPOT@107"

    def test_from_hyperliquid_coin_from_cache(self):
        """Test coin resolution from cache."""
        HyperliquidMapper.COIN_TO_SYMBOL["CACHED"] = "CACHEDUSDT"
        assert HyperliquidMapper.from_hyperliquid_coin("CACHED") == "CACHEDUSDT"
        del HyperliquidMapper.COIN_TO_SYMBOL["CACHED"]


class TestAssetMetadata:
    """Tests for asset metadata handling."""

    def test_set_asset_meta(self):
        """Test setting asset metadata."""
        meta = [
            {"name": "BTC"},
            {"name": "ETH"},
            {"name": "SOL"},
        ]
        HyperliquidMapper.set_asset_meta(meta)

        assert "BTC" in HyperliquidMapper._asset_meta
        assert HyperliquidMapper._coin_to_index["BTC"] == 0
        assert HyperliquidMapper._coin_to_index["ETH"] == 1
        assert HyperliquidMapper._index_to_coin[0] == "BTC"

    def test_get_asset_index(self):
        """Test getting asset index."""
        meta = [{"name": "BTC"}, {"name": "ETH"}]
        HyperliquidMapper.set_asset_meta(meta)

        assert HyperliquidMapper.get_asset_index("BTCUSDT") == 0
        assert HyperliquidMapper.get_asset_index("ETHUSDT") == 1

    def test_get_asset_index_unknown(self):
        """Test getting index for unknown symbol."""
        HyperliquidMapper._coin_to_index.clear()
        with pytest.raises(ValueError, match="Unknown symbol"):
            HyperliquidMapper.get_asset_index("UNKNOWNUSDT")


class TestAssetClassification:
    """Tests for asset class methods."""

    def test_get_asset_class(self):
        """Test asset class lookup."""
        assert HyperliquidMapper.get_asset_class("BTCUSDT") == "crypto_major"
        assert HyperliquidMapper.get_asset_class("GOLDUSDT") == "traditional"
        assert HyperliquidMapper.get_asset_class("SOLUSDT") == "layer1"

    def test_get_asset_class_unknown(self):
        """Test unknown asset defaults to crypto."""
        assert HyperliquidMapper.get_asset_class("UNKNOWNUSDT") == "crypto"

    def test_get_sector(self):
        """Test sector lookup."""
        assert HyperliquidMapper.get_sector("BTCUSDT") == "crypto_major"
        assert HyperliquidMapper.get_sector("AAVEUSDT") == "defi"

    def test_get_medallion_assets(self):
        """Test medallion portfolio assets."""
        assets = HyperliquidMapper.get_medallion_assets()
        assert "BTC" in assets
        assert "ETH" in assets
        assert len(assets) > 10


class TestOrderSideTranslation:
    """Tests for order side translation."""

    def test_to_hyperliquid_side(self):
        """Test unified to Hyperliquid side."""
        assert HyperliquidMapper.to_hyperliquid_side(OrderSide.BUY) == "B"
        assert HyperliquidMapper.to_hyperliquid_side(OrderSide.SELL) == "A"

    def test_to_hyperliquid_is_buy(self):
        """Test unified to Hyperliquid is_buy boolean."""
        assert HyperliquidMapper.to_hyperliquid_is_buy(OrderSide.BUY) is True
        assert HyperliquidMapper.to_hyperliquid_is_buy(OrderSide.SELL) is False

    def test_from_hyperliquid_side(self):
        """Test Hyperliquid to unified side."""
        assert HyperliquidMapper.from_hyperliquid_side("B") == OrderSide.BUY
        assert HyperliquidMapper.from_hyperliquid_side("A") == OrderSide.SELL
        assert HyperliquidMapper.from_hyperliquid_side("b") == OrderSide.BUY


class TestOrderTypeTranslation:
    """Tests for order type translation."""

    def test_limit_order_gtc(self):
        """Test limit order with GTC."""
        result = HyperliquidMapper.to_hyperliquid_order_type(
            OrderType.LIMIT, TimeInForce.GTC
        )
        assert result == {"limit": {"tif": "Gtc"}}

    def test_limit_order_ioc(self):
        """Test limit order with IOC."""
        result = HyperliquidMapper.to_hyperliquid_order_type(
            OrderType.LIMIT, TimeInForce.IOC
        )
        assert result == {"limit": {"tif": "Ioc"}}

    def test_limit_order_post_only(self):
        """Test limit order with post-only."""
        result = HyperliquidMapper.to_hyperliquid_order_type(
            OrderType.LIMIT, TimeInForce.POST_ONLY
        )
        assert result == {"limit": {"tif": "Alo"}}

    def test_limit_order_fok(self):
        """Test limit order with FOK maps to IOC."""
        result = HyperliquidMapper.to_hyperliquid_order_type(
            OrderType.LIMIT, TimeInForce.FOK
        )
        assert result == {"limit": {"tif": "Ioc"}}

    def test_market_order(self):
        """Test market order."""
        result = HyperliquidMapper.to_hyperliquid_order_type(OrderType.MARKET)
        assert result == {"limit": {"tif": "Gtc"}}

    def test_stop_market_order(self):
        """Test stop market order."""
        result = HyperliquidMapper.to_hyperliquid_order_type(
            OrderType.STOP_MARKET, trigger_price=48000.0
        )
        assert "trigger" in result
        assert result["trigger"]["isMarket"] is True
        assert result["trigger"]["tpsl"] == "sl"
        assert result["trigger"]["triggerPx"] == "48000.0"

    def test_take_profit_market_order(self):
        """Test take profit market order."""
        result = HyperliquidMapper.to_hyperliquid_order_type(
            OrderType.TAKE_PROFIT_MARKET, trigger_price=55000.0
        )
        assert result["trigger"]["isMarket"] is True
        assert result["trigger"]["tpsl"] == "tp"

    def test_stop_limit_order(self):
        """Test stop limit order."""
        result = HyperliquidMapper.to_hyperliquid_order_type(
            OrderType.STOP_LIMIT, trigger_price=48000.0
        )
        assert result["trigger"]["isMarket"] is False
        assert result["trigger"]["tpsl"] == "sl"

    def test_take_profit_limit_order(self):
        """Test take profit limit order."""
        result = HyperliquidMapper.to_hyperliquid_order_type(
            OrderType.TAKE_PROFIT_LIMIT, trigger_price=55000.0
        )
        assert result["trigger"]["isMarket"] is False
        assert result["trigger"]["tpsl"] == "tp"


class TestOrderStatusTranslation:
    """Tests for order status translation."""

    def test_from_hyperliquid_order_status(self):
        """Test all status mappings."""
        assert HyperliquidMapper.from_hyperliquid_order_status("open") == OrderStatus.NEW
        assert HyperliquidMapper.from_hyperliquid_order_status("filled") == OrderStatus.FILLED
        assert HyperliquidMapper.from_hyperliquid_order_status("canceled") == OrderStatus.CANCELED
        assert HyperliquidMapper.from_hyperliquid_order_status("triggered") == OrderStatus.NEW
        assert HyperliquidMapper.from_hyperliquid_order_status("rejected") == OrderStatus.REJECTED
        assert HyperliquidMapper.from_hyperliquid_order_status("marginCanceled") == OrderStatus.CANCELED
        assert HyperliquidMapper.from_hyperliquid_order_status("selfTradeCanceled") == OrderStatus.CANCELED
        assert HyperliquidMapper.from_hyperliquid_order_status("reduceOnlyCanceled") == OrderStatus.CANCELED
        assert HyperliquidMapper.from_hyperliquid_order_status("liquidatedCanceled") == OrderStatus.CANCELED
        assert HyperliquidMapper.from_hyperliquid_order_status("unknown") == OrderStatus.NEW


class TestFromUnifiedOrder:
    """Tests for converting UnifiedOrder to wire format."""

    def setup_method(self):
        """Set up test fixtures."""
        HyperliquidMapper.set_asset_meta([{"name": "BTC"}, {"name": "ETH"}])

    def test_basic_limit_order(self):
        """Test basic limit order conversion."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
        )
        wire = HyperliquidMapper.from_unified_order(order)

        assert wire["a"] == 0  # BTC index
        assert wire["b"] is True  # is_buy
        assert wire["p"] == "50000"  # price
        assert wire["s"] == "0.1"  # size
        assert wire["r"] is False  # reduce_only
        assert "limit" in wire["t"]

    def test_market_order(self):
        """Test market order conversion."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.5,
            price=50000.0,
        )
        wire = HyperliquidMapper.from_unified_order(order)

        assert wire["b"] is False  # is_buy = False for sell
        assert wire["s"] == "0.5"

    def test_order_with_client_id(self):
        """Test order with client order ID."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
            client_order_id="abc123",
        )
        wire = HyperliquidMapper.from_unified_order(order)

        assert "c" in wire
        assert wire["c"].startswith("0x")

    def test_reduce_only_order(self):
        """Test reduce-only order."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
            reduce_only=True,
        )
        wire = HyperliquidMapper.from_unified_order(order)

        assert wire["r"] is True


class TestToUnifiedTicker:
    """Tests for ticker conversion."""

    def test_basic_ticker(self):
        """Test basic ticker conversion."""
        mids = {"BTC": "50000.5", "ETH": "3000.0"}
        ticker = HyperliquidMapper.to_unified_ticker(mids, "BTC")

        assert ticker.symbol == "BTCUSDT"
        assert ticker.last_price == 50000.5
        assert ticker.bid_price == 50000.5
        assert ticker.ask_price == 50000.5

    def test_missing_coin(self):
        """Test ticker for missing coin."""
        mids = {"BTC": "50000.0"}
        ticker = HyperliquidMapper.to_unified_ticker(mids, "UNKNOWN")

        assert ticker.last_price == 0.0


class TestToUnifiedOrderbook:
    """Tests for orderbook conversion."""

    def test_basic_orderbook(self):
        """Test basic orderbook conversion."""
        book_data = {
            "levels": [
                [{"px": "49990", "sz": "1.5"}, {"px": "49980", "sz": "2.0"}],
                [{"px": "50010", "sz": "1.0"}, {"px": "50020", "sz": "1.5"}],
            ]
        }
        orderbook = HyperliquidMapper.to_unified_orderbook(book_data, "BTCUSDT")

        assert orderbook.symbol == "BTCUSDT"
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.bids[0] == (49990.0, 1.5)
        assert orderbook.asks[0] == (50010.0, 1.0)

    def test_empty_orderbook(self):
        """Test empty orderbook."""
        book_data = {"levels": [[], []]}
        orderbook = HyperliquidMapper.to_unified_orderbook(book_data, "BTCUSDT")

        assert orderbook.bids == []
        assert orderbook.asks == []

    def test_missing_levels(self):
        """Test orderbook with missing levels."""
        book_data = {}
        orderbook = HyperliquidMapper.to_unified_orderbook(book_data, "BTCUSDT")

        assert orderbook.bids == []
        assert orderbook.asks == []


class TestToUnifiedCandle:
    """Tests for candle conversion."""

    def test_basic_candle(self):
        """Test basic candle conversion."""
        candle_data = {
            "t": 1706644800000,
            "o": "50000.0",
            "h": "50500.0",
            "l": "49500.0",
            "c": "50200.0",
            "v": "1000.5",
            "n": 5000,
        }
        candle = HyperliquidMapper.to_unified_candle(candle_data, "BTCUSDT", "1h")

        assert candle.symbol == "BTCUSDT"
        assert candle.open == 50000.0
        assert candle.high == 50500.0
        assert candle.low == 49500.0
        assert candle.close == 50200.0
        assert candle.volume == 1000.5
        assert candle.trades == 5000
        assert candle.interval == "1h"


class TestToUnifiedPosition:
    """Tests for position conversion."""

    def test_long_position(self):
        """Test long position conversion."""
        pos_data = {
            "position": {
                "coin": "BTC",
                "szi": "0.5",
                "entryPx": "50000.0",
                "unrealizedPnl": "500.0",
                "marginUsed": "1000.0",
                "liquidationPx": "45000.0",
                "leverage": {"type": "cross", "value": 10},
            }
        }
        position = HyperliquidMapper.to_unified_position(pos_data)

        assert position.symbol == "BTCUSDT"
        assert position.side == PositionSide.LONG
        assert position.quantity == 0.5
        assert position.entry_price == 50000.0
        assert position.unrealized_pnl == 500.0
        assert position.leverage == 10
        assert position.liquidation_price == 45000.0
        assert position.margin_mode == MarginMode.CROSS

    def test_short_position(self):
        """Test short position conversion."""
        pos_data = {
            "position": {
                "coin": "ETH",
                "szi": "-2.0",
                "entryPx": "3000.0",
                "unrealizedPnl": "-100.0",
                "leverage": {"type": "isolated", "value": 5},
            }
        }
        position = HyperliquidMapper.to_unified_position(pos_data)

        assert position.side == PositionSide.SHORT
        assert position.quantity == 2.0
        assert position.margin_mode == MarginMode.ISOLATED

    def test_position_no_liquidation_price(self):
        """Test position without liquidation price."""
        pos_data = {
            "position": {
                "coin": "BTC",
                "szi": "0.1",
                "entryPx": "50000.0",
                "leverage": {"value": 1},
            }
        }
        position = HyperliquidMapper.to_unified_position(pos_data)

        assert position.liquidation_price is None


class TestToUnifiedAccountBalance:
    """Tests for account balance conversion."""

    def test_basic_balance(self):
        """Test basic balance conversion."""
        spot_state = {
            "balances": [
                {"coin": "USDC", "total": "10000.0", "hold": "2000.0"},
                {"coin": "OTHER", "total": "500.0", "hold": "0.0"},
            ]
        }
        perps_state = {
            "marginSummary": {"totalNtlPos": "5000.0"}
        }
        balance = HyperliquidMapper.to_unified_account_balance(spot_state, perps_state)

        assert balance.total_balance == 10000.0
        assert balance.available_balance == 8000.0
        assert balance.margin_balance == 2000.0
        assert balance.unrealized_pnl == 5000.0
        assert balance.currency == "USDC"

    def test_balance_no_usdc(self):
        """Test balance without USDC."""
        spot_state = {"balances": []}
        balance = HyperliquidMapper.to_unified_account_balance(spot_state)

        assert balance.total_balance == 0.0
        assert balance.available_balance == 0.0

    def test_balance_no_perps_state(self):
        """Test balance without perps state."""
        spot_state = {
            "balances": [{"coin": "USDC", "total": "1000.0", "hold": "0.0"}]
        }
        balance = HyperliquidMapper.to_unified_account_balance(spot_state, None)

        assert balance.unrealized_pnl == 0.0


class TestToUnifiedOrderResult:
    """Tests for order result conversion."""

    def test_resting_order(self):
        """Test resting order result."""
        response = {
            "response": {
                "data": {
                    "statuses": [{"resting": {"oid": 12345}}]
                }
            }
        }
        result = HyperliquidMapper.to_unified_order_result(response)

        assert result.order_id == "12345"
        assert result.status == OrderStatus.NEW

    def test_filled_order(self):
        """Test filled order result."""
        response = {
            "response": {
                "data": {
                    "statuses": [{
                        "filled": {
                            "oid": 12345,
                            "totalSz": "0.5",
                            "avgPx": "50100.0",
                        }
                    }]
                }
            }
        }
        original = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.5,
            price=50000.0,
        )
        result = HyperliquidMapper.to_unified_order_result(response, original)

        assert result.order_id == "12345"
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 0.5
        assert result.average_price == 50100.0
        assert result.symbol == "BTCUSDT"

    def test_error_order(self):
        """Test error order result."""
        response = {
            "response": {
                "data": {
                    "statuses": [{"error": "Insufficient balance"}]
                }
            }
        }
        result = HyperliquidMapper.to_unified_order_result(response)

        assert result.status == OrderStatus.REJECTED

    def test_empty_statuses(self):
        """Test order result with empty statuses."""
        response = {"response": {"data": {"statuses": []}}}
        result = HyperliquidMapper.to_unified_order_result(response)

        assert result.order_id == ""
        assert result.status == OrderStatus.NEW


class TestToUnifiedTrade:
    """Tests for trade conversion."""

    def test_basic_trade(self):
        """Test basic trade conversion."""
        fill_data = {
            "coin": "BTC",
            "tid": "trade123",
            "px": "50000.0",
            "sz": "0.1",
            "side": "B",
            "time": 1706644800000,
            "crossed": False,
        }
        trade = HyperliquidMapper.to_unified_trade(fill_data)

        assert trade.symbol == "BTCUSDT"
        assert trade.trade_id == "trade123"
        assert trade.price == 50000.0
        assert trade.quantity == 0.1
        assert trade.side == OrderSide.BUY
        assert trade.is_maker is True

    def test_trade_with_oid(self):
        """Test trade using oid as fallback ID."""
        fill_data = {
            "coin": "ETH",
            "oid": "order456",
            "px": "3000.0",
            "sz": "1.0",
            "side": "A",
            "time": 1706644800000,
            "crossed": True,
        }
        trade = HyperliquidMapper.to_unified_trade(fill_data)

        assert trade.trade_id == "order456"
        assert trade.side == OrderSide.SELL
        assert trade.is_maker is False


class TestToSymbolInfo:
    """Tests for symbol info conversion."""

    def test_basic_symbol_info(self):
        """Test basic symbol info conversion."""
        asset_meta = {
            "name": "BTC",
            "szDecimals": 4,
            "maxLeverage": 100,
        }
        info = HyperliquidMapper.to_symbol_info(asset_meta, 0)

        assert info.symbol == "BTCUSDT"
        assert info.base_asset == "BTC"
        assert info.quote_asset == "USDC"
        assert info.quantity_precision == 4
        assert info.max_leverage == 100
        assert info.min_quantity == 0.0001

    def test_symbol_info_defaults(self):
        """Test symbol info with default values."""
        asset_meta = {"name": "NEW"}
        info = HyperliquidMapper.to_symbol_info(asset_meta, 5)

        assert info.max_leverage == 50  # Default
        assert info.quantity_precision == 3  # Default


class TestIntervalMapping:
    """Tests for interval mapping."""

    def test_valid_intervals(self):
        """Test valid interval pass-through."""
        assert HyperliquidMapper.to_hyperliquid_interval("1m") == "1m"
        assert HyperliquidMapper.to_hyperliquid_interval("5m") == "5m"
        assert HyperliquidMapper.to_hyperliquid_interval("1h") == "1h"
        assert HyperliquidMapper.to_hyperliquid_interval("4h") == "4h"
        assert HyperliquidMapper.to_hyperliquid_interval("1d") == "1d"

    def test_interval_aliases(self):
        """Test interval alias mapping."""
        assert HyperliquidMapper.to_hyperliquid_interval("1min") == "1m"
        assert HyperliquidMapper.to_hyperliquid_interval("5min") == "5m"
        assert HyperliquidMapper.to_hyperliquid_interval("1hour") == "1h"
        assert HyperliquidMapper.to_hyperliquid_interval("4hour") == "4h"
        assert HyperliquidMapper.to_hyperliquid_interval("1day") == "1d"
        assert HyperliquidMapper.to_hyperliquid_interval("1week") == "1w"

    def test_unknown_interval(self):
        """Test unknown interval defaults to 1h."""
        assert HyperliquidMapper.to_hyperliquid_interval("unknown") == "1h"


class TestCapabilities:
    """Tests for exchange capabilities."""

    def test_capabilities(self):
        """Test capabilities are correctly defined."""
        caps = HyperliquidMapper.CAPABILITIES

        assert caps.name == "hyperliquid"
        assert caps.supports_futures is True
        assert caps.supports_spot is True
        assert caps.supports_stop_orders is True
        assert caps.supports_take_profit is True
        assert caps.supports_reduce_only is True
        assert caps.supports_post_only is True
        assert caps.max_leverage == 50
        assert caps.websocket_available is True
