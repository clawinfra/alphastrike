"""
Additional coverage tests for exchange models.
"""

from datetime import UTC, datetime

from src.exchange.models import (
    ExchangeCapabilities,
    MarginMode,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    SymbolInfo,
    TimeInForce,
    UnifiedAccountBalance,
    UnifiedCandle,
    UnifiedOrder,
    UnifiedOrderbook,
    UnifiedOrderResult,
    UnifiedPosition,
    UnifiedTicker,
    UnifiedTrade,
)


class TestTimeInForce:
    """Tests for TimeInForce enum."""

    def test_all_values(self):
        """Test all time in force values."""
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"
        assert TimeInForce.POST_ONLY.value == "POST_ONLY"


class TestMarginMode:
    """Tests for MarginMode enum."""

    def test_all_values(self):
        """Test all margin mode values."""
        assert MarginMode.CROSS.value == "CROSS"
        assert MarginMode.ISOLATED.value == "ISOLATED"


class TestUnifiedOrderEdgeCases:
    """Test edge cases for UnifiedOrder."""

    def test_order_with_position_side_none(self):
        """Test order without position side."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )
        assert order.position_side is None

    def test_order_all_fields(self):
        """Test order with all optional fields."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            position_side=PositionSide.LONG,
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
            client_order_id="client123",
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
        )
        assert order.time_in_force == TimeInForce.IOC
        assert order.reduce_only is True


class TestUnifiedOrderResultEdgeCases:
    """Test edge cases for UnifiedOrderResult."""

    def test_is_active_for_new(self):
        """Test is_active for new order."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status=OrderStatus.NEW,
        )
        assert result.is_active is True

    def test_is_active_for_partially_filled(self):
        """Test is_active for partially filled order."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=0.5,
        )
        assert result.is_active is True

    def test_is_active_for_filled(self):
        """Test is_active for filled order."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=None,
            status=OrderStatus.FILLED,
            filled_quantity=1.0,
        )
        assert result.is_active is False

    def test_is_active_for_canceled(self):
        """Test is_active for canceled order."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=None,
            status=OrderStatus.CANCELED,
        )
        assert result.is_active is False

    def test_is_closed_for_rejected(self):
        """Test is_closed for rejected order."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=None,
            status=OrderStatus.REJECTED,
        )
        assert result.is_closed is True
        assert result.is_active is False

    def test_is_filled_property(self):
        """Test is_filled property."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=None,
            status=OrderStatus.FILLED,
        )
        assert result.is_filled is True


class TestUnifiedPositionEdgeCases:
    """Test edge cases for UnifiedPosition."""

    def test_position_key_returns_tuple(self):
        """Test position key returns tuple."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=1.0,
            entry_price=50000.0,
            mark_price=49000.0,
            unrealized_pnl=1000.0,
            leverage=10,
        )
        # position_key returns a tuple (symbol, side)
        assert position.position_key == ("BTCUSDT", PositionSide.SHORT)

    def test_notional_value_short(self):
        """Test notional value for short position."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=2.0,
            entry_price=50000.0,
            mark_price=49000.0,
            unrealized_pnl=2000.0,
            leverage=10,
        )
        assert position.notional_value == 98000.0

    def test_pnl_percentage_negative(self):
        """Test PnL percentage for losing position."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=50000.0,
            mark_price=45000.0,
            unrealized_pnl=-5000.0,
            leverage=10,
        )
        assert position.pnl_percentage == -10.0


class TestUnifiedAccountBalanceEdgeCases:
    """Test edge cases for UnifiedAccountBalance."""

    def test_balance_with_zero_margin(self):
        """Test balance with zero margin."""
        balance = UnifiedAccountBalance(
            total_balance=10000.0,
            available_balance=10000.0,
            margin_balance=0.0,
            unrealized_pnl=0.0,
        )
        assert balance.margin_ratio == 0.0


class TestUnifiedTickerEdgeCases:
    """Test edge cases for UnifiedTicker."""

    def test_spread_with_zero_bid(self):
        """Test spread calculation with zero bid."""
        ticker = UnifiedTicker(
            symbol="BTCUSDT",
            last_price=50000.0,
            bid_price=0.0,
            ask_price=50010.0,
            bid_quantity=0.0,
            ask_quantity=1.0,
            volume_24h=1000.0,
            quote_volume_24h=50000000.0,
            high_24h=51000.0,
            low_24h=49000.0,
            price_change_24h=500.0,
            price_change_pct_24h=1.0,
        )
        # spread formula: if bid_price <= 0: return 0.0
        assert ticker.spread == 0.0

    def test_mid_price_calculation(self):
        """Test mid price calculation."""
        ticker = UnifiedTicker(
            symbol="BTCUSDT",
            last_price=50000.0,
            bid_price=49990.0,
            ask_price=50010.0,
            bid_quantity=1.0,
            ask_quantity=1.0,
            volume_24h=1000.0,
            quote_volume_24h=50000000.0,
            high_24h=51000.0,
            low_24h=49000.0,
            price_change_24h=500.0,
            price_change_pct_24h=1.0,
        )
        assert ticker.mid_price == 50000.0


class TestUnifiedOrderbookEdgeCases:
    """Test edge cases for UnifiedOrderbook."""

    def test_spread_with_no_bids(self):
        """Test spread with no bids."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[],
            asks=[(50010.0, 1.0)],
        )
        # When best_bid is 0, spread returns 0.0
        assert orderbook.spread == 0.0

    def test_spread_with_no_asks(self):
        """Test spread with no asks."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0)],
            asks=[],
        )
        # When best_ask is 0, spread returns 0.0
        assert orderbook.spread == 0.0

    def test_total_bid_depth(self):
        """Test total bid depth."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0), (49990.0, 2.0), (49980.0, 3.0)],
            asks=[(50010.0, 0.5)],
        )
        assert orderbook.total_bid_depth() == 6.0

    def test_total_ask_depth(self):
        """Test total ask depth."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0)],
            asks=[(50010.0, 0.5), (50020.0, 1.5), (50030.0, 2.0)],
        )
        assert orderbook.total_ask_depth() == 4.0

    def test_total_depth_with_levels(self):
        """Test total depth with limited levels."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0), (49990.0, 2.0), (49980.0, 3.0)],
            asks=[(50010.0, 0.5), (50020.0, 1.5)],
        )
        assert orderbook.total_bid_depth(levels=2) == 3.0
        assert orderbook.total_ask_depth(levels=1) == 0.5

    def test_mid_price_no_bids(self):
        """Test mid price with no bids."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[],
            asks=[(50010.0, 1.0)],
        )
        # mid_price returns 0.0 or best_bid or best_ask when one side is empty
        assert orderbook.mid_price == 0.0 or orderbook.mid_price == 50010.0

    def test_best_bid_qty(self):
        """Test best bid quantity."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 2.5), (49990.0, 1.0)],
            asks=[(50010.0, 1.0)],
        )
        assert orderbook.best_bid_qty == 2.5

    def test_best_ask_qty(self):
        """Test best ask quantity."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0)],
            asks=[(50010.0, 3.0), (50020.0, 1.0)],
        )
        assert orderbook.best_ask_qty == 3.0


class TestUnifiedCandleEdgeCases:
    """Test edge cases for UnifiedCandle."""

    def test_bearish_candle(self):
        """Test bearish candle detection."""
        candle = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            open=50100.0,
            high=50200.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
            interval="1m",
        )
        assert candle.is_bullish is False

    def test_body_size_bearish(self):
        """Test body size for bearish candle."""
        candle = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            open=50100.0,
            high=50200.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
            interval="1m",
        )
        assert candle.body_size == 100.0


class TestUnifiedTradeEdgeCases:
    """Test edge cases for UnifiedTrade."""

    def test_trade_notional(self):
        """Test trade notional calculation."""
        trade = UnifiedTrade(
            symbol="BTCUSDT",
            trade_id="123",
            price=50000.0,
            quantity=2.5,
            side=OrderSide.BUY,
            timestamp=datetime.now(UTC),
        )
        assert trade.notional == 125000.0


class TestSymbolInfoEdgeCases:
    """Test edge cases for SymbolInfo."""

    def test_round_price(self):
        """Test rounding price."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.001,
            max_quantity=1000.0,
            min_notional=5.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        # round(50000.125, 2) = 50000.12 (banker's rounding to even)
        assert info.round_price(50000.125) == 50000.12

    def test_round_quantity(self):
        """Test rounding quantity."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=3,
            min_quantity=0.001,
            max_quantity=1000.0,
            min_notional=5.0,
            tick_size=0.01,
            step_size=0.001,
        )
        # round(0.12345, 3) = 0.123
        assert info.round_quantity(0.12345) == 0.123

    def test_validate_quantity_below_min(self):
        """Test quantity validation below minimum."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.001,
            max_quantity=1000.0,
            min_notional=5.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        assert info.validate_quantity(0.0001) is False

    def test_validate_quantity_above_max(self):
        """Test quantity validation above maximum."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.001,
            max_quantity=100.0,
            min_notional=5.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        assert info.validate_quantity(150.0) is False

    def test_validate_notional_below_min(self):
        """Test notional validation below minimum."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.0001,
            max_quantity=1000.0,
            min_notional=5.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        # 0.0001 * 50000 = 5.0, which equals min_notional
        assert info.validate_notional(0.00009, 50000.0) is False

    def test_validate_notional_above_min(self):
        """Test notional validation above minimum."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.0001,
            max_quantity=1000.0,
            min_notional=5.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        assert info.validate_notional(0.1, 50000.0) is True


class TestExchangeCapabilitiesEdgeCases:
    """Test edge cases for ExchangeCapabilities."""

    def test_default_values(self):
        """Test default values."""
        caps = ExchangeCapabilities(name="test")
        assert caps.supports_futures is True
        assert caps.supports_spot is False
        assert caps.supports_margin is False
        assert caps.supports_stop_orders is True
        assert caps.supports_take_profit is True
        assert caps.supports_trailing_stop is False
        assert caps.supports_hedge_mode is True
        assert caps.supports_reduce_only is True
        assert caps.supports_post_only is True
        assert caps.max_leverage == 125
        assert caps.rate_limit_per_second == 10
        assert caps.rate_limit_per_minute == 600
        assert caps.websocket_available is True
        assert caps.authentication_type == "HMAC"
