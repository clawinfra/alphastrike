"""
Unit tests for exchange unified data models.
"""

from datetime import UTC, datetime

import pytest

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
    _utc_now,
)


class TestEnums:
    """Tests for enum types."""

    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP_MARKET.value == "STOP_MARKET"

    def test_position_side_values(self):
        """Test PositionSide enum values."""
        assert PositionSide.LONG.value == "LONG"
        assert PositionSide.SHORT.value == "SHORT"
        assert PositionSide.BOTH.value == "BOTH"

    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.NEW.value == "NEW"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELED.value == "CANCELED"

    def test_margin_mode_values(self):
        """Test MarginMode enum values."""
        assert MarginMode.CROSS.value == "CROSS"
        assert MarginMode.ISOLATED.value == "ISOLATED"


class TestUtcNow:
    """Tests for _utc_now helper function."""

    def test_returns_utc_datetime(self):
        """Test that _utc_now returns timezone-aware UTC datetime."""
        now = _utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo == UTC

    def test_returns_current_time(self):
        """Test that _utc_now returns approximately current time."""
        before = datetime.now(UTC)
        now = _utc_now()
        after = datetime.now(UTC)
        assert before <= now <= after


class TestUnifiedOrder:
    """Tests for UnifiedOrder dataclass."""

    def test_create_market_order(self):
        """Test creating a market order."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
        )
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 0.01
        assert order.price is None
        assert order.reduce_only is False

    def test_create_limit_order(self):
        """Test creating a limit order with price."""
        order = UnifiedOrder(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.5,
            price=2500.00,
            time_in_force=TimeInForce.POST_ONLY,
        )
        assert order.price == 2500.00
        assert order.time_in_force == TimeInForce.POST_ONLY

    def test_create_order_with_stops(self):
        """Test creating order with preset stop loss and take profit."""
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            stop_loss_price=48000.0,
            take_profit_price=52000.0,
        )
        assert order.stop_loss_price == 48000.0
        assert order.take_profit_price == 52000.0


class TestUnifiedOrderResult:
    """Tests for UnifiedOrderResult dataclass."""

    def test_create_order_result(self):
        """Test creating an order result."""
        result = UnifiedOrderResult(
            order_id="123456",
            client_order_id="client_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=None,
            status=OrderStatus.FILLED,
            filled_quantity=0.01,
            average_price=50000.0,
        )
        assert result.order_id == "123456"
        assert result.is_filled is True
        assert result.is_active is False
        assert result.is_closed is True

    def test_is_active_for_new_order(self):
        """Test is_active property for NEW order."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=50000.0,
            status=OrderStatus.NEW,
        )
        assert result.is_active is True
        assert result.is_filled is False
        assert result.is_closed is False

    def test_is_active_for_partially_filled(self):
        """Test is_active for PARTIALLY_FILLED order."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=50000.0,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=0.005,
        )
        assert result.is_active is True

    def test_timestamp_is_utc(self):
        """Test that default timestamp is UTC."""
        result = UnifiedOrderResult(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            price=None,
            status=OrderStatus.FILLED,
        )
        assert result.timestamp.tzinfo == UTC


class TestUnifiedPosition:
    """Tests for UnifiedPosition dataclass."""

    def test_create_long_position(self):
        """Test creating a long position."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
        )
        assert position.symbol == "BTCUSDT"
        assert position.side == PositionSide.LONG
        assert position.quantity == 0.1

    def test_notional_value(self):
        """Test notional value calculation."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
        )
        assert position.notional_value == 5100.0  # 0.1 * 51000

    def test_pnl_percentage(self):
        """Test PnL percentage calculation."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
        )
        assert position.pnl_percentage == pytest.approx(2.0)  # 2% gain

    def test_pnl_percentage_zero_entry(self):
        """Test PnL percentage with zero entry price."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=0.0,
            mark_price=51000.0,
            unrealized_pnl=0.0,
            leverage=10,
        )
        assert position.pnl_percentage == 0.0

    def test_position_key(self):
        """Test position key tuple."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
        )
        assert position.position_key == ("BTCUSDT", PositionSide.LONG)


class TestUnifiedAccountBalance:
    """Tests for UnifiedAccountBalance dataclass."""

    def test_create_balance(self):
        """Test creating account balance."""
        balance = UnifiedAccountBalance(
            total_balance=10000.0,
            available_balance=8000.0,
            margin_balance=2000.0,
            unrealized_pnl=500.0,
        )
        assert balance.total_balance == 10000.0
        assert balance.available_balance == 8000.0

    def test_margin_ratio(self):
        """Test margin ratio calculation."""
        balance = UnifiedAccountBalance(
            total_balance=10000.0,
            available_balance=8000.0,
            margin_balance=2000.0,
            unrealized_pnl=500.0,
        )
        assert balance.margin_ratio == pytest.approx(0.2)  # 2000 / 10000

    def test_margin_ratio_zero_balance(self):
        """Test margin ratio with zero total balance."""
        balance = UnifiedAccountBalance(
            total_balance=0.0,
            available_balance=0.0,
            margin_balance=0.0,
            unrealized_pnl=0.0,
        )
        assert balance.margin_ratio == 0.0


class TestUnifiedTicker:
    """Tests for UnifiedTicker dataclass."""

    def test_create_ticker(self):
        """Test creating a ticker."""
        ticker = UnifiedTicker(
            symbol="BTCUSDT",
            last_price=50000.0,
            bid_price=49990.0,
            ask_price=50010.0,
            bid_quantity=1.0,
            ask_quantity=1.5,
            volume_24h=1000.0,
            quote_volume_24h=50000000.0,
            high_24h=51000.0,
            low_24h=49000.0,
            price_change_24h=500.0,
            price_change_pct_24h=1.0,
        )
        assert ticker.symbol == "BTCUSDT"
        assert ticker.last_price == 50000.0

    def test_spread_calculation(self):
        """Test bid-ask spread calculation."""
        ticker = UnifiedTicker(
            symbol="BTCUSDT",
            last_price=50000.0,
            bid_price=49990.0,
            ask_price=50010.0,
            bid_quantity=1.0,
            ask_quantity=1.5,
            volume_24h=1000.0,
            quote_volume_24h=50000000.0,
            high_24h=51000.0,
            low_24h=49000.0,
            price_change_24h=500.0,
            price_change_pct_24h=1.0,
        )
        # Spread = (50010 - 49990) / 50000 = 0.0004
        assert ticker.spread == pytest.approx(0.0004)

    def test_mid_price(self):
        """Test mid price calculation."""
        ticker = UnifiedTicker(
            symbol="BTCUSDT",
            last_price=50000.0,
            bid_price=49990.0,
            ask_price=50010.0,
            bid_quantity=1.0,
            ask_quantity=1.5,
            volume_24h=1000.0,
            quote_volume_24h=50000000.0,
            high_24h=51000.0,
            low_24h=49000.0,
            price_change_24h=500.0,
            price_change_pct_24h=1.0,
        )
        assert ticker.mid_price == 50000.0


class TestUnifiedOrderbook:
    """Tests for UnifiedOrderbook dataclass."""

    def test_create_orderbook(self):
        """Test creating an orderbook."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0), (49990.0, 2.0), (49980.0, 1.5)],
            asks=[(50010.0, 0.5), (50020.0, 1.0), (50030.0, 2.0)],
        )
        assert orderbook.symbol == "BTCUSDT"
        assert len(orderbook.bids) == 3
        assert len(orderbook.asks) == 3

    def test_best_bid_ask(self):
        """Test best bid/ask extraction."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0), (49990.0, 2.0)],
            asks=[(50010.0, 0.5), (50020.0, 1.0)],
        )
        assert orderbook.best_bid == 50000.0
        assert orderbook.best_ask == 50010.0
        assert orderbook.best_bid_qty == 1.0
        assert orderbook.best_ask_qty == 0.5

    def test_empty_orderbook(self):
        """Test empty orderbook properties."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[],
            asks=[],
        )
        assert orderbook.best_bid == 0.0
        assert orderbook.best_ask == 0.0
        assert orderbook.spread == 0.0
        assert orderbook.mid_price == 0.0

    def test_spread(self):
        """Test orderbook spread calculation."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0)],
            asks=[(50010.0, 0.5)],
        )
        # Spread = (50010 - 50000) / 50005 ≈ 0.0002
        assert orderbook.spread == pytest.approx(0.0002, rel=1e-3)

    def test_total_depth(self):
        """Test total depth calculation."""
        orderbook = UnifiedOrderbook(
            symbol="BTCUSDT",
            bids=[(50000.0, 1.0), (49990.0, 2.0), (49980.0, 1.5)],
            asks=[(50010.0, 0.5), (50020.0, 1.0), (50030.0, 2.0)],
        )
        assert orderbook.total_bid_depth() == 4.5
        assert orderbook.total_ask_depth() == 3.5
        assert orderbook.total_bid_depth(levels=2) == 3.0
        assert orderbook.total_ask_depth(levels=2) == 1.5


class TestUnifiedCandle:
    """Tests for UnifiedCandle dataclass."""

    def test_create_candle(self):
        """Test creating a candle."""
        candle = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        assert candle.symbol == "BTCUSDT"
        assert candle.open == 50000.0

    def test_candle_is_frozen(self):
        """Test that candle is immutable (frozen dataclass)."""
        candle = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            candle.close = 51000.0  # type: ignore

    def test_body_size(self):
        """Test candle body size calculation."""
        candle = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        assert candle.body_size == 50.0

    def test_range(self):
        """Test candle range calculation."""
        candle = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        assert candle.range == 200.0

    def test_is_bullish(self):
        """Test bullish candle detection."""
        bullish = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        bearish = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            open=50050.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        assert bullish.is_bullish is True
        assert bearish.is_bullish is False

    def test_to_dict(self):
        """Test candle to dictionary conversion."""
        ts = datetime.now(UTC)
        candle = UnifiedCandle(
            symbol="BTCUSDT",
            timestamp=ts,
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            interval="5m",
        )
        d = candle.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["open"] == 50000.0
        assert d["interval"] == "5m"
        assert d["timestamp"] == ts.isoformat()


class TestUnifiedTrade:
    """Tests for UnifiedTrade dataclass."""

    def test_create_trade(self):
        """Test creating a trade."""
        trade = UnifiedTrade(
            symbol="BTCUSDT",
            trade_id="123456",
            price=50000.0,
            quantity=0.01,
            side=OrderSide.BUY,
            timestamp=datetime.now(UTC),
        )
        assert trade.symbol == "BTCUSDT"
        assert trade.trade_id == "123456"

    def test_notional(self):
        """Test trade notional calculation."""
        trade = UnifiedTrade(
            symbol="BTCUSDT",
            trade_id="123456",
            price=50000.0,
            quantity=0.01,
            side=OrderSide.BUY,
            timestamp=datetime.now(UTC),
        )
        assert trade.notional == 500.0


class TestSymbolInfo:
    """Tests for SymbolInfo dataclass."""

    def test_create_symbol_info(self):
        """Test creating symbol info."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.001,
            max_quantity=1000.0,
            min_notional=10.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        assert info.symbol == "BTCUSDT"
        assert info.base_asset == "BTC"

    def test_round_price(self):
        """Test price rounding."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.001,
            max_quantity=1000.0,
            min_notional=10.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        assert info.round_price(50000.456789) == 50000.46

    def test_round_quantity(self):
        """Test quantity rounding."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.001,
            max_quantity=1000.0,
            min_notional=10.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        assert info.round_quantity(0.123456789) == 0.1235

    def test_validate_quantity(self):
        """Test quantity validation."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.001,
            max_quantity=1000.0,
            min_notional=10.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        assert info.validate_quantity(0.01) is True
        assert info.validate_quantity(0.0001) is False  # Below min
        assert info.validate_quantity(2000.0) is False  # Above max

    def test_validate_notional(self):
        """Test notional validation."""
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=4,
            min_quantity=0.001,
            max_quantity=1000.0,
            min_notional=10.0,
            tick_size=0.01,
            step_size=0.0001,
        )
        assert info.validate_notional(0.001, 50000.0) is True  # 50 USDT
        assert info.validate_notional(0.0001, 50000.0) is False  # 5 USDT < 10


class TestExchangeCapabilities:
    """Tests for ExchangeCapabilities dataclass."""

    def test_create_capabilities(self):
        """Test creating exchange capabilities."""
        caps = ExchangeCapabilities(name="weex")
        assert caps.name == "weex"
        assert caps.supports_futures is True
        assert caps.max_leverage == 125

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = ExchangeCapabilities(
            name="hyperliquid",
            supports_futures=True,
            supports_spot=False,
            max_leverage=50,
            authentication_type="WALLET",
        )
        assert caps.max_leverage == 50
        assert caps.authentication_type == "WALLET"
