"""
Unified Exchange-Agnostic Data Models

These models represent the canonical format for all trading data.
Exchange adapters translate between these models and exchange-specific formats.

Design Principles:
- Exchange-agnostic: No exchange-specific fields or formats
- Type-safe: All fields have explicit types
- Immutable-friendly: Use frozen dataclasses where appropriate
- Self-documenting: Clear field names and docstrings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class OrderSide(str, Enum):
    """Unified order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Unified order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class PositionSide(str, Enum):
    """Unified position side for hedge mode."""

    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"  # For one-way mode


class TimeInForce(str, Enum):
    """Unified time in force."""

    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    POST_ONLY = "POST_ONLY"  # Maker only


class OrderStatus(str, Enum):
    """Unified order status."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"


class MarginMode(str, Enum):
    """Margin mode for futures positions."""

    CROSS = "CROSS"
    ISOLATED = "ISOLATED"


@dataclass
class UnifiedOrder:
    """
    Unified order request model.

    Exchange adapters convert this to exchange-specific format before sending.

    Example:
        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
            position_side=PositionSide.LONG,
        )
        result = await adapter.rest.place_order(order)
    """

    symbol: str  # Unified format: "BTCUSDT" (uppercase, no separator)
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None  # Required for LIMIT orders
    position_side: PositionSide | None = None  # For hedge mode
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str | None = None
    reduce_only: bool = False
    stop_price: float | None = None  # For stop orders
    take_profit_price: float | None = None  # Preset TP
    stop_loss_price: float | None = None  # Preset SL
    leverage: int | None = None  # Optional leverage override


@dataclass
class UnifiedOrderResult:
    """
    Unified order result/response from exchange.

    Contains the order status and fill information.
    """

    order_id: str
    client_order_id: str | None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None
    status: OrderStatus
    filled_quantity: float = 0.0
    average_price: float | None = None
    commission: float = 0.0
    commission_asset: str = "USDT"
    timestamp: datetime = field(default_factory=_utc_now)
    raw_response: dict = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be cancelled)."""
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_closed(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


@dataclass
class UnifiedPosition:
    """
    Unified position model.

    Represents an open futures position.
    """

    symbol: str
    side: PositionSide
    quantity: float  # Absolute quantity (always positive)
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: float | None = None
    margin_mode: MarginMode = MarginMode.CROSS
    margin: float = 0.0  # Position margin
    timestamp: datetime = field(default_factory=_utc_now)

    @property
    def notional_value(self) -> float:
        """Calculate the notional value of the position."""
        return self.quantity * self.mark_price

    @property
    def pnl_percentage(self) -> float:
        """Calculate PnL as percentage of entry."""
        if self.entry_price <= 0:
            return 0.0
        return (self.mark_price - self.entry_price) / self.entry_price * 100

    @property
    def position_key(self) -> tuple[str, PositionSide]:
        """Return unique key for this position."""
        return (self.symbol, self.side)


@dataclass
class UnifiedAccountBalance:
    """
    Unified account balance information.

    Represents the trading account's financial state.
    """

    total_balance: float  # Total equity (balance + unrealized PnL)
    available_balance: float  # Available for new orders
    margin_balance: float  # Used as margin
    unrealized_pnl: float  # Total unrealized PnL
    currency: str = "USDT"
    timestamp: datetime = field(default_factory=_utc_now)

    @property
    def margin_ratio(self) -> float:
        """Calculate margin usage ratio."""
        if self.total_balance <= 0:
            return 0.0
        return self.margin_balance / self.total_balance


@dataclass
class UnifiedTicker:
    """
    Unified ticker data for a trading pair.

    Contains current price and 24h statistics.
    """

    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    bid_quantity: float
    ask_quantity: float
    volume_24h: float  # Base currency volume
    quote_volume_24h: float  # Quote currency volume
    high_24h: float
    low_24h: float
    price_change_24h: float  # Absolute change
    price_change_pct_24h: float  # Percentage change
    timestamp: datetime = field(default_factory=_utc_now)

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread as ratio."""
        if self.bid_price <= 0:
            return 0.0
        mid = (self.bid_price + self.ask_price) / 2
        return (self.ask_price - self.bid_price) / mid

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2


@dataclass
class UnifiedOrderbook:
    """
    Unified orderbook data.

    Contains bid and ask levels sorted by price.
    """

    symbol: str
    bids: list[tuple[float, float]]  # [(price, quantity), ...] sorted descending
    asks: list[tuple[float, float]]  # [(price, quantity), ...] sorted ascending
    timestamp: datetime = field(default_factory=_utc_now)

    @property
    def best_bid(self) -> float:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else 0.0

    @property
    def best_bid_qty(self) -> float:
        """Get best bid quantity."""
        return self.bids[0][1] if self.bids else 0.0

    @property
    def best_ask_qty(self) -> float:
        """Get best ask quantity."""
        return self.asks[0][1] if self.asks else 0.0

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread as ratio of mid price."""
        if self.best_bid > 0 and self.best_ask > 0:
            mid = (self.best_bid + self.best_ask) / 2
            return (self.best_ask - self.best_bid) / mid
        return 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    def total_bid_depth(self, levels: int | None = None) -> float:
        """Calculate total bid depth in base currency."""
        bids = self.bids[:levels] if levels else self.bids
        return sum(qty for _, qty in bids)

    def total_ask_depth(self, levels: int | None = None) -> float:
        """Calculate total ask depth in base currency."""
        asks = self.asks[:levels] if levels else self.asks
        return sum(qty for _, qty in asks)


@dataclass(frozen=True)
class UnifiedCandle:
    """
    Unified candlestick/OHLCV data.

    Immutable to ensure data integrity.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float  # Base currency volume
    quote_volume: float = 0.0  # Quote currency volume
    trades: int = 0  # Number of trades
    interval: str = "1m"

    @property
    def body_size(self) -> float:
        """Calculate candle body size (absolute)."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """Calculate candle range (high - low)."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open)."""
        return self.close > self.open

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
            "interval": self.interval,
        }


@dataclass
class UnifiedTrade:
    """
    Unified trade tick data.

    Represents a single executed trade on the exchange.
    """

    symbol: str
    trade_id: str
    price: float
    quantity: float
    side: OrderSide  # Taker side
    timestamp: datetime
    is_maker: bool = False  # True if buyer was maker

    @property
    def notional(self) -> float:
        """Calculate trade notional value."""
        return self.price * self.quantity


@dataclass
class SymbolInfo:
    """
    Trading symbol information and constraints.

    Contains trading rules like precision, min/max sizes, and fees.
    """

    symbol: str  # Unified format
    base_asset: str  # e.g., "BTC"
    quote_asset: str  # e.g., "USDT"
    price_precision: int  # Decimal places for price
    quantity_precision: int  # Decimal places for quantity
    min_quantity: float  # Minimum order quantity
    max_quantity: float  # Maximum order quantity
    min_notional: float  # Minimum order value
    tick_size: float  # Minimum price increment
    step_size: float  # Minimum quantity increment
    min_leverage: int = 1
    max_leverage: int = 125
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0006  # 0.06%

    def round_price(self, price: float) -> float:
        """Round price to valid precision."""
        return round(price, self.price_precision)

    def round_quantity(self, quantity: float) -> float:
        """Round quantity to valid precision."""
        return round(quantity, self.quantity_precision)

    def validate_quantity(self, quantity: float) -> bool:
        """Check if quantity meets constraints."""
        return self.min_quantity <= quantity <= self.max_quantity

    def validate_notional(self, quantity: float, price: float) -> bool:
        """Check if order meets minimum notional."""
        return quantity * price >= self.min_notional


@dataclass
class ExchangeCapabilities:
    """
    Describes what features an exchange supports.

    Used to adapt behavior based on exchange limitations.
    """

    name: str  # Exchange identifier
    supports_futures: bool = True
    supports_spot: bool = False
    supports_margin: bool = False
    supports_stop_orders: bool = True
    supports_take_profit: bool = True
    supports_trailing_stop: bool = False
    supports_hedge_mode: bool = True
    supports_reduce_only: bool = True
    supports_post_only: bool = True
    max_leverage: int = 125
    rate_limit_per_second: int = 10
    rate_limit_per_minute: int = 600
    websocket_available: bool = True
    authentication_type: str = "HMAC"  # HMAC, ED25519, WALLET
