"""
Hyperliquid Data Mappers

Translates between unified exchange models and Hyperliquid-specific formats.

Hyperliquid Symbol Format:
- Perpetuals: Coin name (BTC, ETH, SOL)
- Spot: @{index} format (e.g., @107 for HYPE/USDC)

Unified Format: BTCUSDT, ETHUSDT (for compatibility with existing system)

Asset Classes:
- Crypto: BTC, ETH, SOL, etc.
- Commodities: GOLD, SILVER
- Forex: EUR, GBP, JPY (as EURUSDT, etc.)
- Indices: USA500, HK50
- Stocks: TSLA, NVDA, AAPL
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

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


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(UTC)


def float_to_wire(x: float) -> str:
    """
    Convert float to Hyperliquid wire format string.

    Matches official SDK behavior - uses Decimal.normalize() to remove
    trailing zeros (e.g., 75000.0 -> "75000", not "75000.0").
    """
    # Round to 8 decimal places
    rounded = f"{x:.8f}"
    # Normalize using Decimal to remove trailing zeros
    normalized = Decimal(rounded).normalize()
    return f"{normalized:f}"


class HyperliquidMapper:
    """
    Bidirectional mapper between unified models and Hyperliquid API formats.

    Handles:
    - Symbol normalization (BTCUSDT <-> BTC, asset index)
    - Order side translation (BUY <-> B, SELL <-> A)
    - Response parsing into unified models
    - Multi-asset class support
    """

    # Hyperliquid capabilities
    CAPABILITIES = ExchangeCapabilities(
        name="hyperliquid",
        supports_futures=True,
        supports_spot=True,
        supports_margin=False,
        supports_stop_orders=True,
        supports_take_profit=True,
        supports_trailing_stop=False,
        supports_hedge_mode=False,
        supports_reduce_only=True,
        supports_post_only=True,
        max_leverage=50,
        rate_limit_per_second=20,
        rate_limit_per_minute=1200,
        websocket_available=True,
        authentication_type="EIP-712",
    )

    # Asset class categorization with detailed sectors for Medallion strategy
    ASSET_CLASSES = {
        # Crypto Major (highest liquidity, BTC-correlated)
        "BTC": "crypto_major",
        "ETH": "crypto_major",
        "BNB": "crypto_major",
        "XRP": "crypto_major",
        # Layer 1/L2 (infrastructure tokens)
        "SOL": "layer1",
        "AVAX": "layer1",
        "NEAR": "layer1",
        "APT": "layer1",
        "SUI": "layer1",
        "DOT": "layer1",
        "ATOM": "layer1",
        "ADA": "layer1",
        "ARB": "layer2",
        "OP": "layer2",
        "MATIC": "layer2",
        # DeFi (protocol tokens)
        "AAVE": "defi",
        "UNI": "defi",
        "LINK": "defi",
        "MKR": "defi",
        "SNX": "defi",
        "CRV": "defi",
        "LDO": "defi",
        "COMP": "defi",
        "SUSHI": "defi",
        "GMX": "defi",
        "PENDLE": "defi",
        "ONDO": "defi",
        "ENA": "defi",
        "ETHFI": "defi",
        # AI/Compute (narrative-driven)
        "FET": "ai",
        "RNDR": "ai",
        "RENDER": "ai",
        "TAO": "ai",
        "AR": "ai",
        "IO": "ai",
        # Gaming/Metaverse
        "IMX": "gaming",
        "GALA": "gaming",
        "ILV": "gaming",
        "YGG": "gaming",
        "PIXEL": "gaming",
        "BIGTIME": "gaming",
        # Meme (high beta, sentiment-driven)
        "DOGE": "meme",
        "kPEPE": "meme",
        "kSHIB": "meme",
        "WIF": "meme",
        "kBONK": "meme",
        "kFLOKI": "meme",
        "BOME": "meme",
        "POPCAT": "meme",
        "TURBO": "meme",
        "BRETT": "meme",
        "MEW": "meme",
        "HYPE": "meme",
        # Traditional - CRITICAL DIVERSIFIERS (low crypto correlation)
        "PAXG": "traditional",  # Gold-backed, ~0.08 correlation to BTC
        "SPX": "traditional",   # S&P 500 index, ~0.15 correlation to BTC
        "GOLD": "traditional",  # Commodity gold
        "SILVER": "traditional",
        "WTI": "traditional",   # Oil
        # Forex (vs USD)
        "EUR": "forex",
        "GBP": "forex",
        "JPY": "forex",
        "CHF": "forex",
        "AUD": "forex",
        "CAD": "forex",
        "NZD": "forex",
        # Indices
        "USA500": "index",
        "USA100": "index",
        "USA30": "index",
        "UK100": "index",
        "HK50": "index",
        "DE40": "index",
        "JP225": "index",
        # Stocks
        "TSLA": "stock",
        "NVDA": "stock",
        "AAPL": "stock",
        "MSFT": "stock",
        "GOOG": "stock",
        "AMZN": "stock",
        "META": "stock",
        "AMD": "stock",
        "COIN": "stock",
        "MSTR": "stock",
    }

    # Medallion Portfolio: 20 diversified assets for optimal risk/return
    MEDALLION_PORTFOLIO = {
        # Tier 1: Crypto Major (25% weight)
        "crypto_major": ["BTC", "ETH", "BNB", "XRP"],
        # Tier 2: L1/L2 (15% weight)
        "layer1": ["SOL", "AVAX", "NEAR", "APT"],
        # Tier 3: DeFi (10% weight)
        "defi": ["AAVE", "UNI", "LINK"],
        # Tier 4: AI (5% weight)
        "ai": ["RNDR", "FET"],
        # Tier 5: Meme (10% weight)
        "meme": ["DOGE", "HYPE"],
        # Tier 6: Traditional - KEY DIVERSIFIERS (30% weight)
        "traditional": ["PAXG", "SPX"],
    }

    # Target weights per sector
    SECTOR_WEIGHTS = {
        "crypto_major": 0.25,
        "layer1": 0.15,
        "defi": 0.10,
        "ai": 0.05,
        "meme": 0.10,
        "traditional": 0.30,  # Critical for drawdown reduction
        "forex": 0.05,
    }

    @classmethod
    def get_medallion_assets(cls) -> list[str]:
        """Get the list of assets for Medallion-style diversified trading."""
        assets = []
        for sector_assets in cls.MEDALLION_PORTFOLIO.values():
            assets.extend(sector_assets)
        return assets

    @classmethod
    def get_sector(cls, symbol: str) -> str:
        """Get sector for a symbol."""
        coin = cls.to_hyperliquid_coin(symbol)
        return cls.ASSET_CLASSES.get(coin, "crypto_major")

    # Unified symbol to Hyperliquid coin mapping
    # We use XXXUSDT format unified, Hyperliquid uses just XXX
    SYMBOL_TO_COIN: dict[str, str] = {}
    COIN_TO_SYMBOL: dict[str, str] = {}

    # Cache for asset metadata (populated from /info meta endpoint)
    _asset_meta: dict[str, dict[str, Any]] = {}
    _coin_to_index: dict[str, int] = {}
    _index_to_coin: dict[int, str] = {}

    @classmethod
    def set_asset_meta(cls, meta: list[dict[str, Any]]) -> None:
        """
        Set asset metadata from Hyperliquid /info meta response.

        Args:
            meta: List of asset metadata dicts from API
        """
        cls._asset_meta.clear()
        cls._coin_to_index.clear()
        cls._index_to_coin.clear()

        for idx, asset in enumerate(meta):
            name = asset.get("name", "")
            cls._asset_meta[name] = asset
            cls._coin_to_index[name] = idx
            cls._index_to_coin[idx] = name

            # Build symbol mappings
            unified_symbol = f"{name}USDT"
            cls.SYMBOL_TO_COIN[unified_symbol] = name
            cls.COIN_TO_SYMBOL[name] = unified_symbol

    # ==================== Symbol Translation ====================

    @classmethod
    def to_hyperliquid_coin(cls, symbol: str) -> str:
        """
        Convert unified symbol to Hyperliquid coin name.

        BTCUSDT -> BTC
        ETHUSDT -> ETH
        GOLDUSDT -> GOLD

        Args:
            symbol: Unified format (e.g., "BTCUSDT")

        Returns:
            Hyperliquid coin name (e.g., "BTC")
        """
        # Check cache first
        if symbol in cls.SYMBOL_TO_COIN:
            return cls.SYMBOL_TO_COIN[symbol]

        # Strip USDT/USD suffix
        if symbol.endswith("USDT"):
            return symbol[:-4]
        if symbol.endswith("USD"):
            return symbol[:-3]

        return symbol

    @classmethod
    def from_hyperliquid_coin(cls, coin: str) -> str:
        """
        Convert Hyperliquid coin to unified symbol format.

        BTC -> BTCUSDT
        ETH -> ETHUSDT

        Args:
            coin: Hyperliquid coin name (e.g., "BTC")

        Returns:
            Unified format (e.g., "BTCUSDT")
        """
        # Check cache first
        if coin in cls.COIN_TO_SYMBOL:
            return cls.COIN_TO_SYMBOL[coin]

        # Spot pairs use @{index} format
        if coin.startswith("@"):
            return f"SPOT{coin}"

        return f"{coin}USDT"

    @classmethod
    def get_asset_index(cls, symbol: str) -> int:
        """
        Get Hyperliquid asset index for a symbol.

        Args:
            symbol: Unified symbol (e.g., "BTCUSDT")

        Returns:
            Asset index for Hyperliquid API

        Raises:
            ValueError: If symbol not found in metadata
        """
        coin = cls.to_hyperliquid_coin(symbol)

        if coin in cls._coin_to_index:
            return cls._coin_to_index[coin]

        raise ValueError(
            f"Unknown symbol {symbol}. "
            "Call refresh_metadata() to update asset list."
        )

    @classmethod
    def get_asset_class(cls, symbol: str) -> str:
        """
        Get asset class for a symbol.

        Args:
            symbol: Unified or coin symbol

        Returns:
            Asset class: crypto, commodity, forex, index, or stock
        """
        coin = cls.to_hyperliquid_coin(symbol)
        return cls.ASSET_CLASSES.get(coin, "crypto")

    # ==================== Order Side Translation ====================

    @staticmethod
    def to_hyperliquid_side(side: OrderSide) -> str:
        """Convert unified OrderSide to Hyperliquid format."""
        return "B" if side == OrderSide.BUY else "A"

    @staticmethod
    def to_hyperliquid_is_buy(side: OrderSide) -> bool:
        """Convert unified OrderSide to Hyperliquid is_buy boolean."""
        return side == OrderSide.BUY

    @staticmethod
    def from_hyperliquid_side(hl_side: str) -> OrderSide:
        """Convert Hyperliquid order side to unified format."""
        return OrderSide.BUY if hl_side.upper() == "B" else OrderSide.SELL

    # ==================== Order Type Translation ====================

    @staticmethod
    def to_hyperliquid_order_type(
        order_type: OrderType,
        time_in_force: TimeInForce = TimeInForce.GTC,
        trigger_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Convert unified OrderType to Hyperliquid order type spec.

        Args:
            order_type: Unified order type
            time_in_force: Time in force
            trigger_price: Trigger price for stop orders

        Returns:
            Hyperliquid order type specification
        """
        tif_map = {
            TimeInForce.GTC: "Gtc",
            TimeInForce.IOC: "Ioc",
            TimeInForce.POST_ONLY: "Alo",  # Add liquidity only
            TimeInForce.FOK: "Ioc",  # Map FOK to IOC
        }

        trigger_px_str = str(trigger_price) if trigger_price else "0"

        # Handle limit and market orders
        if order_type in (OrderType.MARKET, OrderType.LIMIT):
            return {"limit": {"tif": tif_map.get(time_in_force, "Gtc")}}

        # Handle trigger orders (stop loss and take profit)
        trigger_configs = {
            OrderType.STOP_MARKET: {"isMarket": True, "tpsl": "sl"},
            OrderType.TAKE_PROFIT_MARKET: {"isMarket": True, "tpsl": "tp"},
            OrderType.STOP_LIMIT: {"isMarket": False, "tpsl": "sl"},
            OrderType.TAKE_PROFIT_LIMIT: {"isMarket": False, "tpsl": "tp"},
        }

        if order_type in trigger_configs:
            config = trigger_configs[order_type]
            return {
                "trigger": {
                    "isMarket": config["isMarket"],
                    "triggerPx": trigger_px_str,
                    "tpsl": config["tpsl"],
                }
            }

        return {"limit": {"tif": "Gtc"}}

    @staticmethod
    def from_hyperliquid_order_status(status: str) -> OrderStatus:
        """Convert Hyperliquid order status to unified format."""
        status_map = {
            "open": OrderStatus.NEW,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "triggered": OrderStatus.NEW,  # Triggered orders become active
            "rejected": OrderStatus.REJECTED,
            "marginCanceled": OrderStatus.CANCELED,
            "selfTradeCanceled": OrderStatus.CANCELED,
            "reduceOnlyCanceled": OrderStatus.CANCELED,
            "liquidatedCanceled": OrderStatus.CANCELED,
        }
        return status_map.get(status, OrderStatus.NEW)

    # ==================== Model Translation: Unified -> Hyperliquid ====================

    @classmethod
    def from_unified_order(cls, order: UnifiedOrder) -> dict[str, Any]:
        """
        Convert UnifiedOrder to Hyperliquid order wire format.

        Args:
            order: Unified order model

        Returns:
            Hyperliquid order specification dict
        """
        asset_index = cls.get_asset_index(order.symbol)

        # Determine price - for market orders, use a far price
        if order.order_type == OrderType.MARKET:
            # Market orders still need a price in Hyperliquid
            # Use a very favorable price that will definitely fill
            limit_px = order.price or 0
        else:
            limit_px = order.price or 0

        order_type_spec = cls.to_hyperliquid_order_type(
            order.order_type,
            order.time_in_force,
            order.stop_loss_price or order.take_profit_price,
        )

        wire: dict[str, Any] = {
            "a": asset_index,
            "b": cls.to_hyperliquid_is_buy(order.side),
            "p": float_to_wire(limit_px),
            "s": float_to_wire(order.quantity),
            "r": order.reduce_only,
            "t": order_type_spec,
        }

        if order.client_order_id:
            # Client order ID must be 128-bit hex (32 chars)
            cloid = order.client_order_id
            if not cloid.startswith("0x"):
                cloid = "0x" + cloid.ljust(32, "0")[:32]
            wire["c"] = cloid

        return wire

    # ==================== Model Translation: Hyperliquid -> Unified ====================

    @classmethod
    def to_unified_ticker(cls, mids: dict[str, str], coin: str) -> UnifiedTicker:
        """
        Convert Hyperliquid mid price to UnifiedTicker.

        Args:
            mids: All mid prices dict
            coin: Coin name

        Returns:
            UnifiedTicker
        """
        symbol = cls.from_hyperliquid_coin(coin)
        mid_price = float(mids.get(coin, "0"))

        return UnifiedTicker(
            symbol=symbol,
            last_price=mid_price,
            bid_price=mid_price,  # Mid approximation
            ask_price=mid_price,
            bid_quantity=0,
            ask_quantity=0,
            volume_24h=0,
            quote_volume_24h=0,
            high_24h=0,
            low_24h=0,
            price_change_24h=0,
            price_change_pct_24h=0,
            timestamp=_utc_now(),
        )

    @classmethod
    def to_unified_orderbook(
        cls,
        book_data: dict[str, Any],
        symbol: str,
    ) -> UnifiedOrderbook:
        """
        Convert Hyperliquid L2 book to UnifiedOrderbook.

        Args:
            book_data: Hyperliquid L2 book response
            symbol: Unified symbol

        Returns:
            UnifiedOrderbook
        """
        # Hyperliquid format: {"levels": [[bids], [asks]]}
        levels = book_data.get("levels", [[], []])

        bids = []
        asks = []

        if len(levels) >= 2:
            for level in levels[0]:  # Bids
                bids.append((float(level.get("px", 0)), float(level.get("sz", 0))))
            for level in levels[1]:  # Asks
                asks.append((float(level.get("px", 0)), float(level.get("sz", 0))))

        return UnifiedOrderbook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=_utc_now(),
        )

    @classmethod
    def to_unified_candle(
        cls,
        candle_data: dict[str, Any],
        symbol: str,
        interval: str = "1h",
    ) -> UnifiedCandle:
        """
        Convert Hyperliquid candle to UnifiedCandle.

        Hyperliquid format: {t, T, s, i, o, h, l, c, v, n}

        Args:
            candle_data: Hyperliquid candle dict
            symbol: Unified symbol
            interval: Candle interval

        Returns:
            UnifiedCandle
        """
        return UnifiedCandle(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(int(candle_data.get("t", 0)) / 1000, UTC),
            open=float(candle_data.get("o", 0)),
            high=float(candle_data.get("h", 0)),
            low=float(candle_data.get("l", 0)),
            close=float(candle_data.get("c", 0)),
            volume=float(candle_data.get("v", 0)),
            quote_volume=0,  # Not provided
            trades=int(candle_data.get("n", 0)),
            interval=interval,
        )

    @classmethod
    def to_unified_position(cls, pos_data: dict[str, Any]) -> UnifiedPosition:
        """
        Convert Hyperliquid position to UnifiedPosition.

        Args:
            pos_data: Hyperliquid position dict from assetPositions

        Returns:
            UnifiedPosition
        """
        position = pos_data.get("position", {})
        coin = position.get("coin", "")
        symbol = cls.from_hyperliquid_coin(coin)

        # Signed size: positive = long, negative = short
        szi = float(position.get("szi", 0))
        side = PositionSide.LONG if szi >= 0 else PositionSide.SHORT
        quantity = abs(szi)

        leverage_info = position.get("leverage", {})
        margin_mode = (
            MarginMode.CROSS
            if leverage_info.get("type") == "cross"
            else MarginMode.ISOLATED
        )

        liq_px = position.get("liquidationPx")

        return UnifiedPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=float(position.get("entryPx", 0)),
            mark_price=0,  # Not in position response
            unrealized_pnl=float(position.get("unrealizedPnl", 0)),
            leverage=int(leverage_info.get("value", 1)),
            liquidation_price=float(liq_px) if liq_px else None,
            margin_mode=margin_mode,
            margin=float(position.get("marginUsed", 0)),
            timestamp=_utc_now(),
        )

    @classmethod
    def to_unified_account_balance(
        cls,
        state: dict[str, Any],
    ) -> UnifiedAccountBalance:
        """
        Convert Hyperliquid clearinghouse state to UnifiedAccountBalance.

        Args:
            state: Hyperliquid clearinghouseState response

        Returns:
            UnifiedAccountBalance
        """
        margin_summary = state.get("marginSummary", {})
        cross_margin = state.get("crossMarginSummary", {})

        return UnifiedAccountBalance(
            total_balance=float(margin_summary.get("accountValue", 0)),
            available_balance=float(state.get("withdrawable", 0)),
            margin_balance=float(cross_margin.get("totalMarginUsed", 0)),
            unrealized_pnl=float(
                margin_summary.get("totalNtlPos", 0)
            ),  # Approximate
            currency="USDC",  # Hyperliquid uses USDC
            timestamp=_utc_now(),
        )

    @classmethod
    def to_unified_order_result(
        cls,
        response: dict[str, Any],
        original_order: UnifiedOrder | None = None,
    ) -> UnifiedOrderResult:
        """
        Convert Hyperliquid order response to UnifiedOrderResult.

        Args:
            response: Hyperliquid order response
            original_order: Original order for context

        Returns:
            UnifiedOrderResult
        """
        # Parse response statuses
        statuses = (
            response.get("response", {}).get("data", {}).get("statuses", [])
        )

        order_id = ""
        status = OrderStatus.NEW
        filled_qty = 0.0
        avg_price = None

        if statuses:
            first_status = statuses[0]

            if "resting" in first_status:
                order_id = str(first_status["resting"].get("oid", ""))
                status = OrderStatus.NEW

            elif "filled" in first_status:
                filled_info = first_status["filled"]
                order_id = str(filled_info.get("oid", ""))
                filled_qty = float(filled_info.get("totalSz", 0))
                avg_price = float(filled_info.get("avgPx", 0))
                status = OrderStatus.FILLED

            elif "error" in first_status:
                status = OrderStatus.REJECTED

        return UnifiedOrderResult(
            order_id=order_id,
            client_order_id=original_order.client_order_id if original_order else None,
            symbol=original_order.symbol if original_order else "",
            side=original_order.side if original_order else OrderSide.BUY,
            order_type=original_order.order_type if original_order else OrderType.LIMIT,
            quantity=original_order.quantity if original_order else 0,
            price=original_order.price if original_order else None,
            status=status,
            filled_quantity=filled_qty,
            average_price=avg_price,
            commission=0,
            commission_asset="USDC",
            timestamp=_utc_now(),
            raw_response=response,
        )

    @classmethod
    def to_unified_trade(
        cls,
        fill_data: dict[str, Any],
    ) -> UnifiedTrade:
        """
        Convert Hyperliquid fill to UnifiedTrade.

        Args:
            fill_data: Hyperliquid fill dict

        Returns:
            UnifiedTrade
        """
        coin = fill_data.get("coin", "")
        symbol = cls.from_hyperliquid_coin(coin)

        return UnifiedTrade(
            symbol=symbol,
            trade_id=str(fill_data.get("tid", fill_data.get("oid", ""))),
            price=float(fill_data.get("px", 0)),
            quantity=float(fill_data.get("sz", 0)),
            side=cls.from_hyperliquid_side(fill_data.get("side", "B")),
            timestamp=datetime.fromtimestamp(
                int(fill_data.get("time", 0)) / 1000, UTC
            ),
            is_maker=not fill_data.get("crossed", True),
        )

    @classmethod
    def to_symbol_info(cls, asset_meta: dict[str, Any], index: int) -> SymbolInfo:
        """
        Convert Hyperliquid asset meta to SymbolInfo.

        Args:
            asset_meta: Asset metadata from meta response
            index: Asset index

        Returns:
            SymbolInfo
        """
        name = asset_meta.get("name", "")
        symbol = cls.from_hyperliquid_coin(name)

        return SymbolInfo(
            symbol=symbol,
            base_asset=name,
            quote_asset="USDC",
            price_precision=6,  # Hyperliquid uses 6 decimals
            quantity_precision=asset_meta.get("szDecimals", 3),
            min_quantity=10 ** (-asset_meta.get("szDecimals", 3)),
            max_quantity=1000000,
            min_notional=10,  # $10 minimum
            tick_size=0.000001,
            step_size=10 ** (-asset_meta.get("szDecimals", 3)),
            min_leverage=1,
            max_leverage=asset_meta.get("maxLeverage", 50),
            maker_fee=0.0002,  # 2 bps
            taker_fee=0.0005,  # 5 bps
        )

    # ==================== Interval Mapping ====================

    # Valid Hyperliquid intervals
    VALID_INTERVALS = frozenset({
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "8h", "12h",
        "1d", "3d", "1w", "1M",
    })

    # Alternative interval mappings
    INTERVAL_ALIASES = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1hour": "1h",
        "4hour": "4h",
        "1day": "1d",
        "1week": "1w",
    }

    @classmethod
    def to_hyperliquid_interval(cls, interval: str) -> str:
        """
        Convert unified interval to Hyperliquid format.

        Hyperliquid supports: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        lower_interval = interval.lower()

        if lower_interval in cls.VALID_INTERVALS:
            return interval

        return cls.INTERVAL_ALIASES.get(lower_interval, "1h")
