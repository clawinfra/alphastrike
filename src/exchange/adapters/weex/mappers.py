"""
WEEX Data Mappers

Translates between unified exchange models and WEEX-specific formats.

WEEX Symbol Format: cmt_btcusdt (lowercase, prefixed with cmt_)
Unified Format: BTCUSDT (uppercase, no separator)
"""

from __future__ import annotations

from datetime import UTC, datetime
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


class WEEXMapper:
    """
    Bidirectional mapper between unified models and WEEX API formats.

    Handles:
    - Symbol normalization (BTCUSDT <-> cmt_btcusdt)
    - Enum translation (OrderSide.BUY <-> "buy")
    - Response parsing into unified models
    - Request formatting from unified models
    """

    # WEEX capabilities
    CAPABILITIES = ExchangeCapabilities(
        name="weex",
        supports_futures=True,
        supports_spot=False,
        supports_margin=False,
        supports_stop_orders=True,
        supports_take_profit=True,
        supports_trailing_stop=False,
        supports_hedge_mode=True,
        supports_reduce_only=True,
        supports_post_only=True,
        max_leverage=125,
        rate_limit_per_second=10,
        rate_limit_per_minute=600,
        websocket_available=True,
        authentication_type="HMAC",
    )

    # ==================== Symbol Translation ====================

    @staticmethod
    def to_weex_symbol(symbol: str) -> str:
        """
        Convert unified symbol to WEEX format.

        BTCUSDT -> cmt_btcusdt

        Args:
            symbol: Unified format (e.g., "BTCUSDT")

        Returns:
            WEEX format (e.g., "cmt_btcusdt")
        """
        return f"cmt_{symbol.lower()}"

    @staticmethod
    def from_weex_symbol(weex_symbol: str) -> str:
        """
        Convert WEEX symbol to unified format.

        cmt_btcusdt -> BTCUSDT

        Args:
            weex_symbol: WEEX format (e.g., "cmt_btcusdt")

        Returns:
            Unified format (e.g., "BTCUSDT")
        """
        # Remove "cmt_" prefix if present
        if weex_symbol.lower().startswith("cmt_"):
            return weex_symbol[4:].upper()
        return weex_symbol.upper()

    # ==================== Enum Translation ====================

    @staticmethod
    def to_weex_order_side(side: OrderSide) -> str:
        """Convert unified OrderSide to WEEX format."""
        return side.value.lower()  # BUY -> buy, SELL -> sell

    @staticmethod
    def from_weex_order_side(weex_side: str) -> OrderSide:
        """Convert WEEX order side to unified format."""
        return OrderSide.BUY if weex_side.lower() == "buy" else OrderSide.SELL

    @staticmethod
    def to_weex_order_type(order_type: OrderType) -> str:
        """Convert unified OrderType to WEEX format."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_MARKET: "market",
            OrderType.STOP_LIMIT: "limit",
            OrderType.TAKE_PROFIT_MARKET: "market",
            OrderType.TAKE_PROFIT_LIMIT: "limit",
        }
        return mapping.get(order_type, "market")

    @staticmethod
    def from_weex_order_type(weex_type: str) -> OrderType:
        """Convert WEEX order type to unified format."""
        return OrderType.LIMIT if weex_type.lower() == "limit" else OrderType.MARKET

    @staticmethod
    def to_weex_position_side(side: PositionSide) -> str:
        """Convert unified PositionSide to WEEX holdSide format."""
        mapping = {
            PositionSide.LONG: "long",
            PositionSide.SHORT: "short",
            PositionSide.BOTH: "both",
        }
        return mapping.get(side, "long")

    @staticmethod
    def from_weex_position_side(weex_side: str) -> PositionSide:
        """Convert WEEX holdSide to unified PositionSide."""
        mapping = {
            "long": PositionSide.LONG,
            "short": PositionSide.SHORT,
            "both": PositionSide.BOTH,
        }
        return mapping.get(weex_side.lower(), PositionSide.LONG)

    @staticmethod
    def to_weex_time_in_force(tif: TimeInForce) -> str:
        """Convert unified TimeInForce to WEEX format."""
        mapping = {
            TimeInForce.GTC: "normal",
            TimeInForce.IOC: "ioc",
            TimeInForce.FOK: "fok",
            TimeInForce.POST_ONLY: "post_only",
        }
        return mapping.get(tif, "normal")

    @staticmethod
    def from_weex_order_status(weex_status: str) -> OrderStatus:
        """Convert WEEX order status to unified format."""
        status_map = {
            "new": OrderStatus.NEW,
            "open": OrderStatus.NEW,
            "submitted": OrderStatus.NEW,
            "pending": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "partial-fill": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "full-fill": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELED,
            "canceled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        return status_map.get(weex_status.lower(), OrderStatus.NEW)

    # ==================== Model Translation: Unified -> WEEX ====================

    @staticmethod
    def from_unified_order(order: UnifiedOrder) -> dict[str, Any]:
        """
        Convert UnifiedOrder to WEEX API request format.

        Args:
            order: Unified order model

        Returns:
            WEEX API request body dict
        """
        data: dict[str, Any] = {
            "symbol": WEEXMapper.to_weex_symbol(order.symbol),
            "marginCoin": "USDT",
            "side": WEEXMapper.to_weex_order_side(order.side),
            "orderType": WEEXMapper.to_weex_order_type(order.order_type),
            "size": str(order.quantity),
            "timeInForceValue": WEEXMapper.to_weex_time_in_force(order.time_in_force),
        }

        if order.price is not None:
            data["price"] = str(order.price)

        if order.position_side:
            data["holdSide"] = WEEXMapper.to_weex_position_side(order.position_side)

        if order.client_order_id:
            data["clientOid"] = order.client_order_id

        if order.reduce_only:
            data["reduceOnly"] = "true"

        if order.stop_loss_price:
            data["presetStopLossPrice"] = str(order.stop_loss_price)

        if order.take_profit_price:
            data["presetTakeProfitPrice"] = str(order.take_profit_price)

        return data

    # ==================== Model Translation: WEEX -> Unified ====================

    @staticmethod
    def to_unified_order_result(
        response: dict[str, Any],
        original_order: UnifiedOrder | None = None,
    ) -> UnifiedOrderResult:
        """
        Convert WEEX order response to UnifiedOrderResult.

        Args:
            response: WEEX API response dict
            original_order: Original order request for filling in details

        Returns:
            UnifiedOrderResult
        """
        # Get symbol from response or fall back to original order
        weex_symbol = response.get("symbol", "")
        if weex_symbol:
            symbol = WEEXMapper.from_weex_symbol(weex_symbol)
        elif original_order:
            symbol = original_order.symbol
        else:
            symbol = ""

        # Parse side from response or fall back to original order
        side_str = response.get("side", "")
        if side_str:
            side = WEEXMapper.from_weex_order_side(side_str)
        elif original_order:
            side = original_order.side
        else:
            side = OrderSide.BUY

        # Parse order type from response or fall back to original order
        type_str = response.get("orderType", "")
        if type_str:
            order_type = WEEXMapper.from_weex_order_type(type_str)
        elif original_order:
            order_type = original_order.order_type
        else:
            order_type = OrderType.MARKET

        # Parse quantity
        quantity = float(response.get("size", 0))
        if quantity == 0 and original_order:
            quantity = original_order.quantity

        # Parse price
        price_str = response.get("price")
        if price_str:
            price = float(price_str)
        elif original_order:
            price = original_order.price
        else:
            price = None

        # Parse average price
        avg_px_str = response.get("avgPx")
        average_price = float(avg_px_str) if avg_px_str else None

        return UnifiedOrderResult(
            order_id=str(response.get("orderId", "")),
            client_order_id=response.get("clientOid"),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=WEEXMapper.from_weex_order_status(response.get("status", "submitted")),
            filled_quantity=float(response.get("filledQty", 0)),
            average_price=average_price,
            commission=float(response.get("fee", 0)),
            commission_asset=response.get("feeCcy", "USDT"),
            timestamp=_utc_now(),
            raw_response=response,
        )

    @staticmethod
    def to_unified_position(pos_data: dict[str, Any]) -> UnifiedPosition:
        """
        Convert WEEX position data to UnifiedPosition.

        Args:
            pos_data: WEEX position dict

        Returns:
            UnifiedPosition
        """
        weex_symbol = pos_data.get("symbol", "")
        symbol = WEEXMapper.from_weex_symbol(weex_symbol)

        liq_price = pos_data.get("liquidationPrice")

        return UnifiedPosition(
            symbol=symbol,
            side=WEEXMapper.from_weex_position_side(pos_data.get("holdSide", "long")),
            quantity=float(pos_data.get("total", 0)),
            entry_price=float(pos_data.get("averageOpenPrice", 0)),
            mark_price=float(pos_data.get("markPrice", 0)),
            unrealized_pnl=float(pos_data.get("unrealizedPL", 0)),
            leverage=int(pos_data.get("leverage", 1)),
            liquidation_price=float(liq_price) if liq_price else None,
            margin_mode=(
                MarginMode.ISOLATED
                if pos_data.get("marginMode") == "isolated"
                else MarginMode.CROSS
            ),
            margin=float(pos_data.get("margin", 0)),
            timestamp=_utc_now(),
        )

    @staticmethod
    def to_unified_account_balance(account_data: dict[str, Any]) -> UnifiedAccountBalance:
        """
        Convert WEEX account data to UnifiedAccountBalance.

        Args:
            account_data: WEEX account dict

        Returns:
            UnifiedAccountBalance
        """
        return UnifiedAccountBalance(
            total_balance=float(account_data.get("usdtEquity", 0)),
            available_balance=float(account_data.get("available", 0)),
            margin_balance=float(account_data.get("crossMaxAvailable", 0)),
            unrealized_pnl=float(account_data.get("unrealizedPL", 0)),
            currency="USDT",
            timestamp=_utc_now(),
        )

    @staticmethod
    def to_unified_ticker(ticker_data: dict[str, Any], symbol: str) -> UnifiedTicker:
        """
        Convert WEEX ticker data to UnifiedTicker.

        Args:
            ticker_data: WEEX ticker dict
            symbol: Unified symbol

        Returns:
            UnifiedTicker
        """
        last_price = float(ticker_data.get("last") or ticker_data.get("lastPr") or 0)
        open_price = float(ticker_data.get("open24h") or 0)

        if open_price > 0:
            price_change = last_price - open_price
            price_change_pct = price_change / open_price * 100
        else:
            price_change = 0.0
            price_change_pct = 0.0

        return UnifiedTicker(
            symbol=symbol,
            last_price=last_price,
            bid_price=float(ticker_data.get("bidPr") or ticker_data.get("bestBid") or 0),
            ask_price=float(ticker_data.get("askPr") or ticker_data.get("bestAsk") or 0),
            bid_quantity=float(ticker_data.get("bidSz") or 0),
            ask_quantity=float(ticker_data.get("askSz") or 0),
            volume_24h=float(ticker_data.get("baseVolume") or ticker_data.get("volume24h") or 0),
            quote_volume_24h=float(ticker_data.get("quoteVolume") or 0),
            high_24h=float(ticker_data.get("high24h") or 0),
            low_24h=float(ticker_data.get("low24h") or 0),
            price_change_24h=price_change,
            price_change_pct_24h=price_change_pct,
            timestamp=_utc_now(),
        )

    @staticmethod
    def to_unified_orderbook(
        orderbook_data: dict[str, Any],
        symbol: str,
    ) -> UnifiedOrderbook:
        """
        Convert WEEX orderbook data to UnifiedOrderbook.

        Args:
            orderbook_data: WEEX orderbook dict with bids/asks
            symbol: Unified symbol

        Returns:
            UnifiedOrderbook
        """
        # WEEX format: {"bids": [[price, size], ...], "asks": [[price, size], ...]}
        bids = [(float(level[0]), float(level[1])) for level in orderbook_data.get("bids", [])]
        asks = [(float(level[0]), float(level[1])) for level in orderbook_data.get("asks", [])]

        return UnifiedOrderbook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=_utc_now(),
        )

    @staticmethod
    def to_unified_candle(
        candle_data: list | dict,
        symbol: str,
        interval: str = "1m",
    ) -> UnifiedCandle:
        """
        Convert WEEX candle data to UnifiedCandle.

        WEEX format: [timestamp, open, high, low, close, volume, quoteVolume]

        Args:
            candle_data: WEEX candle array or dict
            symbol: Unified symbol
            interval: Candle interval

        Returns:
            UnifiedCandle
        """
        if isinstance(candle_data, list):
            return WEEXMapper._candle_from_list(candle_data, symbol, interval)
        return WEEXMapper._candle_from_dict(candle_data, symbol, interval)

    @staticmethod
    def _candle_from_list(data: list, symbol: str, interval: str) -> UnifiedCandle:
        """Parse candle from list format."""
        return UnifiedCandle(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(int(data[0]) / 1000),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            quote_volume=float(data[6]) if len(data) > 6 else 0.0,
            trades=int(data[7]) if len(data) > 7 else 0,
            interval=interval,
        )

    @staticmethod
    def _candle_from_dict(data: dict, symbol: str, interval: str) -> UnifiedCandle:
        """Parse candle from dict format."""
        ts = data.get("ts") or data.get("timestamp") or 0
        return UnifiedCandle(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(int(ts) / 1000),
            open=float(data.get("open") or 0),
            high=float(data.get("high") or 0),
            low=float(data.get("low") or 0),
            close=float(data.get("close") or 0),
            volume=float(data.get("volume") or data.get("baseVolume") or 0),
            quote_volume=float(data.get("quoteVolume") or 0),
            trades=int(data.get("trades") or 0),
            interval=interval,
        )

    @staticmethod
    def to_unified_trade(trade_data: dict[str, Any], symbol: str) -> UnifiedTrade:
        """
        Convert WEEX trade data to UnifiedTrade.

        Args:
            trade_data: WEEX trade dict
            symbol: Unified symbol

        Returns:
            UnifiedTrade
        """
        ts = trade_data.get("ts") or trade_data.get("timestamp") or 0
        timestamp = datetime.fromtimestamp(int(ts) / 1000) if ts else _utc_now()

        return UnifiedTrade(
            symbol=symbol,
            trade_id=str(trade_data.get("tradeId") or trade_data.get("id") or ""),
            price=float(trade_data.get("price") or 0),
            quantity=float(trade_data.get("size") or trade_data.get("qty") or 0),
            side=WEEXMapper.from_weex_order_side(trade_data.get("side", "buy")),
            timestamp=timestamp,
            is_maker=trade_data.get("isMaker", False),
        )

    @staticmethod
    def to_symbol_info(contract_data: dict[str, Any]) -> SymbolInfo:
        """
        Convert WEEX contract info to SymbolInfo.

        Args:
            contract_data: WEEX contract info dict

        Returns:
            SymbolInfo
        """
        weex_symbol = contract_data.get("symbol", "")
        symbol = WEEXMapper.from_weex_symbol(weex_symbol)

        # Extract base and quote from symbol (e.g., BTCUSDT -> BTC, USDT)
        # WEEX uses baseCoin and quoteCoin fields
        base_asset = contract_data.get("baseCoin", symbol[:-4] if len(symbol) > 4 else symbol)
        quote_asset = contract_data.get("quoteCoin", "USDT")

        return SymbolInfo(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            price_precision=int(contract_data.get("pricePlace", 2)),
            quantity_precision=int(contract_data.get("volumePlace", 4)),
            min_quantity=float(contract_data.get("minTradeNum", 0.001)),
            max_quantity=float(contract_data.get("maxTradeNum", 10000)),
            min_notional=float(contract_data.get("minTradeAmount", 5)),
            tick_size=float(contract_data.get("priceEndStep", 0.01)),
            step_size=float(contract_data.get("sizeMultiplier", 0.001)),
            min_leverage=int(contract_data.get("minLever", 1)),
            max_leverage=int(contract_data.get("maxLever", 125)),
            maker_fee=float(contract_data.get("makerFeeRate", 0.0002)),
            taker_fee=float(contract_data.get("takerFeeRate", 0.0006)),
        )

    # ==================== Interval Mapping ====================

    @staticmethod
    def to_weex_interval(interval: str) -> str:
        """
        Convert unified interval to WEEX granularity format.

        Args:
            interval: Unified interval (1m, 5m, 15m, 1h, 4h, 1d)

        Returns:
            WEEX granularity string
        """
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
            "1w": "1W",
        }
        return mapping.get(interval.lower(), "1m")

    @staticmethod
    def from_weex_interval(weex_interval: str) -> str:
        """
        Convert WEEX granularity to unified interval format.

        Args:
            weex_interval: WEEX granularity string

        Returns:
            Unified interval (1m, 5m, etc.)
        """
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1H": "1h",
            "4H": "4h",
            "1D": "1d",
            "1W": "1w",
        }
        return mapping.get(weex_interval, "1m")
