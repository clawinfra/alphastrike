"""
WEEX Exchange Adapter

Implements ExchangeAdapter protocol for WEEX CEX.
Refactored from src/data/rest_client.py for unified exchange abstraction.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import aiohttp

from src.core.config import get_settings
from src.exchange.adapters.weex.mappers import WEEXMapper
from src.exchange.exceptions import (
    AuthenticationError,
    ExchangeError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderError,
    OrderNotFoundError,
    RateLimitError,
)
from src.exchange.models import (
    ExchangeCapabilities,
    OrderSide,
    PositionSide,
    SymbolInfo,
    UnifiedAccountBalance,
    UnifiedCandle,
    UnifiedOrder,
    UnifiedOrderbook,
    UnifiedOrderResult,
    UnifiedPosition,
    UnifiedTicker,
    UnifiedTrade,
)
from src.exchange.protocols import ExchangeAdapter, ExchangeRESTProtocol, ExchangeWebSocketProtocol

if TYPE_CHECKING:
    from src.exchange.adapters.weex.websocket import WEEXWebSocket

logger = logging.getLogger(__name__)


class WEEXRESTClient(ExchangeRESTProtocol):
    """
    WEEX REST API client implementing ExchangeRESTProtocol.

    Handles authentication, rate limiting, and request/response translation.

    Example:
        client = WEEXRESTClient()
        await client.initialize()
        balance = await client.get_account_balance()
        await client.close()
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize WEEX REST client.

        Args:
            api_key: WEEX API key (uses config if None)
            api_secret: WEEX API secret (uses config if None)
            api_passphrase: WEEX API passphrase (uses config if None)
            base_url: Base URL for API (uses config if None)
        """
        settings = get_settings()
        self.api_key = api_key or settings.exchange.api_key
        self.api_secret = api_secret or settings.exchange.api_secret
        self.api_passphrase = api_passphrase or settings.exchange.api_passphrase
        self.base_url = (base_url or settings.exchange.rest_url).rstrip("/")

        self._session: aiohttp.ClientSession | None = None
        self._rate_limit_remaining = settings.exchange.rate_limit_requests
        self._rate_limit_reset = 0.0
        self._request_lock = asyncio.Lock()

        # Retry configuration
        self._max_retries = 3
        self._base_delay = 1.0
        self._max_delay = 30.0

        # Symbol cache
        self._symbol_cache: dict[str, SymbolInfo] = {}

    # ==================== Lifecycle ====================

    async def initialize(self) -> None:
        """Initialize HTTP session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"WEEX REST client initialized for {self.base_url}")

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("WEEX REST client closed")

    @property
    def exchange_name(self) -> str:
        """Return the exchange identifier."""
        return "weex"

    @property
    def capabilities(self) -> ExchangeCapabilities:
        """Return exchange capabilities."""
        return WEEXMapper.CAPABILITIES

    # ==================== Authentication ====================

    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        path: str,
        body: str = "",
    ) -> str:
        """
        Generate HMAC-SHA256 signature for request.

        WEEX signature format:
        base64(hmac_sha256(timestamp + method + path + body))
        """
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _get_headers(
        self,
        method: str,
        path: str,
        body: str = "",
    ) -> dict[str, str]:
        """Generate authenticated headers for request."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, path, body)

        return {
            "Content-Type": "application/json",
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.api_passphrase,
            "locale": "en-US",
        }

    # ==================== HTTP Request ====================

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        data: dict | None = None,
        authenticated: bool = True,
    ) -> dict:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API endpoint path
            params: Query parameters
            data: Request body
            authenticated: Whether to include authentication

        Returns:
            Response data as dict

        Raises:
            ExchangeError: On request failure
        """
        if not self._session:
            await self.initialize()

        if self._session is None:
            raise ExchangeError("Session not initialized", exchange=self.exchange_name)
        session = self._session

        url = f"{self.base_url}{path}"
        body = ""

        if data:
            body = json.dumps(data)

        headers = {}
        if authenticated:
            full_path = path
            if params:
                query = "&".join(f"{k}={v}" for k, v in params.items())
                full_path = f"{path}?{query}"
            headers = self._get_headers(method, full_path, body if method != "GET" else "")

        for attempt in range(self._max_retries):
            try:
                async with self._request_lock:
                    # Check rate limit
                    if self._rate_limit_remaining <= 0:
                        wait_time = max(0, self._rate_limit_reset - time.time())
                        if wait_time > 0:
                            logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                            await asyncio.sleep(wait_time)

                    async with session.request(
                        method,
                        url,
                        params=params,
                        data=body if body else None,
                        headers=headers,
                    ) as response:
                        # Update rate limit tracking
                        if "X-RateLimit-Remaining" in response.headers:
                            self._rate_limit_remaining = int(
                                response.headers["X-RateLimit-Remaining"]
                            )
                        if "X-RateLimit-Reset" in response.headers:
                            self._rate_limit_reset = float(response.headers["X-RateLimit-Reset"])

                        response_text = await response.text()

                        if response.status == 429:
                            raise RateLimitError(
                                "Rate limit exceeded",
                                exchange="weex",
                            )

                        if response.status == 401:
                            raise AuthenticationError(
                                "Invalid API credentials",
                                exchange="weex",
                            )

                        if response.status >= 400:
                            logger.error(f"API error: {response.status} - {response_text}")
                            raise ExchangeError(
                                f"API error {response.status}: {response_text}",
                                exchange="weex",
                            )

                        result = json.loads(response_text)

                        # WEEX wraps responses in {"code": "00000", "data": {...}}
                        if isinstance(result, dict):
                            if result.get("code") != "00000":
                                error_msg = result.get("msg", "Unknown error")
                                error_code = result.get("code", "")
                                self._handle_error_code(error_code, error_msg)
                            return result.get("data", result)

                        return result

            except (TimeoutError, aiohttp.ClientError) as e:
                delay = min(self._base_delay * (2**attempt), self._max_delay)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self._max_retries}): {e}, "
                    f"retrying in {delay:.1f}s"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise ExchangeError(
                        f"Request failed after {self._max_retries} attempts: {e}",
                        exchange="weex",
                    )

            except RateLimitError:
                delay = min(self._base_delay * (2**attempt), self._max_delay)
                logger.warning(f"Rate limited, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                if attempt >= self._max_retries - 1:
                    raise

        raise ExchangeError("Request failed: max retries exceeded", exchange="weex")

    def _handle_error_code(self, code: str, message: str) -> None:
        """Translate WEEX error codes to appropriate exceptions."""
        # Map common WEEX error codes
        if code in ("40001", "40002", "40003"):
            raise AuthenticationError(message, exchange="weex", code=code)
        elif code in ("40007", "40008"):
            raise RateLimitError(message, exchange="weex")
        elif code in ("40009", "40010"):
            raise InsufficientBalanceError(message, exchange="weex", code=code)
        elif code in ("40011", "40012", "40013"):
            raise InvalidOrderError(message, exchange="weex", code=code)
        elif code == "40014":
            raise OrderNotFoundError(message, exchange="weex", code=code)
        else:
            raise ExchangeError(message, exchange="weex", code=code)

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> UnifiedTicker:
        """Get ticker information for a symbol."""
        weex_symbol = WEEXMapper.to_weex_symbol(symbol)
        path = "/api/mix/v1/market/ticker"
        params = {"symbol": weex_symbol}

        data = await self._request("GET", path, params=params, authenticated=False)
        return WEEXMapper.to_unified_ticker(data, symbol)

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> UnifiedOrderbook:
        """Get orderbook depth for a symbol."""
        weex_symbol = WEEXMapper.to_weex_symbol(symbol)
        path = "/api/mix/v1/market/depth"
        params = {
            "symbol": weex_symbol,
            "limit": str(limit),
        }

        data = await self._request("GET", path, params=params, authenticated=False)
        return WEEXMapper.to_unified_orderbook(data, symbol)

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[UnifiedCandle]:
        """Get historical candlestick data."""
        weex_symbol = WEEXMapper.to_weex_symbol(symbol)
        weex_interval = WEEXMapper.to_weex_interval(interval)

        path = "/api/mix/v1/market/candles"
        params: dict[str, Any] = {
            "symbol": weex_symbol,
            "granularity": weex_interval,
            "limit": str(limit),
        }

        if start_time:
            params["startTime"] = str(int(start_time.timestamp() * 1000))
        if end_time:
            params["endTime"] = str(int(end_time.timestamp() * 1000))

        data = await self._request("GET", path, params=params, authenticated=False)

        candles = []
        if isinstance(data, list):
            for candle_data in data:
                candles.append(WEEXMapper.to_unified_candle(candle_data, symbol, interval))

        return candles

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[UnifiedTrade]:
        """Get recent trades for a symbol."""
        weex_symbol = WEEXMapper.to_weex_symbol(symbol)
        path = "/api/mix/v1/market/fills"
        params = {
            "symbol": weex_symbol,
            "limit": str(limit),
        }

        data = await self._request("GET", path, params=params, authenticated=False)

        trades = []
        if isinstance(data, list):
            for trade_data in data:
                trades.append(WEEXMapper.to_unified_trade(trade_data, symbol))

        return trades

    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for perpetual futures."""
        weex_symbol = WEEXMapper.to_weex_symbol(symbol)
        path = "/api/mix/v1/market/current-fundRate"
        params = {"symbol": weex_symbol}

        data = await self._request("GET", path, params=params, authenticated=False)
        return float(data.get("fundingRate", 0))

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get trading rules and constraints for a symbol."""
        # Check cache first
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        weex_symbol = WEEXMapper.to_weex_symbol(symbol)
        path = "/api/mix/v1/market/contracts"
        params = {"productType": "umcbl"}

        data = await self._request("GET", path, params=params, authenticated=False)

        if isinstance(data, list):
            for contract in data:
                if contract.get("symbol") == weex_symbol:
                    info = WEEXMapper.to_symbol_info(contract)
                    self._symbol_cache[symbol] = info
                    return info

        raise ExchangeError(
            f"Symbol {symbol} not found",
            exchange="weex",
        )

    async def get_all_symbols(self) -> list[SymbolInfo]:
        """Get all available trading symbols."""
        path = "/api/mix/v1/market/contracts"
        params = {"productType": "umcbl"}

        data = await self._request("GET", path, params=params, authenticated=False)

        symbols = []
        if isinstance(data, list):
            for contract in data:
                info = WEEXMapper.to_symbol_info(contract)
                self._symbol_cache[info.symbol] = info
                symbols.append(info)

        return symbols

    # ==================== Account Operations ====================

    async def get_account_balance(self) -> UnifiedAccountBalance:
        """Get account balance information."""
        path = "/api/mix/v1/account/accounts"
        params = {"productType": "umcbl"}

        data = await self._request("GET", path, params=params)

        # Find USDT account
        usdt_account = None
        if isinstance(data, list):
            for account in data:
                if account.get("marginCoin") == "USDT":
                    usdt_account = account
                    break

        if not usdt_account:
            usdt_account = data if isinstance(data, dict) else {}

        return WEEXMapper.to_unified_account_balance(usdt_account)

    async def get_positions(
        self,
        symbol: str | None = None,
    ) -> list[UnifiedPosition]:
        """Get all open positions."""
        path = "/api/mix/v1/position/allPosition"
        params: dict[str, Any] = {"productType": "umcbl"}

        if symbol:
            params["symbol"] = WEEXMapper.to_weex_symbol(symbol)

        data = await self._request("GET", path, params=params)

        positions = []
        if isinstance(data, list):
            for pos in data:
                if float(pos.get("total", 0)) > 0:
                    positions.append(WEEXMapper.to_unified_position(pos))

        return positions

    async def get_position(
        self,
        symbol: str,
        side: PositionSide,
    ) -> UnifiedPosition | None:
        """Get a specific position by symbol and side."""
        positions = await self.get_positions(symbol)
        for pos in positions:
            if pos.side == side:
                return pos
        return None

    # ==================== Order Operations ====================

    async def place_order(self, order: UnifiedOrder) -> UnifiedOrderResult:
        """Place a new order."""
        path = "/api/mix/v1/order/placeOrder"
        data = WEEXMapper.from_unified_order(order)

        try:
            result = await self._request("POST", path, data=data)
            return WEEXMapper.to_unified_order_result(result, order)
        except ExchangeError as e:
            raise OrderError(f"Failed to place order: {e}", exchange="weex") from e

    async def cancel_order(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> bool:
        """Cancel an order."""
        path = "/api/mix/v1/order/cancel-order"

        data: dict[str, Any] = {
            "symbol": WEEXMapper.to_weex_symbol(symbol),
            "marginCoin": "USDT",
        }

        if order_id:
            data["orderId"] = order_id
        elif client_order_id:
            data["clientOid"] = client_order_id
        else:
            raise InvalidOrderError(
                "Must provide order_id or client_order_id",
                exchange="weex",
            )

        try:
            await self._request("POST", path, data=data)
            return True
        except ExchangeError as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order(
        self,
        symbol: str,
        order_id: str,
    ) -> UnifiedOrderResult:
        """Get order details by ID."""
        path = "/api/mix/v1/order/detail"
        params = {
            "symbol": WEEXMapper.to_weex_symbol(symbol),
            "orderId": order_id,
        }

        data = await self._request("GET", path, params=params)
        return WEEXMapper.to_unified_order_result(data)

    async def get_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[UnifiedOrderResult]:
        """Get all open orders."""
        path = "/api/mix/v1/order/current"
        params: dict[str, Any] = {"productType": "umcbl"}

        if symbol:
            params["symbol"] = WEEXMapper.to_weex_symbol(symbol)

        data = await self._request("GET", path, params=params)

        orders = []
        if isinstance(data, list):
            for order_data in data:
                orders.append(WEEXMapper.to_unified_order_result(order_data))

        return orders

    # ==================== Leverage Operations ====================

    async def set_leverage(
        self,
        symbol: str,
        leverage: int,
        side: PositionSide | None = None,
    ) -> bool:
        """Set leverage for a symbol."""
        settings = get_settings()
        if leverage > settings.risk.max_leverage:
            logger.warning(
                f"Requested leverage {leverage} exceeds max {settings.risk.max_leverage}, "
                f"capping at max"
            )
            leverage = settings.risk.max_leverage

        path = "/api/mix/v1/account/setLeverage"

        data: dict[str, Any] = {
            "symbol": WEEXMapper.to_weex_symbol(symbol),
            "marginCoin": "USDT",
            "leverage": str(leverage),
        }

        if side:
            data["holdSide"] = WEEXMapper.to_weex_position_side(side)

        try:
            await self._request("POST", path, data=data)
            logger.info(f"Set leverage to {leverage}x for {symbol}")
            return True
        except ExchangeError as e:
            logger.error(f"Failed to set leverage: {e}")
            return False

    async def get_leverage(self, symbol: str) -> int:
        """Get current leverage for a symbol."""
        path = "/api/mix/v1/account/account"
        params = {
            "symbol": WEEXMapper.to_weex_symbol(symbol),
            "marginCoin": "USDT",
        }

        result = await self._request("GET", path, params=params)
        return int(result.get("crossMarginLeverage", 1))

    # ==================== Conditional Orders ====================

    async def place_stop_loss(
        self,
        symbol: str,
        side: OrderSide,
        trigger_price: float,
        size: float,
        position_side: PositionSide | None = None,
    ) -> UnifiedOrderResult:
        """Place a stop-loss order."""
        path = "/api/mix/v1/plan/placePlan"

        data: dict[str, Any] = {
            "symbol": WEEXMapper.to_weex_symbol(symbol),
            "marginCoin": "USDT",
            "side": WEEXMapper.to_weex_order_side(side),
            "orderType": "market",
            "triggerType": "fill_price",
            "triggerPrice": str(trigger_price),
            "size": str(size),
            "executePrice": "0",  # Market order
            "planType": "loss_plan",
        }

        if position_side:
            data["holdSide"] = WEEXMapper.to_weex_position_side(position_side)

        result = await self._request("POST", path, data=data)
        return WEEXMapper.to_unified_order_result(result)

    async def place_take_profit(
        self,
        symbol: str,
        side: OrderSide,
        trigger_price: float,
        size: float,
        position_side: PositionSide | None = None,
    ) -> UnifiedOrderResult:
        """Place a take-profit order."""
        path = "/api/mix/v1/plan/placePlan"

        data: dict[str, Any] = {
            "symbol": WEEXMapper.to_weex_symbol(symbol),
            "marginCoin": "USDT",
            "side": WEEXMapper.to_weex_order_side(side),
            "orderType": "market",
            "triggerType": "fill_price",
            "triggerPrice": str(trigger_price),
            "size": str(size),
            "executePrice": "0",  # Market order
            "planType": "profit_plan",
        }

        if position_side:
            data["holdSide"] = WEEXMapper.to_weex_position_side(position_side)

        result = await self._request("POST", path, data=data)
        return WEEXMapper.to_unified_order_result(result)

    async def cancel_conditional_order(
        self,
        symbol: str,
        order_id: str,
    ) -> bool:
        """Cancel a stop-loss or take-profit order."""
        path = "/api/mix/v1/plan/cancelPlan"

        data = {
            "symbol": WEEXMapper.to_weex_symbol(symbol),
            "marginCoin": "USDT",
            "orderId": order_id,
        }

        try:
            await self._request("POST", path, data=data)
            return True
        except ExchangeError:
            return False


class WEEXAdapter(ExchangeAdapter):
    """
    WEEX Exchange Adapter combining REST and WebSocket clients.

    Example:
        adapter = WEEXAdapter()
        await adapter.initialize()

        # Use REST API
        balance = await adapter.rest.get_account_balance()

        # Use WebSocket
        await adapter.websocket.connect()
        await adapter.websocket.subscribe_candles(["BTCUSDT"])

        await adapter.close()
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        rest_url: str | None = None,
        ws_url: str | None = None,
    ):
        """
        Initialize WEEX adapter.

        Args:
            api_key: API key (uses config if None)
            api_secret: API secret (uses config if None)
            api_passphrase: API passphrase (uses config if None)
            rest_url: REST API URL (uses config if None)
            ws_url: WebSocket URL (uses config if None)
        """
        self._rest = WEEXRESTClient(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            base_url=rest_url,
        )

        # Lazy import to avoid circular imports
        from src.exchange.adapters.weex.websocket import WEEXWebSocket

        self._websocket: WEEXWebSocket = WEEXWebSocket(
            ws_url=ws_url,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )

    @property
    def rest(self) -> ExchangeRESTProtocol:
        """Get the REST API client."""
        return self._rest

    @property
    def websocket(self) -> ExchangeWebSocketProtocol:
        """Get the WebSocket client."""
        return self._websocket

    def normalize_symbol(self, symbol: str) -> str:
        """
        Convert unified symbol to WEEX format.

        BTCUSDT -> cmt_btcusdt
        """
        return WEEXMapper.to_weex_symbol(symbol)

    def denormalize_symbol(self, exchange_symbol: str) -> str:
        """
        Convert WEEX symbol to unified format.

        cmt_btcusdt -> BTCUSDT
        """
        return WEEXMapper.from_weex_symbol(exchange_symbol)
