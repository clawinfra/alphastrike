"""
AlphaStrike Trading Bot - WEEX REST Client

Async HTTP client for WEEX exchange API interactions.
Handles account queries, order management, and leverage setting.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    LIMIT = "limit"
    MARKET = "market"


class PositionSide(str, Enum):
    """Position side for perpetual futures."""
    LONG = "long"
    SHORT = "short"


class TimeInForce(str, Enum):
    """Time in force options."""
    GTC = "normal"  # Good Till Cancelled
    IOC = "ioc"     # Immediate or Cancel
    FOK = "fok"     # Fill or Kill
    POST_ONLY = "post_only"


@dataclass
class OrderRequest:
    """Request to place an order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: float | None = None
    position_side: PositionSide | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str | None = None
    reduce_only: bool = False
    preset_stop_loss_price: float | None = None
    preset_take_profit_price: float | None = None


@dataclass
class OrderResult:
    """Result from placing an order."""
    order_id: str
    client_order_id: str | None
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: float | None
    status: str
    filled_size: float = 0.0
    avg_fill_price: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: dict = field(default_factory=dict)


@dataclass
class Position:
    """Current position information."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: float | None = None
    margin_mode: str = "cross"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccountBalance:
    """Account balance information."""
    total_balance: float
    available_balance: float
    margin_balance: float
    unrealized_pnl: float
    currency: str = "USDT"
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RESTClientError(Exception):
    """Base exception for REST client errors."""
    pass


class RateLimitError(RESTClientError):
    """Rate limit exceeded."""
    pass


class AuthenticationError(RESTClientError):
    """Authentication failed."""
    pass


class OrderError(RESTClientError):
    """Order placement failed."""
    pass


class RESTClient:
    """
    Async REST client for WEEX exchange API.

    Handles authentication, rate limiting, and retry logic.

    Usage:
        client = RESTClient()
        await client.initialize()
        balance = await client.get_account_balance()
        result = await client.place_order(order_request)
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
        Initialize REST client.

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

    async def initialize(self) -> None:
        """Initialize HTTP session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"REST client initialized for {self.base_url}")

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("REST client closed")

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
            RESTClientError: On request failure
        """
        if not self._session:
            await self.initialize()

        assert self._session is not None, "Session not initialized"
        session = self._session  # Local reference for type narrowing

        url = f"{self.base_url}{path}"
        body = ""

        if data:
            import json
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
                            self._rate_limit_reset = float(
                                response.headers["X-RateLimit-Reset"]
                            )

                        response_text = await response.text()

                        if response.status == 429:
                            raise RateLimitError("Rate limit exceeded")

                        if response.status == 401:
                            raise AuthenticationError("Invalid API credentials")

                        if response.status >= 400:
                            logger.error(
                                f"API error: {response.status} - {response_text}"
                            )
                            raise RESTClientError(
                                f"API error {response.status}: {response_text}"
                            )

                        import json
                        result = json.loads(response_text)

                        # WEEX wraps responses in {"code": "00000", "data": {...}}
                        if isinstance(result, dict):
                            if result.get("code") != "00000":
                                error_msg = result.get("msg", "Unknown error")
                                raise RESTClientError(f"API error: {error_msg}")
                            return result.get("data", result)

                        return result

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self._max_retries}): {e}, "
                    f"retrying in {delay:.1f}s"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise RESTClientError(f"Request failed after {self._max_retries} attempts: {e}")

            except RateLimitError:
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                logger.warning(f"Rate limited, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                if attempt >= self._max_retries - 1:
                    raise

        raise RESTClientError("Request failed: max retries exceeded")

    # ==================== Account Operations ====================

    async def get_account_balance(self) -> AccountBalance:
        """
        Get account balance information.

        Returns:
            AccountBalance with total, available, and margin balance
        """
        path = "/api/mix/v1/account/accounts"
        params = {"productType": "umcbl"}  # USDT-margined contracts

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

        return AccountBalance(
            total_balance=float(usdt_account.get("usdtEquity", 0)),
            available_balance=float(usdt_account.get("available", 0)),
            margin_balance=float(usdt_account.get("crossMaxAvailable", 0)),
            unrealized_pnl=float(usdt_account.get("unrealizedPL", 0)),
            currency="USDT",
        )

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of Position objects
        """
        path = "/api/mix/v1/position/allPosition"
        params: dict[str, Any] = {"productType": "umcbl"}

        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", path, params=params)

        positions = []
        if isinstance(data, list):
            for pos in data:
                if float(pos.get("total", 0)) > 0:
                    positions.append(
                        Position(
                            symbol=pos.get("symbol", ""),
                            side=PositionSide.LONG if pos.get("holdSide") == "long" else PositionSide.SHORT,
                            size=float(pos.get("total", 0)),
                            entry_price=float(pos.get("averageOpenPrice", 0)),
                            mark_price=float(pos.get("markPrice", 0)),
                            unrealized_pnl=float(pos.get("unrealizedPL", 0)),
                            leverage=int(pos.get("leverage", 1)),
                            liquidation_price=float(pos.get("liquidationPrice", 0)) if pos.get("liquidationPrice") else None,
                            margin_mode=pos.get("marginMode", "cross"),
                        )
                    )

        return positions

    async def get_position(self, symbol: str, side: PositionSide) -> Position | None:
        """
        Get specific position for symbol and side.

        Args:
            symbol: Trading pair symbol
            side: Position side (long/short)

        Returns:
            Position if exists, None otherwise
        """
        positions = await self.get_positions(symbol)
        for pos in positions:
            if pos.side == side:
                return pos
        return None

    # ==================== Order Operations ====================

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place a new order.

        Args:
            request: OrderRequest with order details

        Returns:
            OrderResult with order information

        Raises:
            OrderError: If order placement fails
        """
        path = "/api/mix/v1/order/placeOrder"

        data: dict[str, Any] = {
            "symbol": request.symbol,
            "marginCoin": "USDT",
            "side": request.side.value,
            "orderType": request.order_type.value,
            "size": str(request.size),
            "timeInForceValue": request.time_in_force.value,
        }

        if request.price is not None:
            data["price"] = str(request.price)

        if request.position_side:
            data["holdSide"] = request.position_side.value

        if request.client_order_id:
            data["clientOid"] = request.client_order_id

        if request.reduce_only:
            data["reduceOnly"] = "true"

        if request.preset_stop_loss_price:
            data["presetStopLossPrice"] = str(request.preset_stop_loss_price)

        if request.preset_take_profit_price:
            data["presetTakeProfitPrice"] = str(request.preset_take_profit_price)

        try:
            result = await self._request("POST", path, data=data)

            return OrderResult(
                order_id=result.get("orderId", ""),
                client_order_id=result.get("clientOid"),
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                size=request.size,
                price=request.price,
                status="submitted",
                raw_response=result,
            )

        except RESTClientError as e:
            raise OrderError(f"Failed to place order: {e}")

    async def cancel_order(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            True if cancellation successful
        """
        path = "/api/mix/v1/order/cancel-order"

        data: dict[str, Any] = {
            "symbol": symbol,
            "marginCoin": "USDT",
        }

        if order_id:
            data["orderId"] = order_id
        elif client_order_id:
            data["clientOid"] = client_order_id
        else:
            raise OrderError("Must provide order_id or client_order_id")

        try:
            await self._request("POST", path, data=data)
            return True
        except RESTClientError as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order(
        self,
        symbol: str,
        order_id: str,
    ) -> dict:
        """
        Get order details.

        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID

        Returns:
            Order details dict
        """
        path = "/api/mix/v1/order/detail"
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }

        return await self._request("GET", path, params=params)

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open order dicts
        """
        path = "/api/mix/v1/order/current"
        params: dict[str, Any] = {"productType": "umcbl"}

        if symbol:
            params["symbol"] = symbol

        result = await self._request("GET", path, params=params)
        return result if isinstance(result, list) else []

    # ==================== Leverage Operations ====================

    async def set_leverage(
        self,
        symbol: str,
        leverage: int,
        position_side: PositionSide | None = None,
    ) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair symbol
            leverage: Leverage value (1-125)
            position_side: Optional position side for hedge mode

        Returns:
            True if successful
        """
        settings = get_settings()
        if leverage > settings.risk.max_leverage:
            logger.warning(
                f"Requested leverage {leverage} exceeds max {settings.risk.max_leverage}, "
                f"capping at max"
            )
            leverage = settings.risk.max_leverage

        path = "/api/mix/v1/account/setLeverage"

        data: dict[str, Any] = {
            "symbol": symbol,
            "marginCoin": "USDT",
            "leverage": str(leverage),
        }

        if position_side:
            data["holdSide"] = position_side.value

        try:
            await self._request("POST", path, data=data)
            logger.info(f"Set leverage to {leverage}x for {symbol}")
            return True
        except RESTClientError as e:
            logger.error(f"Failed to set leverage: {e}")
            return False

    async def get_leverage(self, symbol: str) -> int:
        """
        Get current leverage for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current leverage value
        """
        path = "/api/mix/v1/account/account"
        params = {
            "symbol": symbol,
            "marginCoin": "USDT",
        }

        result = await self._request("GET", path, params=params)
        return int(result.get("crossMarginLeverage", 1))

    # ==================== Stop Orders ====================

    async def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        trigger_price: float,
        size: float,
        position_side: PositionSide | None = None,
    ) -> dict:
        """
        Place a stop-loss order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy to close short, sell to close long)
            trigger_price: Price to trigger the stop
            size: Position size to close
            position_side: Position side for hedge mode

        Returns:
            Order result dict
        """
        path = "/api/mix/v1/plan/placePlan"

        data: dict[str, Any] = {
            "symbol": symbol,
            "marginCoin": "USDT",
            "side": side.value,
            "orderType": "market",
            "triggerType": "fill_price",
            "triggerPrice": str(trigger_price),
            "size": str(size),
            "executePrice": "0",  # Market order
            "planType": "loss_plan",
        }

        if position_side:
            data["holdSide"] = position_side.value

        return await self._request("POST", path, data=data)

    async def place_take_profit_order(
        self,
        symbol: str,
        side: OrderSide,
        trigger_price: float,
        size: float,
        position_side: PositionSide | None = None,
    ) -> dict:
        """
        Place a take-profit order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy to close short, sell to close long)
            trigger_price: Price to trigger the take profit
            size: Position size to close
            position_side: Position side for hedge mode

        Returns:
            Order result dict
        """
        path = "/api/mix/v1/plan/placePlan"

        data: dict[str, Any] = {
            "symbol": symbol,
            "marginCoin": "USDT",
            "side": side.value,
            "orderType": "market",
            "triggerType": "fill_price",
            "triggerPrice": str(trigger_price),
            "size": str(size),
            "executePrice": "0",  # Market order
            "planType": "profit_plan",
        }

        if position_side:
            data["holdSide"] = position_side.value

        return await self._request("POST", path, data=data)

    async def cancel_plan_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel a stop/TP order.

        Args:
            symbol: Trading pair symbol
            order_id: Plan order ID

        Returns:
            True if cancellation successful
        """
        path = "/api/mix/v1/plan/cancelPlan"

        data = {
            "symbol": symbol,
            "marginCoin": "USDT",
            "orderId": order_id,
        }

        try:
            await self._request("POST", path, data=data)
            return True
        except RESTClientError:
            return False

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> dict:
        """
        Get ticker information.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dict
        """
        path = "/api/mix/v1/market/ticker"
        params = {"symbol": symbol}

        return await self._request("GET", path, params=params, authenticated=False)

    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """
        Get orderbook depth.

        Args:
            symbol: Trading pair symbol
            limit: Depth limit (default 20)

        Returns:
            Orderbook data with bids and asks
        """
        path = "/api/mix/v1/market/depth"
        params = {
            "symbol": symbol,
            "limit": str(limit),
        }

        return await self._request("GET", path, params=params, authenticated=False)

    async def get_funding_rate(self, symbol: str) -> float:
        """
        Get current funding rate.

        Args:
            symbol: Trading pair symbol

        Returns:
            Funding rate as float
        """
        path = "/api/mix/v1/market/current-fundRate"
        params = {"symbol": symbol}

        result = await self._request("GET", path, params=params, authenticated=False)
        return float(result.get("fundingRate", 0))

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict]:
        """
        Get historical candles.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval (1m, 5m, 15m, 1H, 4H, 1D)
            limit: Number of candles (max 1000)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms

        Returns:
            List of candle dicts
        """
        path = "/api/mix/v1/market/candles"
        params: dict[str, Any] = {
            "symbol": symbol,
            "granularity": interval,
            "limit": str(limit),
        }

        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)

        result = await self._request("GET", path, params=params, authenticated=False)
        return result if isinstance(result, list) else []


# Module-level singleton
_client_instance: RESTClient | None = None


async def get_rest_client() -> RESTClient:
    """
    Get the REST client singleton instance.

    Initializes on first call.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = RESTClient()
        await _client_instance.initialize()
    return _client_instance


async def close_rest_client() -> None:
    """Close the REST client singleton."""
    global _client_instance
    if _client_instance:
        await _client_instance.close()
        _client_instance = None
