"""
Hyperliquid DEX Exchange Adapter

Implements ExchangeAdapter protocol for Hyperliquid decentralized exchange.
Supports multi-asset trading across crypto, commodities, forex, indices, and stocks.

Environment Variables:
    EXCHANGE_WALLET_PRIVATE_KEY: Ethereum private key (hex with 0x prefix)
    EXCHANGE_WALLET_ADDRESS: Wallet address (optional, derived from key if empty)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import aiohttp

from src.core.config import get_settings
from src.exchange.adapters.hyperliquid.auth import HyperliquidSigner
from src.exchange.adapters.hyperliquid.mappers import HyperliquidMapper
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
    OrderType,
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
    from src.exchange.adapters.hyperliquid.websocket import HyperliquidWebSocket

logger = logging.getLogger(__name__)

# Default URLs
MAINNET_URL = "https://api.hyperliquid.xyz"
TESTNET_URL = "https://api.hyperliquid-testnet.xyz"
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"


class HyperliquidRESTClient(ExchangeRESTProtocol):
    """
    Hyperliquid REST API client implementing ExchangeRESTProtocol.

    Handles wallet-based authentication, request formatting, and response parsing.
    Uses /info endpoint for queries and /exchange endpoint for trading.

    Example:
        client = HyperliquidRESTClient(
            private_key="0x...",
            wallet_address="0x...",
        )
        await client.initialize()
        balance = await client.get_account_balance()
        await client.close()
    """

    def __init__(
        self,
        private_key: str | None = None,
        wallet_address: str | None = None,
        base_url: str | None = None,
        testnet: bool = False,
    ):
        """
        Initialize Hyperliquid REST client.

        Credentials are loaded from environment variables if not provided:
        - EXCHANGE_WALLET_PRIVATE_KEY: Wallet private key
        - EXCHANGE_WALLET_ADDRESS: Wallet address (optional)

        Args:
            private_key: Ethereum private key for signing (hex with 0x prefix)
            wallet_address: Ethereum wallet address
            base_url: Base URL for API (uses default if None)
            testnet: Whether to use testnet
        """
        settings = get_settings()

        # Load credentials from config if not provided
        # Only load wallet_address from settings if private_key is also from settings
        # This ensures custom private_key will derive its own address
        if private_key is None:
            private_key = settings.exchange.wallet_private_key or None
            if wallet_address is None:
                wallet_address = settings.exchange.wallet_address or None
        # If private_key was provided explicitly, wallet_address should be derived
        # from it (handled by HyperliquidSigner), not loaded from settings

        self.testnet = testnet
        self.base_url = base_url or (TESTNET_URL if testnet else MAINNET_URL)

        # Initialize signer
        self._signer = HyperliquidSigner(
            private_key=private_key,
            wallet_address=wallet_address,
            testnet=testnet,
        )

        self._session: aiohttp.ClientSession | None = None
        self._request_lock = asyncio.Lock()

        # Rate limiting
        self._rate_limit_remaining = 20
        self._last_request_time = 0.0
        self._min_request_interval = 0.05  # 50ms between requests

        # Retry configuration
        self._max_retries = 3
        self._base_delay = 1.0
        self._max_delay = 30.0

        # Symbol cache
        self._symbol_cache: dict[str, SymbolInfo] = {}
        self._metadata_loaded = False

    # ==================== Lifecycle ====================

    async def initialize(self) -> None:
        """Initialize HTTP session and load metadata."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"Hyperliquid REST client initialized for {self.base_url}")

        # Load asset metadata
        if not self._metadata_loaded:
            await self._load_metadata()

    async def _load_metadata(self) -> None:
        """Load asset metadata from Hyperliquid."""
        try:
            meta = await self._info_request({"type": "meta"})
            universe = meta.get("universe", [])
            HyperliquidMapper.set_asset_meta(universe)
            self._metadata_loaded = True
            logger.info(f"Loaded metadata for {len(universe)} assets")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("Hyperliquid REST client closed")

    @property
    def exchange_name(self) -> str:
        """Return the exchange identifier."""
        return "hyperliquid"

    @property
    def capabilities(self) -> ExchangeCapabilities:
        """Return exchange capabilities."""
        return HyperliquidMapper.CAPABILITIES

    @property
    def wallet_address(self) -> str | None:
        """Get the configured wallet address."""
        return self._signer.wallet_address

    # ==================== HTTP Request ====================

    async def _info_request(self, data: dict[str, Any]) -> Any:
        """
        Make request to /info endpoint.

        Args:
            data: Request body with 'type' field

        Returns:
            Response data
        """
        return await self._request("POST", "/info", data=data, authenticated=False)

    async def _exchange_request(self, data: dict[str, Any]) -> Any:
        """
        Make authenticated request to /exchange endpoint.

        Args:
            data: Signed request body

        Returns:
            Response data
        """
        return await self._request("POST", "/exchange", data=data, authenticated=True)

    async def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        authenticated: bool = False,
    ) -> Any:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method
            path: API endpoint path
            data: Request body
            authenticated: Whether this is an authenticated request

        Returns:
            Response data

        Raises:
            ExchangeError: On request failure
        """
        if not self._session:
            await self.initialize()

        if self._session is None:
            raise ExchangeError("Session not initialized", exchange=self.exchange_name)
        session = self._session

        url = f"{self.base_url}{path}"

        headers = {"Content-Type": "application/json"}

        for attempt in range(self._max_retries):
            try:
                async with self._request_lock:
                    # Rate limiting
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self._min_request_interval:
                        await asyncio.sleep(self._min_request_interval - elapsed)

                    self._last_request_time = time.time()

                    async with session.request(
                        method,
                        url,
                        json=data,
                        headers=headers,
                    ) as response:
                        response_text = await response.text()

                        if response.status == 429:
                            raise RateLimitError(
                                "Rate limit exceeded",
                                exchange=self.exchange_name,
                            )

                        if response.status == 401:
                            raise AuthenticationError(
                                "Invalid credentials",
                                exchange=self.exchange_name,
                            )

                        if response.status >= 400:
                            logger.error(f"API error: {response.status} - {response_text}")
                            self._handle_error(response_text)

                        result = json.loads(response_text)

                        # Check for error in response body
                        if isinstance(result, dict) and result.get("status") == "err":
                            error_msg = result.get("response", "Unknown error")
                            raise ExchangeError(
                                f"API error: {error_msg}",
                                exchange=self.exchange_name,
                            )

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
                        exchange=self.exchange_name,
                    )

            except RateLimitError:
                delay = min(self._base_delay * (2**attempt), self._max_delay)
                logger.warning(f"Rate limited, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                if attempt >= self._max_retries - 1:
                    raise

        raise ExchangeError("Request failed: max retries exceeded", exchange=self.exchange_name)

    def _handle_error(self, error_text: str) -> None:
        """Parse and raise appropriate exception for error response."""
        try:
            error_data = json.loads(error_text)
            message = str(error_data)
        except json.JSONDecodeError:
            message = error_text

        lower_msg = message.lower()

        if "insufficient" in lower_msg or "balance" in lower_msg:
            raise InsufficientBalanceError(message, exchange=self.exchange_name)
        elif "not found" in lower_msg or "unknown" in lower_msg:
            raise OrderNotFoundError(message, exchange=self.exchange_name)
        elif "invalid" in lower_msg:
            raise InvalidOrderError(message, exchange=self.exchange_name)
        else:
            raise ExchangeError(message, exchange=self.exchange_name)

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> UnifiedTicker:
        """Get ticker information for a symbol."""
        coin = HyperliquidMapper.to_hyperliquid_coin(symbol)
        data = await self._info_request({"type": "allMids"})

        return HyperliquidMapper.to_unified_ticker(data, coin)

    async def get_all_tickers(self) -> dict[str, UnifiedTicker]:
        """Get all mid prices."""
        data = await self._info_request({"type": "allMids"})

        tickers = {}
        for coin, _price in data.items():
            symbol = HyperliquidMapper.from_hyperliquid_coin(coin)
            tickers[symbol] = HyperliquidMapper.to_unified_ticker(data, coin)

        return tickers

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> UnifiedOrderbook:
        """Get orderbook depth for a symbol."""
        coin = HyperliquidMapper.to_hyperliquid_coin(symbol)

        data = await self._info_request({
            "type": "l2Book",
            "coin": coin,
            "nSigFigs": 5,
        })

        return HyperliquidMapper.to_unified_orderbook(data, symbol)

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[UnifiedCandle]:
        """
        Get historical candlestick data with pagination support.

        Hyperliquid API may limit candles per request, so this method
        fetches in batches and concatenates results.
        """
        coin = HyperliquidMapper.to_hyperliquid_coin(symbol)
        hl_interval = HyperliquidMapper.to_hyperliquid_interval(interval)

        # Interval to milliseconds mapping
        interval_ms = {
            "1m": 60000, "5m": 300000, "15m": 900000, "30m": 1800000,
            "1h": 3600000, "4h": 14400000, "1d": 86400000,
        }
        interval_duration = interval_ms.get(interval, 3600000)

        # Default time range
        now_ms = int(time.time() * 1000)
        end_ms = int(end_time.timestamp() * 1000) if end_time else now_ms
        start_ms = int(start_time.timestamp() * 1000) if start_time else end_ms - (limit * interval_duration)

        # Fetch in batches (Hyperliquid typically returns ~500 candles max)
        batch_size = 500
        all_candles: list[UnifiedCandle] = []
        current_end = end_ms

        while len(all_candles) < limit and current_end > start_ms:
            # Calculate batch start
            batch_start = max(start_ms, current_end - (batch_size * interval_duration))

            data = await self._info_request({
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": hl_interval,
                    "startTime": batch_start,
                    "endTime": current_end,
                },
            })

            if not isinstance(data, list) or not data:
                break

            # Convert and prepend (older candles first)
            batch_candles = []
            for candle_data in data:
                batch_candles.append(
                    HyperliquidMapper.to_unified_candle(candle_data, symbol, interval)
                )

            # Prepend to maintain chronological order
            all_candles = batch_candles + all_candles

            # Move window back for next batch
            if batch_candles:
                # Use the oldest candle timestamp for next batch end
                oldest_ts = int(batch_candles[0].timestamp.timestamp() * 1000)
                current_end = oldest_ts - interval_duration
            else:
                break

            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)

        # Trim to requested limit (most recent candles)
        if len(all_candles) > limit:
            all_candles = all_candles[-limit:]

        return all_candles

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[UnifiedTrade]:
        """Get recent trades for a symbol."""
        # Hyperliquid doesn't have a direct recent trades endpoint
        # This would require WebSocket subscription for real-time trades
        logger.warning("get_recent_trades not fully supported for Hyperliquid")
        return []

    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for perpetual futures."""
        # Funding rate is embedded in meta/contexts
        # For now, return 0 - this can be enhanced
        logger.debug("Funding rate query - check meta response for details")
        return 0.0

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get trading rules and constraints for a symbol."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        # Ensure metadata is loaded
        if not self._metadata_loaded:
            await self._load_metadata()

        coin = HyperliquidMapper.to_hyperliquid_coin(symbol)

        if coin in HyperliquidMapper._asset_meta:
            meta = HyperliquidMapper._asset_meta[coin]
            index = HyperliquidMapper._coin_to_index[coin]
            info = HyperliquidMapper.to_symbol_info(meta, index)
            self._symbol_cache[symbol] = info
            return info

        raise ExchangeError(f"Symbol {symbol} not found", exchange=self.exchange_name)

    async def get_all_symbols(self) -> list[SymbolInfo]:
        """Get all available trading symbols."""
        if not self._metadata_loaded:
            await self._load_metadata()

        symbols = []
        for coin, meta in HyperliquidMapper._asset_meta.items():
            index = HyperliquidMapper._coin_to_index.get(coin, 0)
            info = HyperliquidMapper.to_symbol_info(meta, index)
            self._symbol_cache[info.symbol] = info
            symbols.append(info)

        return symbols

    # ==================== Account Operations ====================

    async def get_account_balance(self) -> UnifiedAccountBalance:
        """Get account balance information.

        Fetches both spot and perpetual balances:
        - Spot USDC: The actual tradable balance (used as margin for perps)
        - Perps margin: Amount currently used in perpetual positions
        """
        if not self._signer.wallet_address:
            raise AuthenticationError(
                "Wallet address required for account operations",
                exchange=self.exchange_name,
            )

        # Fetch spot balance (actual USDC holdings)
        spot_data = await self._info_request({
            "type": "spotClearinghouseState",
            "user": self._signer.wallet_address,
        })

        # Fetch perps state (for unrealized PnL)
        perps_data = await self._info_request({
            "type": "clearinghouseState",
            "user": self._signer.wallet_address,
        })

        return HyperliquidMapper.to_unified_account_balance(spot_data, perps_data)

    async def get_positions(
        self,
        symbol: str | None = None,
    ) -> list[UnifiedPosition]:
        """Get all open positions."""
        if not self._signer.wallet_address:
            raise AuthenticationError(
                "Wallet address required",
                exchange=self.exchange_name,
            )

        data = await self._info_request({
            "type": "clearinghouseState",
            "user": self._signer.wallet_address,
        })

        positions = []
        for asset_pos in data.get("assetPositions", []):
            pos = HyperliquidMapper.to_unified_position(asset_pos)

            # Filter by symbol if specified
            if symbol and pos.symbol != symbol:
                continue

            # Only include positions with non-zero size
            if pos.quantity > 0:
                positions.append(pos)

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
        if not self._signer.wallet_address:
            raise AuthenticationError(
                "Private key required for trading",
                exchange=self.exchange_name,
            )

        # Convert to Hyperliquid wire format
        order_wire = HyperliquidMapper.from_unified_order(order)

        # Sign the order
        signed_request = self._signer.sign_order_action(
            orders=[order_wire],
            grouping="na",
        )

        try:
            result = await self._exchange_request(signed_request)
            return HyperliquidMapper.to_unified_order_result(result, order)
        except ExchangeError as e:
            raise OrderError(f"Failed to place order: {e}", exchange=self.exchange_name) from e

    async def cancel_order(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> bool:
        """Cancel an order."""
        if not self._signer.wallet_address:
            raise AuthenticationError(
                "Private key required",
                exchange=self.exchange_name,
            )

        if not order_id and not client_order_id:
            raise InvalidOrderError(
                "Must provide order_id or client_order_id",
                exchange=self.exchange_name,
            )

        asset_index = HyperliquidMapper.get_asset_index(symbol)

        if order_id:
            signed_request = self._signer.sign_cancel_action(
                cancels=[{"a": asset_index, "o": int(order_id)}]
            )
        else:
            # Cancel by client order ID requires different action type
            signed_request = {
                "action": {
                    "type": "cancelByCloid",
                    "cancels": [{"asset": asset_index, "cloid": client_order_id}],
                },
                "nonce": int(time.time() * 1000),
                "signature": self._signer._sign_agent(
                    {"type": "cancelByCloid"}, int(time.time() * 1000)
                ),
            }

        try:
            result = await self._exchange_request(signed_request)
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            return len(statuses) > 0 and statuses[0] == "success"
        except ExchangeError as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order(
        self,
        symbol: str,
        order_id: str,
    ) -> UnifiedOrderResult:
        """Get order details by ID."""
        if not self._signer.wallet_address:
            raise AuthenticationError(
                "Wallet address required",
                exchange=self.exchange_name,
            )

        data = await self._info_request({
            "type": "orderStatus",
            "user": self._signer.wallet_address,
            "oid": int(order_id),
        })

        if data.get("status") == "unknownOid":
            raise OrderNotFoundError(
                f"Order {order_id} not found",
                exchange=self.exchange_name,
            )

        order_info = data.get("order", {})
        order_data = order_info.get("order", {})

        # Parse order type from response
        order_type_str = order_data.get("orderType", "Limit").lower()
        order_type = OrderType.LIMIT if order_type_str == "limit" else OrderType.MARKET

        return UnifiedOrderResult(
            order_id=str(order_data.get("oid", order_id)),
            client_order_id=order_data.get("cloid"),
            symbol=symbol,
            side=HyperliquidMapper.from_hyperliquid_side(order_data.get("side", "B")),
            order_type=order_type,
            quantity=float(order_data.get("origSz", 0)),
            price=float(order_data.get("limitPx", 0)),
            status=HyperliquidMapper.from_hyperliquid_order_status(
                order_info.get("status", "open")
            ),
            filled_quantity=float(order_data.get("origSz", 0)) - float(order_data.get("sz", 0)),
            average_price=None,
            commission=0,
            commission_asset="USDC",
            timestamp=datetime.fromtimestamp(order_data.get("timestamp", 0) / 1000),
            raw_response=data,
        )

    async def get_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[UnifiedOrderResult]:
        """Get all open orders."""
        if not self._signer.wallet_address:
            raise AuthenticationError(
                "Wallet address required",
                exchange=self.exchange_name,
            )

        data = await self._info_request({
            "type": "openOrders",
            "user": self._signer.wallet_address,
        })

        orders = []
        for order_data in data:
            coin = order_data.get("coin", "")
            order_symbol = HyperliquidMapper.from_hyperliquid_coin(coin)

            # Filter by symbol if specified
            if symbol and order_symbol != symbol:
                continue

            orders.append(UnifiedOrderResult(
                order_id=str(order_data.get("oid", "")),
                client_order_id=order_data.get("cloid"),
                symbol=order_symbol,
                side=HyperliquidMapper.from_hyperliquid_side(order_data.get("side", "B")),
                order_type=OrderType.LIMIT,
                quantity=float(order_data.get("origSz", order_data.get("sz", 0))),
                price=float(order_data.get("limitPx", 0)),
                status=HyperliquidMapper.from_hyperliquid_order_status("open"),
                filled_quantity=0,
                average_price=None,
                commission=0,
                commission_asset="USDC",
                timestamp=datetime.fromtimestamp(order_data.get("timestamp", 0) / 1000),
                raw_response=order_data,
            ))

        return orders

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
        order = UnifiedOrder(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=size,
            price=trigger_price,  # Use as limit price
            stop_loss_price=trigger_price,
            reduce_only=True,
            position_side=position_side,
        )

        return await self.place_order(order)

    async def place_take_profit(
        self,
        symbol: str,
        side: OrderSide,
        trigger_price: float,
        size: float,
        position_side: PositionSide | None = None,
    ) -> UnifiedOrderResult:
        """Place a take-profit order."""
        order = UnifiedOrder(
            symbol=symbol,
            side=side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=size,
            price=trigger_price,
            take_profit_price=trigger_price,
            reduce_only=True,
            position_side=position_side,
        )

        return await self.place_order(order)

    async def cancel_conditional_order(
        self,
        symbol: str,
        order_id: str,
    ) -> bool:
        """Cancel a stop-loss or take-profit order."""
        return await self.cancel_order(symbol, order_id=order_id)

    # ==================== Leverage Operations ====================

    async def set_leverage(
        self,
        symbol: str,
        leverage: int,
        side: PositionSide | None = None,
    ) -> bool:
        """Set leverage for a symbol."""
        if not self._signer.wallet_address:
            raise AuthenticationError(
                "Private key required",
                exchange=self.exchange_name,
            )

        asset_index = HyperliquidMapper.get_asset_index(symbol)

        signed_request = self._signer.sign_update_leverage(
            asset=asset_index,
            is_cross=True,
            leverage=leverage,
        )

        try:
            await self._exchange_request(signed_request)
            logger.info(f"Set leverage to {leverage}x for {symbol}")
            return True
        except ExchangeError as e:
            logger.error(f"Failed to set leverage: {e}")
            return False

    async def get_leverage(self, symbol: str) -> int:
        """Get current leverage for a symbol."""
        positions = await self.get_positions(symbol)

        if positions:
            return positions[0].leverage

        return 1  # Default leverage

    # ==================== User Trades ====================

    async def get_user_trades(
        self,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[UnifiedTrade]:
        """Get user's trade history."""
        if not self._signer.wallet_address:
            raise AuthenticationError(
                "Wallet address required",
                exchange=self.exchange_name,
            )

        if start_time:
            data = await self._info_request({
                "type": "userFillsByTime",
                "user": self._signer.wallet_address,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000) if end_time else int(time.time() * 1000),
            })
        else:
            data = await self._info_request({
                "type": "userFills",
                "user": self._signer.wallet_address,
            })

        trades = []
        for fill in data[-limit:]:
            trade = HyperliquidMapper.to_unified_trade(fill)

            # Filter by symbol if specified
            if symbol and trade.symbol != symbol:
                continue

            trades.append(trade)

        return trades


class HyperliquidAdapter(ExchangeAdapter):
    """
    Hyperliquid Exchange Adapter combining REST and WebSocket clients.

    Supports multi-asset trading across:
    - Crypto perpetuals (BTC, ETH, SOL, etc.)
    - Commodities (GOLD, SILVER)
    - Forex (EUR, GBP, JPY)
    - Indices (USA500, HK50)
    - Stocks (TSLA, NVDA, AAPL)

    Example:
        adapter = HyperliquidAdapter(
            private_key="0x...",
            wallet_address="0x...",
        )
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
        private_key: str | None = None,
        wallet_address: str | None = None,
        base_url: str | None = None,
        ws_url: str | None = None,
        testnet: bool = False,
    ):
        """
        Initialize Hyperliquid adapter.

        Credentials are loaded from environment variables if not provided:
        - EXCHANGE_WALLET_PRIVATE_KEY: Wallet private key
        - EXCHANGE_WALLET_ADDRESS: Wallet address (optional)

        Args:
            private_key: Ethereum private key for signing
            wallet_address: Ethereum wallet address
            base_url: REST API URL (uses default if None)
            ws_url: WebSocket URL (uses default if None)
            testnet: Whether to use testnet
        """
        # REST client loads credentials from config if not provided
        self._rest = HyperliquidRESTClient(
            private_key=private_key,
            wallet_address=wallet_address,
            base_url=base_url,
            testnet=testnet,
        )

        # Lazy import to avoid circular imports
        from src.exchange.adapters.hyperliquid.websocket import HyperliquidWebSocket

        self._websocket: HyperliquidWebSocket = HyperliquidWebSocket(
            ws_url=ws_url or (TESTNET_WS_URL if testnet else MAINNET_WS_URL),
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
        Convert unified symbol to Hyperliquid format.

        BTCUSDT -> BTC
        """
        return HyperliquidMapper.to_hyperliquid_coin(symbol)

    def denormalize_symbol(self, exchange_symbol: str) -> str:
        """
        Convert Hyperliquid symbol to unified format.

        BTC -> BTCUSDT
        """
        return HyperliquidMapper.from_hyperliquid_coin(exchange_symbol)

    async def initialize(self) -> None:
        """Initialize both REST and WebSocket clients."""
        await self._rest.initialize()

    async def close(self) -> None:
        """Close both REST and WebSocket connections."""
        await self._rest.close()
        if self._websocket.is_connected:
            await self._websocket.disconnect()
