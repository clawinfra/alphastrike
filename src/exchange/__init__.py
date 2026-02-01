"""
AlphaStrike Exchange Abstraction Layer

Provides a unified interface for interacting with any CEX or DEX.
Exchange-specific details are handled by adapters.

Usage:
    from src.exchange import get_exchange_adapter

    adapter = await get_exchange_adapter()  # Uses configured exchange
    balance = await adapter.rest.get_account_balance()

    # Or specify exchange explicitly
    adapter = await get_exchange_adapter("weex")
"""

from src.exchange.exceptions import (
    AuthenticationError,
    ExchangeConnectionError,
    ExchangeError,
    ExchangeTimeoutError,
    InsufficientBalanceError,
    InvalidOrderError,
    MaintenanceError,
    NetworkError,
    OrderError,
    OrderNotFoundError,
    PositionError,
    RateLimitError,
    RequestTimeoutError,
    SymbolNotFoundError,
)
from src.exchange.factory import ExchangeFactory, get_exchange_adapter
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
from src.exchange.protocols import (
    ExchangeAdapter,
    ExchangeRESTProtocol,
    ExchangeWebSocketProtocol,
)

__all__ = [
    # Factory
    "ExchangeFactory",
    "get_exchange_adapter",
    # Protocols
    "ExchangeAdapter",
    "ExchangeRESTProtocol",
    "ExchangeWebSocketProtocol",
    # Models
    "UnifiedOrder",
    "UnifiedOrderResult",
    "UnifiedPosition",
    "UnifiedAccountBalance",
    "UnifiedTicker",
    "UnifiedOrderbook",
    "UnifiedCandle",
    "UnifiedTrade",
    "SymbolInfo",
    "ExchangeCapabilities",
    # Enums
    "OrderSide",
    "OrderType",
    "PositionSide",
    "TimeInForce",
    "OrderStatus",
    "MarginMode",
    # Exceptions
    "ExchangeError",
    "AuthenticationError",
    "RateLimitError",
    "InsufficientBalanceError",
    "OrderError",
    "InvalidOrderError",
    "OrderNotFoundError",
    "PositionError",
    "SymbolNotFoundError",
    "ExchangeConnectionError",
    "ExchangeTimeoutError",
    "MaintenanceError",
    # Aliases
    "NetworkError",
    "RequestTimeoutError",
]
