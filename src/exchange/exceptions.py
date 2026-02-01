"""
Unified Exchange Exceptions

Exchange-agnostic exceptions that adapters translate from exchange-specific errors.
"""

from __future__ import annotations


class ExchangeError(Exception):
    """Base exception for all exchange errors."""

    def __init__(
        self,
        message: str,
        exchange: str | None = None,
        code: str | None = None,
        raw_error: dict | None = None,
    ):
        super().__init__(message)
        self.exchange = exchange
        self.code = code
        self.raw_error = raw_error or {}


class AuthenticationError(ExchangeError):
    """Authentication failed (invalid API key, signature, etc.)."""

    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class InsufficientBalanceError(ExchangeError):
    """Insufficient balance for the operation."""

    def __init__(
        self,
        message: str,
        required: float | None = None,
        available: float | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.required = required
        self.available = available


class OrderError(ExchangeError):
    """Order placement or management failed."""

    pass


class InvalidOrderError(OrderError):
    """Order parameters are invalid."""

    pass


class OrderNotFoundError(OrderError):
    """Order not found."""

    pass


class PositionError(ExchangeError):
    """Position-related error."""

    pass


class SymbolNotFoundError(ExchangeError):
    """Trading symbol not found or not supported."""

    pass


class ExchangeConnectionError(ExchangeError):
    """Network connection error."""

    pass


class ExchangeTimeoutError(ExchangeError):
    """Request timeout."""

    pass


# Aliases to avoid confusion with built-in exceptions
# (ConnectionError and TimeoutError are Python built-ins)
NetworkError = ExchangeConnectionError
RequestTimeoutError = ExchangeTimeoutError


class MaintenanceError(ExchangeError):
    """Exchange is under maintenance."""

    pass
