"""
Unit tests for exchange exception hierarchy.
"""

import pytest

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


class TestExchangeErrorBase:
    """Tests for base ExchangeError."""

    def test_create_basic_error(self):
        """Test creating a basic exchange error."""
        error = ExchangeError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.exchange is None
        assert error.code is None
        assert error.raw_error == {}

    def test_create_error_with_details(self):
        """Test creating error with full details."""
        raw = {"error_code": "123", "message": "test"}
        error = ExchangeError(
            "API error",
            exchange="weex",
            code="40001",
            raw_error=raw,
        )
        assert error.exchange == "weex"
        assert error.code == "40001"
        assert error.raw_error == raw

    def test_error_is_exception(self):
        """Test that ExchangeError inherits from Exception."""
        assert issubclass(ExchangeError, Exception)

    def test_can_raise_and_catch(self):
        """Test raising and catching ExchangeError."""
        with pytest.raises(ExchangeError) as exc_info:
            raise ExchangeError("Test error", exchange="binance")
        assert exc_info.value.exchange == "binance"


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_inherits_from_exchange_error(self):
        """Test inheritance chain."""
        assert issubclass(AuthenticationError, ExchangeError)

    def test_create_auth_error(self):
        """Test creating authentication error."""
        error = AuthenticationError(
            "Invalid API key",
            exchange="weex",
            code="40101",
        )
        assert str(error) == "Invalid API key"
        assert error.exchange == "weex"

    def test_catch_as_exchange_error(self):
        """Test catching as base ExchangeError."""
        with pytest.raises(ExchangeError):
            raise AuthenticationError("Invalid signature")


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_inherits_from_exchange_error(self):
        """Test inheritance chain."""
        assert issubclass(RateLimitError, ExchangeError)

    def test_create_rate_limit_error(self):
        """Test creating rate limit error."""
        error = RateLimitError(
            "Rate limit exceeded",
            retry_after=60.0,
            exchange="weex",
        )
        assert error.retry_after == 60.0
        assert error.exchange == "weex"

    def test_rate_limit_without_retry(self):
        """Test rate limit error without retry_after."""
        error = RateLimitError("Too many requests")
        assert error.retry_after is None


class TestInsufficientBalanceError:
    """Tests for InsufficientBalanceError."""

    def test_inherits_from_exchange_error(self):
        """Test inheritance chain."""
        assert issubclass(InsufficientBalanceError, ExchangeError)

    def test_create_balance_error(self):
        """Test creating insufficient balance error."""
        error = InsufficientBalanceError(
            "Not enough USDT",
            required=1000.0,
            available=500.0,
            exchange="weex",
        )
        assert error.required == 1000.0
        assert error.available == 500.0

    def test_balance_error_without_amounts(self):
        """Test balance error without specific amounts."""
        error = InsufficientBalanceError("Insufficient margin")
        assert error.required is None
        assert error.available is None


class TestOrderErrors:
    """Tests for order-related errors."""

    def test_order_error_hierarchy(self):
        """Test order error inheritance."""
        assert issubclass(OrderError, ExchangeError)
        assert issubclass(InvalidOrderError, OrderError)
        assert issubclass(OrderNotFoundError, OrderError)

    def test_create_order_error(self):
        """Test creating order error."""
        error = OrderError("Order failed", exchange="weex", code="40301")
        assert str(error) == "Order failed"

    def test_create_invalid_order_error(self):
        """Test creating invalid order error."""
        error = InvalidOrderError(
            "Invalid quantity",
            exchange="weex",
            code="40302",
        )
        assert str(error) == "Invalid quantity"

    def test_create_order_not_found_error(self):
        """Test creating order not found error."""
        error = OrderNotFoundError(
            "Order 123 not found",
            exchange="weex",
        )
        assert str(error) == "Order 123 not found"

    def test_catch_invalid_order_as_order_error(self):
        """Test catching InvalidOrderError as OrderError."""
        with pytest.raises(OrderError):
            raise InvalidOrderError("Bad params")

    def test_catch_order_error_as_exchange_error(self):
        """Test catching OrderError as ExchangeError."""
        with pytest.raises(ExchangeError):
            raise OrderNotFoundError("Not found")


class TestPositionError:
    """Tests for PositionError."""

    def test_inherits_from_exchange_error(self):
        """Test inheritance chain."""
        assert issubclass(PositionError, ExchangeError)

    def test_create_position_error(self):
        """Test creating position error."""
        error = PositionError(
            "Position not found",
            exchange="weex",
        )
        assert str(error) == "Position not found"


class TestSymbolNotFoundError:
    """Tests for SymbolNotFoundError."""

    def test_inherits_from_exchange_error(self):
        """Test inheritance chain."""
        assert issubclass(SymbolNotFoundError, ExchangeError)

    def test_create_symbol_error(self):
        """Test creating symbol not found error."""
        error = SymbolNotFoundError(
            "Symbol XYZUSDT not found",
            exchange="weex",
        )
        assert str(error) == "Symbol XYZUSDT not found"


class TestConnectionErrors:
    """Tests for connection-related errors."""

    def test_connection_error_not_builtin(self):
        """Test that ExchangeConnectionError is not the built-in."""
        import builtins

        assert ExchangeConnectionError is not builtins.ConnectionError

    def test_timeout_error_not_builtin(self):
        """Test that ExchangeTimeoutError is not the built-in."""
        import builtins

        assert ExchangeTimeoutError is not builtins.TimeoutError

    def test_connection_error_inheritance(self):
        """Test ExchangeConnectionError inheritance."""
        assert issubclass(ExchangeConnectionError, ExchangeError)

    def test_timeout_error_inheritance(self):
        """Test ExchangeTimeoutError inheritance."""
        assert issubclass(ExchangeTimeoutError, ExchangeError)

    def test_create_connection_error(self):
        """Test creating connection error."""
        error = ExchangeConnectionError(
            "Connection refused",
            exchange="weex",
        )
        assert str(error) == "Connection refused"

    def test_create_timeout_error(self):
        """Test creating timeout error."""
        error = ExchangeTimeoutError(
            "Request timed out after 30s",
            exchange="weex",
        )
        assert str(error) == "Request timed out after 30s"


class TestErrorAliases:
    """Tests for error aliases."""

    def test_network_error_alias(self):
        """Test NetworkError is alias for ExchangeConnectionError."""
        assert NetworkError is ExchangeConnectionError

    def test_request_timeout_error_alias(self):
        """Test RequestTimeoutError is alias for ExchangeTimeoutError."""
        assert RequestTimeoutError is ExchangeTimeoutError

    def test_can_use_alias_to_catch(self):
        """Test using alias to catch exception."""
        with pytest.raises(NetworkError):
            raise ExchangeConnectionError("Connection failed")

        with pytest.raises(RequestTimeoutError):
            raise ExchangeTimeoutError("Timed out")


class TestMaintenanceError:
    """Tests for MaintenanceError."""

    def test_inherits_from_exchange_error(self):
        """Test inheritance chain."""
        assert issubclass(MaintenanceError, ExchangeError)

    def test_create_maintenance_error(self):
        """Test creating maintenance error."""
        error = MaintenanceError(
            "Exchange is under maintenance",
            exchange="weex",
        )
        assert str(error) == "Exchange is under maintenance"
