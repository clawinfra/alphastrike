"""
Unit tests for exchange factory.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.exchange.exceptions import ExchangeError
from src.exchange.factory import (
    ExchangeFactory,
    _adapter_registry,
    get_adapter_class,
    get_exchange_adapter,
    get_exchange_factory,
    list_adapters,
    register_adapter,
)
from src.exchange.protocols import ExchangeAdapter


class MockAdapter(ExchangeAdapter):
    """Mock adapter for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._rest = MagicMock()
        self._rest.capabilities = MagicMock()
        self._rest.capabilities.name = "mock"
        self._rest.capabilities.max_leverage = 100
        self._rest.capabilities.authentication_type = "HMAC"
        self._websocket = MagicMock()
        self._initialized = False

    @property
    def rest(self):
        return self._rest

    @property
    def websocket(self):
        return self._websocket

    async def initialize(self):
        self._initialized = True

    async def close(self):
        self._initialized = False

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.lower()

    def denormalize_symbol(self, symbol: str) -> str:
        return symbol.upper()


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean adapter registry before and after each test."""
    original = _adapter_registry.copy()
    _adapter_registry.clear()
    yield
    _adapter_registry.clear()
    _adapter_registry.update(original)


class TestRegisterAdapter:
    """Tests for adapter registration."""

    def test_register_adapter(self):
        """Test registering an adapter."""
        register_adapter("mock", MockAdapter)
        assert "mock" in _adapter_registry
        assert _adapter_registry["mock"] is MockAdapter

    def test_register_adapter_lowercase(self):
        """Test that adapter names are lowercased."""
        register_adapter("MOCK", MockAdapter)
        assert "mock" in _adapter_registry
        assert "MOCK" not in _adapter_registry

    def test_register_multiple_adapters(self):
        """Test registering multiple adapters."""
        register_adapter("mock1", MockAdapter)
        register_adapter("mock2", MockAdapter)
        assert len(_adapter_registry) == 2


class TestListAdapters:
    """Tests for listing adapters."""

    def test_list_empty(self):
        """Test listing when no adapters registered."""
        assert list_adapters() == []

    def test_list_adapters(self):
        """Test listing registered adapters."""
        register_adapter("mock1", MockAdapter)
        register_adapter("mock2", MockAdapter)
        adapters = list_adapters()
        assert "mock1" in adapters
        assert "mock2" in adapters


class TestGetAdapterClass:
    """Tests for getting adapter class."""

    def test_get_existing_adapter(self):
        """Test getting an existing adapter class."""
        register_adapter("mock", MockAdapter)
        cls = get_adapter_class("mock")
        assert cls is MockAdapter

    def test_get_adapter_case_insensitive(self):
        """Test that lookup is case-insensitive."""
        register_adapter("mock", MockAdapter)
        assert get_adapter_class("MOCK") is MockAdapter
        assert get_adapter_class("Mock") is MockAdapter

    def test_get_nonexistent_adapter(self):
        """Test getting a non-existent adapter."""
        assert get_adapter_class("nonexistent") is None


class TestExchangeFactory:
    """Tests for ExchangeFactory class."""

    @pytest.mark.asyncio
    async def test_create_adapter(self):
        """Test creating an adapter."""
        register_adapter("mock", MockAdapter)
        factory = ExchangeFactory()

        adapter = await factory.create("mock")

        assert isinstance(adapter, MockAdapter)
        assert adapter._initialized is True

    @pytest.mark.asyncio
    async def test_create_adapter_with_kwargs(self):
        """Test creating adapter with custom kwargs."""
        register_adapter("mock", MockAdapter)
        factory = ExchangeFactory()

        adapter = await factory.create("mock", custom_param="value")

        assert adapter.kwargs["custom_param"] == "value"

    @pytest.mark.asyncio
    async def test_create_nonexistent_adapter(self):
        """Test creating non-existent adapter raises error."""
        factory = ExchangeFactory()

        with pytest.raises(ExchangeError) as exc_info:
            await factory.create("nonexistent")

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_create_uses_config_when_no_name(self):
        """Test that factory uses config when no name provided."""
        register_adapter("mock", MockAdapter)
        factory = ExchangeFactory()

        with patch("src.core.config.get_settings") as mock_settings:
            mock_settings.return_value.exchange.name = "mock"
            adapter = await factory.create()

        assert isinstance(adapter, MockAdapter)

    @pytest.mark.asyncio
    async def test_tracks_active_adapters(self):
        """Test that factory tracks active adapters."""
        register_adapter("mock", MockAdapter)
        factory = ExchangeFactory()

        adapter = await factory.create("mock")

        assert adapter in factory._active_adapters

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all adapters."""
        register_adapter("mock", MockAdapter)
        factory = ExchangeFactory()

        adapter1 = await factory.create("mock")
        adapter2 = await factory.create("mock")

        await factory.close_all()

        assert adapter1._initialized is False
        assert adapter2._initialized is False
        assert len(factory._active_adapters) == 0


class TestGetExchangeFactory:
    """Tests for get_exchange_factory singleton."""

    def test_returns_factory(self):
        """Test that get_exchange_factory returns a factory."""
        # Reset singleton
        import src.exchange.factory as factory_module

        factory_module._factory = None

        factory = get_exchange_factory()

        assert isinstance(factory, ExchangeFactory)

    def test_returns_singleton(self):
        """Test that get_exchange_factory returns same instance."""
        import src.exchange.factory as factory_module

        factory_module._factory = None

        factory1 = get_exchange_factory()
        factory2 = get_exchange_factory()

        assert factory1 is factory2


class TestGetExchangeAdapter:
    """Tests for get_exchange_adapter convenience function."""

    @pytest.mark.asyncio
    async def test_get_adapter(self):
        """Test getting an adapter."""
        register_adapter("mock", MockAdapter)

        with patch("src.core.config.get_settings") as mock_settings:
            mock_settings.return_value.exchange.name = "mock"
            adapter = await get_exchange_adapter("mock")

        assert isinstance(adapter, MockAdapter)

    @pytest.mark.asyncio
    async def test_get_adapter_uses_factory(self):
        """Test that get_exchange_adapter uses the factory."""
        register_adapter("mock", MockAdapter)

        import src.exchange.factory as factory_module

        factory_module._factory = None

        with patch("src.core.config.get_settings") as mock_settings:
            mock_settings.return_value.exchange.name = "mock"
            await get_exchange_adapter("mock")

        assert factory_module._factory is not None
