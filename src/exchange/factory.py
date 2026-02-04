"""
Exchange Factory

Creates exchange adapters based on configuration.
Supports runtime exchange selection and adapter registration.

Usage:
    # Get adapter for configured exchange (from settings)
    adapter = await get_exchange_adapter()

    # Or specify exchange explicitly
    adapter = await get_exchange_adapter("weex")

    # Use the adapter
    balance = await adapter.rest.get_account_balance()
    await adapter.close()
"""

from __future__ import annotations

import logging
from typing import Any

from src.exchange.exceptions import ExchangeError
from src.exchange.protocols import ExchangeAdapter

logger = logging.getLogger(__name__)


# Registry of available exchange adapters
_adapter_registry: dict[str, type[ExchangeAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[ExchangeAdapter]) -> None:
    """
    Register an exchange adapter.

    Called by adapter modules to register themselves.

    Args:
        name: Exchange identifier (lowercase, e.g., "weex")
        adapter_cls: Adapter class implementing ExchangeAdapter
    """
    _adapter_registry[name.lower()] = adapter_cls
    logger.debug(f"Registered exchange adapter: {name}")


def list_adapters() -> list[str]:
    """
    List all registered exchange adapters.

    Returns:
        List of exchange names
    """
    return list(_adapter_registry.keys())


def get_adapter_class(name: str) -> type[ExchangeAdapter] | None:
    """
    Get adapter class by name.

    Args:
        name: Exchange identifier

    Returns:
        Adapter class or None if not found
    """
    return _adapter_registry.get(name.lower())


class ExchangeFactory:
    """
    Factory for creating exchange adapters.

    Manages adapter creation with proper initialization and cleanup.

    Example:
        factory = ExchangeFactory()

        # Create adapter for configured exchange
        adapter = await factory.create()

        # Or specify exchange explicitly
        adapter = await factory.create("binance")

        # Remember to close when done
        await adapter.close()
    """

    def __init__(self) -> None:
        """Initialize the factory."""
        self._active_adapters: list[ExchangeAdapter] = []

    async def create(
        self,
        exchange_name: str | None = None,
        **kwargs: Any,
    ) -> ExchangeAdapter:
        """
        Create an exchange adapter.

        Args:
            exchange_name: Exchange identifier (uses config if None)
            **kwargs: Additional configuration passed to adapter

        Returns:
            Initialized ExchangeAdapter

        Raises:
            ExchangeError: If exchange not found or initialization fails
        """
        # Import here to avoid circular imports
        from src.core.config import get_settings

        settings = get_settings()

        # Use configured exchange if not specified
        name: str
        if exchange_name is None:
            configured = getattr(settings.exchange, "name", None)
            if configured is not None and hasattr(configured, "value"):
                name = configured.value
            elif configured is not None:
                name = str(configured)
            else:
                name = "weex"
        else:
            name = exchange_name

        exchange_name = name.lower()

        # Ensure adapters are registered
        self._ensure_adapters_registered()

        # Get adapter class from registry
        adapter_cls = get_adapter_class(exchange_name)
        if adapter_cls is None:
            available = ", ".join(list_adapters())
            raise ExchangeError(
                f"Exchange '{exchange_name}' not found. Available: {available}",
                exchange=exchange_name,
            )

        # Create and initialize adapter
        try:
            adapter = adapter_cls(**kwargs)
            await adapter.initialize()
            self._active_adapters.append(adapter)

            logger.info(
                "Created exchange adapter",
                extra={
                    "exchange": exchange_name,
                    "capabilities": {
                        "name": adapter.rest.capabilities.name,
                        "max_leverage": adapter.rest.capabilities.max_leverage,
                        "auth_type": adapter.rest.capabilities.authentication_type,
                    },
                },
            )

            return adapter

        except Exception as e:
            logger.error(f"Failed to create adapter for {exchange_name}: {e}")
            raise ExchangeError(
                f"Failed to initialize {exchange_name} adapter: {e}",
                exchange=exchange_name,
            ) from e

    def _ensure_adapters_registered(self) -> None:
        """Ensure all adapter modules are imported and registered."""
        if not _adapter_registry:
            # Import adapter modules to trigger registration
            try:
                from src.exchange.adapters import weex  # noqa: F401
            except ImportError as e:
                logger.warning(f"Failed to import weex adapter: {e}")

            try:
                from src.exchange.adapters import hyperliquid  # noqa: F401
            except ImportError as e:
                logger.warning(f"Failed to import hyperliquid adapter: {e}")

    async def close_all(self) -> None:
        """Close all active adapters."""
        for adapter in self._active_adapters:
            try:
                await adapter.close()
            except Exception as e:
                logger.error(f"Error closing adapter: {e}")

        self._active_adapters.clear()
        logger.info("Closed all exchange adapters")


# Module-level factory singleton
_factory: ExchangeFactory | None = None


def get_exchange_factory() -> ExchangeFactory:
    """
    Get the exchange factory singleton.

    Returns:
        ExchangeFactory instance
    """
    global _factory
    if _factory is None:
        _factory = ExchangeFactory()
    return _factory


async def get_exchange_adapter(
    exchange_name: str | None = None,
    **kwargs: Any,
) -> ExchangeAdapter:
    """
    Convenience function to get an exchange adapter.

    This is the primary entry point for obtaining an adapter.

    Args:
        exchange_name: Exchange to use (uses config if None)
        **kwargs: Additional configuration

    Returns:
        Initialized ExchangeAdapter

    Example:
        adapter = await get_exchange_adapter()
        balance = await adapter.rest.get_account_balance()
        print(f"Balance: {balance.total_balance}")
    """
    factory = get_exchange_factory()
    return await factory.create(exchange_name, **kwargs)


async def close_all_adapters() -> None:
    """
    Close all active exchange adapters.

    Call this during application shutdown.
    """
    factory = get_exchange_factory()
    await factory.close_all()
