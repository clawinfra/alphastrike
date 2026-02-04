"""
Hyperliquid DEX Adapter

Implements ExchangeAdapter protocol for Hyperliquid decentralized exchange.
Supports multi-asset trading: crypto, commodities, forex, indices, and stocks.
Includes multi-wallet support for risk distribution.
"""

from __future__ import annotations

from src.exchange.adapters.hyperliquid.adapter import HyperliquidAdapter
from src.exchange.adapters.hyperliquid.multi_wallet import (
    AllocationStrategy,
    MultiWalletManager,
    WalletConfig,
    WalletState,
    load_wallet_configs_from_env,
)
from src.exchange.factory import register_adapter

# Register the adapter when this module is imported
register_adapter("hyperliquid", HyperliquidAdapter)

__all__ = [
    "HyperliquidAdapter",
    "MultiWalletManager",
    "WalletConfig",
    "WalletState",
    "AllocationStrategy",
    "load_wallet_configs_from_env",
]
