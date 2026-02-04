"""
Hyperliquid DEX Adapter

Implements ExchangeAdapter protocol for Hyperliquid decentralized exchange.
Supports multi-asset trading: crypto, commodities, forex, indices, and stocks.
"""

from __future__ import annotations

from src.exchange.adapters.hyperliquid.adapter import HyperliquidAdapter
from src.exchange.factory import register_adapter

# Register the adapter when this module is imported
register_adapter("hyperliquid", HyperliquidAdapter)

__all__ = ["HyperliquidAdapter"]
