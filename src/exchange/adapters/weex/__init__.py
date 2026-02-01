"""
WEEX Exchange Adapter

Implements the ExchangeAdapter protocol for WEEX CEX.
"""

from __future__ import annotations

from src.exchange.adapters.weex.adapter import WEEXAdapter
from src.exchange.factory import register_adapter

# Register the adapter when this module is imported
register_adapter("weex", WEEXAdapter)

__all__ = ["WEEXAdapter"]
