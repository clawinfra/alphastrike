"""
Exchange Adapters Package

Contains exchange-specific implementations of the ExchangeAdapter protocol.
"""

from __future__ import annotations

# Adapters are imported on-demand to avoid circular imports
# and allow registration via the factory pattern.

__all__: list[str] = []
