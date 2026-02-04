"""Shared constants for CLI commands."""

from __future__ import annotations

# Default assets for Hyperliquid trading
DEFAULT_ASSETS: list[str] = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "AVAX", "NEAR", "APT",
    "AAVE", "UNI", "LINK", "FET", "DOGE", "PAXG", "SPX"
]

# Supported exchanges
SUPPORTED_EXCHANGES: list[str] = ["hyperliquid", "weex"]
