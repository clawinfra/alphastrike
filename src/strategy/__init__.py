"""
AlphaStrike Trading Bot - Strategy Layer

Contains market regime detection, exit management, and
adaptive strategy components.
"""

from src.strategy.exit_manager import (
    ExitManager,
    ExitPrices,
    ExitReason,
    PositionExitState,
)
from src.strategy.regime_detector import RegimeDetector, RegimeState

__all__ = [
    "ExitManager",
    "ExitPrices",
    "ExitReason",
    "PositionExitState",
    "RegimeDetector",
    "RegimeState",
]
