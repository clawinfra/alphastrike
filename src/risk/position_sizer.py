"""
AlphaStrike Trading Bot - Position Sizer with Half-Kelly

US-019: Implements adaptive position sizing using the Half-Kelly criterion
with multiple adjustment factors for volatility, drawdown, and confidence.

The Half-Kelly approach provides a more conservative sizing method compared
to full Kelly, reducing variance while still capturing edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from src.core.config import PositionSizingConfig, get_settings

logger = logging.getLogger(__name__)


# Type aliases
VolatilityRegime = Literal["low", "normal", "high", "extreme"]


@dataclass
class PositionSizeResult:
    """Result of position size calculation with component breakdown."""

    position_size: float
    kelly_fraction: float
    adaptive_kelly: float
    confidence_factor: float
    volatility_factor: float
    drawdown_factor: float
    leverage: int
    is_valid: bool
    rejection_reason: str | None


class PositionSizer:
    """
    Adaptive position sizer using Half-Kelly criterion.

    Implements position sizing with multiple adjustment factors:
    - Kelly fraction based on win rate and average win/loss ratio
    - Confidence adjustment based on signal quality (0.5-1.0)
    - Volatility adjustment based on ATR regime (0.6-1.3)
    - Drawdown adjustment based on current drawdown (0.5-1.0)

    The kelly_multiplier parameter (default 0.5 for Half-Kelly) can be
    adjusted between 0.3-0.7 for more conservative/aggressive sizing.
    """

    # Kelly multiplier range (Half-Kelly uses 0.5)
    KELLY_MULTIPLIER_MIN: float = 0.3
    KELLY_MULTIPLIER_MAX: float = 0.7
    DEFAULT_KELLY_MULTIPLIER: float = 0.5

    # Confidence factor bounds
    CONFIDENCE_FACTOR_MIN: float = 0.5
    CONFIDENCE_FACTOR_MAX: float = 1.0

    # Volatility factor bounds and regime mappings
    VOLATILITY_FACTOR_MIN: float = 0.6
    VOLATILITY_FACTOR_MAX: float = 1.3
    VOLATILITY_REGIME_FACTORS: dict[VolatilityRegime, float] = {
        "low": 1.3,      # Low volatility -> larger position
        "normal": 1.0,   # Normal volatility -> base position
        "high": 0.8,     # High volatility -> reduced position
        "extreme": 0.6,  # Extreme volatility -> minimum position
    }

    # Drawdown factor bounds
    DRAWDOWN_FACTOR_MIN: float = 0.5
    DRAWDOWN_FACTOR_MAX: float = 1.0

    # Leverage thresholds based on confidence
    LEVERAGE_CONFIDENCE_THRESHOLDS: list[tuple[float, int]] = [
        (0.85, 5),  # confidence >= 0.85 -> leverage 5
        (0.80, 4),  # confidence >= 0.80 -> leverage 4
        (0.75, 3),  # confidence >= 0.75 -> leverage 3
    ]
    MIN_TRADE_CONFIDENCE: float = 0.75

    def __init__(
        self,
        config: PositionSizingConfig | None = None,
        kelly_multiplier: float | None = None,
    ) -> None:
        """
        Initialize position sizer.

        Args:
            config: Optional configuration. If not provided, uses default settings.
            kelly_multiplier: Optional Kelly multiplier override (0.3-0.7).
                            Default is 0.5 (Half-Kelly).
        """
        self.config = config or get_settings().position

        # Validate and set kelly multiplier
        if kelly_multiplier is not None:
            if not self.KELLY_MULTIPLIER_MIN <= kelly_multiplier <= self.KELLY_MULTIPLIER_MAX:
                raise ValueError(
                    f"kelly_multiplier must be between {self.KELLY_MULTIPLIER_MIN} "
                    f"and {self.KELLY_MULTIPLIER_MAX}, got {kelly_multiplier}"
                )
            self.kelly_multiplier = kelly_multiplier
        else:
            self.kelly_multiplier = self.DEFAULT_KELLY_MULTIPLIER

        logger.info(
            "PositionSizer initialized",
            extra={
                "kelly_multiplier": self.kelly_multiplier,
                "min_notional_value": self.config.min_notional_value,
            },
        )

    def _calculate_kelly_fraction(
        self, win_rate: float, avg_win_loss_ratio: float
    ) -> float:
        """
        Calculate raw Kelly fraction.

        Kelly formula: f* = win_rate - (1 - win_rate) / avg_win_loss_ratio

        Args:
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Ratio of average win to average loss

        Returns:
            Raw Kelly fraction (can be negative if edge is negative)
        """
        if avg_win_loss_ratio <= 0:
            logger.warning(
                "Invalid avg_win_loss_ratio",
                extra={"avg_win_loss_ratio": avg_win_loss_ratio},
            )
            return 0.0

        kelly_fraction = win_rate - (1 - win_rate) / avg_win_loss_ratio
        return kelly_fraction

    def _calculate_confidence_factor(self, confidence: float) -> float:
        """
        Calculate confidence adjustment factor.

        Maps confidence to factor in range [CONFIDENCE_FACTOR_MIN, CONFIDENCE_FACTOR_MAX].
        Higher confidence results in larger position.

        Args:
            confidence: Signal confidence (0-1)

        Returns:
            Confidence factor (0.5-1.0)
        """
        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))

        # Linear mapping from [0, 1] to [CONFIDENCE_FACTOR_MIN, CONFIDENCE_FACTOR_MAX]
        factor_range = self.CONFIDENCE_FACTOR_MAX - self.CONFIDENCE_FACTOR_MIN
        factor = self.CONFIDENCE_FACTOR_MIN + (confidence * factor_range)

        return factor

    def _calculate_volatility_factor(
        self, volatility_regime: VolatilityRegime
    ) -> float:
        """
        Calculate volatility adjustment factor.

        Higher volatility results in smaller position to manage risk.

        Args:
            volatility_regime: Current volatility regime

        Returns:
            Volatility factor (0.6-1.3)
        """
        factor = self.VOLATILITY_REGIME_FACTORS.get(
            volatility_regime,
            self.VOLATILITY_REGIME_FACTORS["normal"]
        )
        return factor

    def _calculate_drawdown_factor(self, drawdown_pct: float) -> float:
        """
        Calculate drawdown adjustment factor.

        Larger drawdowns result in smaller positions to preserve capital.
        Uses linear scaling from 0% drawdown (factor=1.0) to 20% drawdown (factor=0.5).

        Args:
            drawdown_pct: Current drawdown percentage (0-1, e.g., 0.15 = 15% drawdown)

        Returns:
            Drawdown factor (0.5-1.0)
        """
        # Clamp drawdown to valid range
        drawdown_pct = max(0.0, min(1.0, drawdown_pct))

        # Linear scaling: 0% DD -> 1.0, 20% DD -> 0.5, beyond 20% -> floor at 0.5
        max_drawdown_for_scaling = 0.20  # 20%

        if drawdown_pct >= max_drawdown_for_scaling:
            return self.DRAWDOWN_FACTOR_MIN

        # Linear interpolation
        factor_range = self.DRAWDOWN_FACTOR_MAX - self.DRAWDOWN_FACTOR_MIN
        scaling = drawdown_pct / max_drawdown_for_scaling
        factor = self.DRAWDOWN_FACTOR_MAX - (scaling * factor_range)

        return factor

    def get_leverage_for_confidence(self, confidence: float) -> int:
        """
        Determine appropriate leverage based on signal confidence.

        Leverage mapping:
        - confidence >= 0.85: leverage 5
        - confidence >= 0.80: leverage 4
        - confidence >= 0.75: leverage 3
        - confidence < 0.75: no trade (leverage 0)

        Args:
            confidence: Signal confidence (0-1)

        Returns:
            Appropriate leverage level (0 if confidence too low for trade)
        """
        for threshold, leverage in self.LEVERAGE_CONFIDENCE_THRESHOLDS:
            if confidence >= threshold:
                return leverage

        # Confidence too low for any trade
        return 0

    def calculate_position_size(
        self,
        capital: float,
        confidence: float,
        volatility_regime: VolatilityRegime,
        drawdown_pct: float,
        win_rate: float,
        avg_win_loss: float,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using Half-Kelly with adjustments.

        Formula:
            kelly_fraction = win_rate - (1 - win_rate) / avg_win_loss_ratio
            adaptive_kelly = kelly_fraction * kelly_multiplier
            position_size = capital * adaptive_kelly * confidence_factor *
                           volatility_factor * drawdown_factor

        Args:
            capital: Available trading capital
            confidence: Signal confidence (0-1)
            volatility_regime: Current volatility regime ("low", "normal", "high", "extreme")
            drawdown_pct: Current drawdown percentage (0-1)
            win_rate: Historical win rate (0-1)
            avg_win_loss: Average win/loss ratio

        Returns:
            PositionSizeResult with calculated position size and component breakdown
        """
        # Check minimum confidence for trade
        leverage = self.get_leverage_for_confidence(confidence)
        if leverage == 0:
            return PositionSizeResult(
                position_size=0.0,
                kelly_fraction=0.0,
                adaptive_kelly=0.0,
                confidence_factor=0.0,
                volatility_factor=0.0,
                drawdown_factor=0.0,
                leverage=0,
                is_valid=False,
                rejection_reason=f"Confidence {confidence:.2f} below minimum {self.MIN_TRADE_CONFIDENCE}",
            )

        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win_loss)

        # Apply Kelly multiplier (Half-Kelly by default)
        adaptive_kelly = kelly_fraction * self.kelly_multiplier

        # Check for negative or zero edge
        if adaptive_kelly <= 0:
            return PositionSizeResult(
                position_size=0.0,
                kelly_fraction=kelly_fraction,
                adaptive_kelly=adaptive_kelly,
                confidence_factor=0.0,
                volatility_factor=0.0,
                drawdown_factor=0.0,
                leverage=0,
                is_valid=False,
                rejection_reason=f"No edge: kelly_fraction={kelly_fraction:.4f}",
            )

        # Calculate adjustment factors
        confidence_factor = self._calculate_confidence_factor(confidence)
        volatility_factor = self._calculate_volatility_factor(volatility_regime)
        drawdown_factor = self._calculate_drawdown_factor(drawdown_pct)

        # Calculate final position size
        position_size = (
            capital
            * adaptive_kelly
            * confidence_factor
            * volatility_factor
            * drawdown_factor
        )

        # Check minimum notional value
        min_notional = self.config.min_notional_value
        if position_size < min_notional:
            return PositionSizeResult(
                position_size=0.0,
                kelly_fraction=kelly_fraction,
                adaptive_kelly=adaptive_kelly,
                confidence_factor=confidence_factor,
                volatility_factor=volatility_factor,
                drawdown_factor=drawdown_factor,
                leverage=leverage,
                is_valid=False,
                rejection_reason=(
                    f"Position size ${position_size:.2f} below minimum "
                    f"notional ${min_notional:.2f}"
                ),
            )

        logger.debug(
            "Position size calculated",
            extra={
                "capital": capital,
                "position_size": position_size,
                "kelly_fraction": kelly_fraction,
                "adaptive_kelly": adaptive_kelly,
                "confidence_factor": confidence_factor,
                "volatility_factor": volatility_factor,
                "drawdown_factor": drawdown_factor,
                "leverage": leverage,
            },
        )

        return PositionSizeResult(
            position_size=position_size,
            kelly_fraction=kelly_fraction,
            adaptive_kelly=adaptive_kelly,
            confidence_factor=confidence_factor,
            volatility_factor=volatility_factor,
            drawdown_factor=drawdown_factor,
            leverage=leverage,
            is_valid=True,
            rejection_reason=None,
        )
