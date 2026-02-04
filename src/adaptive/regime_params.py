"""
Regime-Aware Parameter Adjustment

Dynamically adjusts trading parameters based on detected market regime.
This is the real-time adaptation layer that sits on top of the optimized
base parameters.

Key Insight: Different regimes require different trading behavior:
- TRENDING_UP: Wider stops, let winners run, favor longs
- TRENDING_DOWN: Tighter stops, quicker exits, favor shorts
- RANGING: Tighter stops, smaller targets, reduce size
- HIGH_VOLATILITY: Much wider stops, smaller size, higher threshold
- EXTREME_VOLATILITY: Minimal trading, tiny size, maximum threshold
- TREND_EXHAUSTION: Prepare for reversal, reduce exposure
"""

import logging
from dataclasses import dataclass

from src.adaptive.asset_config import AdaptiveAssetConfig
from src.core.config import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class RegimeAdjustment:
    """Multipliers to apply to base parameters for a specific regime."""

    regime: MarketRegime

    # Conviction threshold adjustment (additive, not multiplicative)
    conviction_adjustment: float = 0.0  # e.g., +10 means require 70+10=80

    # Stop loss multiplier (1.0 = no change)
    stop_multiplier: float = 1.0

    # Take profit multiplier
    take_profit_multiplier: float = 1.0

    # Position size multiplier
    size_multiplier: float = 1.0

    # Direction bias (-1 to +1, where -1 = shorts only, +1 = longs only)
    direction_bias: float = 0.0

    # Whether to allow trading at all in this regime
    allow_trading: bool = True

    # Description for logging
    description: str = ""


# Default regime adjustments based on quantitative research
DEFAULT_REGIME_ADJUSTMENTS: dict[MarketRegime, RegimeAdjustment] = {
    MarketRegime.TRENDING_UP: RegimeAdjustment(
        regime=MarketRegime.TRENDING_UP,
        conviction_adjustment=-5,  # Slightly lower bar for longs
        stop_multiplier=1.2,  # Wider stops - let trends run
        take_profit_multiplier=1.3,  # Larger targets in trends
        size_multiplier=1.1,  # Slightly larger positions
        direction_bias=0.3,  # Favor longs
        description="Bullish trend: wider stops, larger targets, favor longs",
    ),
    MarketRegime.TRENDING_DOWN: RegimeAdjustment(
        regime=MarketRegime.TRENDING_DOWN,
        conviction_adjustment=-5,  # Slightly lower bar for shorts
        stop_multiplier=1.1,  # Slightly wider stops
        take_profit_multiplier=1.2,  # Larger targets
        size_multiplier=1.0,  # Normal size
        direction_bias=-0.3,  # Favor shorts
        description="Bearish trend: favor shorts, normal sizing",
    ),
    MarketRegime.RANGING: RegimeAdjustment(
        regime=MarketRegime.RANGING,
        conviction_adjustment=10,  # Higher conviction needed - noisy
        stop_multiplier=0.8,  # Tighter stops - less room for movement
        take_profit_multiplier=0.7,  # Smaller targets - range bound
        size_multiplier=0.7,  # Smaller positions - lower edge
        direction_bias=0.0,  # No direction preference
        description="Range-bound: tighter stops, smaller targets, reduced size",
    ),
    MarketRegime.HIGH_VOLATILITY: RegimeAdjustment(
        regime=MarketRegime.HIGH_VOLATILITY,
        conviction_adjustment=15,  # Much higher bar
        stop_multiplier=1.5,  # Much wider stops - volatility needs room
        take_profit_multiplier=1.5,  # Larger targets - volatility creates opportunity
        size_multiplier=0.5,  # Half position size - higher risk per trade
        direction_bias=0.0,  # No direction preference
        description="High volatility: wider stops, smaller size, higher threshold",
    ),
    MarketRegime.EXTREME_VOLATILITY: RegimeAdjustment(
        regime=MarketRegime.EXTREME_VOLATILITY,
        conviction_adjustment=25,  # Maximum threshold
        stop_multiplier=2.0,  # Very wide stops
        take_profit_multiplier=2.0,  # Very large targets
        size_multiplier=0.25,  # Quarter size
        direction_bias=0.0,  # No direction preference
        allow_trading=True,  # Still allow but with extreme caution
        description="Extreme volatility: minimal exposure, maximum caution",
    ),
    MarketRegime.TREND_EXHAUSTION: RegimeAdjustment(
        regime=MarketRegime.TREND_EXHAUSTION,
        conviction_adjustment=20,  # High bar - uncertain
        stop_multiplier=0.8,  # Tighter stops - reversal risk
        take_profit_multiplier=0.6,  # Smaller targets - take profits quickly
        size_multiplier=0.5,  # Half size - uncertain
        direction_bias=0.0,  # Look for reversal signals
        description="Trend exhaustion: prepare for reversal, reduce exposure",
    ),
}


@dataclass
class AdjustedParams:
    """Final adjusted parameters after applying regime adjustments."""

    # Base values (from config)
    base_conviction: float
    base_stop_atr: float
    base_take_profit_atr: float
    base_size_mult: float

    # Adjusted values (after regime)
    adjusted_conviction: float
    adjusted_stop_atr: float
    adjusted_take_profit_atr: float
    adjusted_size_mult: float

    # Direction guidance
    direction_bias: float
    allow_trading: bool

    # Metadata
    regime: MarketRegime
    regime_confidence: float
    adjustment_description: str

    def to_dict(self) -> dict:
        return {
            "base_conviction": self.base_conviction,
            "base_stop_atr": self.base_stop_atr,
            "base_take_profit_atr": self.base_take_profit_atr,
            "base_size_mult": self.base_size_mult,
            "adjusted_conviction": self.adjusted_conviction,
            "adjusted_stop_atr": self.adjusted_stop_atr,
            "adjusted_take_profit_atr": self.adjusted_take_profit_atr,
            "adjusted_size_mult": self.adjusted_size_mult,
            "direction_bias": self.direction_bias,
            "allow_trading": self.allow_trading,
            "regime": self.regime.value,
            "regime_confidence": self.regime_confidence,
            "adjustment_description": self.adjustment_description,
        }


class RegimeAwareParams:
    """
    Adjusts trading parameters in real-time based on detected regime.

    Usage:
        regime_params = RegimeAwareParams()
        adjusted = regime_params.adjust(config, regime_state)
        # Use adjusted.adjusted_* values for trading
    """

    def __init__(
        self,
        custom_adjustments: dict[MarketRegime, RegimeAdjustment] | None = None,
        confidence_scaling: bool = True,
        min_confidence_for_adjustment: float = 0.5,
    ):
        """
        Initialize regime-aware parameter adjuster.

        Args:
            custom_adjustments: Override default regime adjustments
            confidence_scaling: Scale adjustments by regime confidence
            min_confidence_for_adjustment: Don't adjust if confidence below this
        """
        self.adjustments = custom_adjustments or DEFAULT_REGIME_ADJUSTMENTS
        self.confidence_scaling = confidence_scaling
        self.min_confidence = min_confidence_for_adjustment

    def adjust(
        self,
        config: AdaptiveAssetConfig,
        regime: MarketRegime,
        regime_confidence: float = 1.0,
    ) -> AdjustedParams:
        """
        Adjust parameters based on current regime.

        Args:
            config: Base asset configuration
            regime: Detected market regime
            regime_confidence: Confidence in regime detection (0-1)

        Returns:
            AdjustedParams with regime-adjusted values
        """
        # Get adjustment for this regime
        adjustment = self.adjustments.get(
            regime,
            RegimeAdjustment(regime=regime)  # No adjustment if not defined
        )

        # Calculate scaling factor based on confidence
        if self.confidence_scaling and regime_confidence >= self.min_confidence:
            # Scale adjustments linearly with confidence
            # At confidence=0.5, apply 50% of adjustment
            # At confidence=1.0, apply 100% of adjustment
            scale = regime_confidence
        elif regime_confidence < self.min_confidence:
            # Below minimum confidence, don't adjust
            scale = 0.0
        else:
            scale = 1.0

        # Apply adjustments with scaling
        # Conviction: additive adjustment scaled by confidence
        adjusted_conviction = config.conviction_threshold + (
            adjustment.conviction_adjustment * scale
        )
        # Clamp to valid range
        adjusted_conviction = max(50, min(95, adjusted_conviction))

        # Stop/TP: multiplicative adjustment scaled toward 1.0
        # At scale=0.5, multiplier of 1.5 becomes 1.25
        def scale_multiplier(mult: float, s: float) -> float:
            return 1.0 + (mult - 1.0) * s

        adjusted_stop_atr = config.stop_atr_multiplier * scale_multiplier(
            adjustment.stop_multiplier, scale
        )
        adjusted_take_profit_atr = config.take_profit_atr_multiplier * scale_multiplier(
            adjustment.take_profit_multiplier, scale
        )
        adjusted_size_mult = config.position_size_multiplier * scale_multiplier(
            adjustment.size_multiplier, scale
        )

        # Direction bias scaled by confidence
        direction_bias = adjustment.direction_bias * scale

        # Allow trading only if regime says so
        allow_trading = adjustment.allow_trading

        # Enforce minimum position size
        adjusted_size_mult = max(0.1, adjusted_size_mult)

        result = AdjustedParams(
            base_conviction=config.conviction_threshold,
            base_stop_atr=config.stop_atr_multiplier,
            base_take_profit_atr=config.take_profit_atr_multiplier,
            base_size_mult=config.position_size_multiplier,
            adjusted_conviction=adjusted_conviction,
            adjusted_stop_atr=adjusted_stop_atr,
            adjusted_take_profit_atr=adjusted_take_profit_atr,
            adjusted_size_mult=adjusted_size_mult,
            direction_bias=direction_bias,
            allow_trading=allow_trading,
            regime=regime,
            regime_confidence=regime_confidence,
            adjustment_description=adjustment.description,
        )

        logger.debug(
            f"Regime adjustment for {regime.value} (conf={regime_confidence:.2f}): "
            f"conviction {config.conviction_threshold:.0f}→{adjusted_conviction:.0f}, "
            f"stop {config.stop_atr_multiplier:.2f}→{adjusted_stop_atr:.2f}x, "
            f"size {config.position_size_multiplier:.2f}→{adjusted_size_mult:.2f}x"
        )

        return result

    def should_take_trade(
        self,
        adjusted: AdjustedParams,
        signal_direction: str,  # "LONG" or "SHORT"
        signal_conviction: float,
    ) -> tuple[bool, str]:
        """
        Determine if a trade should be taken given regime adjustments.

        Args:
            adjusted: Regime-adjusted parameters
            signal_direction: Direction of the trading signal
            signal_conviction: Conviction score of the signal

        Returns:
            (should_trade, reason)
        """
        # Check if trading allowed
        if not adjusted.allow_trading:
            return False, f"Trading disabled in {adjusted.regime.value} regime"

        # Check conviction threshold
        if signal_conviction < adjusted.adjusted_conviction:
            return False, (
                f"Conviction {signal_conviction:.1f} below adjusted threshold "
                f"{adjusted.adjusted_conviction:.1f}"
            )

        # Check direction bias
        if adjusted.direction_bias > 0.2 and signal_direction == "SHORT":
            # Strong long bias, reject shorts
            return False, (
                f"Direction bias {adjusted.direction_bias:.2f} rejects SHORT in "
                f"{adjusted.regime.value}"
            )
        if adjusted.direction_bias < -0.2 and signal_direction == "LONG":
            # Strong short bias, reject longs
            return False, (
                f"Direction bias {adjusted.direction_bias:.2f} rejects LONG in "
                f"{adjusted.regime.value}"
            )

        return True, "Trade approved"

    def get_regime_summary(self) -> str:
        """Get human-readable summary of all regime adjustments."""
        lines = ["Regime Parameter Adjustments:", "=" * 50]

        for regime, adj in sorted(self.adjustments.items(), key=lambda x: x[0].value):
            lines.extend([
                f"\n{regime.value}:",
                f"  Description: {adj.description}",
                f"  Conviction: {adj.conviction_adjustment:+.0f}",
                f"  Stop mult: {adj.stop_multiplier:.2f}x",
                f"  TP mult: {adj.take_profit_multiplier:.2f}x",
                f"  Size mult: {adj.size_multiplier:.2f}x",
                f"  Direction bias: {adj.direction_bias:+.2f}",
                f"  Allow trading: {adj.allow_trading}",
            ])

        return "\n".join(lines)


def create_symbol_specific_adjustments(
    symbol: str,
    base_adjustments: dict[MarketRegime, RegimeAdjustment] | None = None,
) -> dict[MarketRegime, RegimeAdjustment]:
    """
    Create symbol-specific regime adjustments.

    Different assets respond differently to regimes:
    - BTC: More stable, can use standard adjustments
    - ETH: Higher beta to BTC, more aggressive in trends
    - SOL: High volatility, needs more conservative adjustments
    """
    base = base_adjustments or DEFAULT_REGIME_ADJUSTMENTS.copy()

    # Symbol-specific overrides
    if symbol.upper().startswith("SOL"):
        # SOL is more volatile - be more conservative
        for regime in base:
            adj = base[regime]
            # Increase conviction requirements
            adj.conviction_adjustment += 5
            # Reduce position sizes further
            adj.size_multiplier *= 0.8
            # Widen stops more
            adj.stop_multiplier *= 1.1

    elif symbol.upper().startswith("ETH"):
        # ETH has higher beta - can be more aggressive in trends
        if MarketRegime.TRENDING_UP in base:
            base[MarketRegime.TRENDING_UP].size_multiplier *= 1.1
        if MarketRegime.TRENDING_DOWN in base:
            base[MarketRegime.TRENDING_DOWN].size_multiplier *= 1.1

    # BTC uses defaults

    return base
