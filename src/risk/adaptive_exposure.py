"""
US-020: Adaptive Exposure Manager for AlphaStrike trading bot.

Dynamically adjusts exposure limits based on market conditions including
volatility, market regime, drawdown, and position concentration.
"""

from dataclasses import dataclass


@dataclass
class ExposureLimits:
    """Container for adaptive exposure limits."""

    per_trade: float
    per_pair: float
    total: float


class AdaptiveExposure:
    """
    Manages adaptive exposure limits based on market conditions.

    Adjusts base exposure limits using scaling factors derived from:
    - Volatility (ATR ratio)
    - Market regime (trending vs ranging)
    - Current drawdown
    - Position concentration
    """

    # Base exposure limits (as percentages)
    BASE_PER_TRADE = 0.10  # 10%
    BASE_PER_PAIR = 0.25  # 25%
    BASE_TOTAL = 0.80  # 80%

    # Scaling factor ranges
    VOLATILITY_FACTOR_MIN = 0.6
    VOLATILITY_FACTOR_MAX = 1.3

    REGIME_FACTOR_MIN = 0.7
    REGIME_FACTOR_MAX = 1.2

    DRAWDOWN_FACTOR_MIN = 0.5
    DRAWDOWN_FACTOR_MAX = 1.0

    CONCENTRATION_FACTOR_MIN = 0.7
    CONCENTRATION_FACTOR_MAX = 1.2

    def calculate_limits(
        self,
        volatility: float,
        regime: str,
        drawdown: float,
        position_count: int,
    ) -> ExposureLimits:
        """
        Calculate adaptive exposure limits based on current market conditions.

        Args:
            volatility: ATR ratio (current ATR / average ATR). Higher = more volatile.
            regime: Market regime ('trending', 'ranging', or 'volatile').
            drawdown: Current drawdown as a percentage (0.0 to 1.0).
            position_count: Current number of open positions.

        Returns:
            ExposureLimits with adapted per_trade, per_pair, and total limits.
        """
        vol_factor = self.get_volatility_factor(volatility)
        regime_factor = self.get_regime_factor(regime)
        dd_factor = self.get_drawdown_factor(drawdown)
        concentration_factor = self.get_concentration_factor(position_count)

        # Apply formulas
        adaptive_per_trade = (
            self.BASE_PER_TRADE * vol_factor * regime_factor * dd_factor
        )
        adaptive_per_pair = self.BASE_PER_PAIR * vol_factor * concentration_factor
        adaptive_total = self.BASE_TOTAL * vol_factor * regime_factor * dd_factor

        return ExposureLimits(
            per_trade=adaptive_per_trade,
            per_pair=adaptive_per_pair,
            total=adaptive_total,
        )

    def get_volatility_factor(self, atr_ratio: float) -> float:
        """
        Calculate volatility scaling factor from ATR ratio.

        Higher volatility leads to tighter limits (lower factor).

        Args:
            atr_ratio: Current ATR / average ATR. 1.0 = normal volatility.

        Returns:
            Scaling factor between 0.6 (high vol) and 1.3 (low vol).
        """
        # Inverse relationship: high volatility -> lower factor
        # atr_ratio of 0.5 -> factor of 1.3 (low vol, more exposure)
        # atr_ratio of 1.0 -> factor of 1.0 (normal)
        # atr_ratio of 2.0 -> factor of 0.6 (high vol, less exposure)
        if atr_ratio <= 0:
            return self.VOLATILITY_FACTOR_MAX

        # Linear interpolation: factor = 1.6 - 0.35 * atr_ratio
        # At atr_ratio=0.5: 1.6 - 0.175 = 1.425 -> clamped to 1.3
        # At atr_ratio=1.0: 1.6 - 0.35 = 1.25 -> ~1.0 with adjustment
        # At atr_ratio=2.0: 1.6 - 0.7 = 0.9 -> ~0.6 with adjustment

        # Adjusted formula for better mapping
        factor = 1.3 - (atr_ratio - 0.5) * 0.7 / 1.5

        return max(self.VOLATILITY_FACTOR_MIN, min(self.VOLATILITY_FACTOR_MAX, factor))

    def get_regime_factor(self, regime: str) -> float:
        """
        Calculate regime scaling factor.

        Trending markets allow more exposure, ranging/volatile markets less.

        Args:
            regime: Market regime ('trending', 'ranging', or 'volatile').

        Returns:
            Scaling factor between 0.7 and 1.2.
        """
        regime_factors = {
            "trending": 1.2,  # More exposure in trending markets
            "ranging": 0.9,  # Moderate exposure in ranging markets
            "volatile": 0.7,  # Tighter limits in volatile markets
        }

        return regime_factors.get(regime.lower(), 1.0)

    def get_drawdown_factor(self, drawdown_pct: float) -> float:
        """
        Calculate drawdown scaling factor.

        Higher drawdown leads to tighter limits.

        Args:
            drawdown_pct: Current drawdown as percentage (0.0 to 1.0).
                         0.0 = no drawdown, 0.2 = 20% drawdown.

        Returns:
            Scaling factor between 0.5 (deep drawdown) and 1.0 (no drawdown).
        """
        # Clamp drawdown to valid range
        drawdown_pct = max(0.0, min(1.0, drawdown_pct))

        # Linear scaling: 0% drawdown -> 1.0, 100% drawdown -> 0.5
        factor = self.DRAWDOWN_FACTOR_MAX - drawdown_pct * (
            self.DRAWDOWN_FACTOR_MAX - self.DRAWDOWN_FACTOR_MIN
        )

        return factor

    def get_concentration_factor(self, position_count: int) -> float:
        """
        Calculate concentration scaling factor.

        Fewer positions allow more exposure per position.

        Args:
            position_count: Current number of open positions.

        Returns:
            Scaling factor between 0.7 (many positions) and 1.2 (few positions).
        """
        # 0-1 positions: max factor (1.2)
        # 2-3 positions: moderate factor (1.0)
        # 4-5 positions: reduced factor (0.85)
        # 6+ positions: min factor (0.7)

        if position_count <= 1:
            return self.CONCENTRATION_FACTOR_MAX
        elif position_count <= 3:
            return 1.0
        elif position_count <= 5:
            return 0.85
        else:
            return self.CONCENTRATION_FACTOR_MIN
