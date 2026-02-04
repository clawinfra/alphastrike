"""
Medallion-Style Position Sizer

Position sizing following Jim Simons' Medallion Fund principles:
1. Small positions across many instruments (5-8% max per position)
2. Correlation-adjusted sizing (reduce size if highly correlated to existing)
3. Volatility-adjusted sizing (inverse relationship)
4. Portfolio heat cap (35% max correlated exposure)
5. Dynamic leverage based on diversification

Key insight: The secret to Medallion's 66% returns with <5% drawdown is
not any single trade - it's the DIVERSIFICATION across uncorrelated bets.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    size_usd: float
    entry_price: float


@dataclass
class SizingDecision:
    """Result of position sizing calculation."""
    original_size: float
    adjusted_size: float
    scaling_factor: float
    reason: str
    blocked: bool = False
    components: dict[str, Any] | None = None  # Breakdown of adjustments

    def __post_init__(self):
        if self.components is None:
            self.components = {}


class MedallionPositionSizer:
    """
    Position sizer following Medallion principles.

    Philosophy:
    - Never bet big on any single trade
    - Diversification is the only free lunch
    - Correlation-aware sizing prevents concentration risk
    - Volatility scaling protects during turbulent periods
    """

    # Maximum portfolio heat (correlation-adjusted exposure)
    MAX_PORTFOLIO_HEAT = 0.35  # 35% of balance

    # Maximum single position size
    MAX_SINGLE_POSITION = 0.08  # 8% of balance (was 15-20%)

    # Base position size
    BASE_POSITION_SIZE = 0.05  # 5% of balance

    # Minimum position size (below this, don't trade)
    MIN_POSITION_SIZE = 0.01  # 1% of balance

    # Default asset volatilities (daily, as decimal)
    DEFAULT_VOLATILITIES = {
        "BTCUSDT": 0.025,   # 2.5%
        "ETHUSDT": 0.030,   # 3.0%
        "SOLUSDT": 0.045,   # 4.5%
        "BNBUSDT": 0.035,   # 3.5%
        "XRPUSDT": 0.040,   # 4.0%
        "AVAXUSDT": 0.045,
        "NEARUSDT": 0.050,
        "APTUSDT": 0.050,
        "AAVEUSDT": 0.050,
        "UNIUSDT": 0.045,
        "LINKUSDT": 0.040,
        "RNDRUSDT": 0.055,
        "FETUSDT": 0.055,
        "DOGEUSDT": 0.060,
        "HYPEUSDT": 0.080,
        # Traditional - much lower volatility
        "PAXGUSDT": 0.008,  # 0.8% - Gold is stable
        "SPXUSDT": 0.012,   # 1.2% - S&P 500
    }

    # Default correlations to BTC (simplified - use full matrix in production)
    DEFAULT_BTC_CORRELATIONS = {
        "BTCUSDT": 1.00,
        "ETHUSDT": 0.85,
        "SOLUSDT": 0.75,
        "BNBUSDT": 0.82,
        "XRPUSDT": 0.78,
        "AVAXUSDT": 0.74,
        "NEARUSDT": 0.70,
        "APTUSDT": 0.68,
        "AAVEUSDT": 0.65,
        "UNIUSDT": 0.62,
        "LINKUSDT": 0.68,
        "RNDRUSDT": 0.60,
        "FETUSDT": 0.58,
        "DOGEUSDT": 0.72,
        "HYPEUSDT": 0.55,
        # Traditional - KEY: near-zero correlation
        "PAXGUSDT": 0.08,
        "SPXUSDT": 0.15,
    }

    def __init__(
        self,
        max_portfolio_heat: float = 0.35,
        max_single_position: float = 0.08,
        base_position_size: float = 0.05,
    ):
        """
        Initialize Medallion position sizer.

        Args:
            max_portfolio_heat: Maximum correlation-adjusted exposure (default 35%)
            max_single_position: Maximum single position size (default 8%)
            base_position_size: Base position size before adjustments (default 5%)
        """
        self.max_portfolio_heat = max_portfolio_heat
        self.max_single_position = max_single_position
        self.base_position_size = base_position_size

    def calculate_position_size(
        self,
        symbol: str,
        direction: str,  # "LONG" or "SHORT"
        conviction: float,  # 0-100 conviction score
        balance: float,
        current_positions: dict[str, Position],
        sector_bias: float = 1.0,  # From sector rotation (0.8 - 1.2)
    ) -> SizingDecision:
        """
        Calculate optimal position size using Medallion principles.

        Args:
            symbol: Symbol to trade (e.g., "BTCUSDT")
            direction: Trade direction ("LONG" or "SHORT")
            conviction: Conviction score from ConvictionScorer (0-100)
            balance: Current account balance
            current_positions: Dict of symbol -> Position for open positions
            sector_bias: Sector rotation bias multiplier (default 1.0)

        Returns:
            SizingDecision with adjusted size and reasoning
        """
        components = {}

        # 1. Base size from conviction (higher conviction = larger size)
        # Conviction 50 = base size, 100 = 1.5x base, 0 = 0.5x base
        conviction_scale = 0.5 + (conviction / 100) * 0.5  # 0.5 to 1.0
        base_size = self.base_position_size * conviction_scale * balance
        components["base"] = base_size

        # 2. Volatility adjustment (inverse relationship)
        vol = self.DEFAULT_VOLATILITIES.get(symbol, 0.03)
        normal_vol = 0.03  # 3% is "normal" crypto vol
        vol_factor = min(1.5, max(0.3, normal_vol / vol))
        size = base_size * vol_factor
        components["vol_factor"] = vol_factor

        # 3. Correlation adjustment to existing positions
        corr_factor = self._calculate_correlation_factor(
            symbol, direction, current_positions
        )
        size *= corr_factor
        components["corr_factor"] = corr_factor

        # 4. Apply sector rotation bias
        size *= sector_bias
        components["sector_bias"] = sector_bias

        # 5. Check portfolio heat
        current_heat = self._calculate_portfolio_heat(current_positions, balance)
        components["current_heat"] = current_heat

        if current_heat >= self.max_portfolio_heat:
            # Portfolio is full - block new position
            return SizingDecision(
                original_size=base_size,
                adjusted_size=0.0,
                scaling_factor=0.0,
                reason=f"Portfolio heat {current_heat:.1%} >= max {self.max_portfolio_heat:.0%}",
                blocked=True,
                components=components,
            )

        # Calculate remaining heat capacity
        remaining_heat = self.max_portfolio_heat - current_heat
        asset_corr = self.DEFAULT_BTC_CORRELATIONS.get(symbol, 0.7)
        max_from_heat = (remaining_heat * balance) / max(0.3, asset_corr)
        components["max_from_heat"] = max_from_heat

        # 6. Apply caps
        # Cap at max single position
        max_single = self.max_single_position * balance
        size = min(size, max_single, max_from_heat)
        components["max_single"] = max_single

        # 7. Check minimum size
        min_size = self.MIN_POSITION_SIZE * balance
        if size < min_size:
            return SizingDecision(
                original_size=base_size,
                adjusted_size=0.0,
                scaling_factor=0.0,
                reason=f"Size ${size:.0f} below minimum ${min_size:.0f}",
                blocked=True,
                components=components,
            )

        # Calculate final scaling factor
        scaling_factor = size / base_size if base_size > 0 else 1.0

        reason = self._build_reason(components, size, base_size)

        return SizingDecision(
            original_size=base_size,
            adjusted_size=size,
            scaling_factor=scaling_factor,
            reason=reason,
            blocked=False,
            components=components,
        )

    def _calculate_correlation_factor(
        self,
        symbol: str,
        direction: str,
        current_positions: dict[str, Position],
    ) -> float:
        """
        Calculate correlation penalty/bonus for new position.

        If highly correlated to existing positions in same direction: reduce size
        If opposite direction (hedging): slight bonus
        If uncorrelated asset (PAXG, SPX): bonus
        """
        if not current_positions:
            return 1.0  # No adjustment for first position

        # Check if this is an uncorrelated asset
        symbol_corr = self.DEFAULT_BTC_CORRELATIONS.get(symbol, 0.7)
        if symbol_corr < 0.2:
            # Uncorrelated asset gets bonus - diversifying!
            return 1.2

        # Calculate correlation penalty based on existing positions
        penalty = 1.0
        for existing_symbol, pos in current_positions.items():
            # Estimate correlation between symbols
            existing_corr = self.DEFAULT_BTC_CORRELATIONS.get(existing_symbol, 0.7)
            # Simplified: estimate pair correlation from BTC correlations
            pair_corr = min(symbol_corr, existing_corr) * 0.9

            if pair_corr > 0.5:
                same_direction = (pos.side == direction)
                if same_direction:
                    # Same direction with correlated asset - penalize
                    penalty *= (1 - pair_corr * 0.3)  # Up to 30% reduction
                else:
                    # Opposite direction - hedging, slight bonus
                    penalty *= (1 + pair_corr * 0.1)  # Up to 10% bonus

        return max(0.3, min(1.3, penalty))

    def _calculate_portfolio_heat(
        self,
        positions: dict[str, Position],
        balance: float,
    ) -> float:
        """
        Calculate correlation-adjusted portfolio exposure (heat).

        Key insight: If you have 3 positions at 0.8 correlation,
        your effective risk is ~2.4x, not 3x. But still high!
        """
        if not positions or balance <= 0:
            return 0.0

        # Calculate raw exposure
        raw_exposure = sum(p.size_usd for p in positions.values())

        if len(positions) == 1:
            return raw_exposure / balance

        # Calculate correlation-adjusted exposure
        # Simplified: use average pairwise correlation
        symbols = list(positions.keys())
        correlations = []

        for i, sym1 in enumerate(symbols):
            corr1 = self.DEFAULT_BTC_CORRELATIONS.get(sym1, 0.7)
            for sym2 in symbols[i+1:]:
                corr2 = self.DEFAULT_BTC_CORRELATIONS.get(sym2, 0.7)
                # Estimate pair correlation
                pair_corr = min(corr1, corr2) * 0.9
                correlations.append(pair_corr)

        avg_corr = float(np.mean(correlations)) if correlations else 0.7

        # Portfolio heat formula: raw_exposure * sqrt(1 + (n-1)*avg_corr) / sqrt(n)
        # This accounts for diversification benefit from uncorrelated assets
        n = len(positions)
        diversification_factor = np.sqrt((1 + (n - 1) * avg_corr) / n)
        correlated_exposure = raw_exposure * diversification_factor

        return correlated_exposure / balance

    def _build_reason(
        self,
        components: dict,
        final_size: float,
        base_size: float,
    ) -> str:
        """Build human-readable reason for sizing decision."""
        reasons = []

        vol_factor = components.get("vol_factor", 1.0)
        if vol_factor < 0.8:
            reasons.append(f"high vol ({vol_factor:.1f}x)")
        elif vol_factor > 1.2:
            reasons.append(f"low vol ({vol_factor:.1f}x)")

        corr_factor = components.get("corr_factor", 1.0)
        if corr_factor < 0.9:
            reasons.append(f"corr penalty ({corr_factor:.1f}x)")
        elif corr_factor > 1.1:
            reasons.append(f"diversifier ({corr_factor:.1f}x)")

        sector_bias = components.get("sector_bias", 1.0)
        if sector_bias != 1.0:
            reasons.append(f"sector ({sector_bias:.1f}x)")

        heat = components.get("current_heat", 0)
        if heat > 0.2:
            reasons.append(f"heat {heat:.0%}")

        if not reasons:
            return "standard sizing"

        return ", ".join(reasons)

    def get_portfolio_summary(
        self,
        positions: dict[str, Position],
        balance: float,
    ) -> dict:
        """
        Get portfolio summary for monitoring.

        Returns dict with heat, exposure, and diversification metrics.
        """
        if not positions:
            return {
                "num_positions": 0,
                "raw_exposure": 0,
                "raw_exposure_pct": 0,
                "portfolio_heat": 0,
                "heat_pct": 0,
                "max_heat_pct": self.max_portfolio_heat * 100,
                "headroom_pct": self.max_portfolio_heat * 100,
                "avg_correlation": 0,
                "diversification_benefit": 0,
            }

        raw_exposure = sum(p.size_usd for p in positions.values())
        heat = self._calculate_portfolio_heat(positions, balance)

        # Calculate average correlation
        symbols = list(positions.keys())
        correlations = []
        for i, sym1 in enumerate(symbols):
            corr1 = self.DEFAULT_BTC_CORRELATIONS.get(sym1, 0.7)
            for sym2 in symbols[i+1:]:
                corr2 = self.DEFAULT_BTC_CORRELATIONS.get(sym2, 0.7)
                pair_corr = min(corr1, corr2) * 0.9
                correlations.append(pair_corr)

        avg_corr = float(np.mean(correlations)) if correlations else 0.0

        # Diversification benefit: how much heat reduction from diversification
        div_benefit = 1 - (heat / (raw_exposure / balance)) if raw_exposure > 0 else 0

        return {
            "num_positions": len(positions),
            "raw_exposure": raw_exposure,
            "raw_exposure_pct": (raw_exposure / balance) * 100 if balance > 0 else 0,
            "portfolio_heat": heat * balance,
            "heat_pct": heat * 100,
            "max_heat_pct": self.max_portfolio_heat * 100,
            "headroom_pct": (self.max_portfolio_heat - heat) * 100,
            "avg_correlation": avg_corr,
            "diversification_benefit": div_benefit * 100,
        }
