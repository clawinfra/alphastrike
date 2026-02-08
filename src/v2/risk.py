"""
V2 Risk Manager — Kelly Sizing + Correlation-Aware Limits

Key differences from V1:
1. Kelly criterion position sizing (half-Kelly for safety)
2. Correlation group limits (no stacking correlated bets)
3. Funding rate cost estimation in position sizing
4. Realistic per-asset slippage model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.v2.models import Prediction
from src.v2.regime import Regime, RegimeState

logger = logging.getLogger(__name__)


# Correlation groups — assets that move together
# Max exposure per group prevents "false diversification"
CORRELATION_GROUPS: dict[str, list[str]] = {
    "majors": ["BTC", "ETH", "BNB"],
    "l1_alts": ["SOL", "AVAX", "NEAR", "APT"],
    "defi": ["AAVE", "UNI", "LINK"],
    "speculative": ["FET", "DOGE", "XRP"],
    "traditional": ["PAXG", "SPX"],
}

# Reverse lookup: asset → group
ASSET_GROUP: dict[str, str] = {}
for group, assets in CORRELATION_GROUPS.items():
    for asset in assets:
        ASSET_GROUP[asset] = group


# Per-asset slippage estimates (basis points)
# Based on typical Hyperliquid liquidity depth
ASSET_SLIPPAGE_BPS: dict[str, float] = {
    "BTC": 3.0,
    "ETH": 4.0,
    "BNB": 6.0,
    "SOL": 6.0,
    "XRP": 8.0,
    "AVAX": 10.0,
    "NEAR": 12.0,
    "APT": 12.0,
    "AAVE": 15.0,
    "UNI": 15.0,
    "LINK": 10.0,
    "FET": 20.0,
    "DOGE": 10.0,
    "PAXG": 15.0,
    "SPX": 8.0,
}

DEFAULT_SLIPPAGE_BPS = 10.0


@dataclass
class RiskLimits:
    """Risk limits configuration."""

    # Position sizing
    max_single_position: float = 0.05  # 5% of balance per position
    max_portfolio_exposure: float = 0.40  # 40% total (margin, not notional)
    max_group_exposure: float = 0.15  # 15% per correlation group
    max_leverage: float = 5.0  # max leverage
    min_position_usd: float = 50.0  # minimum position size in USD

    # Kelly sizing
    kelly_fraction: float = 0.5  # half-Kelly (safer)
    max_kelly_size: float = 0.08  # cap Kelly at 8% even with strong signal

    # Drawdown circuit breakers
    daily_drawdown_halt: float = 0.05  # 5% daily DD → stop trading
    total_drawdown_halt: float = 0.12  # 12% total DD → stop trading

    # Costs
    taker_fee: float = 0.0005  # 0.05%
    funding_rate_per_8h: float = 0.0001  # 0.01% per 8h


@dataclass
class PositionSize:
    """Result of position sizing calculation."""

    size_usd: float  # dollar amount (margin)
    leverage: float
    notional_usd: float  # size * leverage
    kelly_raw: float  # raw Kelly fraction
    kelly_applied: float  # after capping
    blocked: bool
    block_reason: str = ""
    estimated_slippage_bps: float = 0.0
    estimated_round_trip_cost: float = 0.0  # total cost (fees + slippage + funding)


class V2RiskManager:
    """
    Risk manager with Kelly sizing and correlation awareness.

    Usage:
        risk = V2RiskManager()
        size = risk.calculate_position_size(
            prediction=prediction,
            symbol="SOL",
            balance=10000,
            current_positions=positions,
            regime=regime_state,
        )
        if not size.blocked:
            # execute trade
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

        # Track state
        self._daily_start_balance: float = 0.0
        self._peak_balance: float = 0.0

    def set_initial_balance(self, balance: float) -> None:
        """Set initial balance for drawdown tracking."""
        self._daily_start_balance = balance
        self._peak_balance = balance

    def update_balance(self, balance: float) -> None:
        """Update balance for drawdown calculations."""
        if balance > self._peak_balance:
            self._peak_balance = balance

    def reset_daily(self, balance: float) -> None:
        """Reset daily tracking (call at start of each day)."""
        self._daily_start_balance = balance

    def calculate_position_size(
        self,
        prediction: Prediction,
        symbol: str,
        balance: float,
        current_positions: dict[str, dict],
        regime: RegimeState,
    ) -> PositionSize:
        """
        Calculate position size using Kelly criterion with risk guards.

        Args:
            prediction: Model prediction with return and uncertainty
            symbol: Asset symbol (e.g., "SOLUSDT")
            balance: Current portfolio balance
            current_positions: Dict of {symbol: {side, size_usd, ...}}
            regime: Current regime state

        Returns:
            PositionSize with sizing details or blocked status
        """
        # Extract base asset (SOLUSDT → SOL)
        base_asset = symbol.replace("USDT", "").replace("USD", "")

        # === Circuit breakers ===

        # Check drawdown limits
        daily_dd = self._daily_drawdown(balance)
        total_dd = self._total_drawdown(balance)

        if daily_dd >= self.limits.daily_drawdown_halt:
            return self._blocked(f"daily drawdown {daily_dd:.1%} >= {self.limits.daily_drawdown_halt:.1%}")

        if total_dd >= self.limits.total_drawdown_halt:
            return self._blocked(f"total drawdown {total_dd:.1%} >= {self.limits.total_drawdown_halt:.1%}")

        # Check regime — no new positions in CHAOS
        if regime.regime == Regime.CHAOS:
            return self._blocked("CHAOS regime — no new positions")

        # Check prediction quality
        if prediction.signal == "HOLD":
            return self._blocked("signal is HOLD")

        if abs(prediction.predicted_return) < self.limits.taker_fee * 2:
            return self._blocked(
                f"predicted return {prediction.predicted_return:.4f} below breakeven "
                f"(2x fee = {self.limits.taker_fee * 2:.4f})"
            )

        # === Exposure checks ===

        current_exposure = self._total_exposure(current_positions, balance)
        if current_exposure >= self.limits.max_portfolio_exposure:
            return self._blocked(
                f"portfolio exposure {current_exposure:.1%} >= {self.limits.max_portfolio_exposure:.1%}"
            )

        # Check correlation group exposure
        group = ASSET_GROUP.get(base_asset, "other")
        group_exposure = self._group_exposure(group, current_positions, balance)
        if group_exposure >= self.limits.max_group_exposure:
            return self._blocked(
                f"{group} group exposure {group_exposure:.1%} >= {self.limits.max_group_exposure:.1%}"
            )

        # Check if already have position in this symbol
        if symbol in current_positions:
            return self._blocked(f"already have position in {symbol}")

        # === Kelly sizing ===

        kelly_raw = self._kelly_fraction(prediction)
        kelly_capped = min(kelly_raw, self.limits.max_kelly_size)

        # Apply half-Kelly
        kelly_applied = kelly_capped * self.limits.kelly_fraction

        # Regime adjustment
        regime_multiplier = self._regime_multiplier(regime)
        kelly_adjusted = kelly_applied * regime_multiplier

        # Calculate size
        size_usd = balance * kelly_adjusted
        size_usd = min(size_usd, balance * self.limits.max_single_position)
        size_usd = min(size_usd, balance * (self.limits.max_portfolio_exposure - current_exposure))
        size_usd = min(size_usd, balance * (self.limits.max_group_exposure - group_exposure))

        if size_usd < self.limits.min_position_usd:
            return self._blocked(f"size ${size_usd:.0f} below minimum ${self.limits.min_position_usd:.0f}")

        # Cost estimation
        leverage = min(self.limits.max_leverage, 5.0)
        notional = size_usd * leverage
        slippage_bps = ASSET_SLIPPAGE_BPS.get(base_asset, DEFAULT_SLIPPAGE_BPS)
        round_trip_cost = (
            notional * self.limits.taker_fee * 2  # entry + exit fees
            + notional * slippage_bps / 10000 * 2  # entry + exit slippage
            + notional * self.limits.funding_rate_per_8h * 4  # ~32h average holding
        )

        return PositionSize(
            size_usd=size_usd,
            leverage=leverage,
            notional_usd=notional,
            kelly_raw=kelly_raw,
            kelly_applied=kelly_adjusted,
            blocked=False,
            estimated_slippage_bps=slippage_bps,
            estimated_round_trip_cost=round_trip_cost,
        )

    def _kelly_fraction(self, prediction: Prediction) -> float:
        """
        Kelly criterion: f* = μ / σ²

        Where:
          μ = predicted return
          σ = predicted std (uncertainty)

        We use the model's predicted_return as μ and residual_std as σ.
        """
        mu = abs(prediction.predicted_return)
        sigma = max(prediction.predicted_std, 0.001)  # floor to prevent division by zero

        # Subtract costs from expected return
        cost_per_trade = self.limits.taker_fee * 2 + 0.001  # fees + avg slippage
        net_mu = mu - cost_per_trade

        if net_mu <= 0:
            return 0.0

        kelly = net_mu / (sigma**2)
        return float(np.clip(kelly, 0, 0.20))  # hard cap at 20%

    def _regime_multiplier(self, regime: RegimeState) -> float:
        """Regime-based position size multiplier."""
        multipliers = {
            Regime.TREND_UP: 1.0,
            Regime.TREND_DOWN: 0.5,  # reduce exposure
            Regime.RANGE: 0.8,  # slightly smaller for mean-reversion
            Regime.CHAOS: 0.0,  # no trading
        }
        base = multipliers.get(regime.regime, 0.5)

        # Scale by regime confidence
        return base * (0.5 + 0.5 * regime.confidence)

    def _total_exposure(self, positions: dict, balance: float) -> float:
        """Total portfolio exposure as fraction of balance."""
        if balance <= 0:
            return 1.0
        total = sum(pos.get("size_usd", 0) for pos in positions.values())
        return total / balance

    def _group_exposure(self, group: str, positions: dict, balance: float) -> float:
        """Exposure for a correlation group."""
        if balance <= 0:
            return 1.0
        group_assets = CORRELATION_GROUPS.get(group, [])
        total = 0.0
        for symbol, pos in positions.items():
            base = symbol.replace("USDT", "").replace("USD", "")
            if base in group_assets:
                total += pos.get("size_usd", 0)
        return total / balance

    def _daily_drawdown(self, balance: float) -> float:
        """Current daily drawdown."""
        if self._daily_start_balance <= 0:
            return 0.0
        loss = self._daily_start_balance - balance
        return max(0, loss / self._daily_start_balance)

    def _total_drawdown(self, balance: float) -> float:
        """Current total drawdown from peak."""
        if self._peak_balance <= 0:
            return 0.0
        loss = self._peak_balance - balance
        return max(0, loss / self._peak_balance)

    def _blocked(self, reason: str) -> PositionSize:
        """Return a blocked position size."""
        return PositionSize(
            size_usd=0.0,
            leverage=0.0,
            notional_usd=0.0,
            kelly_raw=0.0,
            kelly_applied=0.0,
            blocked=True,
            block_reason=reason,
        )
