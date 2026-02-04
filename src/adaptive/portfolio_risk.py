"""
Portfolio Risk Manager - Correlation-Aware Position Sizing

The key insight from Simons: Don't treat correlated assets as independent.
BTC, ETH, SOL move together (~0.8 correlation). If you're long all 3,
your effective risk is 3x what you think.

This module provides:
1. Real-time correlation tracking
2. Portfolio heat calculation (total correlated exposure)
3. Volatility regime detection
4. Dynamic position scaling based on portfolio risk
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VolatilityRegime:
    """Current volatility regime."""
    level: str  # "low", "normal", "high", "extreme"
    current_vol: float  # Current realized volatility
    percentile: float  # Current vol as percentile of historical
    scaling_factor: float  # Position size multiplier (lower for high vol)


@dataclass
class PortfolioHeat:
    """Current portfolio risk exposure."""
    raw_exposure: float  # Sum of all position sizes
    correlated_exposure: float  # Correlation-adjusted exposure
    heat_pct: float  # Portfolio heat as % of balance
    max_heat_pct: float  # Maximum allowed heat
    headroom: float  # Remaining capacity before max heat
    symbols_long: list[str] = field(default_factory=list)
    symbols_short: list[str] = field(default_factory=list)


@dataclass
class RiskDecision:
    """Risk management decision."""
    action: str  # "allow", "reduce", "block", "close_all"
    original_size: float
    adjusted_size: float
    reason: str
    scaling_factor: float


class PortfolioRiskManager:
    """
    Manages portfolio-level risk with correlation awareness.

    Key features:
    1. Tracks real-time correlation between assets
    2. Calculates "portfolio heat" (correlated exposure)
    3. Detects volatility regime changes
    4. Dynamically scales positions based on total risk

    Usage:
        manager = PortfolioRiskManager(max_heat_pct=0.15)

        # On each price update
        manager.update_price("BTCUSDT", price, timestamp)

        # Before opening a position
        decision = manager.check_new_position("ETHUSDT", side="LONG", size=1000)
        if decision.action == "allow":
            open_position(size=decision.adjusted_size)
    """

    # Default correlation matrix for Medallion-style diversified portfolio
    # Includes traditional assets (PAXG, SPX) with near-zero crypto correlation
    DEFAULT_CORRELATIONS = {
        # =================================================================
        # CRYPTO MAJOR correlations (high, 0.75-0.85)
        # =================================================================
        ("BTCUSDT", "ETHUSDT"): 0.85,
        ("BTCUSDT", "BNBUSDT"): 0.82,
        ("BTCUSDT", "XRPUSDT"): 0.78,
        ("ETHUSDT", "BNBUSDT"): 0.78,
        ("ETHUSDT", "XRPUSDT"): 0.76,
        ("BNBUSDT", "XRPUSDT"): 0.72,

        # =================================================================
        # LAYER 1 correlations (moderate-high, 0.65-0.80)
        # =================================================================
        ("BTCUSDT", "SOLUSDT"): 0.75,
        ("BTCUSDT", "AVAXUSDT"): 0.74,
        ("BTCUSDT", "NEARUSDT"): 0.70,
        ("BTCUSDT", "APTUSDT"): 0.68,
        ("ETHUSDT", "SOLUSDT"): 0.80,
        ("ETHUSDT", "AVAXUSDT"): 0.77,
        ("ETHUSDT", "NEARUSDT"): 0.72,
        ("ETHUSDT", "APTUSDT"): 0.70,
        ("SOLUSDT", "AVAXUSDT"): 0.72,
        ("SOLUSDT", "NEARUSDT"): 0.68,
        ("SOLUSDT", "APTUSDT"): 0.70,
        ("AVAXUSDT", "NEARUSDT"): 0.65,
        ("AVAXUSDT", "APTUSDT"): 0.66,
        ("NEARUSDT", "APTUSDT"): 0.72,

        # =================================================================
        # DEFI correlations (moderate, 0.60-0.75)
        # =================================================================
        ("ETHUSDT", "AAVEUSDT"): 0.72,
        ("ETHUSDT", "UNIUSDT"): 0.70,
        ("ETHUSDT", "LINKUSDT"): 0.68,
        ("BTCUSDT", "AAVEUSDT"): 0.65,
        ("BTCUSDT", "UNIUSDT"): 0.62,
        ("BTCUSDT", "LINKUSDT"): 0.68,
        ("AAVEUSDT", "UNIUSDT"): 0.68,
        ("AAVEUSDT", "LINKUSDT"): 0.60,
        ("UNIUSDT", "LINKUSDT"): 0.58,
        ("SOLUSDT", "AAVEUSDT"): 0.55,

        # =================================================================
        # AI/COMPUTE correlations (moderate, 0.55-0.70)
        # =================================================================
        ("BTCUSDT", "RNDRUSDT"): 0.60,
        ("BTCUSDT", "FETUSDT"): 0.58,
        ("ETHUSDT", "RNDRUSDT"): 0.65,
        ("ETHUSDT", "FETUSDT"): 0.62,
        ("RNDRUSDT", "FETUSDT"): 0.75,

        # =================================================================
        # MEME correlations (variable, 0.50-0.72)
        # =================================================================
        ("BTCUSDT", "DOGEUSDT"): 0.72,
        ("BTCUSDT", "HYPEUSDT"): 0.55,
        ("ETHUSDT", "DOGEUSDT"): 0.70,
        ("ETHUSDT", "HYPEUSDT"): 0.52,
        ("DOGEUSDT", "HYPEUSDT"): 0.60,
        ("SOLUSDT", "DOGEUSDT"): 0.68,

        # =================================================================
        # TRADITIONAL - CRITICAL DIVERSIFIERS (near-zero crypto correlation)
        # These are the KEY to achieving Medallion-style drawdown reduction
        # =================================================================
        # PAXG (Gold) - Safe haven, inverse in crashes
        ("BTCUSDT", "PAXGUSDT"): 0.08,   # Near-zero - KEY DIVERSIFIER
        ("ETHUSDT", "PAXGUSDT"): 0.06,
        ("SOLUSDT", "PAXGUSDT"): 0.05,
        ("BNBUSDT", "PAXGUSDT"): 0.07,
        ("XRPUSDT", "PAXGUSDT"): 0.04,
        ("DOGEUSDT", "PAXGUSDT"): 0.03,
        ("AAVEUSDT", "PAXGUSDT"): 0.05,
        ("AVAXUSDT", "PAXGUSDT"): 0.06,
        ("NEARUSDT", "PAXGUSDT"): 0.05,

        # SPX (S&P 500) - Equity market correlation
        ("BTCUSDT", "SPXUSDT"): 0.15,    # Low but positive - KEY DIVERSIFIER
        ("ETHUSDT", "SPXUSDT"): 0.12,
        ("SOLUSDT", "SPXUSDT"): 0.10,
        ("BNBUSDT", "SPXUSDT"): 0.13,
        ("XRPUSDT", "SPXUSDT"): 0.08,
        ("DOGEUSDT", "SPXUSDT"): 0.05,
        ("AAVEUSDT", "SPXUSDT"): 0.10,
        ("AVAXUSDT", "SPXUSDT"): 0.11,
        ("NEARUSDT", "SPXUSDT"): 0.09,

        # Traditional assets correlate with each other
        ("PAXGUSDT", "SPXUSDT"): 0.25,   # Gold-equity moderate correlation
    }

    def __init__(
        self,
        symbols: list[str],
        max_heat_pct: float = 0.20,  # Max 20% portfolio heat
        max_single_position_pct: float = 0.08,  # Max 8% per position
        vol_lookback: int = 24,  # Hours for volatility calculation
        correlation_lookback: int = 168,  # 1 week for correlation
        vol_scaling: bool = True,  # Scale positions by volatility
        correlation_aware: bool = True,  # Use correlation in heat calc
    ):
        self.symbols = symbols
        self.max_heat_pct = max_heat_pct
        self.max_single_position_pct = max_single_position_pct
        self.vol_lookback = vol_lookback
        self.correlation_lookback = correlation_lookback
        self.vol_scaling = vol_scaling
        self.correlation_aware = correlation_aware

        # Price history for correlation/volatility calculation
        self._price_history: dict[str, deque] = {
            s: deque(maxlen=correlation_lookback) for s in symbols
        }
        self._return_history: dict[str, deque] = {
            s: deque(maxlen=correlation_lookback) for s in symbols
        }

        # Current state
        self._positions: dict[str, dict] = {}  # symbol -> {side, size, entry_price}
        self._correlations: dict[tuple[str, str], float] = dict(self.DEFAULT_CORRELATIONS)
        self._volatilities: dict[str, float] = {s: 0.02 for s in symbols}  # Default 2% vol
        self._vol_percentiles: dict[str, float] = {s: 50.0 for s in symbols}

        # Historical volatility for percentile calculation
        self._vol_history: dict[str, deque] = {
            s: deque(maxlen=500) for s in symbols
        }

        # Balance tracking
        self._balance: float = 10000.0

    def set_balance(self, balance: float) -> None:
        """Update current balance."""
        self._balance = balance

    def update_price(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
    ) -> None:
        """
        Update price and recalculate volatility/correlation.

        Call this on each candle close for all symbols.
        """
        if symbol not in self._price_history:
            return

        history = self._price_history[symbol]
        returns = self._return_history[symbol]

        if history:
            # Calculate return
            prev_price = history[-1]
            ret = (price - prev_price) / prev_price
            returns.append(ret)

            # Update volatility (realized vol)
            if len(returns) >= self.vol_lookback:
                recent_returns = list(returns)[-self.vol_lookback:]
                vol = np.std(recent_returns) * np.sqrt(24)  # Annualized hourly vol
                self._volatilities[symbol] = vol
                self._vol_history[symbol].append(vol)

                # Update volatility percentile
                if len(self._vol_history[symbol]) >= 50:
                    all_vols = list(self._vol_history[symbol])
                    percentile = (sum(1 for v in all_vols if v < vol) / len(all_vols)) * 100
                    self._vol_percentiles[symbol] = percentile

        history.append(price)

        # Update correlations periodically
        if len(returns) >= 20 and len(returns) % 10 == 0:
            self._update_correlations()

    def _update_correlations(self) -> None:
        """Recalculate correlations between assets."""
        for i, sym1 in enumerate(self.symbols):
            for sym2 in self.symbols[i+1:]:
                ret1 = list(self._return_history[sym1])
                ret2 = list(self._return_history[sym2])

                # Need at least 20 observations
                min_len = min(len(ret1), len(ret2))
                if min_len < 20:
                    continue

                # Use recent returns
                ret1 = ret1[-min_len:]
                ret2 = ret2[-min_len:]

                # Calculate correlation
                corr = np.corrcoef(ret1, ret2)[0, 1]
                if not np.isnan(corr):
                    self._correlations[(sym1, sym2)] = corr
                    self._correlations[(sym2, sym1)] = corr

    def get_correlation(self, sym1: str, sym2: str) -> float:
        """Get correlation between two symbols."""
        if sym1 == sym2:
            return 1.0
        key = (sym1, sym2)
        if key in self._correlations:
            return self._correlations[key]
        key = (sym2, sym1)
        return self._correlations.get(key, 0.7)  # Default moderate correlation

    def get_volatility_regime(self, symbol: str) -> VolatilityRegime:
        """Get current volatility regime for a symbol."""
        vol = self._volatilities.get(symbol, 0.02)
        percentile = self._vol_percentiles.get(symbol, 50.0)

        # Determine regime
        if percentile < 25:
            level = "low"
            scaling = 1.2  # Can size up in low vol
        elif percentile < 75:
            level = "normal"
            scaling = 1.0
        elif percentile < 90:
            level = "high"
            scaling = 0.7  # Reduce size in high vol
        else:
            level = "extreme"
            scaling = 0.4  # Significantly reduce in extreme vol

        return VolatilityRegime(
            level=level,
            current_vol=vol,
            percentile=percentile,
            scaling_factor=scaling,
        )

    def register_position(
        self,
        symbol: str,
        side: str,  # "LONG" or "SHORT"
        size: float,
        entry_price: float,
    ) -> None:
        """Register an open position."""
        self._positions[symbol] = {
            "side": side,
            "size": size,
            "entry_price": entry_price,
        }

    def close_position(self, symbol: str) -> None:
        """Remove a closed position."""
        if symbol in self._positions:
            del self._positions[symbol]

    def calculate_portfolio_heat(self) -> PortfolioHeat:
        """
        Calculate current portfolio heat (correlation-adjusted exposure).

        The key insight: If you're long BTC, ETH, SOL with 0.8 correlation,
        your effective exposure is ~2.4x (not 3x) due to diversification,
        but still much higher than any single position.
        """
        if not self._positions:
            return PortfolioHeat(
                raw_exposure=0.0,
                correlated_exposure=0.0,
                heat_pct=0.0,
                max_heat_pct=self.max_heat_pct,
                headroom=self.max_heat_pct,
            )

        # Calculate raw exposure
        raw_exposure = sum(p["size"] for p in self._positions.values())

        # Calculate correlation-adjusted exposure
        if not self.correlation_aware or len(self._positions) == 1:
            correlated_exposure = raw_exposure
        else:
            # Use portfolio variance formula: Var(P) = sum(w_i^2 * var_i) + sum(w_i*w_j*cov_ij)
            symbols = list(self._positions.keys())
            sizes = [self._positions[s]["size"] for s in symbols]
            total_size = sum(sizes)
            weights = [s / total_size for s in sizes]

            # Calculate portfolio variance
            portfolio_var = 0.0
            for i, sym1 in enumerate(symbols):
                vol1 = self._volatilities.get(sym1, 0.02)
                portfolio_var += weights[i] ** 2 * vol1 ** 2

                for j, sym2 in enumerate(symbols):
                    if j > i:
                        vol2 = self._volatilities.get(sym2, 0.02)
                        corr = self.get_correlation(sym1, sym2)
                        portfolio_var += 2 * weights[i] * weights[j] * vol1 * vol2 * corr

            # Correlation multiplier: sqrt(portfolio_var / sum(individual_vars))
            individual_var = sum(w**2 * self._volatilities.get(s, 0.02)**2
                                for s, w in zip(symbols, weights))
            if individual_var > 0:
                corr_multiplier = np.sqrt(portfolio_var / individual_var)
            else:
                corr_multiplier = 1.0

            correlated_exposure = raw_exposure * min(corr_multiplier, 1.5)

        heat_pct = correlated_exposure / self._balance if self._balance > 0 else 0
        headroom = max(0, self.max_heat_pct - heat_pct)

        return PortfolioHeat(
            raw_exposure=raw_exposure,
            correlated_exposure=correlated_exposure,
            heat_pct=heat_pct,
            max_heat_pct=self.max_heat_pct,
            headroom=headroom,
            symbols_long=[s for s, p in self._positions.items() if p["side"] == "LONG"],
            symbols_short=[s for s, p in self._positions.items() if p["side"] == "SHORT"],
        )

    def check_new_position(
        self,
        symbol: str,
        side: str,
        size: float,
    ) -> RiskDecision:
        """
        Check if a new position should be allowed given portfolio risk.

        Returns a RiskDecision with potentially adjusted size.
        """
        # Calculate current heat
        current_heat = self.calculate_portfolio_heat()

        # Get volatility regime
        vol_regime = self.get_volatility_regime(symbol)

        # Apply volatility scaling
        vol_adjusted_size = size * vol_regime.scaling_factor if self.vol_scaling else size

        # Check single position limit
        max_single = self._balance * self.max_single_position_pct
        if vol_adjusted_size > max_single:
            vol_adjusted_size = max_single

        # Calculate new heat if this position is added
        # Simplified: assume new position is correlated with existing positions
        avg_correlation = 0.8  # Conservative estimate for crypto
        if self._positions:
            new_correlated_exposure = current_heat.correlated_exposure + vol_adjusted_size * avg_correlation
        else:
            new_correlated_exposure = vol_adjusted_size

        new_heat_pct = new_correlated_exposure / self._balance if self._balance > 0 else 0

        # Decision logic
        if new_heat_pct > self.max_heat_pct * 1.5:
            # Way over limit - block entirely
            return RiskDecision(
                action="block",
                original_size=size,
                adjusted_size=0.0,
                reason=f"Portfolio heat {new_heat_pct*100:.1f}% exceeds max {self.max_heat_pct*100:.1f}%",
                scaling_factor=0.0,
            )
        elif new_heat_pct > self.max_heat_pct:
            # Over limit - reduce size
            available = (self.max_heat_pct - current_heat.heat_pct) * self._balance / avg_correlation
            if available <= 0:
                return RiskDecision(
                    action="block",
                    original_size=size,
                    adjusted_size=0.0,
                    reason="No portfolio heat headroom",
                    scaling_factor=0.0,
                )
            adjusted_size = min(vol_adjusted_size, available)
            return RiskDecision(
                action="reduce",
                original_size=size,
                adjusted_size=adjusted_size,
                reason=f"Reduced to stay under {self.max_heat_pct*100:.0f}% heat",
                scaling_factor=adjusted_size / size,
            )
        else:
            # Under limit - allow (possibly with vol scaling)
            return RiskDecision(
                action="allow",
                original_size=size,
                adjusted_size=vol_adjusted_size,
                reason=f"Within limits (heat: {new_heat_pct*100:.1f}%)",
                scaling_factor=vol_adjusted_size / size if size > 0 else 1.0,
            )

    def get_emergency_action(self) -> str | None:
        """
        Check if emergency action is needed.

        Returns "close_all" if portfolio is in danger, None otherwise.
        """
        heat = self.calculate_portfolio_heat()

        # Emergency close if way over limit
        if heat.heat_pct > self.max_heat_pct * 2:
            return "close_all"

        return None

    def get_risk_summary(self) -> str:
        """Get human-readable risk summary."""
        heat = self.calculate_portfolio_heat()

        lines = [
            "=" * 50,
            "PORTFOLIO RISK SUMMARY",
            "=" * 50,
            f"Balance: ${self._balance:,.2f}",
            f"Raw Exposure: ${heat.raw_exposure:,.2f}",
            f"Correlated Exposure: ${heat.correlated_exposure:,.2f}",
            f"Portfolio Heat: {heat.heat_pct*100:.1f}% / {heat.max_heat_pct*100:.0f}% max",
            f"Headroom: {heat.headroom*100:.1f}%",
            "",
            "Positions:",
        ]

        for symbol, pos in self._positions.items():
            vol_regime = self.get_volatility_regime(symbol)
            lines.append(
                f"  {symbol}: {pos['side']} ${pos['size']:,.2f} "
                f"(vol: {vol_regime.level})"
            )

        lines.extend([
            "",
            "Volatility Regimes:",
        ])
        for symbol in self.symbols:
            regime = self.get_volatility_regime(symbol)
            lines.append(
                f"  {symbol}: {regime.level} ({regime.percentile:.0f}%ile) "
                f"-> {regime.scaling_factor:.1f}x sizing"
            )

        return "\n".join(lines)
