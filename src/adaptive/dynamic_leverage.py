"""
Dynamic Leverage Calculator

Calculates optimal leverage based on:
- Current volatility (ATR / price)
- Recent performance (win rate, drawdown)
- Market regime (trending vs ranging)
- Kelly criterion approximation

Integrates with hot reload for real-time updates.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LeverageState:
    """Persisted leverage state for hot reload."""

    current_leverage: float = 5.0
    base_leverage: float = 5.0
    min_leverage: float = 1.0
    max_leverage: float = 10.0

    # Adjustment factors (multiplicative)
    volatility_factor: float = 1.0
    drawdown_factor: float = 1.0
    performance_factor: float = 1.0

    # Market conditions
    current_volatility: float = 0.02  # ATR as % of price
    normal_volatility: float = 0.02
    current_drawdown: float = 0.0
    rolling_win_rate: float = 0.5

    # History
    last_adjustment: Optional[str] = None
    adjustment_reason: str = ""
    adjustment_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_leverage": self.current_leverage,
            "base_leverage": self.base_leverage,
            "min_leverage": self.min_leverage,
            "max_leverage": self.max_leverage,
            "volatility_factor": self.volatility_factor,
            "drawdown_factor": self.drawdown_factor,
            "performance_factor": self.performance_factor,
            "current_volatility": self.current_volatility,
            "normal_volatility": self.normal_volatility,
            "current_drawdown": self.current_drawdown,
            "rolling_win_rate": self.rolling_win_rate,
            "last_adjustment": self.last_adjustment,
            "adjustment_reason": self.adjustment_reason,
            "adjustment_history": self.adjustment_history[-20:],  # Keep last 20
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LeverageState":
        return cls(
            current_leverage=data.get("current_leverage", 5.0),
            base_leverage=data.get("base_leverage", 5.0),
            min_leverage=data.get("min_leverage", 1.0),
            max_leverage=data.get("max_leverage", 10.0),
            volatility_factor=data.get("volatility_factor", 1.0),
            drawdown_factor=data.get("drawdown_factor", 1.0),
            performance_factor=data.get("performance_factor", 1.0),
            current_volatility=data.get("current_volatility", 0.02),
            normal_volatility=data.get("normal_volatility", 0.02),
            current_drawdown=data.get("current_drawdown", 0.0),
            rolling_win_rate=data.get("rolling_win_rate", 0.5),
            last_adjustment=data.get("last_adjustment"),
            adjustment_reason=data.get("adjustment_reason", ""),
            adjustment_history=data.get("adjustment_history", []),
        )


class DynamicLeverageManager:
    """
    Manages dynamic leverage adjustments based on market conditions.

    Philosophy (Jim Simons-inspired):
    - Leverage is a function of edge * inverse variance
    - Higher volatility → lower leverage
    - In drawdown → reduce leverage to preserve capital
    - Strong performance → slightly increase leverage (half-Kelly)
    - Never exceed max leverage regardless of conditions
    """

    def __init__(
        self,
        state_file: Optional[Path] = None,
        base_leverage: float = 5.0,
        min_leverage: float = 1.0,
        max_leverage: float = 10.0,
    ):
        self.state_file = state_file or Path("data/state/leverage_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.state = LeverageState(
            base_leverage=base_leverage,
            current_leverage=base_leverage,
            min_leverage=min_leverage,
            max_leverage=max_leverage,
        )

        # Try to load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk if available."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self.state = LeverageState.from_dict(data)
                logger.info(f"Leverage state loaded: {self.state.current_leverage}x")
            except Exception as e:
                logger.warning(f"Failed to load leverage state: {e}")

    def save_state(self) -> None:
        """Save state to disk for hot reload."""
        try:
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            temp_file.rename(self.state_file)
            logger.info(f"Leverage state saved: {self.state.current_leverage}x")
        except Exception as e:
            logger.error(f"Failed to save leverage state: {e}")

    def get_leverage(self) -> float:
        """Get current leverage."""
        return self.state.current_leverage

    def calculate_optimal_leverage(
        self,
        current_volatility: float,
        current_drawdown: float,
        rolling_win_rate: float,
        avg_win_r: float = 1.5,
        avg_loss_r: float = 1.0,
    ) -> tuple[float, str]:
        """
        Calculate optimal leverage based on current conditions.

        Args:
            current_volatility: ATR as % of price (e.g., 0.02 = 2%)
            current_drawdown: Current drawdown as decimal (e.g., 0.05 = 5%)
            rolling_win_rate: Recent win rate (0-1)
            avg_win_r: Average winning trade in R-multiples
            avg_loss_r: Average losing trade in R-multiples

        Returns:
            (optimal_leverage, reason)
        """
        base = self.state.base_leverage
        reasons = []

        # 1. Volatility adjustment (inverse relationship)
        # High vol → lower leverage, low vol → can increase
        vol_ratio = self.state.normal_volatility / max(current_volatility, 0.005)
        vol_factor = min(1.5, max(0.3, vol_ratio))

        if vol_factor < 0.7:
            reasons.append(f"high volatility ({current_volatility:.1%})")
        elif vol_factor > 1.2:
            reasons.append(f"low volatility ({current_volatility:.1%})")

        # 2. Drawdown adjustment (critical for capital preservation)
        # Thresholds: >15% emergency, >10% high, >5% elevated
        dd_thresholds = [
            (0.15, 0.3, "EMERGENCY DD"),
            (0.10, 0.5, "high DD"),
            (0.05, 0.7, "elevated DD"),
        ]
        dd_factor = 1.0
        for threshold, factor, label in dd_thresholds:
            if current_drawdown > threshold:
                dd_factor = factor
                reasons.append(f"{label} ({current_drawdown:.1%})")
                break

        # 3. Performance adjustment (Kelly-inspired)
        # Kelly fraction = (p*b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss
        if rolling_win_rate > 0 and avg_loss_r > 0:
            b = avg_win_r / avg_loss_r
            kelly = (rolling_win_rate * b - (1 - rolling_win_rate)) / b
            kelly = max(0, kelly)

            # Use half-Kelly for safety
            half_kelly = kelly * 0.5

            if half_kelly > 0.15:  # Strong edge
                perf_factor = 1.2
                reasons.append(f"strong edge (Kelly={kelly:.1%})")
            elif half_kelly < 0.05:  # Weak/no edge
                perf_factor = 0.6
                reasons.append(f"weak edge (Kelly={kelly:.1%})")
            else:
                perf_factor = 1.0
        else:
            perf_factor = 1.0

        # Win rate override
        if rolling_win_rate < 0.35:
            perf_factor = min(perf_factor, 0.5)
            reasons.append(f"low win rate ({rolling_win_rate:.0%})")
        elif rolling_win_rate > 0.60:
            perf_factor = max(perf_factor, 1.1)
            reasons.append(f"high win rate ({rolling_win_rate:.0%})")

        # Calculate final leverage
        optimal = base * vol_factor * dd_factor * perf_factor
        optimal = max(self.state.min_leverage, min(self.state.max_leverage, optimal))

        reason = ", ".join(reasons) if reasons else "normal conditions"
        return optimal, reason

    def update_conditions(
        self,
        current_volatility: Optional[float] = None,
        current_drawdown: Optional[float] = None,
        rolling_win_rate: Optional[float] = None,
    ) -> tuple[float, bool, str]:
        """
        Update market conditions and recalculate leverage.

        Returns:
            (new_leverage, changed, reason)
        """
        if current_volatility is not None:
            self.state.current_volatility = current_volatility
        if current_drawdown is not None:
            self.state.current_drawdown = current_drawdown
        if rolling_win_rate is not None:
            self.state.rolling_win_rate = rolling_win_rate

        old_leverage = self.state.current_leverage
        new_leverage, reason = self.calculate_optimal_leverage(
            self.state.current_volatility,
            self.state.current_drawdown,
            self.state.rolling_win_rate,
        )

        # Only change if difference > 10% (avoid noise)
        if abs(new_leverage - old_leverage) / old_leverage > 0.10:
            self.state.current_leverage = new_leverage
            self.state.last_adjustment = datetime.now(timezone.utc).isoformat()
            self.state.adjustment_reason = reason
            self.state.adjustment_history.append({
                "timestamp": self.state.last_adjustment,
                "old": old_leverage,
                "new": new_leverage,
                "reason": reason,
            })
            self.save_state()
            logger.info(f"Leverage adjusted: {old_leverage:.1f}x → {new_leverage:.1f}x ({reason})")
            return new_leverage, True, reason

        return old_leverage, False, "no significant change needed"

    def set_leverage(self, new_leverage: float, reason: str) -> tuple[float, str]:
        """
        Manually set leverage (e.g., from LLM decision).

        Args:
            new_leverage: Desired leverage
            reason: Reason for change

        Returns:
            (actual_leverage, message)
        """
        old_leverage = self.state.current_leverage

        # Clamp to valid range
        clamped = max(self.state.min_leverage, min(self.state.max_leverage, new_leverage))

        self.state.current_leverage = clamped
        self.state.last_adjustment = datetime.now(timezone.utc).isoformat()
        self.state.adjustment_reason = reason
        self.state.adjustment_history.append({
            "timestamp": self.state.last_adjustment,
            "old": old_leverage,
            "new": clamped,
            "reason": f"LLM: {reason}",
        })

        self.save_state()

        if clamped != new_leverage:
            msg = f"Leverage: {old_leverage:.1f}x → {clamped:.1f}x (clamped from {new_leverage:.1f}x)"
        else:
            msg = f"Leverage: {old_leverage:.1f}x → {clamped:.1f}x"

        logger.info(msg)
        return clamped, msg

    def get_status(self) -> dict:
        """Get current leverage status for monitoring."""
        return {
            "current_leverage": self.state.current_leverage,
            "base_leverage": self.state.base_leverage,
            "range": f"{self.state.min_leverage}x - {self.state.max_leverage}x",
            "factors": {
                "volatility": self.state.volatility_factor,
                "drawdown": self.state.drawdown_factor,
                "performance": self.state.performance_factor,
            },
            "conditions": {
                "volatility": f"{self.state.current_volatility:.2%}",
                "drawdown": f"{self.state.current_drawdown:.2%}",
                "win_rate": f"{self.state.rolling_win_rate:.1%}",
            },
            "last_adjustment": self.state.last_adjustment,
            "reason": self.state.adjustment_reason,
        }
