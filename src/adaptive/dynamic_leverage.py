"""
Dynamic Leverage Calculator

Calculates optimal leverage using a conservative emergency-only approach:
- Reduces leverage during severe drawdown (>10%)
- Reduces leverage during extreme volatility (>5% ATR)
- Otherwise maintains base leverage to maximize returns

Integrates with hot reload for real-time updates.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LeverageState:
    """Persisted leverage state for hot reload."""

    current_leverage: float = 5.0
    base_leverage: float = 5.0
    min_leverage: float = 1.0
    max_leverage: float = 10.0

    # Market conditions
    current_volatility: float = 0.02  # ATR as % of price
    current_drawdown: float = 0.0
    rolling_win_rate: float = 0.5

    # History
    last_adjustment: str | None = None
    adjustment_reason: str = ""
    adjustment_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_leverage": self.current_leverage,
            "base_leverage": self.base_leverage,
            "min_leverage": self.min_leverage,
            "max_leverage": self.max_leverage,
            "current_volatility": self.current_volatility,
            "current_drawdown": self.current_drawdown,
            "rolling_win_rate": self.rolling_win_rate,
            "last_adjustment": self.last_adjustment,
            "adjustment_reason": self.adjustment_reason,
            "adjustment_history": self.adjustment_history[-20:],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LeverageState":
        return cls(
            current_leverage=data.get("current_leverage", 5.0),
            base_leverage=data.get("base_leverage", 5.0),
            min_leverage=data.get("min_leverage", 1.0),
            max_leverage=data.get("max_leverage", 10.0),
            current_volatility=data.get("current_volatility", 0.02),
            current_drawdown=data.get("current_drawdown", 0.0),
            rolling_win_rate=data.get("rolling_win_rate", 0.5),
            last_adjustment=data.get("last_adjustment"),
            adjustment_reason=data.get("adjustment_reason", ""),
            adjustment_history=data.get("adjustment_history", []),
        )


class DynamicLeverageManager:
    """
    Manages dynamic leverage adjustments based on market conditions.

    Conservative approach:
    - Only reduce leverage during emergencies (severe drawdown or extreme volatility)
    - Otherwise maintain base leverage to maximize returns
    - Never exceed max leverage regardless of conditions
    """

    def __init__(
        self,
        state_file: Path | None = None,
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
                with open(self.state_file) as f:
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
    ) -> tuple[float, str]:
        """
        Calculate optimal leverage based on current conditions.

        Conservative implementation: only reduce leverage during emergencies.
        Otherwise, use base leverage to maximize returns.

        Args:
            current_volatility: ATR as % of price (e.g., 0.02 = 2%)
            current_drawdown: Current drawdown as decimal (e.g., 0.05 = 5%)
            rolling_win_rate: Recent win rate (0-1, tracked for monitoring)

        Returns:
            (optimal_leverage, reason)
        """
        base = self.state.base_leverage

        # CONSERVATIVE: Only reduce leverage during emergencies
        # Otherwise use base leverage to maximize returns
        optimal = base
        reason = "normal conditions"

        # Emergency condition 1: Severe drawdown (>10%)
        if current_drawdown > 0.10:
            optimal = base * 0.5
            reason = f"EMERGENCY DD ({current_drawdown:.1%})"
        # Emergency condition 2: Extreme volatility (>5% ATR)
        elif current_volatility > 0.05:
            optimal = base * 0.6
            reason = f"extreme volatility ({current_volatility:.1%})"

        # Clamp to valid range
        optimal = max(self.state.min_leverage, min(self.state.max_leverage, optimal))

        return optimal, reason

    def update_conditions(
        self,
        current_volatility: float | None = None,
        current_drawdown: float | None = None,
        rolling_win_rate: float | None = None,
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
            self.state.last_adjustment = datetime.now(UTC).isoformat()
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
        self.state.last_adjustment = datetime.now(UTC).isoformat()
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
            "conditions": {
                "volatility": f"{self.state.current_volatility:.2%}",
                "drawdown": f"{self.state.current_drawdown:.2%}",
                "win_rate": f"{self.state.rolling_win_rate:.1%}",
            },
            "last_adjustment": self.state.last_adjustment,
            "reason": self.state.adjustment_reason,
        }
