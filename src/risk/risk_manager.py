"""
US-021: Risk Manager with Multi-Layer Validation for AlphaStrike trading bot.

Provides centralized risk validation for all order requests with checks for:
- Close orders (always allowed)
- Exposure limits (per-trade, per-pair, total)
- Drawdown limits (daily and total)
- Leverage limits
- Never adding to losing positions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.core.config import get_settings
from src.data.rest_client import OrderRequest
from src.risk.adaptive_exposure import AdaptiveExposure
from src.risk.portfolio import PortfolioManager

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """
    Result of a risk validation check.

    Attributes:
        allowed: Whether the order is allowed to proceed
        reason: Explanation if order is blocked (None if allowed)
        checks_passed: List of validation checks that passed
        checks_failed: List of validation checks that failed
    """

    allowed: bool
    reason: str | None
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


class RiskManager:
    """
    Multi-layer risk validation manager for order requests.

    Performs sequential validation checks on orders before execution:
    1. Close orders bypass all checks (reduce_only=True)
    2. Exposure check against adaptive limits
    3. Drawdown check (daily 5-7%, total 15%)
    4. Leverage check (max 20x from config)
    5. Never add to losers check

    Example:
        manager = RiskManager()
        result = manager.validate_order(order, portfolio)
        if result.allowed:
            # Execute order
        else:
            logger.warning(f"Order blocked: {result.reason}")
    """

    # Drawdown limits (as ratios)
    DAILY_DRAWDOWN_MIN = 0.05  # 5%
    DAILY_DRAWDOWN_MAX = 0.07  # 7%
    TOTAL_DRAWDOWN_LIMIT = 0.15  # 15%

    def __init__(
        self,
        adaptive_exposure: AdaptiveExposure | None = None,
        volatility: float = 1.0,
        regime: str = "ranging",
    ) -> None:
        """
        Initialize RiskManager.

        Args:
            adaptive_exposure: AdaptiveExposure instance for dynamic limits.
                             If None, a new instance is created.
            volatility: Current ATR ratio for exposure calculations.
            regime: Current market regime ('trending', 'ranging', 'volatile').
        """
        self._adaptive_exposure = adaptive_exposure or AdaptiveExposure()
        self._volatility = volatility
        self._regime = regime
        self._settings = get_settings()

        logger.info(
            "RiskManager initialized",
            extra={
                "max_leverage": self._settings.risk.max_leverage,
                "daily_drawdown_limit": self.DAILY_DRAWDOWN_MAX,
                "total_drawdown_limit": self.TOTAL_DRAWDOWN_LIMIT,
            },
        )

    def update_market_conditions(
        self,
        volatility: float,
        regime: str,
    ) -> None:
        """
        Update market conditions for adaptive exposure calculations.

        Args:
            volatility: Current ATR ratio (1.0 = normal volatility).
            regime: Market regime ('trending', 'ranging', 'volatile').
        """
        self._volatility = volatility
        self._regime = regime

    def validate_order(
        self,
        order: OrderRequest,
        portfolio: PortfolioManager,
    ) -> RiskCheck:
        """
        Validate an order against all risk checks.

        Checks are performed in order:
        1. Close orders (reduce_only) - always allowed
        2. Exposure limits
        3. Drawdown limits
        4. Leverage limits
        5. Never add to losers

        Args:
            order: The order request to validate.
            portfolio: Current portfolio state.

        Returns:
            RiskCheck with validation result and details.
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        # 1. Close orders always allowed
        if self._check_close_order(order):
            checks_passed.append("close_order")
            return RiskCheck(
                allowed=True,
                reason=None,
                checks_passed=["close_order (bypass all checks)"],
                checks_failed=[],
            )

        # 2. Exposure check
        exposure_ok, exposure_reason = self._check_exposure(order, portfolio)
        if exposure_ok:
            checks_passed.append("exposure_limits")
        else:
            checks_failed.append("exposure_limits")
            return RiskCheck(
                allowed=False,
                reason=exposure_reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        # 3. Drawdown check
        drawdown_ok, drawdown_reason = self._check_drawdown(portfolio)
        if drawdown_ok:
            checks_passed.append("drawdown_limits")
        else:
            checks_failed.append("drawdown_limits")
            return RiskCheck(
                allowed=False,
                reason=drawdown_reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        # 4. Leverage check
        leverage_ok, leverage_reason = self._check_leverage(order, portfolio)
        if leverage_ok:
            checks_passed.append("leverage_limit")
        else:
            checks_failed.append("leverage_limit")
            return RiskCheck(
                allowed=False,
                reason=leverage_reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        # 5. Never add to losers check
        loser_ok, loser_reason = self._check_add_to_loser(order, portfolio)
        if loser_ok:
            checks_passed.append("no_add_to_loser")
        else:
            checks_failed.append("no_add_to_loser")
            return RiskCheck(
                allowed=False,
                reason=loser_reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        # All checks passed
        return RiskCheck(
            allowed=True,
            reason=None,
            checks_passed=checks_passed,
            checks_failed=[],
        )

    def _check_close_order(self, order: OrderRequest) -> bool:
        """
        Check if order is a close/reduce-only order.

        Close orders always bypass risk checks to ensure
        positions can be closed even in extreme conditions.

        Args:
            order: The order to check.

        Returns:
            True if this is a close order that should bypass checks.
        """
        return order.reduce_only

    def _check_exposure(
        self,
        order: OrderRequest,
        portfolio: PortfolioManager,
    ) -> tuple[bool, str | None]:
        """
        Check order against exposure limits.

        Validates:
        - Per-trade exposure limit
        - Per-pair exposure limit
        - Total portfolio exposure limit

        Args:
            order: The order to check.
            portfolio: Current portfolio state.

        Returns:
            Tuple of (passed, reason if failed).
        """
        # Get adaptive limits based on current conditions
        limits = self._adaptive_exposure.calculate_limits(
            volatility=self._volatility,
            regime=self._regime,
            drawdown=portfolio.get_drawdown(),
            position_count=portfolio.get_position_count(),
        )

        # Calculate order notional value
        # Use provided price or estimate from current price (assume market order)
        order_price = order.price if order.price else 0.0
        order_notional = order.size * order_price

        # Calculate exposure ratios
        if portfolio.balance <= 0:
            return (False, "Portfolio balance is zero or negative")

        per_trade_exposure = order_notional / portfolio.balance
        current_symbol_exposure = portfolio.get_exposure_by_symbol(order.symbol)
        new_symbol_exposure = current_symbol_exposure + (order_notional / portfolio.balance)
        current_total_exposure = portfolio.get_total_exposure()
        new_total_exposure = current_total_exposure + (order_notional / portfolio.balance)

        # Check per-trade limit
        if per_trade_exposure > limits.per_trade:
            return (
                False,
                f"Per-trade exposure {per_trade_exposure:.2%} exceeds limit {limits.per_trade:.2%}",
            )

        # Check per-pair limit
        if new_symbol_exposure > limits.per_pair:
            return (
                False,
                f"Per-pair exposure {new_symbol_exposure:.2%} would exceed limit {limits.per_pair:.2%}",
            )

        # Check total exposure limit
        if new_total_exposure > limits.total:
            return (
                False,
                f"Total exposure {new_total_exposure:.2%} would exceed limit {limits.total:.2%}",
            )

        return (True, None)

    def _check_drawdown(
        self,
        portfolio: PortfolioManager,
    ) -> tuple[bool, str | None]:
        """
        Check if current drawdown exceeds limits.

        Validates:
        - Daily drawdown (5-7% depending on conditions)
        - Total drawdown (15% hard limit)

        Args:
            portfolio: Current portfolio state.

        Returns:
            Tuple of (passed, reason if failed).
        """
        # Calculate daily P&L as percentage of daily start balance
        daily_pnl = portfolio.get_daily_pnl()
        if portfolio.daily_start_balance > 0:
            daily_drawdown = -daily_pnl / portfolio.daily_start_balance
        else:
            daily_drawdown = 0.0

        # Use total drawdown from portfolio
        total_drawdown = portfolio.get_drawdown()

        # Check total drawdown (hard limit)
        if total_drawdown >= self.TOTAL_DRAWDOWN_LIMIT:
            return (
                False,
                f"Total drawdown {total_drawdown:.2%} exceeds limit {self.TOTAL_DRAWDOWN_LIMIT:.2%}",
            )

        # Check daily drawdown (adaptive limit)
        daily_limit = self._get_daily_drawdown_limit(total_drawdown)
        if daily_drawdown >= daily_limit:
            return (
                False,
                f"Daily drawdown {daily_drawdown:.2%} exceeds limit {daily_limit:.2%}",
            )

        return (True, None)

    def _get_daily_drawdown_limit(self, total_drawdown: float) -> float:
        """
        Get adaptive daily drawdown limit based on current total drawdown.

        When closer to total limit, use tighter daily limit.

        Args:
            total_drawdown: Current total drawdown ratio.

        Returns:
            Daily drawdown limit to use.
        """
        # Scale daily limit based on how close we are to total limit
        # At 0% total DD -> use 7% daily
        # At 10% total DD -> use 5% daily
        drawdown_ratio = total_drawdown / self.TOTAL_DRAWDOWN_LIMIT
        limit_range = self.DAILY_DRAWDOWN_MAX - self.DAILY_DRAWDOWN_MIN

        # Linear interpolation
        daily_limit = self.DAILY_DRAWDOWN_MAX - (drawdown_ratio * limit_range)
        return max(self.DAILY_DRAWDOWN_MIN, min(self.DAILY_DRAWDOWN_MAX, daily_limit))

    def _check_leverage(
        self,
        order: OrderRequest,
        portfolio: PortfolioManager,
    ) -> tuple[bool, str | None]:
        """
        Check if order leverage is within limits.

        Args:
            order: The order to check.
            portfolio: Current portfolio state.

        Returns:
            Tuple of (passed, reason if failed).
        """
        max_leverage = self._settings.risk.max_leverage

        # Check existing position leverage
        # Determine position side based on order side
        if order.position_side:
            side = order.position_side.value.upper()
        else:
            # Infer from order side
            side = "LONG" if order.side.value == "buy" else "SHORT"

        existing_position = portfolio.get_position(order.symbol, side)
        if existing_position and existing_position.leverage > max_leverage:
            return (
                False,
                f"Existing position leverage {existing_position.leverage}x exceeds max {max_leverage}x",
            )

        return (True, None)

    def _check_add_to_loser(
        self,
        order: OrderRequest,
        portfolio: PortfolioManager,
    ) -> tuple[bool, str | None]:
        """
        Check if order would add to a losing position.

        Adding to losers (averaging down) is prohibited as it
        increases risk on already losing trades.

        Args:
            order: The order to check.
            portfolio: Current portfolio state.

        Returns:
            Tuple of (passed, reason if failed).
        """
        # Determine position side
        if order.position_side:
            side = order.position_side.value.upper()
        else:
            # Infer from order side: buy -> LONG, sell -> SHORT
            side = "LONG" if order.side.value == "buy" else "SHORT"

        # Check if there's an existing position
        existing_position = portfolio.get_position(order.symbol, side)
        if existing_position is None:
            # No existing position, this is a new position
            return (True, None)

        # Check if order would add to the position (not reduce)
        # For LONG: buy adds, sell reduces
        # For SHORT: sell adds, buy reduces
        is_adding = (
            (side == "LONG" and order.side.value == "buy")
            or (side == "SHORT" and order.side.value == "sell")
        )

        if not is_adding:
            # Order is reducing position, not adding
            return (True, None)

        # Check if position is losing
        if existing_position.unrealized_pnl < 0:
            return (
                False,
                f"Cannot add to losing position {order.symbol} {side} "
                f"with unrealized P&L {existing_position.unrealized_pnl:.2f}",
            )

        return (True, None)

    def get_risk_summary(self, portfolio: PortfolioManager) -> dict[str, float]:
        """
        Get current risk metrics summary.

        Args:
            portfolio: Current portfolio state.

        Returns:
            Dictionary with current risk metrics.
        """
        daily_pnl = portfolio.get_daily_pnl()
        daily_drawdown = (
            -daily_pnl / portfolio.daily_start_balance
            if portfolio.daily_start_balance > 0
            else 0.0
        )

        limits = self._adaptive_exposure.calculate_limits(
            volatility=self._volatility,
            regime=self._regime,
            drawdown=portfolio.get_drawdown(),
            position_count=portfolio.get_position_count(),
        )

        return {
            "total_exposure": portfolio.get_total_exposure(),
            "total_drawdown": portfolio.get_drawdown(),
            "daily_drawdown": daily_drawdown,
            "position_count": portfolio.get_position_count(),
            "per_trade_limit": limits.per_trade,
            "per_pair_limit": limits.per_pair,
            "total_limit": limits.total,
            "daily_drawdown_limit": self._get_daily_drawdown_limit(portfolio.get_drawdown()),
            "total_drawdown_limit": self.TOTAL_DRAWDOWN_LIMIT,
            "max_leverage": self._settings.risk.max_leverage,
        }
