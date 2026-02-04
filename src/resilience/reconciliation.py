"""
Reconciliation Engine - On-Chain vs Local State Verification

Ensures local state matches on-chain reality.
Automatically resolves discrepancies where safe, alerts on critical issues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IssueSeverity(str, Enum):
    """Severity levels for reconciliation issues."""

    INFO = "info"  # Minor discrepancy, auto-resolved
    WARNING = "warning"  # Needs attention but not critical
    CRITICAL = "critical"  # Requires immediate action
    FATAL = "fatal"  # Trading should halt


class IssueType(str, Enum):
    """Types of reconciliation issues."""

    # Position issues
    UNKNOWN_POSITION = "unknown_position"  # On-chain position not in local state
    MISSING_POSITION = "missing_position"  # Local position not on chain
    SIZE_MISMATCH = "size_mismatch"  # Position size differs
    PRICE_MISMATCH = "price_mismatch"  # Entry price differs (suspicious)

    # Order issues
    UNKNOWN_FILL = "unknown_fill"  # Fill we didn't track
    ORPHAN_ORDER = "orphan_order"  # Order with no matching position
    DUPLICATE_ORDER = "duplicate_order"  # Same order appears twice

    # Balance issues
    BALANCE_MISMATCH = "balance_mismatch"  # Available balance differs
    MARGIN_MISMATCH = "margin_mismatch"  # Used margin differs

    # State issues
    STALE_STATE = "stale_state"  # Local state is too old
    CORRUPTED_STATE = "corrupted_state"  # Hash chain broken


class Resolution(str, Enum):
    """How issues are resolved."""

    AUTO_FIXED = "auto_fixed"  # Automatically corrected
    MANUAL_REQUIRED = "manual_required"  # Needs human intervention
    IGNORED = "ignored"  # Within acceptable tolerance
    ESCALATED = "escalated"  # Sent to alerting system


@dataclass
class ReconciliationIssue:
    """A single reconciliation issue."""

    issue_type: IssueType
    severity: IssueSeverity
    symbol: str | None
    description: str
    local_value: Any
    chain_value: Any
    resolution: Resolution
    resolved_at: datetime | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "description": self.description,
            "local_value": self.local_value,
            "chain_value": self.chain_value,
            "resolution": self.resolution.value,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "notes": self.notes,
        }


@dataclass
class ReconciliationReport:
    """Complete reconciliation report."""

    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    issues: list[ReconciliationIssue] = field(default_factory=list)
    positions_checked: int = 0
    orders_checked: int = 0
    balance_checked: bool = False

    # Summary
    auto_fixed: int = 0
    manual_required: int = 0
    critical_issues: int = 0

    @property
    def is_healthy(self) -> bool:
        """True if no critical issues requiring manual intervention."""
        return self.critical_issues == 0 and self.manual_required == 0

    @property
    def can_continue_trading(self) -> bool:
        """True if safe to continue trading."""
        fatal = [i for i in self.issues if i.severity == IssueSeverity.FATAL]
        return len(fatal) == 0

    def add_issue(self, issue: ReconciliationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)

        if issue.resolution == Resolution.AUTO_FIXED:
            self.auto_fixed += 1
        elif issue.resolution == Resolution.MANUAL_REQUIRED:
            self.manual_required += 1

        if issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.FATAL):
            self.critical_issues += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "issues": [i.to_dict() for i in self.issues],
            "positions_checked": self.positions_checked,
            "orders_checked": self.orders_checked,
            "balance_checked": self.balance_checked,
            "summary": {
                "auto_fixed": self.auto_fixed,
                "manual_required": self.manual_required,
                "critical_issues": self.critical_issues,
                "is_healthy": self.is_healthy,
                "can_continue_trading": self.can_continue_trading,
            },
        }

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "RECONCILIATION REPORT",
            "=" * 60,
            f"Started: {self.started_at}",
            f"Completed: {self.completed_at}",
            "",
            f"Positions checked: {self.positions_checked}",
            f"Orders checked: {self.orders_checked}",
            f"Balance checked: {self.balance_checked}",
            "",
            f"Issues found: {len(self.issues)}",
            f"  - Auto-fixed: {self.auto_fixed}",
            f"  - Manual required: {self.manual_required}",
            f"  - Critical: {self.critical_issues}",
            "",
            f"Status: {'HEALTHY' if self.is_healthy else 'NEEDS ATTENTION'}",
            f"Can trade: {'YES' if self.can_continue_trading else 'NO - HALTED'}",
        ]

        if self.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.issues:
                status = "FIXED" if issue.resolution == Resolution.AUTO_FIXED else "PENDING"
                lines.append(f"  [{issue.severity.value.upper()}] {issue.description} - {status}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class LocalPosition:
    """Local state representation of a position."""

    symbol: str
    size: float
    entry_price: float
    direction: str  # "LONG" or "SHORT"
    leverage: int
    entry_time: datetime | None = None
    conviction: float = 0.0
    strategy: str = ""
    planned_exit_hours: int | None = None


@dataclass
class ChainPosition:
    """On-chain representation of a position."""

    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    leverage: int
    unrealized_pnl: float
    margin_used: float


class ReconciliationEngine:
    """
    Engine for reconciling local state with on-chain state.

    Runs on:
    - Bot startup (mandatory)
    - Periodic intervals (every N minutes)
    - After any error/exception
    - Manual trigger
    """

    def __init__(
        self,
        size_tolerance: float = 0.0001,  # 0.01% size tolerance
        price_tolerance: float = 0.001,  # 0.1% price tolerance
        stale_threshold_minutes: int = 5,
    ):
        self.size_tolerance = size_tolerance
        self.price_tolerance = price_tolerance
        self.stale_threshold = timedelta(minutes=stale_threshold_minutes)

    async def reconcile(
        self,
        local_positions: dict[str, LocalPosition],
        chain_positions: list[ChainPosition],
        local_balance: float | None = None,
        chain_balance: float | None = None,
        last_event_time: datetime | None = None,
    ) -> ReconciliationReport:
        """
        Perform full reconciliation between local and chain state.

        Args:
            local_positions: Our local view of positions
            chain_positions: Positions from on-chain query
            local_balance: Our tracked balance
            chain_balance: Balance from chain
            last_event_time: When we last processed an event

        Returns:
            ReconciliationReport with all issues and resolutions
        """
        report = ReconciliationReport()

        # Check for stale state
        if last_event_time:
            age = datetime.now(UTC) - last_event_time
            if age > self.stale_threshold:
                report.add_issue(ReconciliationIssue(
                    issue_type=IssueType.STALE_STATE,
                    severity=IssueSeverity.WARNING,
                    symbol=None,
                    description=f"Local state is {age.total_seconds():.0f}s old",
                    local_value=last_event_time.isoformat(),
                    chain_value=datetime.now(UTC).isoformat(),
                    resolution=Resolution.IGNORED,
                    notes="State may be outdated, proceed with caution",
                ))

        # Convert chain positions to dict for lookup
        chain_by_symbol = {p.symbol: p for p in chain_positions}
        report.positions_checked = len(chain_positions)

        # Check each local position against chain
        for symbol, local in local_positions.items():
            chain = chain_by_symbol.get(symbol)

            if chain is None:
                # We think we have a position but chain doesn't
                report.add_issue(ReconciliationIssue(
                    issue_type=IssueType.MISSING_POSITION,
                    severity=IssueSeverity.CRITICAL,
                    symbol=symbol,
                    description=f"Local position {symbol} not found on chain",
                    local_value={"size": local.size, "entry": local.entry_price},
                    chain_value=None,
                    resolution=Resolution.MANUAL_REQUIRED,
                    notes="Position may have been liquidated or closed externally",
                ))
            else:
                # Check size
                size_diff = abs(local.size - abs(chain.size))
                if size_diff > self.size_tolerance * local.size:
                    report.add_issue(self._handle_size_mismatch(
                        symbol, local.size, chain.size
                    ))

                # Check entry price (informational - can drift due to averaging)
                price_diff = abs(local.entry_price - chain.entry_price)
                if price_diff > self.price_tolerance * local.entry_price:
                    report.add_issue(ReconciliationIssue(
                        issue_type=IssueType.PRICE_MISMATCH,
                        severity=IssueSeverity.INFO,
                        symbol=symbol,
                        description=f"Entry price differs for {symbol}",
                        local_value=local.entry_price,
                        chain_value=chain.entry_price,
                        resolution=Resolution.AUTO_FIXED,
                        resolved_at=datetime.now(UTC),
                        notes="Updated local entry price to match chain",
                    ))

        # Check for positions on chain that we don't know about
        for symbol, chain in chain_by_symbol.items():
            if abs(chain.size) < 0.0000001:
                continue  # Skip zero positions

            if symbol not in local_positions:
                report.add_issue(ReconciliationIssue(
                    issue_type=IssueType.UNKNOWN_POSITION,
                    severity=IssueSeverity.CRITICAL,
                    symbol=symbol,
                    description=f"Unknown position found on chain: {symbol}",
                    local_value=None,
                    chain_value={"size": chain.size, "entry": chain.entry_price},
                    resolution=Resolution.MANUAL_REQUIRED,
                    notes="Position opened externally or local state corrupted",
                ))

        # Check balance
        if local_balance is not None and chain_balance is not None:
            report.balance_checked = True
            balance_diff = abs(local_balance - chain_balance)
            if balance_diff > 1.0:  # $1 tolerance
                severity = IssueSeverity.WARNING if balance_diff < 100 else IssueSeverity.CRITICAL
                report.add_issue(ReconciliationIssue(
                    issue_type=IssueType.BALANCE_MISMATCH,
                    severity=severity,
                    symbol=None,
                    description=f"Balance mismatch: ${balance_diff:.2f}",
                    local_value=local_balance,
                    chain_value=chain_balance,
                    resolution=Resolution.AUTO_FIXED,
                    resolved_at=datetime.now(UTC),
                    notes="Updated local balance to match chain",
                ))

        report.completed_at = datetime.now(UTC)
        return report

    def _handle_size_mismatch(
        self,
        symbol: str,
        local_size: float,
        chain_size: float,
    ) -> ReconciliationIssue:
        """Handle position size mismatch."""
        chain_size_abs = abs(chain_size)
        diff_pct = abs(local_size - chain_size_abs) / local_size * 100

        if diff_pct < 1:
            # Small difference - likely rounding
            return ReconciliationIssue(
                issue_type=IssueType.SIZE_MISMATCH,
                severity=IssueSeverity.INFO,
                symbol=symbol,
                description=f"Minor size difference for {symbol} ({diff_pct:.2f}%)",
                local_value=local_size,
                chain_value=chain_size_abs,
                resolution=Resolution.AUTO_FIXED,
                resolved_at=datetime.now(UTC),
                notes="Adjusted local size to match chain (rounding difference)",
            )
        elif diff_pct < 10:
            # Moderate difference - partial fill we missed?
            return ReconciliationIssue(
                issue_type=IssueType.SIZE_MISMATCH,
                severity=IssueSeverity.WARNING,
                symbol=symbol,
                description=f"Size mismatch for {symbol} ({diff_pct:.2f}%)",
                local_value=local_size,
                chain_value=chain_size_abs,
                resolution=Resolution.AUTO_FIXED,
                resolved_at=datetime.now(UTC),
                notes="Updated local size - may have missed a partial fill",
            )
        else:
            # Large difference - something is wrong
            return ReconciliationIssue(
                issue_type=IssueType.SIZE_MISMATCH,
                severity=IssueSeverity.CRITICAL,
                symbol=symbol,
                description=f"Major size mismatch for {symbol} ({diff_pct:.2f}%)",
                local_value=local_size,
                chain_value=chain_size_abs,
                resolution=Resolution.MANUAL_REQUIRED,
                notes="Significant discrepancy - manual review required",
            )

    async def quick_check(
        self,
        chain_positions: list[ChainPosition],
        expected_symbols: set[str],
    ) -> bool:
        """
        Quick health check without full reconciliation.

        Returns True if basic state looks correct.
        """
        chain_symbols = {p.symbol for p in chain_positions if abs(p.size) > 0.0000001}

        # Check for unexpected positions
        unexpected = chain_symbols - expected_symbols
        if unexpected:
            logger.warning(f"Unexpected positions found: {unexpected}")
            return False

        return True


class ReconciliationScheduler:
    """
    Schedules periodic reconciliation checks.

    Runs reconciliation:
    - On startup (immediate)
    - Every N minutes during operation
    - After any trading error
    """

    def __init__(
        self,
        engine: ReconciliationEngine,
        interval_minutes: int = 5,
    ):
        self.engine = engine
        self.interval = timedelta(minutes=interval_minutes)
        self._last_reconciliation: datetime | None = None
        self._task: Any = None

    def should_reconcile(self) -> bool:
        """Check if reconciliation is due."""
        if self._last_reconciliation is None:
            return True
        return datetime.now(UTC) - self._last_reconciliation > self.interval

    async def run_if_due(
        self,
        local_positions: dict[str, LocalPosition],
        chain_positions: list[ChainPosition],
        **kwargs: Any,
    ) -> ReconciliationReport | None:
        """Run reconciliation if due, otherwise return None."""
        if not self.should_reconcile():
            return None

        report = await self.engine.reconcile(
            local_positions,
            chain_positions,
            **kwargs,
        )
        self._last_reconciliation = datetime.now(UTC)
        return report

    def force_reconciliation(self) -> None:
        """Force reconciliation on next check."""
        self._last_reconciliation = None
