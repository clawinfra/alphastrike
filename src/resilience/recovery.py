"""
Auto Recovery Manager - Automatic State Recovery After Crashes

Handles:
- Startup recovery from local WAL
- Cloud state recovery (if local WAL lost)
- Reconciliation with on-chain state
- Resuming scheduled exits
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.resilience.event_store import Event, EventStore, EventType
from src.resilience.reconciliation import (
    ChainPosition,
    LocalPosition,
    ReconciliationEngine,
    ReconciliationReport,
)

if TYPE_CHECKING:
    from src.resilience.cloud_backup import CloudBackupManager

logger = logging.getLogger(__name__)


@dataclass
class RecoveredState:
    """State recovered from events and reconciliation."""

    positions: dict[str, LocalPosition] = field(default_factory=dict)
    pending_exits: list[dict[str, Any]] = field(default_factory=list)
    last_sequence: int = 0
    last_event_time: datetime | None = None
    balance: float = 0.0

    # Recovery metadata
    recovered_from: str = ""  # "local_wal", "cloud", "chain_only"
    events_replayed: int = 0
    recovery_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": {
                s: {
                    "symbol": p.symbol,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "direction": p.direction,
                    "leverage": p.leverage,
                    "entry_time": p.entry_time.isoformat() if p.entry_time else None,
                    "conviction": p.conviction,
                    "strategy": p.strategy,
                    "planned_exit_hours": p.planned_exit_hours,
                }
                for s, p in self.positions.items()
            },
            "pending_exits": self.pending_exits,
            "last_sequence": self.last_sequence,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "balance": self.balance,
            "recovered_from": self.recovered_from,
            "events_replayed": self.events_replayed,
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
        }


class AutoRecoveryManager:
    """
    Manages automatic recovery on startup.

    Recovery order:
    1. Try to recover from local WAL
    2. If WAL missing/corrupted, try cloud backup
    3. Reconcile with on-chain state
    4. Resume any pending scheduled exits
    5. Log recovery event
    """

    def __init__(
        self,
        event_store: EventStore,
        reconciliation_engine: ReconciliationEngine,
        cloud_backup: CloudBackupManager | None = None,
        get_chain_positions: Callable[[], Any] | None = None,
        get_chain_balance: Callable[[], Any] | None = None,
    ):
        self.event_store = event_store
        self.reconciliation = reconciliation_engine
        self.cloud_backup = cloud_backup
        self.get_chain_positions = get_chain_positions
        self.get_chain_balance = get_chain_balance

    async def recover(self) -> tuple[RecoveredState, ReconciliationReport]:
        """
        Perform full recovery sequence.

        Returns:
            Tuple of (RecoveredState, ReconciliationReport)
        """
        logger.info("Starting auto-recovery...")

        # Step 1: Try local WAL recovery
        state = await self._recover_from_local()

        # Step 2: If local failed, try cloud
        if state is None and self.cloud_backup:
            state = await self._recover_from_cloud()

        # Step 3: If still no state, start fresh
        if state is None:
            state = RecoveredState(recovered_from="fresh_start")
            logger.warning("No prior state found, starting fresh")

        # Step 4: Reconcile with chain
        report = await self._reconcile_with_chain(state)

        # Step 5: Apply fixes from reconciliation
        state = await self._apply_reconciliation_fixes(state, report)

        # Step 6: Log recovery event
        await self._log_recovery_event(state, report)

        state.recovery_time = datetime.now(UTC)
        logger.info(
            f"Recovery complete: {len(state.positions)} positions, "
            f"{state.events_replayed} events replayed, "
            f"recovered from: {state.recovered_from}"
        )

        return state, report

    async def _recover_from_local(self) -> RecoveredState | None:
        """Recover state from local WAL."""
        # Verify WAL integrity
        is_valid, errors = self.event_store.verify_integrity()
        if not is_valid:
            logger.error(f"WAL integrity check failed: {errors}")
            return None

        if self.event_store.sequence_number == 0:
            logger.info("No events in local WAL")
            return None

        logger.info(f"Recovering from local WAL ({self.event_store.sequence_number} events)")

        state = RecoveredState(recovered_from="local_wal")

        # Find last checkpoint for faster recovery
        checkpoint = await self.event_store.get_last_checkpoint()
        start_sequence = 0

        if checkpoint:
            # Load state from checkpoint
            checkpoint_state = checkpoint.payload.get("state", {})
            start_sequence = checkpoint.payload.get("sequence_at_checkpoint", 0)
            logger.info(f"Found checkpoint at sequence {start_sequence}")

            # Restore positions from checkpoint
            for symbol, pos_data in checkpoint_state.get("positions", {}).items():
                state.positions[symbol] = LocalPosition(
                    symbol=pos_data["symbol"],
                    size=pos_data["size"],
                    entry_price=pos_data["entry_price"],
                    direction=pos_data["direction"],
                    leverage=pos_data.get("leverage", 1),
                    entry_time=datetime.fromisoformat(pos_data["entry_time"]) if pos_data.get("entry_time") else None,
                    conviction=pos_data.get("conviction", 0),
                    strategy=pos_data.get("strategy", ""),
                    planned_exit_hours=pos_data.get("planned_exit_hours"),
                )

        # Replay events since checkpoint
        async for event in self.event_store.replay(from_sequence=start_sequence + 1):
            self._apply_event(state, event)
            state.events_replayed += 1

        state.last_sequence = self.event_store.sequence_number
        return state

    async def _recover_from_cloud(self) -> RecoveredState | None:
        """Recover state from cloud backup."""
        if not self.cloud_backup:
            return None

        logger.info("Attempting cloud recovery...")

        # Try hot tier first (fastest)
        if self.cloud_backup.hot:
            positions = await self.cloud_backup.get_position_state()
            if positions:
                state = RecoveredState(recovered_from="cloud_hot")
                for pos_data in positions:
                    symbol = pos_data["symbol"]
                    state.positions[symbol] = LocalPosition(
                        symbol=symbol,
                        size=pos_data["size"],
                        entry_price=pos_data["entry_price"],
                        direction=pos_data["direction"],
                        leverage=pos_data.get("leverage", 1),
                        entry_time=datetime.fromisoformat(pos_data["entry_time"]) if pos_data.get("entry_time") else None,
                        conviction=pos_data.get("conviction", 0),
                        strategy=pos_data.get("strategy", ""),
                        planned_exit_hours=pos_data.get("planned_exit_hours"),
                    )
                logger.info(f"Recovered {len(state.positions)} positions from cloud hot tier")
                return state

        # Try warm tier (recent events)
        if self.cloud_backup.warm:
            recent_events = await self.cloud_backup.warm.query_events(
                since=datetime.now(UTC) - timedelta(hours=24),
                limit=10000,
            )
            if recent_events:
                state = RecoveredState(recovered_from="cloud_warm")
                for event_data in sorted(recent_events, key=lambda e: e.get("sequence_number", 0)):
                    event = Event.from_dict(event_data)
                    self._apply_event(state, event)
                    state.events_replayed += 1
                logger.info(f"Recovered from {state.events_replayed} cloud events")
                return state

        return None

    def _apply_event(self, state: RecoveredState, event: Event) -> None:
        """Apply a single event to state."""
        payload = event.payload
        symbol = event.symbol

        if event.event_type == EventType.POSITION_OPENED:
            state.positions[symbol] = LocalPosition(
                symbol=symbol,
                size=payload.get("size", 0),
                entry_price=payload.get("entry_price", 0),
                direction=payload.get("direction", "LONG"),
                leverage=payload.get("leverage", 1),
                entry_time=datetime.fromisoformat(payload["entry_time"]) if payload.get("entry_time") else event.timestamp,
                conviction=payload.get("conviction", 0),
                strategy=payload.get("strategy", ""),
                planned_exit_hours=payload.get("planned_exit_hours"),
            )

            # Track pending exit if scheduled
            if payload.get("planned_exit_hours"):
                entry_time = state.positions[symbol].entry_time or event.timestamp
                exit_time = entry_time + timedelta(hours=payload["planned_exit_hours"])
                state.pending_exits.append({
                    "symbol": symbol,
                    "exit_time": exit_time.isoformat(),
                    "reason": "time_exit",
                })

        elif event.event_type == EventType.POSITION_CLOSED:
            if symbol in state.positions:
                del state.positions[symbol]
            # Remove from pending exits
            state.pending_exits = [e for e in state.pending_exits if e["symbol"] != symbol]

        elif event.event_type == EventType.POSITION_INCREASED:
            if symbol in state.positions:
                pos = state.positions[symbol]
                # Average in the new size
                old_value = pos.size * pos.entry_price
                new_value = payload.get("size", 0) * payload.get("price", 0)
                new_size = pos.size + payload.get("size", 0)
                if new_size > 0:
                    pos.entry_price = (old_value + new_value) / new_size
                    pos.size = new_size

        elif event.event_type == EventType.POSITION_REDUCED:
            if symbol in state.positions:
                pos = state.positions[symbol]
                pos.size -= payload.get("size", 0)
                if pos.size <= 0:
                    del state.positions[symbol]

        state.last_event_time = event.timestamp
        state.last_sequence = event.sequence_number

    async def _reconcile_with_chain(self, state: RecoveredState) -> ReconciliationReport:
        """Reconcile recovered state with on-chain state."""
        if not self.get_chain_positions:
            logger.warning("No chain position getter, skipping reconciliation")
            return ReconciliationReport()

        # Get chain positions
        try:
            raw_positions = await self.get_chain_positions()
            chain_positions = [
                ChainPosition(
                    symbol=p.get("symbol", p.get("coin", "")),
                    size=float(p.get("size", p.get("szi", 0))),
                    entry_price=float(p.get("entry_price", p.get("entryPx", 0))),
                    leverage=int(p.get("leverage", 1)),
                    unrealized_pnl=float(p.get("unrealized_pnl", p.get("unrealizedPnl", 0))),
                    margin_used=float(p.get("margin_used", p.get("marginUsed", 0))),
                )
                for p in raw_positions
            ]
        except Exception as e:
            logger.error(f"Failed to get chain positions: {e}")
            return ReconciliationReport()

        # Get chain balance
        chain_balance = None
        if self.get_chain_balance:
            try:
                chain_balance = await self.get_chain_balance()
            except Exception as e:
                logger.error(f"Failed to get chain balance: {e}")

        # Run reconciliation
        report = await self.reconciliation.reconcile(
            local_positions=state.positions,
            chain_positions=chain_positions,
            local_balance=state.balance,
            chain_balance=chain_balance,
            last_event_time=state.last_event_time,
        )

        return report

    async def _apply_reconciliation_fixes(
        self,
        state: RecoveredState,
        report: ReconciliationReport,
    ) -> RecoveredState:
        """Apply auto-fixes from reconciliation report."""
        from src.resilience.reconciliation import IssueType, Resolution

        for issue in report.issues:
            if issue.resolution != Resolution.AUTO_FIXED:
                continue

            symbol = issue.symbol

            if issue.issue_type == IssueType.SIZE_MISMATCH and symbol:
                # Update size to match chain
                if symbol in state.positions:
                    state.positions[symbol].size = abs(issue.chain_value)
                    logger.info(f"Auto-fixed size for {symbol}: {issue.chain_value}")

            elif issue.issue_type == IssueType.PRICE_MISMATCH and symbol:
                # Update entry price to match chain
                if symbol in state.positions:
                    state.positions[symbol].entry_price = issue.chain_value
                    logger.info(f"Auto-fixed entry price for {symbol}: {issue.chain_value}")

            elif issue.issue_type == IssueType.BALANCE_MISMATCH:
                # Update balance to match chain
                state.balance = issue.chain_value
                logger.info(f"Auto-fixed balance: {issue.chain_value}")

            elif issue.issue_type == IssueType.UNKNOWN_POSITION and symbol:
                # Add missing position (best effort - limited metadata)
                chain_data = issue.chain_value or {}
                state.positions[symbol] = LocalPosition(
                    symbol=symbol,
                    size=abs(chain_data.get("size", 0)),
                    entry_price=chain_data.get("entry", 0),
                    direction="LONG" if chain_data.get("size", 0) > 0 else "SHORT",
                    leverage=1,
                    strategy="recovered",
                    conviction=0,
                )
                logger.warning(f"Added unknown position {symbol} from chain (limited metadata)")

        return state

    async def _log_recovery_event(
        self,
        state: RecoveredState,
        report: ReconciliationReport,
    ) -> None:
        """Log recovery event for audit trail."""
        await self.event_store.create_event(
            EventType.STATE_RECOVERED,
            recovered_from=state.recovered_from,
            events_replayed=state.events_replayed,
            positions_recovered=len(state.positions),
            pending_exits=len(state.pending_exits),
            reconciliation_issues=len(report.issues),
            reconciliation_healthy=report.is_healthy,
        )

    async def get_overdue_exits(self, state: RecoveredState) -> list[dict[str, Any]]:
        """Get exits that should have happened while bot was down."""
        now = datetime.now(UTC)
        overdue = []

        for exit_info in state.pending_exits:
            exit_time = datetime.fromisoformat(exit_info["exit_time"])
            if exit_time < now:
                hours_overdue = (now - exit_time).total_seconds() / 3600
                overdue.append({
                    **exit_info,
                    "hours_overdue": hours_overdue,
                })

        return overdue


class GracefulShutdownHandler:
    """
    Handles graceful shutdown with state preservation.

    On shutdown:
    1. Cancel pending orders (optional)
    2. Create checkpoint
    3. Sync to cloud
    4. Log shutdown event
    """

    def __init__(
        self,
        event_store: EventStore,
        cloud_backup: CloudBackupManager | None = None,
        get_current_state: Callable[[], dict[str, Any]] | None = None,
    ):
        self.event_store = event_store
        self.cloud_backup = cloud_backup
        self.get_current_state = get_current_state

    async def shutdown(self, reason: str = "normal") -> None:
        """Perform graceful shutdown."""
        logger.info(f"Starting graceful shutdown (reason: {reason})")

        # Log shutdown event
        await self.event_store.create_event(
            EventType.BOT_STOPPED,
            reason=reason,
            sequence_at_shutdown=self.event_store.sequence_number,
        )

        # Create final checkpoint
        if self.get_current_state:
            state = self.get_current_state()
            await self.event_store.create_checkpoint(state)
            logger.info("Created shutdown checkpoint")

        # Final cloud sync
        if self.cloud_backup:
            # Give pending backups time to complete
            await asyncio.sleep(1)
            logger.info("Cloud backup sync complete")

        logger.info("Graceful shutdown complete")
