"""
Event Store - Append-Only Event Log with Event Sourcing

Provides immutable audit trail for all trading actions.
Events are the source of truth - state is derived from events.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """All possible trading events."""

    # Lifecycle
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    BOT_CRASHED = "bot_crashed"

    # Signals
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"

    # Orders
    ORDER_CREATED = "order_created"
    ORDER_SENT = "order_sent"
    ORDER_ACKNOWLEDGED = "order_ack"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILL = "order_partial"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_FAILED = "order_failed"

    # Positions
    POSITION_OPENED = "position_opened"
    POSITION_INCREASED = "position_increased"
    POSITION_REDUCED = "position_reduced"
    POSITION_CLOSED = "position_closed"

    # Risk
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    TIME_EXIT_TRIGGERED = "time_exit"
    LIQUIDATION_WARNING = "liquidation_warning"

    # Reconciliation
    RECONCILIATION_STARTED = "reconciliation_started"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    RECONCILIATION_RESOLVED = "reconciliation_resolved"
    RECONCILIATION_COMPLETED = "reconciliation_completed"

    # System
    CHECKPOINT_CREATED = "checkpoint"
    STATE_RECOVERED = "state_recovered"


@dataclass
class Event:
    """
    Immutable event record.

    Events are append-only and form the source of truth for the trading system.
    All state can be reconstructed by replaying events.
    """

    event_id: str
    event_type: EventType
    timestamp: datetime
    symbol: str | None
    payload: dict[str, Any]

    # Metadata for audit trail
    sequence_number: int = 0
    previous_hash: str = ""
    event_hash: str = ""

    # Origin tracking
    instance_id: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.event_hash:
            self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash for event integrity."""
        content = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "symbol": self.symbol,
            "payload": self.payload,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "symbol": self.symbol,
            "payload": self.payload,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
            "instance_id": self.instance_id,
            "version": self.version,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Deserialize from dict."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data.get("symbol"),
            payload=data.get("payload", {}),
            sequence_number=data.get("sequence_number", 0),
            previous_hash=data.get("previous_hash", ""),
            event_hash=data.get("event_hash", ""),
            instance_id=data.get("instance_id", ""),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Event:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def generate_event_id() -> str:
    """Generate unique event ID."""
    import secrets
    timestamp = int(time.time() * 1000)
    random_part = secrets.token_hex(4)
    return f"evt_{timestamp}_{random_part}"


def generate_instance_id() -> str:
    """Generate unique instance ID for this bot run."""
    import secrets
    return f"inst_{int(time.time())}_{secrets.token_hex(4)}"


class EventStore:
    """
    Append-only event store with local WAL and async cloud backup.

    Features:
    - Local write-ahead log (WAL) for durability
    - Hash chain for integrity verification
    - Async hooks for cloud backup
    - Checkpoint support for faster recovery
    """

    def __init__(
        self,
        storage_dir: Path | str = "data/events",
        instance_id: str | None = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.instance_id = instance_id or generate_instance_id()
        self._sequence_number = 0
        self._last_hash = "genesis"
        self._wal_file: Any = None
        self._wal_path = self.storage_dir / "events.wal.jsonl"

        # Async backup hooks
        self._backup_hooks: list[Callable[[Event], Any]] = []

        # Event subscribers for state updates
        self._subscribers: list[Callable[[Event], Any]] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load sequence number and last hash from existing WAL."""
        if not self._wal_path.exists():
            logger.info(f"Starting fresh event store at {self._wal_path}")
            return

        try:
            with open(self._wal_path) as f:
                for line in f:
                    if line.strip():
                        event = Event.from_json(line.strip())
                        self._sequence_number = event.sequence_number
                        self._last_hash = event.event_hash

            logger.info(
                f"Loaded event store: sequence={self._sequence_number}, "
                f"last_hash={self._last_hash[:8]}..."
            )
        except Exception as e:
            logger.error(f"Failed to load event store state: {e}")

    async def append(self, event: Event) -> Event:
        """
        Append event to the store.

        This is the ONLY way to modify state in the system.
        """
        async with self._lock:
            # Assign sequence and chain hash
            self._sequence_number += 1
            event.sequence_number = self._sequence_number
            event.previous_hash = self._last_hash
            event.instance_id = self.instance_id
            event.event_hash = event._compute_hash()

            # Write to local WAL (synchronous for durability)
            self._write_to_wal(event)

            # Update chain
            self._last_hash = event.event_hash

            # Notify subscribers (state update)
            for subscriber in self._subscribers:
                try:
                    result = subscriber(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")

            # Async backup hooks (fire and forget)
            for hook in self._backup_hooks:
                try:
                    asyncio.create_task(self._run_hook(hook, event))
                except Exception as e:
                    logger.error(f"Backup hook error: {e}")

            return event

    async def _run_hook(self, hook: Callable, event: Event) -> None:
        """Run backup hook with error handling."""
        try:
            result = hook(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Backup hook failed: {e}")

    def _write_to_wal(self, event: Event) -> None:
        """Write event to local WAL file."""
        with open(self._wal_path, "a") as f:
            f.write(event.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())  # Ensure durability

    def add_backup_hook(self, hook: Callable[[Event], Any]) -> None:
        """Add async backup hook (e.g., cloud upload)."""
        self._backup_hooks.append(hook)

    def subscribe(self, callback: Callable[[Event], Any]) -> None:
        """Subscribe to events for state updates."""
        self._subscribers.append(callback)

    async def create_event(
        self,
        event_type: EventType,
        symbol: str | None = None,
        **payload: Any,
    ) -> Event:
        """Create and append a new event."""
        event = Event(
            event_id=generate_event_id(),
            event_type=event_type,
            timestamp=datetime.now(UTC),
            symbol=symbol,
            payload=payload,
        )
        return await self.append(event)

    async def replay(
        self,
        from_sequence: int = 0,
        to_sequence: int | None = None,
        event_types: list[EventType] | None = None,
        symbol: str | None = None,
    ) -> AsyncIterator[Event]:
        """
        Replay events from the store.

        Used for:
        - State recovery after crash
        - Audit trail queries
        - Analytics
        """
        if not self._wal_path.exists():
            return

        with open(self._wal_path) as f:
            for line in f:
                if not line.strip():
                    continue

                event = Event.from_json(line.strip())

                # Apply filters
                if event.sequence_number < from_sequence:
                    continue
                if to_sequence and event.sequence_number > to_sequence:
                    break
                if event_types and event.event_type not in event_types:
                    continue
                if symbol and event.symbol != symbol:
                    continue

                yield event

    async def get_last_checkpoint(self) -> Event | None:
        """Get the most recent checkpoint event."""
        last_checkpoint = None
        async for event in self.replay(event_types=[EventType.CHECKPOINT_CREATED]):
            last_checkpoint = event
        return last_checkpoint

    async def create_checkpoint(self, state_snapshot: dict[str, Any]) -> Event:
        """Create a checkpoint for faster recovery."""
        return await self.create_event(
            EventType.CHECKPOINT_CREATED,
            state=state_snapshot,
            sequence_at_checkpoint=self._sequence_number,
        )

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify hash chain integrity."""
        errors = []
        expected_hash = "genesis"

        if not self._wal_path.exists():
            return True, []

        with open(self._wal_path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                event = Event.from_json(line.strip())

                # Verify chain
                if event.previous_hash != expected_hash:
                    errors.append(
                        f"Line {i}: hash chain broken. "
                        f"Expected {expected_hash[:8]}, got {event.previous_hash[:8]}"
                    )

                # Verify self-hash
                computed = event._compute_hash()
                if computed != event.event_hash:
                    errors.append(
                        f"Line {i}: event hash mismatch. "
                        f"Stored {event.event_hash[:8]}, computed {computed[:8]}"
                    )

                expected_hash = event.event_hash

        return len(errors) == 0, errors

    @property
    def sequence_number(self) -> int:
        """Current sequence number."""
        return self._sequence_number

    @property
    def last_hash(self) -> str:
        """Last event hash in the chain."""
        return self._last_hash


# Convenience function to create trading events
class TradingEvents:
    """Factory for common trading events."""

    def __init__(self, store: EventStore):
        self.store = store

    async def signal_generated(
        self,
        symbol: str,
        direction: str,
        conviction: float,
        strategy: str,
        features: dict[str, float] | None = None,
    ) -> Event:
        return await self.store.create_event(
            EventType.SIGNAL_GENERATED,
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            strategy=strategy,
            features=features or {},
        )

    async def order_sent(
        self,
        symbol: str,
        order_id: str,
        client_order_id: str,
        side: str,
        size: float,
        price: float,
        order_type: str,
    ) -> Event:
        return await self.store.create_event(
            EventType.ORDER_SENT,
            symbol=symbol,
            order_id=order_id,
            client_order_id=client_order_id,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
        )

    async def order_filled(
        self,
        symbol: str,
        order_id: str,
        fill_price: float,
        fill_size: float,
        fee: float,
    ) -> Event:
        return await self.store.create_event(
            EventType.ORDER_FILLED,
            symbol=symbol,
            order_id=order_id,
            fill_price=fill_price,
            fill_size=fill_size,
            fee=fee,
        )

    async def position_opened(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        leverage: int,
        conviction: float,
        strategy: str,
        planned_exit_hours: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Event:
        return await self.store.create_event(
            EventType.POSITION_OPENED,
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry_price,
            leverage=leverage,
            conviction=conviction,
            strategy=strategy,
            planned_exit_hours=planned_exit_hours,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    async def position_closed(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_percent: float,
        holding_hours: float,
    ) -> Event:
        return await self.store.create_event(
            EventType.POSITION_CLOSED,
            symbol=symbol,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_percent=pnl_percent,
            holding_hours=holding_hours,
        )

    async def reconciliation_mismatch(
        self,
        symbol: str,
        issue_type: str,
        local_value: Any,
        chain_value: Any,
        resolution: str,
    ) -> Event:
        return await self.store.create_event(
            EventType.RECONCILIATION_MISMATCH,
            symbol=symbol,
            issue_type=issue_type,
            local_value=local_value,
            chain_value=chain_value,
            resolution=resolution,
        )
