"""
Cloud Backup Manager - Multi-Tier Cloud Storage

Provides:
- Hot tier: Upstash Redis (1-5ms) - current state, locks
- Warm tier: Turso SQLite (5-10ms) - recent events
- Cold tier: Supabase (50-100ms) - full audit trail

All writes are async and non-blocking to trading operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Configuration for cloud backup."""

    # Upstash Redis (hot tier)
    upstash_url: str = ""
    upstash_token: str = ""

    # Turso SQLite (warm tier)
    turso_url: str = ""
    turso_token: str = ""

    # Supabase (cold tier)
    supabase_url: str = ""
    supabase_key: str = ""

    # Backup settings
    hot_tier_ttl_hours: int = 24
    warm_tier_retention_days: int = 30
    batch_size: int = 100

    @classmethod
    def from_env(cls) -> BackupConfig:
        """Load configuration from environment variables."""
        return cls(
            upstash_url=os.getenv("UPSTASH_REDIS_URL", ""),
            upstash_token=os.getenv("UPSTASH_REDIS_TOKEN", ""),
            turso_url=os.getenv("TURSO_DATABASE_URL", ""),
            turso_token=os.getenv("TURSO_AUTH_TOKEN", ""),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", ""),
        )


class BackupTier(ABC):
    """Abstract base class for backup tiers."""

    @abstractmethod
    async def store_event(self, event: dict[str, Any]) -> bool:
        """Store a single event."""
        pass

    @abstractmethod
    async def store_state(self, key: str, state: dict[str, Any], ttl: int | None = None) -> bool:
        """Store current state (for hot tier)."""
        pass

    @abstractmethod
    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Retrieve current state."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the tier is healthy."""
        pass


class UpstashRedisBackup(BackupTier):
    """
    Hot tier using Upstash Redis.

    Stores:
    - Current positions
    - Pending exits
    - Distributed locks
    - Recent event IDs (deduplication)
    """

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Lazy initialize Redis client."""
        if self._client is None:
            try:
                from upstash_redis import Redis
                self._client = Redis(url=self.url, token=self.token)
            except ImportError:
                logger.warning("upstash-redis not installed. Install with: pip install upstash-redis")
                return None
        return self._client

    async def store_event(self, event: dict[str, Any]) -> bool:
        """Store event ID for deduplication."""
        client = await self._get_client()
        if not client:
            return False

        try:
            # Store event ID with short TTL for deduplication
            event_id = event.get("event_id", "")
            await asyncio.to_thread(
                client.set,
                f"event:{event_id}",
                "1",
                ex=3600,  # 1 hour TTL
            )
            return True
        except Exception as e:
            logger.error(f"Upstash store_event failed: {e}")
            return False

    async def store_state(self, key: str, state: dict[str, Any], ttl: int | None = None) -> bool:
        """Store state with optional TTL."""
        client = await self._get_client()
        if not client:
            return False

        try:
            await asyncio.to_thread(
                client.set,
                key,
                json.dumps(state),
                ex=ttl,
            )
            return True
        except Exception as e:
            logger.error(f"Upstash store_state failed: {e}")
            return False

    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Retrieve state."""
        client = await self._get_client()
        if not client:
            return None

        try:
            result = await asyncio.to_thread(client.get, key)
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.error(f"Upstash get_state failed: {e}")
            return None

    async def acquire_lock(self, lock_name: str, ttl: int = 60) -> bool:
        """Acquire distributed lock (prevents split brain)."""
        client = await self._get_client()
        if not client:
            return True  # Fail open if no Redis

        try:
            result = await asyncio.to_thread(
                client.set,
                f"lock:{lock_name}",
                os.getenv("INSTANCE_ID", "default"),
                nx=True,
                ex=ttl,
            )
            return result is not None
        except Exception as e:
            logger.error(f"Upstash acquire_lock failed: {e}")
            return True  # Fail open

    async def release_lock(self, lock_name: str) -> bool:
        """Release distributed lock."""
        client = await self._get_client()
        if not client:
            return True

        try:
            await asyncio.to_thread(client.delete, f"lock:{lock_name}")
            return True
        except Exception as e:
            logger.error(f"Upstash release_lock failed: {e}")
            return False

    async def health_check(self) -> bool:
        """Check Redis health."""
        client = await self._get_client()
        if not client:
            return False

        try:
            result = await asyncio.to_thread(client.ping)
            return result == "PONG"
        except Exception:
            return False


class TursoSQLiteBackup(BackupTier):
    """
    Warm tier using Turso (edge SQLite).

    Stores:
    - Recent events (30 days)
    - Trade history
    - Queryable audit log
    """

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self._client: Any = None
        self._initialized = False

    async def _get_client(self) -> Any:
        """Lazy initialize Turso client."""
        if self._client is None:
            try:
                import libsql_experimental as libsql
                self._client = libsql.connect(
                    self.url,
                    auth_token=self.token,
                )
                if not self._initialized:
                    await self._init_schema()
                    self._initialized = True
            except ImportError:
                logger.warning("libsql not installed. Install with: pip install libsql-experimental")
                return None
        return self._client

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        if not self._client:
            return

        await asyncio.to_thread(
            self._client.execute,
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT,
                payload TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_hash TEXT NOT NULL,
                instance_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await asyncio.to_thread(
            self._client.execute,
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
        )
        await asyncio.to_thread(
            self._client.execute,
            "CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol)"
        )
        await asyncio.to_thread(
            self._client.execute,
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)"
        )

    async def store_event(self, event: dict[str, Any]) -> bool:
        """Store event in Turso."""
        client = await self._get_client()
        if not client:
            return False

        try:
            await asyncio.to_thread(
                client.execute,
                """
                INSERT OR REPLACE INTO events
                (event_id, event_type, timestamp, symbol, payload, sequence_number, event_hash, instance_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    event.get("event_id"),
                    event.get("event_type"),
                    event.get("timestamp"),
                    event.get("symbol"),
                    json.dumps(event.get("payload", {})),
                    event.get("sequence_number", 0),
                    event.get("event_hash", ""),
                    event.get("instance_id", ""),
                ],
            )
            return True
        except Exception as e:
            logger.error(f"Turso store_event failed: {e}")
            return False

    async def store_state(self, key: str, state: dict[str, Any], ttl: int | None = None) -> bool:
        """Not used for Turso - events only."""
        return True

    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Not used for Turso - use query methods instead."""
        return None

    async def query_events(
        self,
        since: datetime | None = None,
        symbol: str | None = None,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query events from Turso."""
        client = await self._get_client()
        if not client:
            return []

        conditions = []
        params = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM events
            WHERE {where_clause}
            ORDER BY sequence_number DESC
            LIMIT ?
        """
        params.append(limit)

        try:
            result = await asyncio.to_thread(client.execute, query, params)
            return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Turso query failed: {e}")
            return []

    async def health_check(self) -> bool:
        """Check Turso health."""
        client = await self._get_client()
        if not client:
            return False

        try:
            result = await asyncio.to_thread(client.execute, "SELECT 1")
            return result is not None
        except Exception:
            return False


class SupabaseBackup(BackupTier):
    """
    Cold tier using Supabase (Postgres).

    Stores:
    - Full event history (forever)
    - Aggregated analytics
    - Compliance audit trail
    """

    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self._client: Any = None
        self._initialized = False

    async def _get_client(self) -> Any:
        """Lazy initialize Supabase client."""
        if self._client is None:
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
                if not self._initialized:
                    # Schema should be created via Supabase dashboard/migrations
                    self._initialized = True
            except ImportError:
                logger.warning("supabase not installed. Install with: pip install supabase")
                return None
        return self._client

    async def store_event(self, event: dict[str, Any]) -> bool:
        """Store event in Supabase."""
        client = await self._get_client()
        if not client:
            return False

        try:
            await asyncio.to_thread(
                lambda: client.table("trading_events").upsert({
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    "timestamp": event.get("timestamp"),
                    "symbol": event.get("symbol"),
                    "payload": event.get("payload", {}),
                    "sequence_number": event.get("sequence_number", 0),
                    "event_hash": event.get("event_hash", ""),
                    "instance_id": event.get("instance_id", ""),
                }).execute
            )
            return True
        except Exception as e:
            logger.error(f"Supabase store_event failed: {e}")
            return False

    async def store_state(self, key: str, state: dict[str, Any], ttl: int | None = None) -> bool:
        """Store state snapshot."""
        client = await self._get_client()
        if not client:
            return False

        try:
            await asyncio.to_thread(
                lambda: client.table("state_snapshots").upsert({
                    "key": key,
                    "state": state,
                    "updated_at": datetime.now(UTC).isoformat(),
                }).execute
            )
            return True
        except Exception as e:
            logger.error(f"Supabase store_state failed: {e}")
            return False

    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Retrieve state snapshot."""
        client = await self._get_client()
        if not client:
            return None

        try:
            result = await asyncio.to_thread(
                lambda: client.table("state_snapshots")
                .select("state")
                .eq("key", key)
                .single()
                .execute()
            )
            if result.data:
                return result.data.get("state")
            return None
        except Exception as e:
            logger.error(f"Supabase get_state failed: {e}")
            return None

    async def health_check(self) -> bool:
        """Check Supabase health."""
        client = await self._get_client()
        if not client:
            return False

        try:
            # Simple query to check connection
            await asyncio.to_thread(
                lambda: client.table("trading_events").select("event_id").limit(1).execute()
            )
            return True
        except Exception:
            return False


class CloudBackupManager:
    """
    Manages multi-tier cloud backup with graceful degradation.

    If a tier fails, events are queued for retry.
    Trading never blocks on backup failures.
    """

    def __init__(self, config: BackupConfig | None = None):
        self.config = config or BackupConfig.from_env()

        # Initialize tiers (lazy)
        self._hot: UpstashRedisBackup | None = None
        self._warm: TursoSQLiteBackup | None = None
        self._cold: SupabaseBackup | None = None

        # Retry queue for failed backups
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._retry_task: asyncio.Task | None = None

    @property
    def hot(self) -> UpstashRedisBackup | None:
        """Hot tier (Upstash Redis)."""
        if self._hot is None and self.config.upstash_url:
            self._hot = UpstashRedisBackup(
                self.config.upstash_url,
                self.config.upstash_token,
            )
        return self._hot

    @property
    def warm(self) -> TursoSQLiteBackup | None:
        """Warm tier (Turso SQLite)."""
        if self._warm is None and self.config.turso_url:
            self._warm = TursoSQLiteBackup(
                self.config.turso_url,
                self.config.turso_token,
            )
        return self._warm

    @property
    def cold(self) -> SupabaseBackup | None:
        """Cold tier (Supabase)."""
        if self._cold is None and self.config.supabase_url:
            self._cold = SupabaseBackup(
                self.config.supabase_url,
                self.config.supabase_key,
            )
        return self._cold

    async def backup_event(self, event: dict[str, Any]) -> None:
        """
        Backup event to all tiers asynchronously.

        This is non-blocking - failures are retried in background.
        """
        tasks = []

        # Hot tier - event ID for dedup
        if self.hot:
            tasks.append(self._backup_to_tier("hot", self.hot, event))

        # Warm tier - full event
        if self.warm:
            tasks.append(self._backup_to_tier("warm", self.warm, event))

        # Cold tier - full event (async, can be slower)
        if self.cold:
            tasks.append(self._backup_to_tier("cold", self.cold, event))

        # Run all backups concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _backup_to_tier(
        self,
        tier_name: str,
        tier: BackupTier,
        event: dict[str, Any],
    ) -> None:
        """Backup to a specific tier with retry on failure."""
        try:
            success = await tier.store_event(event)
            if not success:
                await self._retry_queue.put((tier_name, event))
        except Exception as e:
            logger.error(f"Backup to {tier_name} failed: {e}")
            await self._retry_queue.put((tier_name, event))

    async def store_position_state(
        self,
        positions: list[dict[str, Any]],
    ) -> None:
        """Store current position state to hot tier."""
        if not self.hot:
            return

        try:
            await self.hot.store_state(
                "positions:current",
                {
                    "positions": positions,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
                ttl=self.config.hot_tier_ttl_hours * 3600,
            )
        except Exception as e:
            logger.error(f"Failed to store position state: {e}")

    async def get_position_state(self) -> list[dict[str, Any]]:
        """Retrieve position state from hot tier."""
        if not self.hot:
            return []

        try:
            state = await self.hot.get_state("positions:current")
            if state:
                return state.get("positions", [])
        except Exception as e:
            logger.error(f"Failed to get position state: {e}")

        return []

    async def acquire_trading_lock(self, symbol: str) -> bool:
        """Acquire distributed lock for a symbol."""
        if not self.hot:
            return True  # No Redis = single instance mode

        return await self.hot.acquire_lock(f"trading:{symbol}")

    async def release_trading_lock(self, symbol: str) -> None:
        """Release distributed lock."""
        if self.hot:
            await self.hot.release_lock(f"trading:{symbol}")

    async def health_check(self) -> dict[str, bool]:
        """Check health of all tiers."""
        results = {}

        if self.hot:
            results["hot"] = await self.hot.health_check()
        if self.warm:
            results["warm"] = await self.warm.health_check()
        if self.cold:
            results["cold"] = await self.cold.health_check()

        return results

    async def start_retry_worker(self) -> None:
        """Start background worker for retrying failed backups."""
        if self._retry_task is None:
            self._retry_task = asyncio.create_task(self._retry_worker())

    async def _retry_worker(self) -> None:
        """Background worker that retries failed backups."""
        while True:
            try:
                tier_name, event = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=60,
                )

                tier = {
                    "hot": self.hot,
                    "warm": self.warm,
                    "cold": self.cold,
                }.get(tier_name)

                if tier:
                    success = await tier.store_event(event)
                    if not success:
                        # Re-queue with backoff
                        await asyncio.sleep(5)
                        await self._retry_queue.put((tier_name, event))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Retry worker error: {e}")
                await asyncio.sleep(1)

    async def close(self) -> None:
        """Clean shutdown."""
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
