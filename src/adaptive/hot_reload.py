"""
Hot Reload Manager

Handles zero-downtime model and configuration updates with state preservation.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """Serializable trading state for hot reload."""

    # Open positions
    positions: list[dict[str, Any]] = field(default_factory=list)

    # Pending orders
    pending_orders: list[dict[str, Any]] = field(default_factory=list)

    # Equity tracking
    current_equity: float = 0.0
    peak_equity: float = 0.0

    # Model versions
    model_versions: dict[str, str] = field(default_factory=dict)

    # Dynamic leverage (NEW)
    current_leverage: float = 5.0
    leverage_reason: str = ""

    # Per-asset parameters (LLM-adjusted)
    asset_params: dict[str, dict] = field(default_factory=dict)

    # Timestamp
    saved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "positions": self.positions,
            "pending_orders": self.pending_orders,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "model_versions": self.model_versions,
            "current_leverage": self.current_leverage,
            "leverage_reason": self.leverage_reason,
            "asset_params": self.asset_params,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradingState":
        return cls(
            positions=data.get("positions", []),
            pending_orders=data.get("pending_orders", []),
            current_equity=data.get("current_equity", 0.0),
            peak_equity=data.get("peak_equity", 0.0),
            model_versions=data.get("model_versions", {}),
            current_leverage=data.get("current_leverage", 5.0),
            leverage_reason=data.get("leverage_reason", ""),
            asset_params=data.get("asset_params", {}),
            saved_at=data.get("saved_at"),
        )


@dataclass
class ReloadResult:
    """Result of a reload operation."""

    success: bool
    message: str
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    rollback_performed: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HotReloadManager:
    """
    Manages zero-downtime model and configuration updates.

    Flow:
    1. Save current state (positions, orders, equity)
    2. Train/load new models in background
    3. Validate new models on holdout data
    4. If validation passes, swap models atomically
    5. If validation fails, keep old models
    6. Resume trading with state restored
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        model_dir: Optional[Path] = None,
        min_validation_accuracy: float = 0.55,
    ):
        self.state_dir = state_dir or Path("data/state")
        self.model_dir = model_dir or Path("models")
        self.min_validation_accuracy = min_validation_accuracy

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._current_state: Optional[TradingState] = None
        self._is_reloading = False
        self._reload_lock = asyncio.Lock()

        # Callbacks
        self._on_reload_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_reload_complete: Optional[Callable[[ReloadResult], Awaitable[None]]] = None
        self._train_models: Optional[Callable[[], Awaitable[tuple[Any, str]]]] = None
        self._validate_models: Optional[Callable[[Any], Awaitable[float]]] = None
        self._swap_models: Optional[Callable[[Any], Awaitable[None]]] = None

    @property
    def is_reloading(self) -> bool:
        return self._is_reloading

    def register_callbacks(
        self,
        on_reload_start: Optional[Callable[[], Awaitable[None]]] = None,
        on_reload_complete: Optional[Callable[[ReloadResult], Awaitable[None]]] = None,
        train_models: Optional[Callable[[], Awaitable[tuple[Any, str]]]] = None,
        validate_models: Optional[Callable[[Any], Awaitable[float]]] = None,
        swap_models: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        """Register callback functions for reload process."""
        if on_reload_start:
            self._on_reload_start = on_reload_start
        if on_reload_complete:
            self._on_reload_complete = on_reload_complete
        if train_models:
            self._train_models = train_models
        if validate_models:
            self._validate_models = validate_models
        if swap_models:
            self._swap_models = swap_models

    async def save_state(self, state: TradingState) -> Path:
        """Save trading state to disk."""
        state.saved_at = datetime.now(timezone.utc).isoformat()
        self._current_state = state

        state_file = self.state_dir / "trading_state.json"
        backup_file = self.state_dir / f"trading_state_{state.saved_at.replace(':', '-')}.json"

        # Write to temp file first, then rename (atomic)
        temp_file = state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        temp_file.rename(state_file)

        # Also save backup
        with open(backup_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        logger.info(f"State saved to {state_file}")
        return state_file

    async def load_state(self) -> Optional[TradingState]:
        """Load trading state from disk."""
        state_file = self.state_dir / "trading_state.json"

        if not state_file.exists():
            return None

        with open(state_file, "r") as f:
            data = json.load(f)

        state = TradingState.from_dict(data)
        self._current_state = state
        logger.info(f"State loaded from {state_file}")
        return state

    async def reload_models(self) -> ReloadResult:
        """
        Perform a hot reload of ML models.

        This is the main entry point for model updates.
        """
        async with self._reload_lock:
            if self._is_reloading:
                return ReloadResult(
                    success=False,
                    message="Reload already in progress"
                )

            self._is_reloading = True

            try:
                # Step 1: Signal reload start
                if self._on_reload_start:
                    await self._on_reload_start()

                # Step 2: Train new models
                if not self._train_models:
                    return ReloadResult(
                        success=False,
                        message="No train_models callback registered"
                    )

                logger.info("Training new models...")
                new_models, new_version = await self._train_models()

                # Step 3: Validate new models
                if self._validate_models:
                    logger.info("Validating new models...")
                    accuracy = await self._validate_models(new_models)

                    if accuracy < self.min_validation_accuracy:
                        result = ReloadResult(
                            success=False,
                            message=f"Validation failed: {accuracy:.2%} < {self.min_validation_accuracy:.2%}",
                            new_version=new_version,
                        )
                        if self._on_reload_complete:
                            await self._on_reload_complete(result)
                        return result

                # Step 4: Swap models
                if self._swap_models:
                    logger.info("Swapping models...")
                    old_version = self._current_state.model_versions.get("ensemble") if self._current_state else None
                    await self._swap_models(new_models)

                    if self._current_state:
                        self._current_state.model_versions["ensemble"] = new_version

                result = ReloadResult(
                    success=True,
                    message="Models reloaded successfully",
                    old_version=old_version if self._current_state else None,
                    new_version=new_version,
                )

                if self._on_reload_complete:
                    await self._on_reload_complete(result)

                return result

            except Exception as e:
                logger.error(f"Reload failed: {e}")
                return ReloadResult(
                    success=False,
                    message=f"Reload failed: {e}",
                    rollback_performed=True,
                )

            finally:
                self._is_reloading = False

    async def reload_configs(self) -> ReloadResult:
        """Reload asset configurations from disk."""
        try:
            from src.adaptive.asset_config import load_all_asset_configs

            configs = load_all_asset_configs()
            logger.info(f"Reloaded {len(configs)} asset configs")

            return ReloadResult(
                success=True,
                message=f"Reloaded {len(configs)} configs: {', '.join(configs.keys())}",
            )

        except Exception as e:
            logger.error(f"Config reload failed: {e}")
            return ReloadResult(
                success=False,
                message=f"Config reload failed: {e}",
            )

    async def graceful_restart(
        self,
        state: TradingState,
        restart_command: Optional[str] = None,
    ) -> None:
        """
        Perform a graceful restart with state preservation.

        1. Save current state
        2. Execute restart command (or signal supervisor)
        3. New process loads state and resumes
        """
        # Save state
        await self.save_state(state)
        logger.info("State saved for graceful restart")

        # Write restart marker
        restart_marker = self.state_dir / "restart_pending"
        restart_marker.write_text(datetime.now(timezone.utc).isoformat())

        if restart_command:
            import subprocess
            logger.info(f"Executing restart: {restart_command}")
            subprocess.Popen(restart_command, shell=True)

        logger.info("Graceful restart initiated")

    async def check_restart_recovery(self) -> Optional[TradingState]:
        """
        Check if we're recovering from a restart and load state.

        Call this at startup to resume from saved state.
        """
        restart_marker = self.state_dir / "restart_pending"

        if restart_marker.exists():
            logger.info("Recovering from restart...")
            restart_marker.unlink()  # Remove marker

            state = await self.load_state()
            if state:
                logger.info(f"Recovered state from {state.saved_at}")
                logger.info(f"  Positions: {len(state.positions)}")
                logger.info(f"  Pending orders: {len(state.pending_orders)}")
                return state

        return None

    def get_state_summary(self) -> dict:
        """Get summary of current state for monitoring."""
        if not self._current_state:
            return {"status": "no_state"}

        return {
            "status": "active",
            "positions": len(self._current_state.positions),
            "pending_orders": len(self._current_state.pending_orders),
            "equity": self._current_state.current_equity,
            "peak_equity": self._current_state.peak_equity,
            "model_versions": self._current_state.model_versions,
            "saved_at": self._current_state.saved_at,
            "is_reloading": self._is_reloading,
        }
