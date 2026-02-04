"""
AlphaStrike Trading Bot - Position Sync with Exchange (US-025)

Provides periodic synchronization between local position state and exchange,
detecting orphan positions, reconciling discrepancies, and managing exit orders.
Exchange is always the source of truth.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from src.data.rest_client import (
    OrderSide,
    PositionSide,
    RESTClient,
    RESTClientError,
)
from src.data.rest_client import Position as ExchangePosition
from src.risk.portfolio import PortfolioManager, Position
from src.strategy.exit_manager import ExitManager, ExitPrices

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a position synchronization operation."""

    success: bool
    positions_synced: int
    orphans_found: int
    discrepancies_resolved: int
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Discrepancy:
    """
    Represents a discrepancy between local and exchange position state.

    Attributes:
        position_key: Tuple of (symbol, side) identifying the position.
        local_size: Position size tracked locally.
        exchange_size: Position size on exchange (source of truth).
        action_taken: Description of corrective action taken.
    """

    position_key: tuple[str, str]
    local_size: float
    exchange_size: float
    action_taken: str


class PositionSync:
    """
    Synchronizes local position state with exchange positions.

    The exchange is always the source of truth. Local state is updated
    to match exchange state during periodic syncs.

    Features:
    - Periodic sync (every 60 seconds by default)
    - Orphan position detection (exchange positions not tracked locally)
    - Discrepancy reconciliation (size/state mismatches)
    - Stop-loss order management
    - Take-profit order management
    - Exit condition processing

    Example:
        rest_client = await get_rest_client()
        portfolio = PortfolioManager(initial_balance=10000.0)
        exit_manager = ExitManager()

        sync = PositionSync(
            rest_client=rest_client,
            portfolio_manager=portfolio,
        )

        result = await sync.sync_positions()
        if result.success:
            print(f"Synced {result.positions_synced} positions")
    """

    # Sync interval in seconds
    DEFAULT_SYNC_INTERVAL: int = 60

    def __init__(
        self,
        rest_client: RESTClient,
        portfolio_manager: PortfolioManager,
        sync_interval: int = DEFAULT_SYNC_INTERVAL,
    ) -> None:
        """
        Initialize PositionSync.

        Args:
            rest_client: REST client for exchange API communication.
            portfolio_manager: Portfolio manager for local state tracking.
            sync_interval: Interval between automatic syncs in seconds.
        """
        self.rest_client = rest_client
        self.portfolio_manager = portfolio_manager
        self.sync_interval = sync_interval

        self._last_sync: datetime | None = None
        self._is_running = False
        self._sync_task: asyncio.Task[None] | None = None

        logger.info(
            "PositionSync initialized",
            extra={"sync_interval": sync_interval},
        )

    async def sync_positions(self) -> SyncResult:
        """
        Perform a full position synchronization with the exchange.

        Fetches positions from exchange and reconciles with local state.
        Exchange is the source of truth - local state is updated to match.

        Returns:
            SyncResult with synchronization statistics and any errors.
        """
        errors: list[str] = []
        positions_synced = 0
        discrepancies_resolved = 0

        try:
            # Fetch positions from exchange
            exchange_positions = await self.rest_client.get_positions()

            # Get local positions
            local_positions = self.portfolio_manager.get_all_positions()

            # Detect orphans (exchange positions not tracked locally)
            orphans = self.detect_orphan_positions(
                exchange_positions=exchange_positions,
                local_positions=local_positions,
            )

            # Reconcile discrepancies
            discrepancies = self.reconcile_discrepancies(
                exchange_positions=exchange_positions,
                local_positions=local_positions,
            )
            discrepancies_resolved = len(discrepancies)

            # Log orphans and discrepancies
            for orphan in orphans:
                logger.warning(
                    "Orphan position detected on exchange",
                    extra={
                        "symbol": orphan.symbol,
                        "side": orphan.side,
                        "size": orphan.size,
                        "entry_price": orphan.entry_price,
                    },
                )

            for disc in discrepancies:
                logger.info(
                    "Discrepancy resolved",
                    extra={
                        "position_key": disc.position_key,
                        "local_size": disc.local_size,
                        "exchange_size": disc.exchange_size,
                        "action": disc.action_taken,
                    },
                )

            # Convert exchange positions to local Position format and sync
            local_format_positions = self._convert_to_local_positions(exchange_positions)
            self.portfolio_manager.sync_with_exchange(local_format_positions)

            positions_synced = len(exchange_positions)
            self._last_sync = datetime.utcnow()

            logger.info(
                "Position sync completed",
                extra={
                    "positions_synced": positions_synced,
                    "orphans_found": len(orphans),
                    "discrepancies_resolved": discrepancies_resolved,
                },
            )

            return SyncResult(
                success=True,
                positions_synced=positions_synced,
                orphans_found=len(orphans),
                discrepancies_resolved=discrepancies_resolved,
                errors=errors,
            )

        except RESTClientError as e:
            error_msg = f"Failed to fetch positions from exchange: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

            return SyncResult(
                success=False,
                positions_synced=0,
                orphans_found=0,
                discrepancies_resolved=0,
                errors=errors,
            )

        except Exception as e:
            error_msg = f"Unexpected error during position sync: {e}"
            logger.exception(error_msg)
            errors.append(error_msg)

            return SyncResult(
                success=False,
                positions_synced=0,
                orphans_found=0,
                discrepancies_resolved=0,
                errors=errors,
            )

    def detect_orphan_positions(
        self,
        exchange_positions: list[ExchangePosition] | None = None,
        local_positions: list[Position] | None = None,
    ) -> list[Position]:
        """
        Find positions on the exchange that are not tracked locally.

        Args:
            exchange_positions: Positions from exchange (fetched if None).
            local_positions: Local positions (fetched from portfolio if None).

        Returns:
            List of Position objects that exist on exchange but not locally.
        """
        if local_positions is None:
            local_positions = self.portfolio_manager.get_all_positions()

        if exchange_positions is None:
            # Return empty list - caller should await sync_positions() instead
            return []

        # Build set of local position keys
        local_keys: set[tuple[str, str]] = set()
        for pos in local_positions:
            local_keys.add((pos.symbol, pos.side))

        # Find orphans - positions on exchange not in local state
        orphans: list[Position] = []
        for ex_pos in exchange_positions:
            # Exchange Position uses PositionSide enum
            side_str = ex_pos.side.value.upper()

            key = (ex_pos.symbol, side_str)

            # Also check with lowercase for compatibility
            key_lower = (ex_pos.symbol, side_str.lower())

            if key not in local_keys and key_lower not in local_keys:
                # Convert to local Position format
                local_pos = self._convert_single_position(ex_pos)
                orphans.append(local_pos)

        return orphans

    def reconcile_discrepancies(
        self,
        exchange_positions: list[ExchangePosition],
        local_positions: list[Position],
    ) -> list[Discrepancy]:
        """
        Reconcile differences between exchange and local position state.

        Exchange is the source of truth. Local state is updated to match.

        Args:
            exchange_positions: Positions from exchange API.
            local_positions: Positions tracked locally.

        Returns:
            List of Discrepancy objects describing what was reconciled.
        """
        discrepancies: list[Discrepancy] = []

        # Build lookup for exchange positions by key
        exchange_by_key: dict[tuple[str, str], ExchangePosition] = {}
        for ex_pos in exchange_positions:
            side_str = ex_pos.side.value.upper()
            key = (ex_pos.symbol, side_str)
            exchange_by_key[key] = ex_pos

        # Check local positions against exchange
        for local_pos in local_positions:
            key = (local_pos.symbol, local_pos.side)
            key_upper = (local_pos.symbol, local_pos.side.upper())

            ex_pos = exchange_by_key.get(key) or exchange_by_key.get(key_upper)

            if ex_pos is None:
                # Position exists locally but not on exchange - needs removal
                discrepancies.append(
                    Discrepancy(
                        position_key=key,
                        local_size=local_pos.size,
                        exchange_size=0.0,
                        action_taken="removed_local_position",
                    )
                )
            elif abs(local_pos.size - ex_pos.size) > 1e-8:
                # Size mismatch - update local to match exchange
                discrepancies.append(
                    Discrepancy(
                        position_key=key,
                        local_size=local_pos.size,
                        exchange_size=ex_pos.size,
                        action_taken="updated_size_to_exchange_value",
                    )
                )

        # Check for positions on exchange not tracked locally (orphans)
        local_keys = {(pos.symbol, pos.side) for pos in local_positions}
        local_keys_upper = {(pos.symbol, pos.side.upper()) for pos in local_positions}

        for key, ex_pos in exchange_by_key.items():
            if key not in local_keys and key not in local_keys_upper:
                discrepancies.append(
                    Discrepancy(
                        position_key=key,
                        local_size=0.0,
                        exchange_size=ex_pos.size,
                        action_taken="added_from_exchange",
                    )
                )

        return discrepancies

    async def manage_stop_loss_orders(
        self,
        position: Position,
        exit_prices: ExitPrices,
    ) -> None:
        """
        Place or update stop-loss orders for a position.

        Args:
            position: Position to manage stop-loss for.
            exit_prices: Exit price levels including stop-loss.
        """
        try:
            # Determine order side for closing the position
            if position.side == "LONG":
                order_side = OrderSide.SELL
                position_side = PositionSide.LONG
            else:
                order_side = OrderSide.BUY
                position_side = PositionSide.SHORT

            # Place stop-loss order
            result = await self.rest_client.place_stop_loss_order(
                symbol=position.symbol,
                side=order_side,
                trigger_price=exit_prices.stop_loss,
                size=position.size,
                position_side=position_side,
            )

            logger.info(
                "Stop-loss order placed",
                extra={
                    "symbol": position.symbol,
                    "side": position.side,
                    "trigger_price": exit_prices.stop_loss,
                    "size": position.size,
                    "order_id": result.get("orderId", "unknown"),
                },
            )

        except RESTClientError as e:
            logger.error(
                f"Failed to place stop-loss order: {e}",
                extra={
                    "symbol": position.symbol,
                    "side": position.side,
                    "stop_loss": exit_prices.stop_loss,
                },
            )
            raise

    async def manage_take_profit_orders(
        self,
        position: Position,
        exit_prices: ExitPrices,
    ) -> None:
        """
        Place or update take-profit orders for a position.

        Places multi-level take-profit orders based on exit_prices configuration.

        Args:
            position: Position to manage take-profit for.
            exit_prices: Exit price levels including TP1, TP2.
        """
        try:
            # Determine order side for closing the position
            if position.side == "LONG":
                order_side = OrderSide.SELL
                position_side = PositionSide.LONG
            else:
                order_side = OrderSide.BUY
                position_side = PositionSide.SHORT

            # Calculate sizes for each TP level
            tp1_size = position.size * exit_prices.tp1_size_pct
            tp2_size = position.size * exit_prices.tp2_size_pct

            # Place TP1 order
            tp1_result = await self.rest_client.place_take_profit_order(
                symbol=position.symbol,
                side=order_side,
                trigger_price=exit_prices.tp1_price,
                size=tp1_size,
                position_side=position_side,
            )

            logger.info(
                "Take-profit 1 order placed",
                extra={
                    "symbol": position.symbol,
                    "side": position.side,
                    "trigger_price": exit_prices.tp1_price,
                    "size": tp1_size,
                    "order_id": tp1_result.get("orderId", "unknown"),
                },
            )

            # Place TP2 order
            tp2_result = await self.rest_client.place_take_profit_order(
                symbol=position.symbol,
                side=order_side,
                trigger_price=exit_prices.tp2_price,
                size=tp2_size,
                position_side=position_side,
            )

            logger.info(
                "Take-profit 2 order placed",
                extra={
                    "symbol": position.symbol,
                    "side": position.side,
                    "trigger_price": exit_prices.tp2_price,
                    "size": tp2_size,
                    "order_id": tp2_result.get("orderId", "unknown"),
                },
            )

        except RESTClientError as e:
            logger.error(
                f"Failed to place take-profit order: {e}",
                extra={
                    "symbol": position.symbol,
                    "side": position.side,
                },
            )
            raise

    async def process_exit_conditions(
        self,
        position: Position,
        exit_manager: ExitManager,
    ) -> bool:
        """
        Check and process exit conditions for a position.

        Evaluates whether the position should be exited based on
        current market conditions and exit manager state.

        Args:
            position: Position to evaluate.
            exit_manager: ExitManager with exit logic and state.

        Returns:
            True if an exit was triggered, False otherwise.
        """
        try:
            # Get current market price
            ticker = await self.rest_client.get_ticker(position.symbol)
            current_price = float(ticker.get("last", 0))

            if current_price <= 0:
                logger.warning(
                    "Invalid market price for exit evaluation",
                    extra={"symbol": position.symbol},
                )
                return False

            # Use a default ATR estimate if not available
            # In production, this should come from a feature pipeline
            estimated_atr = current_price * 0.02  # 2% of price as default ATR

            # Check exit conditions
            should_exit, reason = exit_manager.should_exit(
                position=position,
                current_price=current_price,
                atr=estimated_atr,
                regime="ranging",  # Default regime - should come from regime detector
            )

            if should_exit:
                logger.info(
                    "Exit condition triggered",
                    extra={
                        "symbol": position.symbol,
                        "side": position.side,
                        "reason": reason,
                        "current_price": current_price,
                    },
                )

                # Place market order to close position
                if position.side == "LONG":
                    order_side = OrderSide.SELL
                    position_side = PositionSide.LONG
                else:
                    order_side = OrderSide.BUY
                    position_side = PositionSide.SHORT

                from src.data.rest_client import OrderRequest, OrderType, TimeInForce

                close_order = OrderRequest(
                    symbol=position.symbol,
                    side=order_side,
                    order_type=OrderType.MARKET,
                    size=position.size,
                    position_side=position_side,
                    time_in_force=TimeInForce.IOC,
                    reduce_only=True,
                )

                result = await self.rest_client.place_order(close_order)

                logger.info(
                    "Position closed",
                    extra={
                        "symbol": position.symbol,
                        "side": position.side,
                        "reason": reason,
                        "order_id": result.order_id,
                    },
                )

                # Unregister from exit manager
                exit_manager.unregister_position(position.symbol, position.side)

                return True

            return False

        except RESTClientError as e:
            logger.error(
                f"Failed to process exit conditions: {e}",
                extra={
                    "symbol": position.symbol,
                    "side": position.side,
                },
            )
            return False

        except Exception as e:
            logger.exception(
                f"Unexpected error processing exit conditions: {e}",
                extra={
                    "symbol": position.symbol,
                    "side": position.side,
                },
            )
            return False

    def _convert_to_local_positions(
        self,
        exchange_positions: list[ExchangePosition],
    ) -> list[Position]:
        """
        Convert exchange positions to local Position format.

        Args:
            exchange_positions: Positions from REST client.

        Returns:
            List of Position objects in local portfolio format.
        """
        local_positions: list[Position] = []

        for ex_pos in exchange_positions:
            local_pos = self._convert_single_position(ex_pos)
            local_positions.append(local_pos)

        return local_positions

    def _convert_single_position(self, ex_pos: ExchangePosition) -> Position:
        """
        Convert a single exchange position to local Position format.

        Args:
            ex_pos: Position from REST client.

        Returns:
            Position in local portfolio format.
        """
        # Get side as string for local Position (exchange uses PositionSide enum)
        side_str: Literal["LONG", "SHORT"] = (
            "LONG" if ex_pos.side.value.lower() == "long" else "SHORT"
        )

        # Use timestamp from exchange position
        entry_time = ex_pos.timestamp

        return Position(
            symbol=ex_pos.symbol,
            side=side_str,
            size=ex_pos.size,
            entry_price=ex_pos.entry_price,
            entry_time=entry_time,
            leverage=ex_pos.leverage,
            unrealized_pnl=ex_pos.unrealized_pnl,
            current_price=ex_pos.mark_price,
        )

    async def start_periodic_sync(self) -> None:
        """
        Start the periodic position synchronization loop.

        Runs in the background and syncs positions every sync_interval seconds.
        """
        if self._is_running:
            logger.warning("Periodic sync already running")
            return

        self._is_running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(
            "Started periodic position sync",
            extra={"interval": self.sync_interval},
        )

    async def stop_periodic_sync(self) -> None:
        """Stop the periodic position synchronization loop."""
        if not self._is_running:
            return

        self._is_running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        logger.info("Stopped periodic position sync")

    async def _sync_loop(self) -> None:
        """Internal sync loop for periodic synchronization."""
        while self._is_running:
            try:
                await self.sync_positions()
            except Exception as e:
                logger.exception(f"Error in sync loop: {e}")

            await asyncio.sleep(self.sync_interval)

    @property
    def last_sync(self) -> datetime | None:
        """Get timestamp of last successful sync."""
        return self._last_sync

    @property
    def is_running(self) -> bool:
        """Check if periodic sync is running."""
        return self._is_running
