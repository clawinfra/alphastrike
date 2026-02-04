"""
Multi-Wallet Manager for Hyperliquid

Supports trading across multiple wallets for risk distribution:
- Spread capital across wallets to reduce single-point-of-failure risk
- Independent position limits per wallet
- Configurable allocation strategies (round-robin, weighted, lowest-exposure)
- Aggregate portfolio view across all wallets
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.exchange.adapters.hyperliquid.adapter import HyperliquidAdapter
from src.exchange.models import (
    OrderSide,
    UnifiedAccountBalance,
    UnifiedOrder,
    UnifiedOrderResult,
    UnifiedPosition,
)

logger = logging.getLogger(__name__)


class AllocationStrategy(str, Enum):
    """Strategy for allocating trades across wallets."""

    ROUND_ROBIN = "round_robin"  # Rotate through wallets
    LOWEST_EXPOSURE = "lowest_exposure"  # Use wallet with lowest current exposure
    WEIGHTED = "weighted"  # Allocate based on wallet weights
    HIGHEST_BALANCE = "highest_balance"  # Use wallet with most available balance
    SPECIFIC = "specific"  # Caller specifies which wallet


@dataclass
class WalletConfig:
    """Configuration for a single wallet."""

    name: str  # Human-readable name (e.g., "wallet-1", "main", "hedge")
    private_key: str  # Wallet private key
    address: str = ""  # Derived from private key if not provided
    weight: float = 1.0  # Allocation weight (for weighted strategy)
    max_exposure: float = 1.0  # Max exposure as fraction of balance (0.0-1.0)
    max_positions: int = 10  # Maximum concurrent positions
    enabled: bool = True  # Whether wallet is active for trading
    tags: list[str] = field(default_factory=list)  # Tags for filtering (e.g., ["long-only", "btc"])


@dataclass
class WalletState:
    """Runtime state for a wallet."""

    config: WalletConfig
    adapter: HyperliquidAdapter
    balance: UnifiedAccountBalance | None = None
    positions: list[UnifiedPosition] = field(default_factory=list)
    current_exposure: float = 0.0
    last_trade_time: datetime | None = None
    trade_count: int = 0
    error_count: int = 0
    last_error: str = ""


@dataclass
class AggregatePortfolio:
    """Aggregate view across all wallets."""

    total_balance: float = 0.0
    available_balance: float = 0.0
    total_exposure: float = 0.0
    total_unrealized_pnl: float = 0.0
    positions_by_symbol: dict[str, list[tuple[str, UnifiedPosition]]] = field(default_factory=dict)
    wallet_balances: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "AGGREGATE PORTFOLIO",
            "=" * 60,
            f"Total Balance: ${self.total_balance:,.2f}",
            f"Available: ${self.available_balance:,.2f}",
            f"Total Exposure: {self.total_exposure:.1%}",
            f"Unrealized P&L: ${self.total_unrealized_pnl:+,.2f}",
            "",
            "Wallet Balances:",
        ]
        for name, balance in self.wallet_balances.items():
            lines.append(f"  {name}: ${balance:,.2f}")

        if self.positions_by_symbol:
            lines.append("")
            lines.append("Positions by Symbol:")
            for symbol, wallet_positions in self.positions_by_symbol.items():
                total_size = sum(p.quantity for _, p in wallet_positions)
                lines.append(f"  {symbol}: {total_size:.6f} across {len(wallet_positions)} wallet(s)")
                for wallet_name, pos in wallet_positions:
                    lines.append(f"    - {wallet_name}: {pos.quantity:.6f} @ ${pos.entry_price:,.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class MultiWalletManager:
    """
    Manages multiple Hyperliquid wallets for distributed trading.

    Features:
    - Initialize and manage multiple wallet adapters
    - Allocate trades based on configurable strategies
    - Track positions and exposure per wallet
    - Aggregate portfolio view
    - Automatic failover if a wallet has issues
    """

    def __init__(
        self,
        wallet_configs: list[WalletConfig],
        testnet: bool = True,
        default_strategy: AllocationStrategy = AllocationStrategy.LOWEST_EXPOSURE,
    ):
        self.wallet_configs = wallet_configs
        self.testnet = testnet
        self.default_strategy = default_strategy

        self._wallets: dict[str, WalletState] = {}
        self._round_robin_index = 0
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize all wallet adapters."""
        logger.info(f"Initializing {len(self.wallet_configs)} wallets...")

        for config in self.wallet_configs:
            if not config.enabled:
                logger.info(f"Wallet '{config.name}' is disabled, skipping")
                continue

            try:
                adapter = HyperliquidAdapter(
                    testnet=self.testnet,
                    private_key=config.private_key,
                    wallet_address=config.address or None,
                )
                await adapter.initialize()

                # Get wallet address if not provided
                if not config.address:
                    # Derive address from private key
                    from eth_account import Account
                    account = Account.from_key(config.private_key)
                    config.address = account.address

                self._wallets[config.name] = WalletState(
                    config=config,
                    adapter=adapter,
                )

                logger.info(f"Initialized wallet '{config.name}': {config.address[:10]}...{config.address[-6:]}")

            except Exception as e:
                logger.error(f"Failed to initialize wallet '{config.name}': {e}")

        if not self._wallets:
            raise RuntimeError("No wallets initialized successfully")

        # Initial state sync
        await self.sync_all_wallets()
        self._initialized = True

        logger.info(f"Multi-wallet manager ready with {len(self._wallets)} active wallets")

    async def close(self) -> None:
        """Close all wallet adapters."""
        for name, wallet in self._wallets.items():
            try:
                await wallet.adapter.close()
                logger.info(f"Closed wallet '{name}'")
            except Exception as e:
                logger.error(f"Error closing wallet '{name}': {e}")

    async def sync_all_wallets(self) -> None:
        """Sync balances and positions for all wallets."""
        tasks = [self._sync_wallet(name) for name in self._wallets]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _sync_wallet(self, wallet_name: str) -> None:
        """Sync a single wallet's state."""
        wallet = self._wallets.get(wallet_name)
        if not wallet:
            return

        try:
            # Get balance
            wallet.balance = await wallet.adapter.rest.get_account_balance()

            # Get positions
            wallet.positions = await wallet.adapter.rest.get_positions()

            # Calculate exposure
            if wallet.balance and wallet.balance.total_balance > 0:
                position_value = sum(
                    p.quantity * p.entry_price for p in wallet.positions
                )
                wallet.current_exposure = position_value / wallet.balance.total_balance
            else:
                wallet.current_exposure = 0.0

        except Exception as e:
            wallet.error_count += 1
            wallet.last_error = str(e)
            logger.error(f"Failed to sync wallet '{wallet_name}': {e}")

    def get_aggregate_portfolio(self) -> AggregatePortfolio:
        """Get aggregate portfolio view across all wallets."""
        portfolio = AggregatePortfolio()

        for name, wallet in self._wallets.items():
            if wallet.balance:
                portfolio.total_balance += wallet.balance.total_balance
                portfolio.available_balance += wallet.balance.available_balance
                portfolio.wallet_balances[name] = wallet.balance.total_balance

            for pos in wallet.positions:
                portfolio.total_unrealized_pnl += pos.unrealized_pnl or 0

                if pos.symbol not in portfolio.positions_by_symbol:
                    portfolio.positions_by_symbol[pos.symbol] = []
                portfolio.positions_by_symbol[pos.symbol].append((name, pos))

        if portfolio.total_balance > 0:
            portfolio.total_exposure = (
                portfolio.total_balance - portfolio.available_balance
            ) / portfolio.total_balance

        return portfolio

    async def select_wallet(
        self,
        strategy: AllocationStrategy | None = None,
        symbol: str | None = None,
        side: OrderSide | None = None,
        size_usd: float | None = None,
        wallet_name: str | None = None,
        tags: list[str] | None = None,
    ) -> WalletState | None:
        """
        Select a wallet for a trade based on strategy.

        Args:
            strategy: Allocation strategy (uses default if None)
            symbol: Trading symbol (for checking existing positions)
            side: Order side
            size_usd: Order size in USD (for checking capacity)
            wallet_name: Specific wallet name (for SPECIFIC strategy)
            tags: Filter wallets by tags

        Returns:
            Selected WalletState or None if no suitable wallet
        """
        strategy = strategy or self.default_strategy

        # Filter eligible wallets
        eligible = self._get_eligible_wallets(symbol, side, size_usd, tags)

        if not eligible:
            logger.warning("No eligible wallets for trade")
            return None

        # Specific wallet requested
        if strategy == AllocationStrategy.SPECIFIC:
            if wallet_name and wallet_name in eligible:
                return eligible[wallet_name]
            logger.warning(f"Requested wallet '{wallet_name}' not eligible")
            return None

        # Round robin
        if strategy == AllocationStrategy.ROUND_ROBIN:
            wallet_names = list(eligible.keys())
            self._round_robin_index = (self._round_robin_index + 1) % len(wallet_names)
            return eligible[wallet_names[self._round_robin_index]]

        # Lowest exposure
        if strategy == AllocationStrategy.LOWEST_EXPOSURE:
            return min(eligible.values(), key=lambda w: w.current_exposure)

        # Highest balance
        if strategy == AllocationStrategy.HIGHEST_BALANCE:
            return max(
                eligible.values(),
                key=lambda w: w.balance.available_balance if w.balance else 0,
            )

        # Weighted (probabilistic based on weight)
        if strategy == AllocationStrategy.WEIGHTED:
            import random
            total_weight = sum(w.config.weight for w in eligible.values())
            if total_weight == 0:
                return list(eligible.values())[0]

            r = random.uniform(0, total_weight)
            cumulative = 0
            for wallet in eligible.values():
                cumulative += wallet.config.weight
                if r <= cumulative:
                    return wallet

        return list(eligible.values())[0]

    def _get_eligible_wallets(
        self,
        symbol: str | None = None,
        side: OrderSide | None = None,
        size_usd: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, WalletState]:
        """Get wallets eligible for a trade."""
        eligible = {}

        for name, wallet in self._wallets.items():
            # Check if enabled
            if not wallet.config.enabled:
                continue

            # Check tags
            if tags:
                if not any(tag in wallet.config.tags for tag in tags):
                    continue

            # Check max positions
            if len(wallet.positions) >= wallet.config.max_positions:
                continue

            # Check exposure limit
            if wallet.current_exposure >= wallet.config.max_exposure:
                continue

            # Check available balance
            if size_usd and wallet.balance:
                if wallet.balance.available_balance < size_usd * 0.5:  # Need at least 50% margin
                    continue

            # Check for errors (skip wallets with recent errors)
            if wallet.error_count > 5:
                continue

            eligible[name] = wallet

        return eligible

    async def place_order(
        self,
        order: UnifiedOrder,
        strategy: AllocationStrategy | None = None,
        wallet_name: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[str, UnifiedOrderResult] | None:
        """
        Place an order using selected wallet.

        Args:
            order: The order to place
            strategy: Wallet selection strategy
            wallet_name: Specific wallet (for SPECIFIC strategy)
            tags: Filter wallets by tags

        Returns:
            Tuple of (wallet_name, order_result) or None if failed
        """
        async with self._lock:
            # Calculate order size in USD for wallet selection
            size_usd = order.quantity * (order.price or 0)

            # Select wallet
            wallet = await self.select_wallet(
                strategy=strategy,
                symbol=order.symbol,
                side=order.side,
                size_usd=size_usd,
                wallet_name=wallet_name,
                tags=tags,
            )

            if not wallet:
                logger.error(f"No eligible wallet for order: {order.symbol} {order.side}")
                return None

            try:
                result = await wallet.adapter.rest.place_order(order)
                wallet.trade_count += 1
                wallet.last_trade_time = datetime.now(UTC)

                logger.info(
                    f"Order placed via '{wallet.config.name}': "
                    f"{order.symbol} {order.side.value} {order.quantity} @ {order.price}"
                )

                # Sync wallet state after trade
                await self._sync_wallet(wallet.config.name)

                return wallet.config.name, result

            except Exception as e:
                wallet.error_count += 1
                wallet.last_error = str(e)
                logger.error(f"Order failed on wallet '{wallet.config.name}': {e}")

                # Try another wallet if available
                if strategy != AllocationStrategy.SPECIFIC:
                    # Disable this wallet temporarily and retry
                    wallet.config.enabled = False
                    try:
                        return await self.place_order(order, strategy, wallet_name, tags)
                    finally:
                        wallet.config.enabled = True

                return None

    async def close_position(
        self,
        symbol: str,
        wallet_name: str | None = None,
    ) -> list[tuple[str, UnifiedOrderResult]]:
        """
        Close positions for a symbol across wallets.

        Args:
            symbol: Symbol to close
            wallet_name: Specific wallet (closes on all if None)

        Returns:
            List of (wallet_name, order_result) tuples
        """
        results = []

        wallets_to_check = (
            {wallet_name: self._wallets[wallet_name]}
            if wallet_name and wallet_name in self._wallets
            else self._wallets
        )

        for name, wallet in wallets_to_check.items():
            for pos in wallet.positions:
                if pos.symbol != symbol or pos.quantity == 0:
                    continue

                # Create close order
                from src.exchange.models import OrderType, TimeInForce
                close_order = UnifiedOrder(
                    symbol=symbol,
                    side=OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=abs(pos.quantity),
                    price=pos.mark_price,  # Use mark price for limit
                    time_in_force=TimeInForce.IOC,
                    reduce_only=True,
                )

                try:
                    result = await wallet.adapter.rest.place_order(close_order)
                    results.append((name, result))
                    logger.info(f"Closed {pos.quantity} {symbol} on wallet '{name}'")
                except Exception as e:
                    logger.error(f"Failed to close {symbol} on wallet '{name}': {e}")

        return results

    async def get_position(
        self,
        symbol: str,
        wallet_name: str | None = None,
    ) -> list[tuple[str, UnifiedPosition]]:
        """Get positions for a symbol across wallets."""
        positions = []

        wallets_to_check = (
            {wallet_name: self._wallets[wallet_name]}
            if wallet_name and wallet_name in self._wallets
            else self._wallets
        )

        for name, wallet in wallets_to_check.items():
            for pos in wallet.positions:
                if pos.symbol == symbol and pos.quantity != 0:
                    positions.append((name, pos))

        return positions

    def get_wallet(self, name: str) -> WalletState | None:
        """Get a specific wallet by name."""
        return self._wallets.get(name)

    def list_wallets(self) -> list[str]:
        """List all wallet names."""
        return list(self._wallets.keys())

    def get_status(self) -> dict[str, Any]:
        """Get status of all wallets."""
        return {
            name: {
                "address": wallet.config.address[:10] + "..." + wallet.config.address[-6:],
                "enabled": wallet.config.enabled,
                "balance": wallet.balance.total_balance if wallet.balance else 0,
                "exposure": f"{wallet.current_exposure:.1%}",
                "positions": len(wallet.positions),
                "trades": wallet.trade_count,
                "errors": wallet.error_count,
                "last_error": wallet.last_error[:50] if wallet.last_error else None,
            }
            for name, wallet in self._wallets.items()
        }


def load_wallet_configs_from_env() -> list[WalletConfig]:
    """
    Load wallet configurations from environment variables.

    Format:
        HYPERLIQUID_WALLETS=wallet1,wallet2,wallet3
        HYPERLIQUID_WALLET_wallet1_KEY=0x...
        HYPERLIQUID_WALLET_wallet1_WEIGHT=1.0
        HYPERLIQUID_WALLET_wallet1_MAX_EXPOSURE=0.8
        HYPERLIQUID_WALLET_wallet1_TAGS=main,long-only

    Or simple format (single wallet):
        EXCHANGE_WALLET_PRIVATE_KEY=0x...
    """
    import os

    configs = []

    # Check for multi-wallet config
    wallet_names = os.getenv("HYPERLIQUID_WALLETS", "").split(",")
    wallet_names = [w.strip() for w in wallet_names if w.strip()]

    if wallet_names:
        for name in wallet_names:
            prefix = f"HYPERLIQUID_WALLET_{name}_"
            private_key = os.getenv(f"{prefix}KEY", "")

            if not private_key:
                logger.warning(f"No private key for wallet '{name}', skipping")
                continue

            configs.append(WalletConfig(
                name=name,
                private_key=private_key,
                address=os.getenv(f"{prefix}ADDRESS", ""),
                weight=float(os.getenv(f"{prefix}WEIGHT", "1.0")),
                max_exposure=float(os.getenv(f"{prefix}MAX_EXPOSURE", "1.0")),
                max_positions=int(os.getenv(f"{prefix}MAX_POSITIONS", "10")),
                enabled=os.getenv(f"{prefix}ENABLED", "true").lower() == "true",
                tags=os.getenv(f"{prefix}TAGS", "").split(",") if os.getenv(f"{prefix}TAGS") else [],
            ))

    # Fallback to single wallet
    if not configs:
        private_key = os.getenv("EXCHANGE_WALLET_PRIVATE_KEY", "")
        if private_key:
            configs.append(WalletConfig(
                name="default",
                private_key=private_key,
                address=os.getenv("EXCHANGE_WALLET_ADDRESS", ""),
            ))

    return configs
