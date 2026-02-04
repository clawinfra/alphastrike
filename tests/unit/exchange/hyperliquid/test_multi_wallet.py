"""
Tests for Hyperliquid multi-wallet manager.

Coverage targets: wallet configuration, allocation strategies, portfolio aggregation.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

from src.exchange.adapters.hyperliquid.multi_wallet import (
    MultiWalletManager,
    WalletConfig,
    WalletState,
    AggregatePortfolio,
    AllocationStrategy,
    load_wallet_configs_from_env,
)
from src.exchange.models import (
    OrderSide,
    OrderType,
    PositionSide,
    UnifiedAccountBalance,
    UnifiedOrder,
    UnifiedOrderResult,
    UnifiedPosition,
    OrderStatus,
)


class TestWalletConfig:
    """Tests for WalletConfig dataclass."""

    def test_basic_config(self):
        """Test basic wallet configuration."""
        config = WalletConfig(
            name="test-wallet",
            private_key="0x" + "a" * 64,
        )

        assert config.name == "test-wallet"
        assert config.address == ""
        assert config.weight == 1.0
        assert config.max_exposure == 1.0
        assert config.max_positions == 10
        assert config.enabled is True
        assert config.tags == []

    def test_full_config(self):
        """Test full wallet configuration."""
        config = WalletConfig(
            name="hedge-wallet",
            private_key="0x" + "b" * 64,
            address="0x1234",
            weight=0.5,
            max_exposure=0.8,
            max_positions=5,
            enabled=False,
            tags=["hedge", "btc-only"],
        )

        assert config.weight == 0.5
        assert config.max_exposure == 0.8
        assert config.max_positions == 5
        assert config.enabled is False
        assert "hedge" in config.tags


class TestWalletState:
    """Tests for WalletState dataclass."""

    def test_basic_state(self):
        """Test basic wallet state."""
        config = WalletConfig(name="test", private_key="0x" + "a" * 64)
        adapter = MagicMock()

        state = WalletState(config=config, adapter=adapter)

        assert state.config == config
        assert state.adapter == adapter
        assert state.balance is None
        assert state.positions == []
        assert state.current_exposure == 0.0
        assert state.trade_count == 0
        assert state.error_count == 0


class TestAggregatePortfolio:
    """Tests for AggregatePortfolio dataclass."""

    def test_empty_portfolio(self):
        """Test empty portfolio."""
        portfolio = AggregatePortfolio()

        assert portfolio.total_balance == 0.0
        assert portfolio.available_balance == 0.0
        assert portfolio.total_exposure == 0.0
        assert portfolio.wallet_balances == {}

    def test_portfolio_str(self):
        """Test portfolio string representation."""
        portfolio = AggregatePortfolio(
            total_balance=10000.0,
            available_balance=8000.0,
            total_exposure=0.2,
            total_unrealized_pnl=500.0,
            wallet_balances={"wallet1": 5000.0, "wallet2": 5000.0},
        )

        output = str(portfolio)

        assert "AGGREGATE PORTFOLIO" in output
        assert "$10,000.00" in output
        assert "wallet1" in output

    def test_portfolio_str_with_positions(self):
        """Test portfolio string with positions."""
        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
            margin=500.0,
            timestamp=datetime.now(UTC),
        )
        portfolio = AggregatePortfolio(
            total_balance=10000.0,
            positions_by_symbol={"BTCUSDT": [("wallet1", position)]},
        )

        output = str(portfolio)

        assert "BTCUSDT" in output
        assert "wallet1" in output


class TestAllocationStrategy:
    """Tests for AllocationStrategy enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert AllocationStrategy.ROUND_ROBIN.value == "round_robin"
        assert AllocationStrategy.LOWEST_EXPOSURE.value == "lowest_exposure"
        assert AllocationStrategy.WEIGHTED.value == "weighted"
        assert AllocationStrategy.HIGHEST_BALANCE.value == "highest_balance"
        assert AllocationStrategy.SPECIFIC.value == "specific"


class TestMultiWalletManagerInit:
    """Tests for MultiWalletManager initialization."""

    def test_init(self):
        """Test manager initialization."""
        configs = [
            WalletConfig(name="wallet1", private_key="0x" + "a" * 64),
            WalletConfig(name="wallet2", private_key="0x" + "b" * 64),
        ]

        manager = MultiWalletManager(
            wallet_configs=configs,
            testnet=True,
            default_strategy=AllocationStrategy.ROUND_ROBIN,
        )

        assert len(manager.wallet_configs) == 2
        assert manager.testnet is True
        assert manager.default_strategy == AllocationStrategy.ROUND_ROBIN
        assert manager._initialized is False


class TestMultiWalletManagerLifecycle:
    """Tests for manager lifecycle methods."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test manager initialization."""
        configs = [
            WalletConfig(name="wallet1", private_key="0x" + "a" * 64),
        ]

        manager = MultiWalletManager(wallet_configs=configs, testnet=True)

        with patch("src.exchange.adapters.hyperliquid.multi_wallet.HyperliquidAdapter") as MockAdapter:
            mock_adapter = AsyncMock()
            mock_adapter.rest.get_account_balance = AsyncMock(
                return_value=UnifiedAccountBalance(
                    total_balance=1000.0,
                    available_balance=900.0,
                    margin_balance=100.0,
                    unrealized_pnl=0.0,
                    currency="USDC",
                    timestamp=datetime.now(UTC),
                )
            )
            mock_adapter.rest.get_positions = AsyncMock(return_value=[])
            MockAdapter.return_value = mock_adapter

            await manager.initialize()

            assert manager._initialized is True
            assert len(manager._wallets) == 1

    @pytest.mark.asyncio
    async def test_initialize_disabled_wallet(self):
        """Test initialization skips disabled wallets."""
        configs = [
            WalletConfig(name="disabled", private_key="0x" + "a" * 64, enabled=False),
        ]

        manager = MultiWalletManager(wallet_configs=configs, testnet=True)

        with patch("src.exchange.adapters.hyperliquid.multi_wallet.HyperliquidAdapter"):
            with pytest.raises(RuntimeError, match="No wallets initialized"):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test manager close."""
        configs = [WalletConfig(name="wallet1", private_key="0x" + "a" * 64)]
        manager = MultiWalletManager(wallet_configs=configs, testnet=True)

        # Set up mock wallet
        mock_adapter = AsyncMock()
        manager._wallets["wallet1"] = WalletState(
            config=configs[0],
            adapter=mock_adapter,
        )

        await manager.close()

        mock_adapter.close.assert_called_once()


class TestMultiWalletManagerSync:
    """Tests for wallet sync methods."""

    @pytest.mark.asyncio
    async def test_sync_wallet(self):
        """Test syncing a single wallet."""
        config = WalletConfig(name="wallet1", private_key="0x" + "a" * 64)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)

        mock_adapter = AsyncMock()
        mock_adapter.rest.get_account_balance = AsyncMock(
            return_value=UnifiedAccountBalance(
                total_balance=10000.0,
                available_balance=8000.0,
                margin_balance=2000.0,
                unrealized_pnl=500.0,
                currency="USDC",
                timestamp=datetime.now(UTC),
            )
        )
        mock_adapter.rest.get_positions = AsyncMock(return_value=[
            UnifiedPosition(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=0.1,
                entry_price=50000.0,
                mark_price=51000.0,
                unrealized_pnl=100.0,
                leverage=10,
                margin=500.0,
                timestamp=datetime.now(UTC),
            )
        ])

        manager._wallets["wallet1"] = WalletState(config=config, adapter=mock_adapter)

        await manager._sync_wallet("wallet1")

        wallet = manager._wallets["wallet1"]
        assert wallet.balance is not None
        assert wallet.balance.total_balance == 10000.0
        assert len(wallet.positions) == 1
        assert wallet.current_exposure > 0

    @pytest.mark.asyncio
    async def test_sync_wallet_error(self):
        """Test sync wallet handles errors."""
        config = WalletConfig(name="wallet1", private_key="0x" + "a" * 64)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)

        mock_adapter = AsyncMock()
        mock_adapter.rest.get_account_balance = AsyncMock(side_effect=Exception("API Error"))

        manager._wallets["wallet1"] = WalletState(config=config, adapter=mock_adapter)

        await manager._sync_wallet("wallet1")

        wallet = manager._wallets["wallet1"]
        assert wallet.error_count == 1
        assert "API Error" in wallet.last_error

    @pytest.mark.asyncio
    async def test_sync_wallet_not_found(self):
        """Test sync wallet with unknown wallet."""
        manager = MultiWalletManager(wallet_configs=[], testnet=True)

        await manager._sync_wallet("nonexistent")  # Should not raise


class TestMultiWalletManagerAggregation:
    """Tests for portfolio aggregation."""

    def test_get_aggregate_portfolio(self):
        """Test aggregate portfolio calculation."""
        configs = [
            WalletConfig(name="wallet1", private_key="0x" + "a" * 64),
            WalletConfig(name="wallet2", private_key="0x" + "b" * 64),
        ]
        manager = MultiWalletManager(wallet_configs=configs, testnet=True)

        # Set up mock wallets
        manager._wallets["wallet1"] = WalletState(
            config=configs[0],
            adapter=MagicMock(),
            balance=UnifiedAccountBalance(
                total_balance=5000.0,
                available_balance=4000.0,
                margin_balance=1000.0,
                unrealized_pnl=100.0,
                currency="USDC",
                timestamp=datetime.now(UTC),
            ),
            positions=[
                UnifiedPosition(
                    symbol="BTCUSDT",
                    side=PositionSide.LONG,
                    quantity=0.1,
                    entry_price=50000.0,
                    mark_price=51000.0,
                    unrealized_pnl=100.0,
                    leverage=10,
                    margin=500.0,
                    timestamp=datetime.now(UTC),
                )
            ],
        )
        manager._wallets["wallet2"] = WalletState(
            config=configs[1],
            adapter=MagicMock(),
            balance=UnifiedAccountBalance(
                total_balance=5000.0,
                available_balance=5000.0,
                margin_balance=0.0,
                unrealized_pnl=0.0,
                currency="USDC",
                timestamp=datetime.now(UTC),
            ),
        )

        portfolio = manager.get_aggregate_portfolio()

        assert portfolio.total_balance == 10000.0
        assert portfolio.available_balance == 9000.0
        assert portfolio.total_unrealized_pnl == 100.0
        assert len(portfolio.wallet_balances) == 2


class TestMultiWalletManagerSelection:
    """Tests for wallet selection."""

    @pytest.fixture
    def manager_with_wallets(self):
        """Create manager with test wallets."""
        configs = [
            WalletConfig(name="wallet1", private_key="0x" + "a" * 64, weight=2.0),
            WalletConfig(name="wallet2", private_key="0x" + "b" * 64, weight=1.0),
        ]
        manager = MultiWalletManager(
            wallet_configs=configs,
            testnet=True,
            default_strategy=AllocationStrategy.LOWEST_EXPOSURE,
        )

        for i, config in enumerate(configs):
            manager._wallets[config.name] = WalletState(
                config=config,
                adapter=MagicMock(),
                balance=UnifiedAccountBalance(
                    total_balance=5000.0,
                    available_balance=4000.0 - i * 1000,  # wallet1 has more available
                    margin_balance=1000.0 + i * 1000,
                    unrealized_pnl=0.0,
                    currency="USDC",
                    timestamp=datetime.now(UTC),
                ),
                current_exposure=0.1 * i,  # wallet1 has lower exposure
            )

        return manager

    @pytest.mark.asyncio
    async def test_select_wallet_lowest_exposure(self, manager_with_wallets):
        """Test wallet selection by lowest exposure."""
        wallet = await manager_with_wallets.select_wallet(
            strategy=AllocationStrategy.LOWEST_EXPOSURE
        )

        assert wallet is not None
        assert wallet.config.name == "wallet1"  # Lower exposure

    @pytest.mark.asyncio
    async def test_select_wallet_highest_balance(self, manager_with_wallets):
        """Test wallet selection by highest balance."""
        wallet = await manager_with_wallets.select_wallet(
            strategy=AllocationStrategy.HIGHEST_BALANCE
        )

        assert wallet is not None
        assert wallet.config.name == "wallet1"  # Higher available balance

    @pytest.mark.asyncio
    async def test_select_wallet_round_robin(self, manager_with_wallets):
        """Test wallet selection by round robin."""
        first = await manager_with_wallets.select_wallet(
            strategy=AllocationStrategy.ROUND_ROBIN
        )
        second = await manager_with_wallets.select_wallet(
            strategy=AllocationStrategy.ROUND_ROBIN
        )

        assert first.config.name != second.config.name

    @pytest.mark.asyncio
    async def test_select_wallet_specific(self, manager_with_wallets):
        """Test specific wallet selection."""
        wallet = await manager_with_wallets.select_wallet(
            strategy=AllocationStrategy.SPECIFIC,
            wallet_name="wallet2",
        )

        assert wallet is not None
        assert wallet.config.name == "wallet2"

    @pytest.mark.asyncio
    async def test_select_wallet_specific_not_found(self, manager_with_wallets):
        """Test specific wallet selection when not found."""
        wallet = await manager_with_wallets.select_wallet(
            strategy=AllocationStrategy.SPECIFIC,
            wallet_name="nonexistent",
        )

        assert wallet is None

    @pytest.mark.asyncio
    async def test_select_wallet_weighted(self, manager_with_wallets):
        """Test weighted wallet selection."""
        # Run multiple times to test randomness
        selections = []
        for _ in range(100):
            wallet = await manager_with_wallets.select_wallet(
                strategy=AllocationStrategy.WEIGHTED
            )
            selections.append(wallet.config.name)

        # wallet1 has weight 2.0, wallet2 has weight 1.0
        # So wallet1 should be selected more often
        wallet1_count = selections.count("wallet1")
        assert wallet1_count > 30  # Should be selected at least 30% of time

    @pytest.mark.asyncio
    async def test_select_wallet_no_eligible(self):
        """Test wallet selection with no eligible wallets."""
        configs = [
            WalletConfig(name="disabled", private_key="0x" + "a" * 64, enabled=False),
        ]
        manager = MultiWalletManager(wallet_configs=configs, testnet=True)
        manager._wallets["disabled"] = WalletState(
            config=configs[0],
            adapter=MagicMock(),
        )

        wallet = await manager.select_wallet()

        assert wallet is None


class TestMultiWalletManagerEligibility:
    """Tests for wallet eligibility filtering."""

    def test_get_eligible_wallets_disabled(self):
        """Test disabled wallets are filtered out."""
        config = WalletConfig(name="disabled", private_key="0x" + "a" * 64, enabled=False)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)
        manager._wallets["disabled"] = WalletState(config=config, adapter=MagicMock())

        eligible = manager._get_eligible_wallets()

        assert len(eligible) == 0

    def test_get_eligible_wallets_max_positions(self):
        """Test wallets at max positions are filtered out."""
        config = WalletConfig(name="full", private_key="0x" + "a" * 64, max_positions=1)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)
        manager._wallets["full"] = WalletState(
            config=config,
            adapter=MagicMock(),
            positions=[MagicMock()],  # Already has one position
        )

        eligible = manager._get_eligible_wallets()

        assert len(eligible) == 0

    def test_get_eligible_wallets_max_exposure(self):
        """Test wallets at max exposure are filtered out."""
        config = WalletConfig(name="exposed", private_key="0x" + "a" * 64, max_exposure=0.5)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)
        manager._wallets["exposed"] = WalletState(
            config=config,
            adapter=MagicMock(),
            current_exposure=0.6,  # Above max
        )

        eligible = manager._get_eligible_wallets()

        assert len(eligible) == 0

    def test_get_eligible_wallets_tags(self):
        """Test wallet filtering by tags."""
        configs = [
            WalletConfig(name="btc", private_key="0x" + "a" * 64, tags=["btc"]),
            WalletConfig(name="eth", private_key="0x" + "b" * 64, tags=["eth"]),
        ]
        manager = MultiWalletManager(wallet_configs=configs, testnet=True)
        for config in configs:
            manager._wallets[config.name] = WalletState(config=config, adapter=MagicMock())

        eligible = manager._get_eligible_wallets(tags=["btc"])

        assert len(eligible) == 1
        assert "btc" in eligible


class TestMultiWalletManagerOrders:
    """Tests for order placement."""

    @pytest.mark.asyncio
    async def test_place_order(self):
        """Test placing an order."""
        config = WalletConfig(name="wallet1", private_key="0x" + "a" * 64)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)

        mock_adapter = AsyncMock()
        mock_adapter.rest.place_order = AsyncMock(
            return_value=UnifiedOrderResult(
                order_id="123",
                client_order_id=None,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                price=50000.0,
                status=OrderStatus.NEW,
                filled_quantity=0.0,
                average_price=None,
                commission=0.0,
                commission_asset="USDC",
                timestamp=datetime.now(UTC),
            )
        )
        mock_adapter.rest.get_account_balance = AsyncMock(
            return_value=UnifiedAccountBalance(
                total_balance=10000.0,
                available_balance=9000.0,
                margin_balance=1000.0,
                unrealized_pnl=0.0,
                currency="USDC",
                timestamp=datetime.now(UTC),
            )
        )
        mock_adapter.rest.get_positions = AsyncMock(return_value=[])

        manager._wallets["wallet1"] = WalletState(
            config=config,
            adapter=mock_adapter,
            balance=UnifiedAccountBalance(
                total_balance=10000.0,
                available_balance=9000.0,
                margin_balance=1000.0,
                unrealized_pnl=0.0,
                currency="USDC",
                timestamp=datetime.now(UTC),
            ),
        )

        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
        )

        result = await manager.place_order(order)

        assert result is not None
        assert result[0] == "wallet1"
        assert result[1].order_id == "123"

    @pytest.mark.asyncio
    async def test_place_order_no_wallet(self):
        """Test placing an order with no eligible wallet."""
        manager = MultiWalletManager(wallet_configs=[], testnet=True)

        order = UnifiedOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
        )

        result = await manager.place_order(order)

        assert result is None


class TestMultiWalletManagerPositions:
    """Tests for position management."""

    @pytest.mark.asyncio
    async def test_get_position(self):
        """Test getting positions by symbol."""
        config = WalletConfig(name="wallet1", private_key="0x" + "a" * 64)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)

        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
            margin=500.0,
            timestamp=datetime.now(UTC),
        )

        manager._wallets["wallet1"] = WalletState(
            config=config,
            adapter=MagicMock(),
            positions=[position],
        )

        positions = await manager.get_position("BTCUSDT")

        assert len(positions) == 1
        assert positions[0][0] == "wallet1"
        assert positions[0][1].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_position_specific_wallet(self):
        """Test getting positions for specific wallet."""
        configs = [
            WalletConfig(name="wallet1", private_key="0x" + "a" * 64),
            WalletConfig(name="wallet2", private_key="0x" + "b" * 64),
        ]
        manager = MultiWalletManager(wallet_configs=configs, testnet=True)

        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
            margin=500.0,
            timestamp=datetime.now(UTC),
        )

        manager._wallets["wallet1"] = WalletState(
            config=configs[0],
            adapter=MagicMock(),
            positions=[position],
        )
        manager._wallets["wallet2"] = WalletState(
            config=configs[1],
            adapter=MagicMock(),
            positions=[],
        )

        positions = await manager.get_position("BTCUSDT", wallet_name="wallet1")

        assert len(positions) == 1

    @pytest.mark.asyncio
    async def test_close_position(self):
        """Test closing a position."""
        config = WalletConfig(name="wallet1", private_key="0x" + "a" * 64)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)

        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
            margin=500.0,
            timestamp=datetime.now(UTC),
        )

        mock_adapter = AsyncMock()
        mock_adapter.rest.place_order = AsyncMock(
            return_value=UnifiedOrderResult(
                order_id="close123",
                client_order_id=None,
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                price=51000.0,
                status=OrderStatus.FILLED,
                filled_quantity=0.1,
                average_price=51000.0,
                commission=0.0,
                commission_asset="USDC",
                timestamp=datetime.now(UTC),
            )
        )

        manager._wallets["wallet1"] = WalletState(
            config=config,
            adapter=mock_adapter,
            positions=[position],
        )

        results = await manager.close_position("BTCUSDT")

        assert len(results) == 1
        assert results[0][0] == "wallet1"

    @pytest.mark.asyncio
    async def test_close_position_error(self):
        """Test closing a position with error."""
        config = WalletConfig(name="wallet1", private_key="0x" + "a" * 64)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)

        position = UnifiedPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100.0,
            leverage=10,
            margin=500.0,
            timestamp=datetime.now(UTC),
        )

        mock_adapter = AsyncMock()
        mock_adapter.rest.place_order = AsyncMock(side_effect=Exception("Close failed"))

        manager._wallets["wallet1"] = WalletState(
            config=config,
            adapter=mock_adapter,
            positions=[position],
        )

        results = await manager.close_position("BTCUSDT")

        # Should return empty list on error
        assert len(results) == 0


class TestMultiWalletManagerUtilities:
    """Tests for utility methods."""

    def test_get_wallet(self):
        """Test getting a specific wallet."""
        config = WalletConfig(name="wallet1", private_key="0x" + "a" * 64)
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)
        manager._wallets["wallet1"] = WalletState(config=config, adapter=MagicMock())

        wallet = manager.get_wallet("wallet1")

        assert wallet is not None
        assert wallet.config.name == "wallet1"

    def test_get_wallet_not_found(self):
        """Test getting a non-existent wallet."""
        manager = MultiWalletManager(wallet_configs=[], testnet=True)

        wallet = manager.get_wallet("nonexistent")

        assert wallet is None

    def test_list_wallets(self):
        """Test listing all wallets."""
        configs = [
            WalletConfig(name="wallet1", private_key="0x" + "a" * 64),
            WalletConfig(name="wallet2", private_key="0x" + "b" * 64),
        ]
        manager = MultiWalletManager(wallet_configs=configs, testnet=True)
        for config in configs:
            manager._wallets[config.name] = WalletState(config=config, adapter=MagicMock())

        names = manager.list_wallets()

        assert len(names) == 2
        assert "wallet1" in names
        assert "wallet2" in names

    def test_get_status(self):
        """Test getting wallet status."""
        config = WalletConfig(name="wallet1", private_key="0x" + "a" * 64, address="0x1234567890abcdef")
        manager = MultiWalletManager(wallet_configs=[config], testnet=True)
        manager._wallets["wallet1"] = WalletState(
            config=config,
            adapter=MagicMock(),
            balance=UnifiedAccountBalance(
                total_balance=1000.0,
                available_balance=900.0,
                margin_balance=100.0,
                unrealized_pnl=0.0,
                currency="USDC",
                timestamp=datetime.now(UTC),
            ),
            current_exposure=0.1,
            trade_count=5,
            error_count=1,
            last_error="Test error",
        )

        status = manager.get_status()

        assert "wallet1" in status
        assert status["wallet1"]["balance"] == 1000.0
        assert status["wallet1"]["trades"] == 5
        assert status["wallet1"]["errors"] == 1


class TestLoadWalletConfigsFromEnv:
    """Tests for loading wallet configs from environment."""

    def test_load_multi_wallet(self):
        """Test loading multi-wallet configuration."""
        env = {
            "HYPERLIQUID_WALLETS": "wallet1,wallet2",
            "HYPERLIQUID_WALLET_wallet1_KEY": "0x" + "a" * 64,
            "HYPERLIQUID_WALLET_wallet1_WEIGHT": "2.0",
            "HYPERLIQUID_WALLET_wallet1_TAGS": "main,long",
            "HYPERLIQUID_WALLET_wallet2_KEY": "0x" + "b" * 64,
        }

        with patch.dict(os.environ, env, clear=True):
            configs = load_wallet_configs_from_env()

        assert len(configs) == 2
        assert configs[0].name == "wallet1"
        assert configs[0].weight == 2.0
        assert "main" in configs[0].tags

    def test_load_single_wallet_fallback(self):
        """Test loading single wallet fallback."""
        env = {
            "EXCHANGE_WALLET_PRIVATE_KEY": "0x" + "a" * 64,
            "EXCHANGE_WALLET_ADDRESS": "0x1234",
        }

        with patch.dict(os.environ, env, clear=True):
            configs = load_wallet_configs_from_env()

        assert len(configs) == 1
        assert configs[0].name == "default"

    def test_load_no_wallets(self):
        """Test loading with no wallet config."""
        with patch.dict(os.environ, {}, clear=True):
            configs = load_wallet_configs_from_env()

        assert len(configs) == 0

    def test_load_wallet_missing_key(self):
        """Test loading wallet with missing key."""
        env = {
            "HYPERLIQUID_WALLETS": "wallet1,wallet2",
            "HYPERLIQUID_WALLET_wallet1_KEY": "0x" + "a" * 64,
            # wallet2 has no key
        }

        with patch.dict(os.environ, env, clear=True):
            configs = load_wallet_configs_from_env()

        assert len(configs) == 1
        assert configs[0].name == "wallet1"
