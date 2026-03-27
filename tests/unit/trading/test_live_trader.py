"""
Tests for AlphaStrike V3 Live Trader.

Covers:
- EIP-712 signing correctness
- Safety rail enforcement (max position, daily loss, coin allowlist, etc.)
- Order building and submission flow
- Position tracking and state management
- Stop-loss checking
- Dry-run mode
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# We need to import from scripts/ which isn't a package, so we manipulate sys.path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from live_trader import (
    ALLOWED_COINS,
    COIN_TO_ASSET,
    DAILY_LOSS_LIMIT_USD,
    MAX_POSITION_USD,
    STOP_LOSS_PCT,
    HLLiveTrader,
    _action_hash,
    _float_to_hl_string,
    _round_price,
    _round_size,
    sign_action,
)

# Test private key (throwaway, NOT a real key)
TEST_PRIVATE_KEY = "0x" + "ab" * 32  # 64 hex chars = 32 bytes


# ── Signing Tests ─────────────────────────────────────────────────────────────


class TestSigning:
    """Tests for EIP-712 signing functions."""

    def test_action_hash_deterministic(self):
        """Action hash should be deterministic for same inputs."""
        action = {"type": "order", "orders": [{"a": 0}], "grouping": "na"}
        nonce = 1000000000000
        h1 = _action_hash(action, None, nonce)
        h2 = _action_hash(action, None, nonce)
        assert h1 == h2
        assert h1.startswith("0x")
        assert len(h1) == 66  # 0x + 64 hex chars

    def test_action_hash_different_nonce(self):
        """Different nonces should produce different hashes."""
        action = {"type": "order", "orders": [{"a": 0}], "grouping": "na"}
        h1 = _action_hash(action, None, 1000)
        h2 = _action_hash(action, None, 2000)
        assert h1 != h2

    def test_action_hash_with_vault_address(self):
        """Vault address should change the hash."""
        action = {"type": "order", "orders": [{"a": 0}], "grouping": "na"}
        nonce = 1000
        h1 = _action_hash(action, None, nonce)
        h2 = _action_hash(action, "0x1234567890abcdef", nonce)
        assert h1 != h2

    def test_sign_action_returns_signature(self):
        """sign_action should return r, s, v components."""
        action = {
            "type": "order",
            "orders": [
                {
                    "a": 0,
                    "b": True,
                    "p": "50000.0",
                    "s": "0.001",
                    "r": False,
                    "t": {"limit": {"tif": "Gtc"}},
                }
            ],
            "grouping": "na",
        }
        nonce = int(time.time() * 1000)
        sig = sign_action(action, nonce, TEST_PRIVATE_KEY)

        assert "r" in sig
        assert "s" in sig
        assert "v" in sig
        assert sig["r"].startswith("0x")
        assert sig["s"].startswith("0x")
        assert sig["v"] in (27, 28)

    def test_sign_action_no_0x_prefix(self):
        """sign_action should handle keys without 0x prefix."""
        action = {"type": "order", "orders": [{"a": 0}], "grouping": "na"}
        nonce = 1000
        key_no_prefix = "ab" * 32
        sig = sign_action(action, nonce, key_no_prefix)
        assert "r" in sig
        assert "s" in sig

    def test_sign_action_deterministic(self):
        """Same inputs should produce same signature."""
        action = {"type": "order", "orders": [{"a": 0}], "grouping": "na"}
        nonce = 123456789000
        sig1 = sign_action(action, nonce, TEST_PRIVATE_KEY)
        sig2 = sign_action(action, nonce, TEST_PRIVATE_KEY)
        assert sig1 == sig2


# ── Helper Function Tests ────────────────────────────────────────────────────


class TestHelpers:
    """Tests for utility/helper functions."""

    def test_float_to_hl_string_basic(self):
        assert _float_to_hl_string(50000.0, 1) == "50000"
        assert _float_to_hl_string(50000.5, 1) == "50000.5"
        assert _float_to_hl_string(0.001, 3) == "0.001"

    def test_round_price_btc(self):
        assert _round_price(50000.1234, "BTC") == "50000.1"

    def test_round_price_eth(self):
        assert _round_price(3000.1234, "ETH") == "3000.12"

    def test_round_price_sol(self):
        assert _round_price(150.12345, "SOL") == "150.123"

    def test_round_size_btc(self):
        assert _round_size(0.001234, "BTC") == "0.00123"

    def test_round_size_eth(self):
        assert _round_size(0.12345, "ETH") == "0.1235"

    def test_round_size_sol(self):
        assert _round_size(1.234, "SOL") == "1.23"


# ── Safety Rail Tests ─────────────────────────────────────────────────────────


class TestSafetyRails:
    """Tests that hardcoded safety rails are enforced."""

    @patch("live_trader._load_live_state")
    def _make_trader(self, mock_load, state: dict | None = None):
        mock_load.return_value = state or {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        return HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)

    def test_max_position_size_enforced(self):
        trader = self._make_trader()
        with pytest.raises(ValueError, match="exceeds max"):
            trader._check_safety("BTC", 51.0)

    def test_max_position_size_at_limit(self):
        """$50 exactly should be allowed."""
        trader = self._make_trader()
        trader._check_safety("BTC", 50.0)  # Should not raise

    def test_disallowed_coin(self):
        trader = self._make_trader()
        with pytest.raises(ValueError, match="not in allowed list"):
            trader._check_safety("DOGE", 10.0)

    def test_allowed_coins(self):
        """BTC, ETH, SOL should all pass."""
        trader = self._make_trader()
        for coin in ["BTC", "ETH", "SOL"]:
            trader._check_safety(coin, 10.0)  # Should not raise

    def test_daily_loss_limit_halts_trading(self):
        state = {
            "daily_pnl": -20.01,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 5,
            "halted": True,
        }
        trader = self._make_trader(state=state)
        with pytest.raises(ValueError, match="HALTED"):
            trader._check_safety("BTC", 10.0)

    def test_max_one_position_per_coin(self):
        state = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {
                "BTC": {
                    "side": "LONG",
                    "entry_price": 50000,
                    "size_usd": 25,
                    "size_coins": 0.0005,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_loss": 49000,
                }
            },
            "trade_count": 1,
            "halted": False,
        }
        trader = self._make_trader(state=state)
        with pytest.raises(ValueError, match="Already have an open position"):
            trader._check_safety("BTC", 10.0)

    def test_different_coin_position_allowed(self):
        """Having a BTC position shouldn't block ETH."""
        state = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {
                "BTC": {
                    "side": "LONG",
                    "entry_price": 50000,
                    "size_usd": 25,
                    "size_coins": 0.0005,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_loss": 49000,
                }
            },
            "trade_count": 1,
            "halted": False,
        }
        trader = self._make_trader(state=state)
        trader._check_safety("ETH", 10.0)  # Should not raise

    def test_constants_are_hardcoded(self):
        """Verify the safety constants are what we expect."""
        assert MAX_POSITION_USD == 50.0
        assert STOP_LOSS_PCT == 0.02
        assert DAILY_LOSS_LIMIT_USD == 20.0
        assert ALLOWED_COINS == {"BTC", "ETH", "SOL"}


# ── Order Building Tests ─────────────────────────────────────────────────────


class TestOrderBuilding:
    """Tests for order action construction."""

    @patch("live_trader._load_live_state")
    def _make_trader(self, mock_load):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        return HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)

    def test_build_buy_order(self):
        trader = self._make_trader()
        action = trader._build_order_action("BTC", True, 0.001, 50000.0)

        assert action["type"] == "order"
        assert action["grouping"] == "na"
        assert len(action["orders"]) == 1

        order = action["orders"][0]
        assert order["a"] == COIN_TO_ASSET["BTC"]
        assert order["b"] is True  # is_buy
        assert order["r"] is False  # not reduce_only
        assert order["t"] == {"limit": {"tif": "Ioc"}}

    def test_build_sell_order(self):
        trader = self._make_trader()
        action = trader._build_order_action("ETH", False, 0.1, 3000.0)

        order = action["orders"][0]
        assert order["a"] == COIN_TO_ASSET["ETH"]
        assert order["b"] is False

    def test_build_reduce_only_order(self):
        trader = self._make_trader()
        action = trader._build_order_action("SOL", True, 1.0, 150.0, reduce_only=True)

        order = action["orders"][0]
        assert order["r"] is True


# ── Dry-Run Mode Tests ───────────────────────────────────────────────────────


class TestDryRun:
    """Tests for dry-run mode (signs but doesn't submit)."""

    @patch("live_trader._save_live_state")
    @patch("live_trader._log_live_event")
    @patch("live_trader._fetch_mid_price", return_value=50000.0)
    @patch("live_trader._load_live_state")
    def test_dry_run_signs_but_does_not_submit(
        self, mock_load, mock_price, mock_log, mock_save
    ):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        result = trader.place_order("BTC", "buy", 10.0)

        assert result["status"] == "dry_run"
        assert "signature" in result
        assert "action" in result
        assert result["message"].startswith("Order signed but NOT submitted")

    @patch("live_trader._save_live_state")
    @patch("live_trader._log_live_event")
    @patch("live_trader._fetch_mid_price", return_value=50000.0)
    @patch("live_trader._load_live_state")
    def test_dry_run_tracks_position_locally(
        self, mock_load, mock_price, mock_log, mock_save
    ):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        trader.place_order("BTC", "buy", 10.0)

        assert "BTC" in trader._state["open_positions"]
        pos = trader._state["open_positions"]["BTC"]
        assert pos["side"] == "LONG"
        assert pos["size_usd"] == 10.0

    @patch("live_trader._save_live_state")
    @patch("live_trader._log_live_event")
    @patch("live_trader._fetch_mid_price", return_value=50000.0)
    @patch("live_trader._load_live_state")
    def test_dry_run_sell_creates_short(
        self, mock_load, mock_price, mock_log, mock_save
    ):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        trader.place_order("ETH", "sell", 20.0)

        assert "ETH" in trader._state["open_positions"]
        pos = trader._state["open_positions"]["ETH"]
        assert pos["side"] == "SHORT"


# ── Position Close Tests ─────────────────────────────────────────────────────


class TestClosePosition:
    """Tests for closing positions."""

    @patch("live_trader._save_live_state")
    @patch("live_trader._log_live_event")
    @patch("live_trader._fetch_mid_price", return_value=51000.0)  # Price moved up
    @patch("live_trader._load_live_state")
    def test_close_long_with_profit(
        self, mock_load, mock_price, mock_log, mock_save
    ):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {
                "BTC": {
                    "side": "LONG",
                    "entry_price": 50000.0,
                    "size_usd": 25.0,
                    "size_coins": 0.0005,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_loss": 49000.0,
                }
            },
            "trade_count": 1,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        result = trader.close_position("BTC")

        # PnL should be positive for a long when price goes up
        assert result["pnl"] > 0
        assert "BTC" not in trader._state["open_positions"]

    @patch("live_trader._save_live_state")
    @patch("live_trader._log_live_event")
    @patch("live_trader._fetch_mid_price", return_value=49000.0)  # Price dropped
    @patch("live_trader._load_live_state")
    def test_close_long_with_loss(
        self, mock_load, mock_price, mock_log, mock_save
    ):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {
                "BTC": {
                    "side": "LONG",
                    "entry_price": 50000.0,
                    "size_usd": 25.0,
                    "size_coins": 0.0005,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_loss": 49000.0,
                }
            },
            "trade_count": 1,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        result = trader.close_position("BTC")

        assert result["pnl"] < 0

    def test_close_nonexistent_position(self):
        with patch("live_trader._load_live_state") as mock_load:
            mock_load.return_value = {
                "daily_pnl": 0.0,
                "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "open_positions": {},
                "trade_count": 0,
                "halted": False,
            }
            trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
            with pytest.raises(ValueError, match="No open position"):
                trader.close_position("BTC")

    @patch("live_trader._save_live_state")
    @patch("live_trader._log_live_event")
    @patch("live_trader._fetch_mid_price", return_value=40000.0)
    @patch("live_trader._load_live_state")
    def test_daily_loss_triggers_halt(
        self, mock_load, mock_price, mock_log, mock_save
    ):
        """Closing a losing position that pushes daily PnL past $20 should halt."""
        mock_load.return_value = {
            "daily_pnl": -15.0,  # Already lost $15 today
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {
                "BTC": {
                    "side": "LONG",
                    "entry_price": 50000.0,
                    "size_usd": 50.0,  # Max position
                    "size_coins": 0.001,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_loss": 49000.0,
                }
            },
            "trade_count": 3,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        result = trader.close_position("BTC")

        # BTC dropped 20% (50k→40k) → loss on $50 position = $10
        # daily_pnl = -15 + (-10) = -25, exceeds $20 limit → halt
        assert result["pnl"] < 0
        assert trader._state["halted"] is True


# ── Stop-Loss Check Tests ────────────────────────────────────────────────────


class TestStopLossChecks:
    """Tests for automatic stop-loss monitoring."""

    @patch("live_trader._save_live_state")
    @patch("live_trader._log_live_event")
    @patch("live_trader._fetch_mid_price", return_value=49000.0)
    @patch("live_trader._load_live_state")
    def test_stop_loss_triggers_close(
        self, mock_load, mock_price, mock_log, mock_save
    ):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {
                "BTC": {
                    "side": "LONG",
                    "entry_price": 50000.0,
                    "size_usd": 25.0,
                    "size_coins": 0.0005,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_loss": 49500.0,  # SL at 49500, current is 49000
                }
            },
            "trade_count": 1,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        actions = trader.check_stop_losses()

        assert len(actions) == 1
        assert "STOP-LOSS" in actions[0]
        assert "BTC" not in trader._state.get("open_positions", {})

    @patch("live_trader._fetch_mid_price", return_value=50500.0)
    @patch("live_trader._load_live_state")
    def test_stop_loss_not_triggered_above(self, mock_load, mock_price):
        """Price above SL should not trigger close."""
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {
                "BTC": {
                    "side": "LONG",
                    "entry_price": 50000.0,
                    "size_usd": 25.0,
                    "size_coins": 0.0005,
                    "opened_at": "2026-01-01T00:00:00Z",
                    "stop_loss": 49000.0,
                }
            },
            "trade_count": 1,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        actions = trader.check_stop_losses()

        assert len(actions) == 0
        assert "BTC" in trader._state["open_positions"]


# ── Address Derivation Tests ─────────────────────────────────────────────────


class TestAddressDerivation:
    """Tests for wallet address derivation from private key."""

    @patch("live_trader._load_live_state")
    def test_address_derived_correctly(self, mock_load):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        assert trader.address.startswith("0x")
        assert len(trader.address) == 42

    @patch("live_trader._load_live_state")
    def test_address_consistent(self, mock_load):
        """Same key should always produce same address."""
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        t1 = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        t2 = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        assert t1.address == t2.address


# ── Input Validation Tests ───────────────────────────────────────────────────


class TestInputValidation:
    """Tests for input validation."""

    @patch("live_trader._save_live_state")
    @patch("live_trader._log_live_event")
    @patch("live_trader._fetch_mid_price", return_value=50000.0)
    @patch("live_trader._load_live_state")
    def test_invalid_side_rejected(self, mock_load, mock_price, mock_log, mock_save):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        with pytest.raises(ValueError, match="Side must be"):
            trader.place_order("BTC", "hodl", 10.0)

    @patch("live_trader._load_live_state")
    def test_close_disallowed_coin(self, mock_load):
        mock_load.return_value = {
            "daily_pnl": 0.0,
            "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "open_positions": {},
            "trade_count": 0,
            "halted": False,
        }
        trader = HLLiveTrader(private_key=TEST_PRIVATE_KEY, dry_run=True)
        with pytest.raises(ValueError, match="not in allowed list"):
            trader.close_position("DOGE")
