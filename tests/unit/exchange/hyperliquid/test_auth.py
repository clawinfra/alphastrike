"""
Tests for Hyperliquid authentication and signing.

Coverage targets: signer initialization, signing methods, wire format creation.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.exchange.adapters.hyperliquid.auth import (
    HyperliquidSigner,
    create_order_wire,
    create_cancel_wire,
    HYPERLIQUID_DOMAIN,
    MAINNET_DOMAIN,
)


class TestHyperliquidSignerInit:
    """Tests for signer initialization."""

    def test_init_with_private_key(self):
        """Test initialization with private key."""
        pk = "0x" + "a" * 64
        signer = HyperliquidSigner(private_key=pk, testnet=True)

        assert signer.private_key == pk
        assert signer.testnet is True
        assert signer.domain == HYPERLIQUID_DOMAIN
        assert signer.wallet_address is not None

    def test_init_with_wallet_address(self):
        """Test initialization with explicit wallet address."""
        pk = "0x" + "a" * 64
        addr = "0x1234567890abcdef1234567890abcdef12345678"
        signer = HyperliquidSigner(private_key=pk, wallet_address=addr, testnet=True)

        assert signer.wallet_address == addr

    def test_init_mainnet_domain(self):
        """Test initialization with mainnet domain."""
        pk = "0x" + "a" * 64
        signer = HyperliquidSigner(private_key=pk, testnet=False)

        assert signer.domain == MAINNET_DOMAIN

    def test_init_without_private_key(self):
        """Test initialization without private key."""
        signer = HyperliquidSigner(testnet=True)

        assert signer.private_key is None
        assert signer._account is None

    def test_wallet_address_property(self):
        """Test wallet_address property."""
        pk = "0x" + "a" * 64
        signer = HyperliquidSigner(private_key=pk, testnet=True)

        assert signer.wallet_address is not None
        assert signer.wallet_address.startswith("0x")

    def test_init_with_invalid_key_logs_error(self):
        """Test initialization with invalid key logs error."""
        with patch("src.exchange.adapters.hyperliquid.auth.logger") as mock_logger:
            signer = HyperliquidSigner(private_key="invalid", testnet=True)
            assert mock_logger.error.called or signer._account is None

    def test_init_without_eth_account(self):
        """Test initialization when eth_account is not installed."""
        with patch.dict("sys.modules", {"eth_account": None}):
            with patch("src.exchange.adapters.hyperliquid.auth.logger") as mock_logger:
                pk = "0x" + "a" * 64
                signer = HyperliquidSigner(private_key=pk, testnet=True)
                # Should log warning or error


class TestHyperliquidSignerSignTypedData:
    """Tests for sign_typed_data method."""

    @pytest.fixture
    def signer(self):
        """Create signer fixture."""
        pk = "0x" + "a" * 64
        return HyperliquidSigner(private_key=pk, testnet=True)

    def test_sign_typed_data_not_initialized(self):
        """Test sign_typed_data raises when not initialized."""
        signer = HyperliquidSigner(testnet=True)

        with pytest.raises(RuntimeError, match="not initialized"):
            signer.sign_typed_data(
                domain={},
                types={},
                primary_type="Test",
                message={},
            )

    def test_sign_typed_data_success(self, signer):
        """Test successful signing."""
        result = signer.sign_typed_data(
            domain=HYPERLIQUID_DOMAIN,
            types={"Test": [{"name": "value", "type": "uint256"}]},
            primary_type="Test",
            message={"value": 123},
        )

        assert "r" in result
        assert "s" in result
        assert "v" in result
        assert result["r"].startswith("0x")
        assert result["s"].startswith("0x")


class TestHyperliquidSignerOrderActions:
    """Tests for order action signing."""

    @pytest.fixture
    def signer(self):
        """Create signer fixture."""
        pk = "0x" + "a" * 64
        return HyperliquidSigner(private_key=pk, testnet=True)

    def test_sign_order_action(self, signer):
        """Test signing order action."""
        orders = [{
            "a": 0,
            "b": True,
            "p": "50000",
            "s": "0.1",
            "r": False,
            "t": {"limit": {"tif": "Gtc"}},
        }]

        # Mock the SDK import
        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                result = signer.sign_order_action(orders=orders, grouping="na")

        assert "action" in result
        assert "nonce" in result
        assert "signature" in result
        assert result["action"]["type"] == "order"
        assert result["action"]["orders"] == orders
        assert result["action"]["grouping"] == "na"

    def test_sign_order_action_with_nonce(self, signer):
        """Test signing order action with explicit nonce."""
        orders = [{"a": 0, "b": True, "p": "50000", "s": "0.1", "r": False, "t": {}}]
        nonce = 1234567890000

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                result = signer.sign_order_action(orders=orders, grouping="na", nonce=nonce)

        assert result["nonce"] == nonce


class TestHyperliquidSignerCancelActions:
    """Tests for cancel action signing."""

    @pytest.fixture
    def signer(self):
        """Create signer fixture."""
        pk = "0x" + "a" * 64
        return HyperliquidSigner(private_key=pk, testnet=True)

    def test_sign_cancel_action(self, signer):
        """Test signing cancel action."""
        cancels = [{"a": 0, "o": 12345}]

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                result = signer.sign_cancel_action(cancels=cancels)

        assert "action" in result
        assert "nonce" in result
        assert "signature" in result
        assert result["action"]["type"] == "cancel"
        assert result["action"]["cancels"] == cancels

    def test_sign_cancel_action_with_nonce(self, signer):
        """Test signing cancel action with explicit nonce."""
        cancels = [{"a": 0, "o": 12345}]
        nonce = 1234567890000

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                result = signer.sign_cancel_action(cancels=cancels, nonce=nonce)

        assert result["nonce"] == nonce


class TestHyperliquidSignerLeverageActions:
    """Tests for leverage action signing."""

    @pytest.fixture
    def signer(self):
        """Create signer fixture."""
        pk = "0x" + "a" * 64
        return HyperliquidSigner(private_key=pk, testnet=True)

    def test_sign_update_leverage(self, signer):
        """Test signing update leverage action."""
        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                result = signer.sign_update_leverage(
                    asset=0,
                    is_cross=True,
                    leverage=10,
                )

        assert "action" in result
        assert "nonce" in result
        assert "signature" in result
        assert result["action"]["type"] == "updateLeverage"
        assert result["action"]["asset"] == 0
        assert result["action"]["isCross"] is True
        assert result["action"]["leverage"] == 10

    def test_sign_update_leverage_with_nonce(self, signer):
        """Test signing leverage action with explicit nonce."""
        nonce = 1234567890000

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                result = signer.sign_update_leverage(
                    asset=0,
                    is_cross=False,
                    leverage=20,
                    nonce=nonce,
                )

        assert result["nonce"] == nonce


class TestSignAgentManual:
    """Tests for manual agent signing fallback."""

    @pytest.fixture
    def signer(self):
        """Create signer fixture."""
        pk = "0x" + "a" * 64
        return HyperliquidSigner(private_key=pk, testnet=True)

    def test_sign_agent_manual_testnet(self, signer):
        """Test manual signing for testnet uses fallback."""
        action = {"type": "order", "orders": []}
        nonce = 1234567890000

        # Test with SDK available - should use SDK
        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                result = signer._sign_agent(action, nonce)

        assert "r" in result
        assert "s" in result
        assert "v" in result

    def test_sign_agent_manual_mainnet(self):
        """Test manual signing for mainnet."""
        pk = "0x" + "a" * 64
        signer = HyperliquidSigner(private_key=pk, testnet=False)

        action = {"type": "cancel", "cancels": []}
        nonce = 1234567890000

        mock_signature = {"r": "0x123", "s": "0x456", "v": 27}
        with patch.dict("sys.modules", {"hyperliquid": MagicMock(), "hyperliquid.utils": MagicMock(), "hyperliquid.utils.signing": MagicMock()}):
            with patch("hyperliquid.utils.signing.sign_l1_action", return_value=mock_signature):
                result = signer._sign_agent(action, nonce)

        assert "r" in result
        assert "s" in result


class TestSignAgentNotInitialized:
    """Tests for signing without initialization."""

    def test_sign_agent_not_initialized(self):
        """Test _sign_agent raises when not initialized."""
        signer = HyperliquidSigner(testnet=True)

        with pytest.raises(RuntimeError, match="not initialized"):
            signer._sign_agent({"type": "order"}, 123)


class TestCreateOrderWire:
    """Tests for create_order_wire helper."""

    def test_basic_order_wire(self):
        """Test basic order wire creation."""
        wire = create_order_wire(
            asset=0,
            is_buy=True,
            limit_px="50000",
            sz="0.1",
        )

        assert wire["a"] == 0
        assert wire["b"] is True
        assert wire["p"] == "50000"
        assert wire["s"] == "0.1"
        assert wire["r"] is False
        assert wire["t"] == {"limit": {"tif": "Gtc"}}

    def test_order_wire_with_options(self):
        """Test order wire with all options."""
        order_type = {"trigger": {"isMarket": True, "triggerPx": "48000", "tpsl": "sl"}}
        wire = create_order_wire(
            asset=1,
            is_buy=False,
            limit_px="48000",
            sz="0.5",
            reduce_only=True,
            order_type=order_type,
            client_order_id="0x" + "a" * 32,
        )

        assert wire["a"] == 1
        assert wire["b"] is False
        assert wire["r"] is True
        assert wire["t"] == order_type
        assert wire["c"] == "0x" + "a" * 32

    def test_order_wire_without_client_id(self):
        """Test order wire without client ID."""
        wire = create_order_wire(
            asset=0,
            is_buy=True,
            limit_px="50000",
            sz="0.1",
        )

        assert "c" not in wire


class TestCreateCancelWire:
    """Tests for create_cancel_wire helper."""

    def test_basic_cancel_wire(self):
        """Test basic cancel wire creation."""
        wire = create_cancel_wire(asset=0, order_id=12345)

        assert wire["a"] == 0
        assert wire["o"] == 12345

    def test_cancel_wire_large_order_id(self):
        """Test cancel wire with large order ID."""
        wire = create_cancel_wire(asset=5, order_id=9999999999)

        assert wire["a"] == 5
        assert wire["o"] == 9999999999


class TestDomainConstants:
    """Tests for domain constants."""

    def test_hyperliquid_domain(self):
        """Test testnet domain constants."""
        assert HYPERLIQUID_DOMAIN["name"] == "HyperliquidSignTransaction"
        assert HYPERLIQUID_DOMAIN["version"] == "1"
        assert HYPERLIQUID_DOMAIN["chainId"] == 1337

    def test_mainnet_domain(self):
        """Test mainnet domain constants."""
        assert MAINNET_DOMAIN["name"] == "HyperliquidSignTransaction"
        assert MAINNET_DOMAIN["version"] == "1"
        assert MAINNET_DOMAIN["chainId"] == 42161  # Arbitrum


class TestSignAgentManualFallback:
    """Tests for manual signing fallback when SDK is not available."""

    def test_sign_agent_manual_fallback_testnet(self):
        """Test manual fallback signing for testnet (requires msgpack)."""
        pytest.importorskip("msgpack")

        pk = "0x" + "a" * 64
        signer = HyperliquidSigner(private_key=pk, testnet=True)

        action = {"type": "order", "orders": []}
        nonce = 1234567890000

        # Call the manual method directly
        result = signer._sign_agent_manual(action, nonce)

        assert "r" in result
        assert "s" in result
        assert "v" in result
        assert result["r"].startswith("0x")

    def test_sign_agent_manual_fallback_mainnet(self):
        """Test manual fallback signing for mainnet (requires msgpack)."""
        pytest.importorskip("msgpack")

        pk = "0x" + "a" * 64
        signer = HyperliquidSigner(private_key=pk, testnet=False)

        action = {"type": "cancel", "cancels": []}
        nonce = 1234567890000

        result = signer._sign_agent_manual(action, nonce)

        assert "r" in result
        assert "s" in result
        assert "v" in result
