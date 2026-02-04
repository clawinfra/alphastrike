"""
Hyperliquid EIP-712 Authentication

Implements wallet-based signing for Hyperliquid DEX trading operations.
Uses EIP-712 typed data signatures for secure authentication.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Domain separator for EIP-712
HYPERLIQUID_DOMAIN = {
    "name": "HyperliquidSignTransaction",
    "version": "1",
    "chainId": 1337,  # Hyperliquid L1 chain ID
    "verifyingContract": "0x0000000000000000000000000000000000000000",
}

# Domain for mainnet
MAINNET_DOMAIN = {
    "name": "HyperliquidSignTransaction",
    "version": "1",
    "chainId": 42161,  # Arbitrum chain ID for mainnet
    "verifyingContract": "0x0000000000000000000000000000000000000000",
}

# EIP-712 type definitions for different actions
ORDER_TYPE = {
    "Order": [
        {"name": "asset", "type": "uint32"},
        {"name": "isBuy", "type": "bool"},
        {"name": "limitPx", "type": "uint64"},
        {"name": "sz", "type": "uint64"},
        {"name": "reduceOnly", "type": "bool"},
        {"name": "cloid", "type": "bytes16"},
    ]
}

CANCEL_TYPE = {
    "Cancel": [
        {"name": "asset", "type": "uint32"},
        {"name": "oid", "type": "uint64"},
    ]
}

UPDATE_LEVERAGE_TYPE = {
    "UpdateLeverage": [
        {"name": "asset", "type": "uint32"},
        {"name": "isCross", "type": "bool"},
        {"name": "leverage", "type": "uint32"},
    ]
}


class HyperliquidSigner:
    """
    EIP-712 signer for Hyperliquid DEX operations.

    Uses eth_account for signing typed data per EIP-712 standard.

    Example:
        signer = HyperliquidSigner(private_key="0x...")
        signed = signer.sign_order(order_data)
    """

    def __init__(
        self,
        private_key: str | None = None,
        wallet_address: str | None = None,
        testnet: bool = False,
    ):
        """
        Initialize the signer.

        Args:
            private_key: Ethereum private key (hex string with 0x prefix)
            wallet_address: Wallet address (derived from private key if not provided)
            testnet: Whether to use testnet domain
        """
        self.private_key = private_key
        self._account = None
        self._wallet_address = wallet_address
        self.testnet = testnet
        self.domain = HYPERLIQUID_DOMAIN if testnet else MAINNET_DOMAIN

        if private_key:
            self._init_account()

    def _init_account(self) -> None:
        """Initialize eth_account from private key."""
        try:
            from eth_account import Account
            self._account = Account.from_key(self.private_key)
            if not self._wallet_address:
                self._wallet_address = self._account.address
            logger.info(f"Hyperliquid signer initialized for {self._wallet_address}")
        except ImportError:
            logger.warning(
                "eth_account not installed. Install with: pip install eth-account"
            )
        except Exception as e:
            logger.error(f"Failed to initialize signer: {e}")

    @property
    def wallet_address(self) -> str | None:
        """Get the wallet address."""
        return self._wallet_address

    def sign_typed_data(
        self,
        domain: dict[str, Any],
        types: dict[str, list],
        primary_type: str,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Sign EIP-712 typed data.

        Args:
            domain: EIP-712 domain separator
            types: Type definitions
            primary_type: Primary type name
            message: Message to sign

        Returns:
            Signature dict with r, s, v components
        """
        if not self._account:
            raise RuntimeError("Signer not initialized - missing private key")

        try:
            from eth_account.messages import encode_typed_data

            full_message = {
                "domain": domain,
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"},
                    ],
                    **types,
                },
                "primaryType": primary_type,
                "message": message,
            }

            signable = encode_typed_data(full_message=full_message)
            signed = self._account.sign_message(signable)

            return {
                "r": hex(signed.r),
                "s": hex(signed.s),
                "v": signed.v,
            }
        except ImportError:
            raise RuntimeError(
                "eth_account not installed. Install with: pip install eth-account"
            )

    def sign_order_action(
        self,
        orders: list[dict[str, Any]],
        grouping: str = "na",
        nonce: int | None = None,
    ) -> dict[str, Any]:
        """
        Sign an order placement action.

        Args:
            orders: List of order specifications
            grouping: Order grouping type (na, normalTpsl, positionTpsl)
            nonce: Timestamp nonce (current time if not provided)

        Returns:
            Complete signed request body
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action = {
            "type": "order",
            "orders": orders,
            "grouping": grouping,
        }

        signature = self._sign_agent(action, nonce)

        return {
            "action": action,
            "nonce": nonce,
            "signature": signature,
        }

    def sign_cancel_action(
        self,
        cancels: list[dict[str, Any]],
        nonce: int | None = None,
    ) -> dict[str, Any]:
        """
        Sign a cancel order action.

        Args:
            cancels: List of cancel specifications [{a: asset_index, o: order_id}]
            nonce: Timestamp nonce

        Returns:
            Complete signed request body
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action = {
            "type": "cancel",
            "cancels": cancels,
        }

        signature = self._sign_agent(action, nonce)

        return {
            "action": action,
            "nonce": nonce,
            "signature": signature,
        }

    def sign_update_leverage(
        self,
        asset: int,
        is_cross: bool,
        leverage: int,
        nonce: int | None = None,
    ) -> dict[str, Any]:
        """
        Sign an update leverage action.

        Args:
            asset: Asset index
            is_cross: Whether to update cross leverage
            leverage: New leverage value
            nonce: Timestamp nonce

        Returns:
            Complete signed request body
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action = {
            "type": "updateLeverage",
            "asset": asset,
            "isCross": is_cross,
            "leverage": leverage,
        }

        signature = self._sign_agent(action, nonce)

        return {
            "action": action,
            "nonce": nonce,
            "signature": signature,
        }

    def _sign_agent(
        self,
        action: dict[str, Any],
        nonce: int,
    ) -> dict[str, Any]:
        """
        Sign an agent action using the official Hyperliquid SDK signing method.

        Uses the SDK's sign_l1_action which properly implements:
        1. msgpack-based action hashing
        2. Phantom agent construction
        3. EIP-712 typed data signing
        """
        if not self._account:
            raise RuntimeError("Signer not initialized")

        try:
            # Use official SDK signing functions
            from hyperliquid.utils.signing import sign_l1_action

            # sign_l1_action(wallet, action, active_pool, nonce, expires_after, is_mainnet)
            signature = sign_l1_action(
                wallet=self._account,
                action=action,
                active_pool=None,  # No vault/pool for regular orders
                nonce=nonce,
                expires_after=None,  # No expiry
                is_mainnet=not self.testnet,
            )

            return signature

        except ImportError:
            # Fallback to manual implementation if SDK not available
            logger.warning("Official SDK not available, using manual signing")
            return self._sign_agent_manual(action, nonce)
        except Exception as e:
            logger.error(f"Failed to sign action: {e}")
            raise

    def _sign_agent_manual(
        self,
        action: dict[str, Any],
        nonce: int,
    ) -> dict[str, Any]:
        """
        Manual fallback signing (for when SDK is not available).
        Implements the same algorithm as the official SDK.
        """
        try:
            import msgpack
            from eth_abi import encode as eth_encode
            from eth_account.messages import encode_typed_data
            from Crypto.Hash import keccak

            def keccak256(data: bytes) -> bytes:
                k = keccak.new(digest_bits=256)
                k.update(data)
                return k.digest()

            # Step 1: Create action hash (msgpack + nonce)
            data = msgpack.packb(action)
            data += nonce.to_bytes(8, "big")
            data += b"\x00"  # No vault address
            action_hash = keccak256(data)

            # Step 2: Construct phantom agent
            source = "a" if not self.testnet else "b"
            phantom_agent = {"source": source, "connectionId": action_hash}

            # Step 3: Build EIP-712 payload
            payload = {
                "domain": {
                    "chainId": 1337,
                    "name": "Exchange",
                    "verifyingContract": "0x0000000000000000000000000000000000000000",
                    "version": "1",
                },
                "types": {
                    "Agent": [
                        {"name": "source", "type": "string"},
                        {"name": "connectionId", "type": "bytes32"},
                    ],
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"},
                    ],
                },
                "primaryType": "Agent",
                "message": phantom_agent,
            }

            # Step 4: Sign
            structured_data = encode_typed_data(full_message=payload)
            signed = self._account.sign_message(structured_data)

            return {
                "r": hex(signed.r),
                "s": hex(signed.s),
                "v": signed.v,
            }

        except Exception as e:
            logger.error(f"Manual signing failed: {e}")
            raise


def create_order_wire(
    asset: int,
    is_buy: bool,
    limit_px: str,
    sz: str,
    reduce_only: bool = False,
    order_type: dict[str, Any] | None = None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    """
    Create order specification in Hyperliquid wire format.

    Args:
        asset: Asset index
        is_buy: True for buy, False for sell
        limit_px: Limit price as string
        sz: Size as string
        reduce_only: Reduce only flag
        order_type: Order type specification (limit or trigger)
        client_order_id: Optional client order ID (128-bit hex)

    Returns:
        Order specification dict
    """
    order: dict[str, Any] = {
        "a": asset,
        "b": is_buy,
        "p": limit_px,
        "s": sz,
        "r": reduce_only,
        "t": order_type or {"limit": {"tif": "Gtc"}},
    }

    if client_order_id:
        order["c"] = client_order_id

    return order


def create_cancel_wire(
    asset: int,
    order_id: int,
) -> dict[str, Any]:
    """
    Create cancel specification in Hyperliquid wire format.

    Args:
        asset: Asset index
        order_id: Order ID to cancel

    Returns:
        Cancel specification dict
    """
    return {
        "a": asset,
        "o": order_id,
    }
