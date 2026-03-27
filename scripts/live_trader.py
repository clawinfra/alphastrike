#!/usr/bin/env python3
"""
AlphaStrike V3 — Live Trading Module for Hyperliquid

Wraps V3 signal generation (paper_trade_v3.py) with real order execution
on Hyperliquid mainnet using EIP-712 signed requests.

Safety rails are HARDCODED — not configurable:
  - Max position size: $50 USD
  - Stop-loss: 2% from entry
  - Max 1 open position per coin
  - Daily loss limit: $20 (halt trading if hit)
  - Only BTC, ETH, SOL supported

Usage:
    # Dry run — signs but does NOT submit orders
    uv run python scripts/live_trader.py --dry-run --coin BTC --side buy --size 10

    # Live order (REAL MONEY)
    uv run python scripts/live_trader.py --coin BTC --side buy --size 10

    # Auto mode — runs V3 signals and executes
    uv run python scripts/live_trader.py --auto

    # Show open positions
    uv run python scripts/live_trader.py --positions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from eth_account import Account
from eth_account.messages import encode_typed_data

# ── Constants ─────────────────────────────────────────────────────────────────

HL_EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# EIP-712 domain for Hyperliquid mainnet
MAINNET_CHAIN_ID = 1337

# Coin → asset index mapping on Hyperliquid
# These are the perp asset indices (check HL meta endpoint for latest)
COIN_TO_ASSET: dict[str, int] = {
    "BTC": 0,
    "ETH": 1,
    "SOL": 4,
}

SUPPORTED_COINS = set(COIN_TO_ASSET.keys())

# ── HARDCODED SAFETY RAILS (NOT CONFIGURABLE) ────────────────────────────────
MAX_POSITION_USD = 50.0       # Max $50 per position
STOP_LOSS_PCT = 0.02          # 2% stop-loss from entry
MAX_POSITIONS_PER_COIN = 1    # Only 1 open position per coin
DAILY_LOSS_LIMIT_USD = 20.0   # Halt trading if daily loss exceeds $20
ALLOWED_COINS = {"BTC", "ETH", "SOL"}

# State file for tracking daily P&L and positions
STATE_DIR = Path(__file__).parent.parent / "data"
LIVE_STATE_FILE = STATE_DIR / "live_trade_state.json"
LIVE_LOG_FILE = STATE_DIR / "live_trade_log.jsonl"


# ── EIP-712 Signing (from signing.py reference) ──────────────────────────────

def _get_domain() -> dict[str, Any]:
    """Get the EIP-712 domain for Hyperliquid mainnet."""
    return {
        "name": "Exchange",
        "version": "1",
        "chainId": MAINNET_CHAIN_ID,
        "verifyingContract": "0x0000000000000000000000000000000000000000",
    }


def _action_hash(action: dict[str, Any], vault_address: Optional[str], nonce: int) -> str:
    """Compute the SHA256 action hash for signing."""
    action_json = json.dumps(action, separators=(",", ":"), sort_keys=True)
    if vault_address:
        data = f"{action_json}{vault_address}{nonce}"
    else:
        data = f"{action_json}{nonce}"
    return "0x" + hashlib.sha256(data.encode()).hexdigest()


def sign_action(
    action: dict[str, Any],
    nonce: int,
    private_key: str,
    vault_address: Optional[str] = None,
) -> dict[str, Any]:
    """
    Sign a trading action using EIP-712 structured data.

    Returns dict with r, s, v signature components.
    """
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    ah = _action_hash(action, vault_address, nonce)

    types = {
        "EIP712Domain": [
            {"name": "name", "type": "string"},
            {"name": "version", "type": "string"},
            {"name": "chainId", "type": "uint256"},
            {"name": "verifyingContract", "type": "address"},
        ],
        "Agent": [
            {"name": "source", "type": "string"},
            {"name": "connectionId", "type": "bytes32"},
        ],
    }

    message = {
        "source": "a",
        "connectionId": ah,
    }

    typed_data = {
        "types": types,
        "primaryType": "Agent",
        "domain": _get_domain(),
        "message": message,
    }

    encoded = encode_typed_data(full_message=typed_data)
    signed = Account.sign_message(encoded, private_key)

    return {
        "r": hex(signed.r),
        "s": hex(signed.s),
        "v": signed.v,
    }


# ── HL API Helpers ────────────────────────────────────────────────────────────

def _fetch_meta() -> dict:
    """Fetch exchange metadata (asset list)."""
    resp = requests.post(HL_INFO_URL, json={"type": "meta"}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _fetch_mid_price(coin: str) -> float:
    """Fetch current mid price for a coin from L2 orderbook."""
    resp = requests.post(
        HL_INFO_URL,
        json={"type": "l2Book", "coin": coin},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    levels = data.get("levels", [[], []])
    if levels[0] and levels[1]:
        best_bid = float(levels[0][0]["px"])
        best_ask = float(levels[1][0]["px"])
        return (best_bid + best_ask) / 2
    raise ValueError(f"No orderbook data for {coin}")


def _fetch_user_state(address: str) -> dict:
    """Fetch user's open positions and margin info."""
    resp = requests.post(
        HL_INFO_URL,
        json={"type": "clearinghouseState", "user": address},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _float_to_hl_string(value: float, decimals: int = 8) -> str:
    """Format a float to HL-compatible string (no trailing zeros, but enough precision)."""
    formatted = f"{value:.{decimals}f}"
    # Strip trailing zeros but keep at least one decimal
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _round_price(price: float, coin: str) -> str:
    """Round price to appropriate precision for the coin."""
    if coin == "BTC":
        return _float_to_hl_string(round(price, 1), 1)
    elif coin == "ETH":
        return _float_to_hl_string(round(price, 2), 2)
    elif coin == "SOL":
        return _float_to_hl_string(round(price, 3), 3)
    return _float_to_hl_string(price)


def _round_size(size: float, coin: str) -> str:
    """Round size to appropriate precision for the coin."""
    if coin == "BTC":
        # BTC min size increment varies, use 5 decimals
        return _float_to_hl_string(round(size, 5), 5)
    elif coin == "ETH":
        return _float_to_hl_string(round(size, 4), 4)
    elif coin == "SOL":
        return _float_to_hl_string(round(size, 2), 2)
    return _float_to_hl_string(size)


# ── State Management ─────────────────────────────────────────────────────────

def _load_live_state() -> dict:
    """Load live trading state from disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if LIVE_STATE_FILE.exists():
        return json.loads(LIVE_STATE_FILE.read_text())
    return {
        "daily_pnl": 0.0,
        "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "open_positions": {},
        "trade_count": 0,
        "halted": False,
    }


def _save_live_state(state: dict) -> None:
    """Save live trading state to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_STATE_FILE.write_text(json.dumps(state, indent=2))


def _log_live_event(event: dict) -> None:
    """Append a trade event to the log file."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LIVE_LOG_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _reset_daily_pnl_if_needed(state: dict) -> None:
    """Reset daily P&L counter if it's a new day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("daily_pnl_date") != today:
        state["daily_pnl"] = 0.0
        state["daily_pnl_date"] = today
        state["halted"] = False


# ── HLLiveTrader Class ───────────────────────────────────────────────────────

class HLLiveTrader:
    """
    Live trading client for Hyperliquid with hardcoded safety rails.

    Safety rules (NOT configurable):
    - Max $50 per position
    - 2% stop-loss
    - 1 position per coin max
    - $20 daily loss limit → auto-halt
    - Only BTC, ETH, SOL
    """

    def __init__(self, private_key: str, dry_run: bool = False):
        """
        Initialize the live trader.

        Args:
            private_key: Hex private key (with or without 0x prefix)
            dry_run: If True, sign orders but don't submit them
        """
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key
        self._private_key = private_key
        self._account = Account.from_key(private_key)
        self._address = self._account.address
        self._dry_run = dry_run
        self._state = _load_live_state()
        _reset_daily_pnl_if_needed(self._state)

    @property
    def address(self) -> str:
        return self._address

    @property
    def is_halted(self) -> bool:
        return self._state.get("halted", False)

    @property
    def daily_pnl(self) -> float:
        return self._state.get("daily_pnl", 0.0)

    def _check_safety(self, coin: str, size_usd: float) -> None:
        """
        Enforce all safety rails. Raises ValueError on violation.
        """
        # Check coin is allowed
        if coin not in ALLOWED_COINS:
            raise ValueError(f"Coin {coin} not in allowed list: {ALLOWED_COINS}")

        # Check max position size
        if size_usd > MAX_POSITION_USD:
            raise ValueError(
                f"Position size ${size_usd:.2f} exceeds max ${MAX_POSITION_USD:.2f}"
            )

        # Check daily loss limit
        _reset_daily_pnl_if_needed(self._state)
        if self._state.get("halted", False):
            raise ValueError(
                f"Trading HALTED: daily loss limit ${DAILY_LOSS_LIMIT_USD:.2f} exceeded "
                f"(current daily PnL: ${self._state['daily_pnl']:.2f})"
            )

        # Check max 1 position per coin
        open_positions = self._state.get("open_positions", {})
        if coin in open_positions:
            raise ValueError(
                f"Already have an open position in {coin}. "
                f"Close it first before opening a new one."
            )

    def _build_order_action(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        price: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Build the HL order action payload."""
        asset_idx = COIN_TO_ASSET[coin]
        return {
            "type": "order",
            "orders": [
                {
                    "a": asset_idx,
                    "b": is_buy,
                    "p": _round_price(price, coin),
                    "s": _round_size(size, coin),
                    "r": reduce_only,
                    "t": {"limit": {"tif": "Ioc"}},  # Immediate-or-Cancel for market-like
                }
            ],
            "grouping": "na",
        }

    def _submit_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Sign and submit an action to HL exchange."""
        nonce = int(time.time() * 1000)
        signature = sign_action(action, nonce, self._private_key)

        payload = {
            "action": action,
            "nonce": nonce,
            "signature": signature,
            "vaultAddress": None,
        }

        if self._dry_run:
            # In dry-run mode, return what would be sent (redact private key)
            return {
                "status": "dry_run",
                "action": action,
                "nonce": nonce,
                "signature": signature,
                "address": self._address,
                "message": "Order signed but NOT submitted (dry-run mode)",
            }

        # Submit to HL exchange
        resp = requests.post(
            HL_EXCHANGE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def place_order(
        self,
        coin: str,
        side: str,
        size_usd: float,
        price: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Place a market order on Hyperliquid.

        Args:
            coin: Trading pair (BTC, ETH, SOL)
            side: "buy" or "sell"
            size_usd: Position size in USD
            price: Limit price. If None, fetches mid price and adds/subtracts 0.1% slippage.

        Returns:
            Order result dict
        """
        coin = coin.upper()
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Side must be 'buy' or 'sell', got '{side}'")

        # Enforce safety rails
        self._check_safety(coin, size_usd)

        # Get price if not provided
        if price is None:
            mid_price = _fetch_mid_price(coin)
            # Add 0.1% slippage for market-like execution
            if side == "buy":
                price = mid_price * 1.001
            else:
                price = mid_price * 0.999

        # Convert USD size to coin quantity
        size_coins = size_usd / price

        is_buy = side == "buy"
        action = self._build_order_action(coin, is_buy, size_coins, price)

        result = self._submit_action(action)

        # Track position in state
        entry_info = {
            "side": "LONG" if is_buy else "SHORT",
            "entry_price": price,
            "size_usd": size_usd,
            "size_coins": size_coins,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "stop_loss": price * (1 - STOP_LOSS_PCT) if is_buy else price * (1 + STOP_LOSS_PCT),
        }
        self._state.setdefault("open_positions", {})[coin] = entry_info
        self._state["trade_count"] = self._state.get("trade_count", 0) + 1
        _save_live_state(self._state)

        # Log the trade
        _log_live_event({
            "type": "open",
            "ts": datetime.now(timezone.utc).isoformat(),
            "coin": coin,
            "side": side,
            "size_usd": size_usd,
            "price": price,
            "dry_run": self._dry_run,
            "result": result if self._dry_run else str(result)[:500],
        })

        return result

    def close_position(self, coin: str) -> dict[str, Any]:
        """
        Close an open position for a coin.

        Args:
            coin: The coin to close position for

        Returns:
            Order result dict
        """
        coin = coin.upper()
        if coin not in ALLOWED_COINS:
            raise ValueError(f"Coin {coin} not in allowed list: {ALLOWED_COINS}")

        open_positions = self._state.get("open_positions", {})
        if coin not in open_positions:
            raise ValueError(f"No open position for {coin}")

        pos = open_positions[coin]
        entry_price = pos["entry_price"]
        side = pos["side"]
        size_coins = pos["size_coins"]

        # Close = opposite direction, reduce_only=True
        is_buy = side == "SHORT"  # Buy to close short, sell to close long
        close_side = "buy" if is_buy else "sell"

        # Get current price for the close
        current_price = _fetch_mid_price(coin)
        # Slippage for close
        if is_buy:
            close_price = current_price * 1.001
        else:
            close_price = current_price * 0.999

        action = self._build_order_action(coin, is_buy, size_coins, close_price, reduce_only=True)
        result = self._submit_action(action)

        # Calculate P&L
        if side == "LONG":
            pnl = (current_price - entry_price) / entry_price * pos["size_usd"]
        else:
            pnl = (entry_price - current_price) / entry_price * pos["size_usd"]

        # Update daily P&L
        self._state["daily_pnl"] = self._state.get("daily_pnl", 0.0) + pnl
        if self._state["daily_pnl"] <= -DAILY_LOSS_LIMIT_USD:
            self._state["halted"] = True
            print(f"⚠️  TRADING HALTED: Daily loss ${abs(self._state['daily_pnl']):.2f} "
                  f"exceeds limit ${DAILY_LOSS_LIMIT_USD:.2f}")

        # Remove position
        del self._state["open_positions"][coin]
        _save_live_state(self._state)

        # Log the close
        _log_live_event({
            "type": "close",
            "ts": datetime.now(timezone.utc).isoformat(),
            "coin": coin,
            "close_side": close_side,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl": round(pnl, 2),
            "daily_pnl": round(self._state["daily_pnl"], 2),
            "halted": self._state["halted"],
            "dry_run": self._dry_run,
            "result": result if self._dry_run else str(result)[:500],
        })

        return {
            "result": result,
            "pnl": round(pnl, 2),
            "daily_pnl": round(self._state["daily_pnl"], 2),
        }

    def get_positions(self) -> list[dict]:
        """
        Query open positions from Hyperliquid.

        Returns list of position dicts with coin, side, size, unrealized PnL.
        """
        try:
            user_state = _fetch_user_state(self._address)
            positions = []
            for pos in user_state.get("assetPositions", []):
                p = pos.get("position", {})
                if float(p.get("szi", "0")) != 0:
                    positions.append({
                        "coin": p.get("coin", "?"),
                        "size": float(p.get("szi", "0")),
                        "entry_price": float(p.get("entryPx", "0")),
                        "unrealized_pnl": float(p.get("unrealizedPnl", "0")),
                        "margin_used": float(p.get("marginUsed", "0")),
                        "leverage": float(p.get("leverage", {}).get("value", "0")),
                    })
            return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def get_balance(self) -> float:
        """Get account USDC balance."""
        try:
            user_state = _fetch_user_state(self._address)
            # HL returns margin summary with accountValue
            margin = user_state.get("marginSummary", {})
            return float(margin.get("accountValue", "0"))
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    def check_stop_losses(self) -> list[str]:
        """
        Check all open positions against stop-loss levels.
        Close any that have been hit.

        Returns list of action descriptions.
        """
        actions = []
        open_positions = dict(self._state.get("open_positions", {}))

        for coin, pos in open_positions.items():
            try:
                current_price = _fetch_mid_price(coin)
                stop_loss = pos.get("stop_loss", 0)
                side = pos["side"]

                triggered = False
                if side == "LONG" and current_price <= stop_loss:
                    triggered = True
                elif side == "SHORT" and current_price >= stop_loss:
                    triggered = True

                if triggered:
                    result = self.close_position(coin)
                    msg = (
                        f"🛑 STOP-LOSS {coin}: {side} closed @ "
                        f"${current_price:.2f} (entry: ${pos['entry_price']:.2f}, "
                        f"PnL: ${result['pnl']:+.2f})"
                    )
                    actions.append(msg)
                    print(msg)
            except Exception as e:
                actions.append(f"Error checking {coin} stop-loss: {e}")

        return actions


# ── Auto Mode (V3 signals → live execution) ───────────────────────────────────

def run_auto_mode(trader: HLLiveTrader, days: int = 1) -> None:
    """
    Run V3 signal evaluation and execute trades automatically.
    """
    # Import V3 signal evaluation
    sys.path.insert(0, str(Path(__file__).parent))
    from paper_trade_v3 import (
        COINS,
        evaluate_signal,
        fetch_candles,
        fetch_orderbook_imbalance,
    )

    run_ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"AlphaStrike V3 LIVE Auto-Trade — {run_ts}")
    print(f"Address: {trader.address}")
    print(f"Dry-run: {trader._dry_run}")
    print(f"Balance: ${trader.get_balance():.2f} USDC")
    print(f"{'='*60}")

    if trader.is_halted:
        print("⚠️  Trading is HALTED due to daily loss limit. No orders will be placed.")
        return

    # Check stop-losses first
    sl_actions = trader.check_stop_losses()
    if sl_actions:
        print(f"\nStop-loss actions: {len(sl_actions)}")
        for a in sl_actions:
            print(f"  {a}")

    for coin in COINS:
        if coin not in ALLOWED_COINS:
            continue
        try:
            print(f"\n[{coin}] Fetching {days}d candles + orderbook...")
            candles = fetch_candles(coin, days=days)
            if len(candles) < 30:
                print(f"[{coin}] Not enough candles ({len(candles)}), skipping")
                continue

            imbalance = fetch_orderbook_imbalance(coin)
            sig = evaluate_signal(coin, candles, imbalance)

            print(f"[{coin}] RSI={sig['rsi']} | MACD_H={sig['macd_histogram']} | "
                  f"VolR={sig['volume_ratio']} | OBI={sig['orderbook_imbalance']:.3f}")
            print(f"[{coin}] V3 Signal: {sig['v3_signal']}")

            direction = sig["v3_signal"]
            open_pos = trader._state.get("open_positions", {})

            # If we have an open position, check if signal flipped
            if coin in open_pos:
                pos_side = open_pos[coin]["side"]
                if direction != "FLAT" and direction != pos_side:
                    print(f"  ▶ Signal flipped {pos_side} → {direction}, closing position")
                    result = trader.close_position(coin)
                    print(f"  ▶ Closed: PnL ${result['pnl']:+.2f}")

            # Open new position if signal and no existing position
            if direction != "FLAT" and coin not in trader._state.get("open_positions", {}):
                side = "buy" if direction == "LONG" else "sell"
                size_usd = min(MAX_POSITION_USD, 25.0)  # Conservative: $25 default

                print(f"  ▶ Opening {direction} {coin} | ${size_usd:.2f}")
                try:
                    result = trader.place_order(coin, side, size_usd)
                    if trader._dry_run:
                        print(f"  ▶ DRY-RUN: Order signed but not submitted")
                        print(f"    Signature: r={result['signature']['r'][:10]}...")
                    else:
                        print(f"  ▶ Order submitted: {result}")
                except ValueError as e:
                    print(f"  ⚠️  Safety check blocked order: {e}")

        except Exception as e:
            print(f"[{coin}] ERROR: {e}")

    # Summary
    print(f"\n{'─'*60}")
    print(f"Daily PnL: ${trader.daily_pnl:+.2f} | Halted: {trader.is_halted}")
    open_pos = trader._state.get("open_positions", {})
    print(f"Open positions: {len(open_pos)}")
    for coin, pos in open_pos.items():
        print(f"  {pos['side']} {coin} @ ${pos['entry_price']:.2f} "
              f"(SL: ${pos['stop_loss']:.2f}) since {pos['opened_at'][:10]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaStrike V3 Live Trader for Hyperliquid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Sign orders but don't submit")
    parser.add_argument("--coin", type=str, help="Coin to trade (BTC, ETH, SOL)")
    parser.add_argument("--side", type=str, help="Order side (buy, sell)")
    parser.add_argument("--size", type=float, help="Position size in USD")
    parser.add_argument("--price", type=float, help="Limit price (optional, uses mid if omitted)")
    parser.add_argument("--close", type=str, help="Close position for coin")
    parser.add_argument("--positions", action="store_true", help="Show open positions")
    parser.add_argument("--balance", action="store_true", help="Show account balance")
    parser.add_argument("--auto", action="store_true", help="Run V3 signals and auto-trade")
    parser.add_argument("--days", type=int, default=1, help="Candle history days for auto mode")
    args = parser.parse_args()

    # Load private key from environment or decrypt
    private_key = os.environ.get("HL_PRIVATE_KEY")
    if not private_key:
        # Try decrypting from encrypted store
        import subprocess
        try:
            result = subprocess.run(
                ["bash", "/home/bowen/.openclaw/workspace/memory/decrypt.sh", "hl-private-key.txt"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                private_key = result.stdout.strip()
            else:
                print("ERROR: Could not decrypt private key. Set HL_PRIVATE_KEY env var.")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load private key: {e}")
            sys.exit(1)

    trader = HLLiveTrader(private_key=private_key, dry_run=args.dry_run)
    # Clear private_key from local scope immediately
    private_key = None  # noqa: F841

    print(f"Trader initialized | Address: {trader.address} | Dry-run: {args.dry_run}")

    if args.positions:
        positions = trader.get_positions()
        if positions:
            print(f"\nOpen positions ({len(positions)}):")
            for p in positions:
                print(f"  {p['coin']}: size={p['size']:.6f}, "
                      f"entry=${p['entry_price']:.2f}, "
                      f"uPnL=${p['unrealized_pnl']:.2f}, "
                      f"leverage={p['leverage']:.1f}x")
        else:
            print("\nNo open positions on-chain.")

        # Also show local state
        local_pos = trader._state.get("open_positions", {})
        if local_pos:
            print(f"\nLocal tracked positions ({len(local_pos)}):")
            for coin, pos in local_pos.items():
                print(f"  {pos['side']} {coin} @ ${pos['entry_price']:.2f} "
                      f"(SL: ${pos['stop_loss']:.2f})")
        return

    if args.balance:
        balance = trader.get_balance()
        print(f"\nAccount balance: ${balance:.2f} USDC")
        print(f"Daily PnL: ${trader.daily_pnl:+.2f}")
        print(f"Trading halted: {trader.is_halted}")
        return

    if args.close:
        coin = args.close.upper()
        print(f"\nClosing {coin} position...")
        result = trader.close_position(coin)
        print(f"Result: PnL=${result['pnl']:+.2f}, Daily PnL=${result['daily_pnl']:+.2f}")
        return

    if args.auto:
        run_auto_mode(trader, days=args.days)
        return

    # Manual order mode
    if not args.coin or not args.side or not args.size:
        parser.error("For manual orders, --coin, --side, and --size are required "
                      "(or use --auto, --positions, --balance, --close)")

    coin = args.coin.upper()
    print(f"\nPlacing {'DRY-RUN ' if args.dry_run else ''}order: "
          f"{args.side.upper()} {coin} ${args.size:.2f}")
    result = trader.place_order(coin, args.side, args.size, args.price)

    if args.dry_run:
        print(f"\n✅ DRY-RUN complete. Order signed but NOT submitted.")
        print(f"   Action: {json.dumps(result['action'], indent=2)}")
        print(f"   Nonce: {result['nonce']}")
        print(f"   Signature: r={result['signature']['r'][:20]}... "
              f"s={result['signature']['s'][:20]}... v={result['signature']['v']}")
        print(f"   Address: {result['address']}")
    else:
        print(f"\n✅ Order submitted: {result}")


if __name__ == "__main__":
    main()
