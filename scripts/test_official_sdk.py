#!/usr/bin/env python3
"""
Test Order API using Official Hyperliquid SDK

Verifies order placement works on testnet.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Get private key from env
private_key = os.getenv("EXCHANGE_WALLET_PRIVATE_KEY")
if not private_key:
    print("ERROR: EXCHANGE_WALLET_PRIVATE_KEY not set in .env")
    exit(1)

print("=" * 60)
print("HYPERLIQUID TESTNET ORDER TEST (Official SDK)")
print("=" * 60)

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Use testnet
base_url = constants.TESTNET_API_URL
print(f"Using: {base_url}")

# Initialize
info = Info(base_url, skip_ws=True)
exchange = Exchange(None, base_url, account_address=None)

# Get wallet address from private key
from eth_account import Account
account = Account.from_key(private_key)
wallet_address = account.address
print(f"Wallet: {wallet_address}")

# Check balance
user_state = info.user_state(wallet_address)
if user_state:
    margin = user_state.get("marginSummary", {})
    balance = float(margin.get("accountValue", 0))
    print(f"Balance: ${balance:.2f} USDC")
else:
    print("Could not get user state")
    exit(1)

# Get BTC price
mids = info.all_mids()
btc_price = float(mids.get("BTC", 0))
print(f"BTC Price: ${btc_price:,.2f}")

# Test parameters
coin = "BTC"
leverage = 2
size = 0.001  # Small test size
is_buy = True

# Calculate limit price (1% above for buy)
limit_price = round(btc_price * 1.01, 1)

print()
print("-" * 60)
print("TEST: LONG Position")
print("-" * 60)

# Initialize exchange with wallet
exchange = Exchange(account, base_url, account_address=wallet_address)

# Set leverage first
print(f"Setting leverage to {leverage}x...")
try:
    result = exchange.update_leverage(leverage, coin, is_cross=True)
    print(f"Leverage result: {result}")
except Exception as e:
    print(f"Leverage error (may be OK): {e}")

# Place order
print(f"Placing LONG order: {size} BTC @ ${limit_price:,.1f}")
try:
    order_result = exchange.order(
        coin=coin,
        is_buy=is_buy,
        sz=size,
        limit_px=limit_price,
        order_type={"limit": {"tif": "Ioc"}},  # IOC for quick fill
        reduce_only=False,
    )
    print(f"Order result: {order_result}")

    if order_result.get("status") == "ok":
        response = order_result.get("response", {})
        if response.get("type") == "order":
            statuses = response.get("data", {}).get("statuses", [])
            for status in statuses:
                if "filled" in status:
                    filled = status["filled"]
                    print(f"✓ FILLED: {filled.get('totalSz')} @ ${filled.get('avgPx')}")
                elif "resting" in status:
                    print(f"Order resting: {status['resting']}")
                elif "error" in status:
                    print(f"✗ Order error: {status['error']}")
except Exception as e:
    print(f"Order error: {e}")
    import traceback
    traceback.print_exc()

# Check position
print()
print("Checking positions...")
user_state = info.user_state(wallet_address)
if user_state:
    positions = user_state.get("assetPositions", [])
    for pos in positions:
        pos_info = pos.get("position", {})
        coin_pos = pos_info.get("coin")
        size_pos = float(pos_info.get("szi", 0))
        if coin_pos == coin and abs(size_pos) > 0:
            entry = float(pos_info.get("entryPx", 0))
            print(f"Position: {size_pos} {coin_pos} @ ${entry:,.2f}")

# Close position if any
print()
print("Closing any open position...")
try:
    close_result = exchange.order(
        coin=coin,
        is_buy=False,  # Sell to close long
        sz=size,
        limit_px=round(btc_price * 0.99, 1),  # 1% below
        order_type={"limit": {"tif": "Ioc"}},
        reduce_only=True,
    )
    print(f"Close result: {close_result}")
except Exception as e:
    print(f"Close error: {e}")

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
