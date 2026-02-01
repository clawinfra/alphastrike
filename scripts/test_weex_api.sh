#!/usr/bin/env python3
"""
Weex API Live Test Script

This script implements the test cases from docs/weex_api_live_test_plan.md.
Run with: uv run scripts/run_weex_live_test.py

Test Cases:
  TC01: Connectivity & Server Time
  TC02: Market Data Retrieval
  TC03: Account Authentication
  TC04: Live Trade Execution (10 USDT BTCUSDT)
  TC05: Error Handling
  TC06: Leverage Trading
"""

import asyncio
import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings
from src.data.rest_client import RESTClient, WEEXAPIError


async def tc01_connectivity(client: RESTClient) -> bool:
    """TC01: Connectivity & Server Time"""
    print("\n" + "=" * 60)
    print("[TC01] Connectivity & Server Time")
    print("=" * 60)
    print("Objective: Verify basic network connectivity and API availability.")
    print(f"Endpoint: GET /capi/v2/market/time")
    print(f"Base URL: {client.base_url}")

    try:
        time_resp = await client.get_server_time()
        print(f"✅ SUCCESS - Server responded")
        print(f"   Server Time: {time_resp}")
        return True
    except WEEXAPIError as e:
        print(f"❌ FAILED - WEEX API Error: {e.code} - {e.message}")
        return False
    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False


async def tc02_market_data(client: RESTClient) -> bool:
    """TC02: Market Data Retrieval"""
    print("\n" + "=" * 60)
    print("[TC02] Market Data Retrieval")
    print("=" * 60)
    print("Objective: Verify parsing of public market data.")
    print("Endpoint: GET /capi/v2/market/ticker?symbol=cmt_btcusdt")

    try:
        # Use raw request without validation to see actual response
        ticker = await client._request(
            "GET",
            "/capi/v2/market/ticker",
            params={"symbol": "cmt_btcusdt"},
            authenticated=False,
            response_model=None,  # Skip validation
        )
        print(f"✅ SUCCESS - Ticker data retrieved")
        if isinstance(ticker, dict):
            print(f"   Symbol: {ticker.get('symbol', 'N/A')}")
            print(f"   Last Price: {ticker.get('last', 'N/A')}")
            print(f"   High 24h: {ticker.get('high_24h', 'N/A')}")
            print(f"   Low 24h: {ticker.get('low_24h', 'N/A')}")
            print(f"   Volume 24h: {ticker.get('volume_24h', 'N/A')}")
        return True
    except WEEXAPIError as e:
        print(f"❌ FAILED - WEEX API Error: {e.code} - {e.message}")
        return False
    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False


async def tc03_authentication(client: RESTClient) -> tuple[bool, float]:
    """TC03: Account Authentication. Returns (success, usdt_balance)."""
    print("\n" + "=" * 60)
    print("[TC03] Account Authentication")
    print("=" * 60)
    print("Objective: Verify HMAC-SHA256 signature generation is correct.")
    print("Endpoint: GET /capi/v2/account/assets")

    try:
        # Use raw request without validation to see actual response
        assets = await client._request(
            "GET",
            "/capi/v2/account/assets",
            authenticated=True,
            response_model=None,  # Skip validation
        )
        print(f"✅ SUCCESS - Authentication verified")
        
        usdt_balance = 0.0
        if isinstance(assets, list):
            print(f"   Asset count: {len(assets)}")
            for asset in assets:
                if asset.get("coinName") == "USDT":
                    usdt_balance = float(asset.get("available", 0))
                    print(f"   USDT Balance: {usdt_balance}")
                    break
        elif isinstance(assets, dict) and "data" in assets:
            data = assets["data"]
            if isinstance(data, list):
                print(f"   Asset count: {len(data)}")
                for asset in data:
                    if asset.get("coinName") == "USDT":
                        usdt_balance = float(asset.get("available", 0))
                        print(f"   USDT Balance: {usdt_balance}")
                        break
        else:
            print("   No assets found (but authentication worked).")
        
        return True, usdt_balance
    except WEEXAPIError as e:
        print(f"❌ FAILED - WEEX API Error: {e.code} - {e.message}")
        if e.code in [401, 403, 40006]:
            print("   Hint: Check ACCESS-KEY, ACCESS-SIGN, or IP whitelist.")
        return False, 0.0
    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False, 0.0


async def tc04_trade_execution(client: RESTClient, usdt_balance: float) -> bool:
    """TC04: Live Trade Execution (10 USDT BTCUSDT)"""
    print("\n" + "=" * 60)
    print("[TC04] Live Trade Execution (10 USDT BTCUSDT)")
    print("=" * 60)
    print("Objective: Execute an order with notional value of 10 USDT.")
    print("Endpoint: POST /capi/v2/order/placeOrder")

    # Check balance
    min_balance = 15.0
    if usdt_balance < min_balance:
        print(f"⚠️ SKIPPED - Insufficient balance ({usdt_balance:.2f} < {min_balance} USDT)")
        return False

    # Get current price to calculate size
    try:
        ticker = await client._request(
            "GET",
            "/capi/v2/market/ticker",
            params={"symbol": "cmt_btcusdt"},
            authenticated=False,
            response_model=None,
        )
        current_price = float(ticker.get("last", 0))
        print(f"   Current BTC Price: {current_price}")
    except Exception as e:
        print(f"❌ FAILED - Could not get ticker: {e}")
        return False

    # Calculate size for ~11 USDT (slightly over 10 to ensure we meet requirement)
    target_notional = 11.0
    size_btc = target_notional / current_price
    size_to_trade = round(size_btc, 4)
    if size_to_trade == 0:
        size_to_trade = 0.0001

    notional_value = size_to_trade * current_price
    print(f"   Calculated Size: {size_to_trade} BTC")
    print(f"   Notional Value: ~{notional_value:.2f} USDT")

    # Confirmation prompt
    print("\n⚠️ WARNING: This will execute a LIVE trade!")
    confirm = input("   Type 'yes' to proceed: ")
    if confirm.lower() != 'yes':
        print("   Trade cancelled by user.")
        return False

    try:
        import time
        client_oid = f"as_{int(time.time() * 1000)}"
        order = await client._request(
            "POST",
            "/capi/v2/order/placeOrder",
            data={
                "symbol": "cmt_btcusdt",
                "client_oid": client_oid,
                "size": str(size_to_trade),
                "type": "1",  # open_long
                "order_type": "1",  # market
                "match_price": "1",
            },
            authenticated=True,
            response_model=None,
        )
        print(f"✅ SUCCESS - Order placed!")
        if isinstance(order, dict):
            order_id = order.get("orderId") or order.get("order_id") or order.get("ordid")
            print(f"   Order ID: {order_id}")
            print(f"   Client OID: {order.get('client_oid', client_oid)}")
        return True
    except WEEXAPIError as e:
        print(f"❌ FAILED - WEEX API Error: {e.code} - {e.message}")
        return False
    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False


async def tc05_error_handling(client: RESTClient) -> bool:
    """TC05: Error Handling"""
    print("\n" + "=" * 60)
    print("[TC05] Error Handling")
    print("=" * 60)
    print("Objective: Validate system resilience to error conditions.")

    all_passed = True

    # Test 1: Invalid symbol
    print("\n[TC05.1] Invalid Symbol Test")
    try:
        await client.get_ticker("INVALID_SYMBOL_XYZ")
        print("   ⚠️ UNEXPECTED - No error for invalid symbol")
    except WEEXAPIError as e:
        print(f"   ✅ Correctly raised WEEXAPIError: {e.code}")
    except Exception as e:
        print(f"   ✅ Correctly raised error: {type(e).__name__}")

    # Test 2: Invalid order (missing required fields via direct request)
    print("\n[TC05.2] Invalid Order Parameters Test")
    try:
        # Attempt to place an order with invalid data
        await client.place_order(
            symbol="cmt_btcusdt",
            side="invalid_side",  # Invalid side
            order_type="market",
            size=0.0001
        )
        print("   ⚠️ UNEXPECTED - No error for invalid side")
        all_passed = False
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {e}")
    except WEEXAPIError as e:
        print(f"   ✅ Correctly raised WEEXAPIError: {e.code}")
    except Exception as e:
        print(f"   ✅ Correctly raised error: {type(e).__name__}")

    return all_passed


async def tc06_leverage_trading(client: RESTClient) -> bool:
    """TC06: Leverage Trading"""
    print("\n" + "=" * 60)
    print("[TC06] Leverage Trading")
    print("=" * 60)
    print("Objective: Set leverage for a trading pair.")
    print("Endpoint: POST /capi/v2/account/leverage")

    try:
        leverage = await client._request(
            "POST",
            "/capi/v2/account/leverage",
            data={
                "symbol": "cmt_btcusdt",
                "marginMode": 1,  # 1 = Cross margin
                "longLeverage": "5",
                "shortLeverage": "5",
            },
            authenticated=True,
            response_model=None,
        )
        print(f"✅ SUCCESS - Leverage set!")
        if isinstance(leverage, dict):
            print(f"   Symbol: cmt_btcusdt")
            print(f"   Margin Mode: Cross (1)")
            print(f"   Long Leverage: 5x")
            print(f"   Short Leverage: 5x")
            print(f"   Response: {leverage}")
        return True
    except WEEXAPIError as e:
        # Handle code 200 as success (WEEX uses HTTP-style codes for some endpoints)
        if e.code == 200 or e.message == "success":
            print(f"✅ SUCCESS - Leverage set!")
            print(f"   Symbol: cmt_btcusdt")
            print(f"   Margin Mode: Cross (1)")
            print(f"   Long/Short Leverage: 5x")
            return True
        print(f"❌ FAILED - WEEX API Error: {e.code} - {e.message}")
        return False
    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False


async def run_live_test():
    """Main test runner."""
    print("=" * 60)
    print("WEEX API LIVE TEST")
    print("=" * 60)
    print(f"Base URL: {settings.weex_base_url}")
    print(f"Trading Mode: {settings.trading_mode}")
    print(f"Max Leverage: {settings.max_leverage}")

    client = RESTClient()

    results = {
        "TC01": False,
        "TC02": False,
        "TC03": False,
        "TC04": False,
        "TC05": False,
        "TC06": False,
    }

    try:
        # TC01: Connectivity
        results["TC01"] = await tc01_connectivity(client)
        if not results["TC01"]:
            print("\n⛔ Stopping tests - connectivity failed.")
            return results

        # TC02: Market Data
        results["TC02"] = await tc02_market_data(client)

        # TC03: Authentication
        auth_success, usdt_balance = await tc03_authentication(client)
        results["TC03"] = auth_success
        if not auth_success:
            print("\n⛔ Stopping tests - authentication failed.")
            return results

        # TC04: Trade Execution
        results["TC04"] = await tc04_trade_execution(client, usdt_balance)

        # TC05: Error Handling
        results["TC05"] = await tc05_error_handling(client)

        # TC06: Leverage Trading
        results["TC06"] = await tc06_leverage_trading(client)

    finally:
        await client.close()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for tc, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {tc}: {status}")

    return results


if __name__ == "__main__":
    asyncio.run(run_live_test())
