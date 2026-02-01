#!/usr/bin/env python3
"""
Standalone WEEX API Connectivity Test

Tests the WEEX API using the documented v2 endpoints from docs/weex_api.json.
Does not depend on the full config system - uses credentials directly.

Usage:
    python scripts/test_weex_connectivity.py
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

import aiohttp

# WEEX API Credentials (from environment or hardcoded for testing)
API_KEY = "weex_b312cd202f9e97dde056693413959964"
API_SECRET = "8c83020575dfe348749b3269898b37b4ff03ce511413a69577817dd07c8b254d"
PASSPHRASE = "weex89769876976"
BASE_URL = "https://api-contract.weex.com"


@dataclass
class TestResult:
    """Result of a test case."""
    name: str
    passed: bool
    message: str
    data: dict | None = None


def generate_signature(secret: str, timestamp: str, method: str, path: str, body: str = "") -> str:
    """
    Generate HMAC-SHA256 signature for WEEX API.

    Signature format: base64(hmac_sha256(timestamp + method + path + body))
    """
    message = f"{timestamp}{method.upper()}{path}{body}"
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(signature).decode("utf-8")


def get_auth_headers(method: str, path: str, body: str = "") -> dict[str, str]:
    """Generate authenticated headers for WEEX API request."""
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(API_SECRET, timestamp, method, path, body)

    return {
        "Content-Type": "application/json",
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "locale": "en-US",
    }


async def test_server_time(session: aiohttp.ClientSession) -> TestResult:
    """TC01: Test server time endpoint (no auth required)."""
    url = f"{BASE_URL}/capi/v2/market/time"

    try:
        async with session.get(url) as response:
            text = await response.text()

            if response.status != 200:
                return TestResult(
                    name="Server Time",
                    passed=False,
                    message=f"HTTP {response.status}: {text}",
                )

            data = json.loads(text)

            # WEEX wraps responses
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            return TestResult(
                name="Server Time",
                passed=True,
                message=f"Server time: {data.get('timestamp', data)}",
                data=data,
            )

    except Exception as e:
        return TestResult(
            name="Server Time",
            passed=False,
            message=f"Error: {type(e).__name__}: {e}",
        )


async def test_ticker(session: aiohttp.ClientSession) -> TestResult:
    """TC02: Test ticker endpoint (no auth required)."""
    url = f"{BASE_URL}/capi/v2/market/ticker"
    params = {"symbol": "cmt_btcusdt"}

    try:
        async with session.get(url, params=params) as response:
            text = await response.text()

            if response.status != 200:
                return TestResult(
                    name="BTC Ticker",
                    passed=False,
                    message=f"HTTP {response.status}: {text}",
                )

            data = json.loads(text)

            # WEEX wraps responses
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            if isinstance(data, dict):
                price = data.get("last", "N/A")
                return TestResult(
                    name="BTC Ticker",
                    passed=True,
                    message=f"BTC/USDT: ${price}",
                    data=data,
                )

            return TestResult(
                name="BTC Ticker",
                passed=True,
                message=f"Response: {data}",
                data=data if isinstance(data, dict) else {"raw": data},
            )

    except Exception as e:
        return TestResult(
            name="BTC Ticker",
            passed=False,
            message=f"Error: {type(e).__name__}: {e}",
        )


async def test_candles(session: aiohttp.ClientSession) -> TestResult:
    """TC03: Test candle data endpoint (no auth required)."""
    url = f"{BASE_URL}/capi/v2/market/candles"
    params = {
        "symbol": "cmt_btcusdt",
        "granularity": "1m",
        "limit": "5",
    }

    try:
        async with session.get(url, params=params) as response:
            text = await response.text()

            if response.status != 200:
                return TestResult(
                    name="Candle Data",
                    passed=False,
                    message=f"HTTP {response.status}: {text}",
                )

            data = json.loads(text)

            # WEEX wraps responses
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            if isinstance(data, list) and len(data) > 0:
                return TestResult(
                    name="Candle Data",
                    passed=True,
                    message=f"Retrieved {len(data)} candles",
                    data={"candles": data[:2]},  # First 2 for display
                )

            return TestResult(
                name="Candle Data",
                passed=True,
                message=f"Response: {data}",
                data=data if isinstance(data, dict) else {"raw": data},
            )

    except Exception as e:
        return TestResult(
            name="Candle Data",
            passed=False,
            message=f"Error: {type(e).__name__}: {e}",
        )


async def test_authentication(session: aiohttp.ClientSession) -> TestResult:
    """TC04: Test authenticated endpoint (get account assets)."""
    path = "/capi/v2/account/assets"
    url = f"{BASE_URL}{path}"
    headers = get_auth_headers("GET", path)

    try:
        async with session.get(url, headers=headers) as response:
            text = await response.text()

            if response.status == 401:
                return TestResult(
                    name="Authentication",
                    passed=False,
                    message="401 Unauthorized - Invalid credentials or signature",
                )

            if response.status == 403:
                return TestResult(
                    name="Authentication",
                    passed=False,
                    message="403 Forbidden - IP not whitelisted or permissions issue",
                )

            if response.status != 200:
                return TestResult(
                    name="Authentication",
                    passed=False,
                    message=f"HTTP {response.status}: {text[:200]}",
                )

            data = json.loads(text)

            # Check WEEX response code
            if isinstance(data, dict):
                code = data.get("code", "")
                if code != "00000":
                    return TestResult(
                        name="Authentication",
                        passed=False,
                        message=f"API Error: {code} - {data.get('msg', 'Unknown')}",
                        data=data,
                    )

                assets = data.get("data", [])
                if isinstance(assets, list):
                    usdt_balance = 0.0
                    for asset in assets:
                        if asset.get("coinName") == "USDT":
                            usdt_balance = float(asset.get("available", 0))
                            break

                    return TestResult(
                        name="Authentication",
                        passed=True,
                        message=f"Authenticated! USDT Balance: {usdt_balance:.2f}",
                        data={"usdt_balance": usdt_balance, "assets": assets},
                    )

            return TestResult(
                name="Authentication",
                passed=True,
                message=f"Response: {text[:200]}",
                data=data if isinstance(data, dict) else {"raw": text[:200]},
            )

    except Exception as e:
        return TestResult(
            name="Authentication",
            passed=False,
            message=f"Error: {type(e).__name__}: {e}",
        )


async def test_contracts(session: aiohttp.ClientSession) -> TestResult:
    """TC05: Test contracts/symbols endpoint (no auth required)."""
    url = f"{BASE_URL}/capi/v2/market/contracts"

    try:
        async with session.get(url) as response:
            text = await response.text()

            if response.status != 200:
                return TestResult(
                    name="Contracts Info",
                    passed=False,
                    message=f"HTTP {response.status}: {text[:200]}",
                )

            data = json.loads(text)

            # WEEX wraps responses
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            if isinstance(data, list):
                # Find BTC contract
                btc_contract = None
                for contract in data:
                    if contract.get("symbol") == "cmt_btcusdt":
                        btc_contract = contract
                        break

                if btc_contract:
                    return TestResult(
                        name="Contracts Info",
                        passed=True,
                        message=f"Found {len(data)} contracts. BTC max leverage: {btc_contract.get('maxLeverage', 'N/A')}x",
                        data=btc_contract,
                    )

                return TestResult(
                    name="Contracts Info",
                    passed=True,
                    message=f"Found {len(data)} contracts",
                    data={"count": len(data)},
                )

            return TestResult(
                name="Contracts Info",
                passed=True,
                message=f"Response: {text[:200]}",
                data=data if isinstance(data, dict) else {"raw": text[:200]},
            )

    except Exception as e:
        return TestResult(
            name="Contracts Info",
            passed=False,
            message=f"Error: {type(e).__name__}: {e}",
        )


async def main():
    """Run all connectivity tests."""
    print("=" * 60)
    print("WEEX API Connectivity Test")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY[:20]}...")
    print()

    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Run tests
        tests = [
            ("TC01", test_server_time),
            ("TC02", test_ticker),
            ("TC03", test_candles),
            ("TC04", test_authentication),
            ("TC05", test_contracts),
        ]

        results = []

        for tc_id, test_fn in tests:
            print(f"\n[{tc_id}] {test_fn.__doc__.strip().split(chr(10))[0]}")
            print("-" * 50)

            result = await test_fn(session)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"Result: {status}")
            print(f"Message: {result.message}")

            if result.data and result.passed:
                print(f"Data: {json.dumps(result.data, indent=2)[:500]}")

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}: {result.message[:60]}")

        print()
        print(f"Passed: {passed}/{total}")

        if passed == total:
            print("\nAll tests passed! WEEX API is accessible.")
        else:
            print("\nSome tests failed. Check the error messages above.")

        return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
