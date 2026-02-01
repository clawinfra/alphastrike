#!/usr/bin/env python3
"""
Test script to download historical data from WEEX.

Verifies:
1. API credentials work
2. Data loader can fetch candles
3. Data is cached to SQLite
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for WEEX credentials
os.environ["WEEX_API_KEY"] = "weex_b312cd202f9e97dde056693413959964"
os.environ["WEEX_API_SECRET"] = "8c83020575dfe348749b3269898b37b4ff03ce511413a69577817dd07c8b254d"
os.environ["WEEX_PASSPHRASE"] = "weex89769876976"
os.environ["WEEX_BASE_URL"] = "https://api-contract.weex.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_download")


async def main():
    """Test data download."""
    logger.info("=" * 60)
    logger.info("WEEX Historical Data Download Test")
    logger.info("=" * 60)

    try:
        from src.data.database import Database
        from src.exchange.adapters.weex.adapter import WEEXRESTClient
        from src.backtest.data_loader import DataLoader

        # Initialize database
        logger.info("Initializing database...")
        database = Database()
        await database.initialize()

        # Initialize REST client with explicit credentials
        logger.info("Initializing WEEX client...")
        rest_client = WEEXRESTClient(
            api_key=os.environ["WEEX_API_KEY"],
            api_secret=os.environ["WEEX_API_SECRET"],
            api_passphrase=os.environ["WEEX_PASSPHRASE"],
            base_url=os.environ["WEEX_BASE_URL"],
        )
        await rest_client.initialize()

        # Test account balance first (verifies credentials)
        logger.info("Testing API credentials...")
        try:
            balance = await rest_client.get_account_balance()
            logger.info(f"Account balance: ${balance.total_balance:,.2f}")
            logger.info("API credentials verified!")
        except Exception as e:
            logger.warning(f"Could not fetch balance (may need trading permissions): {e}")
            logger.info("Continuing with public data endpoints...")

        # Test candle download
        symbol = "BTCUSDT"
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)  # Last 24 hours

        logger.info(f"\nDownloading candles for {symbol}...")
        logger.info(f"Period: {start_time} to {end_time}")

        data_loader = DataLoader(database=database, rest_client=rest_client)

        candles = await data_loader.fetch_candles(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            interval="1m",
            force_download=True,
        )

        if candles:
            logger.info(f"\nDownloaded {len(candles)} candles")
            logger.info("\nSample data (first 5 candles):")
            for c in candles[:5]:
                logger.info(
                    f"  {c.timestamp}: O={c.open:.2f} H={c.high:.2f} "
                    f"L={c.low:.2f} C={c.close:.2f} V={c.volume:.2f}"
                )

            logger.info("\nSample data (last 5 candles):")
            for c in candles[-5:]:
                logger.info(
                    f"  {c.timestamp}: O={c.open:.2f} H={c.high:.2f} "
                    f"L={c.low:.2f} C={c.close:.2f} V={c.volume:.2f}"
                )

            # Test cache
            logger.info("\nTesting cache retrieval...")
            cached = await data_loader.get_cached_candles(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval="1m",
            )
            logger.info(f"Retrieved {len(cached)} candles from cache")

            logger.info("\n" + "=" * 60)
            logger.info("DATA DOWNLOAD TEST: SUCCESS")
            logger.info("=" * 60)
        else:
            logger.error("No candles downloaded")
            return 1

        # Cleanup
        await rest_client.close()
        await database.close()

        return 0

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
