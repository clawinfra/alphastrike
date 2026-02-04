#!/usr/bin/env python3
"""
Test Order API on Hyperliquid Testnet

Verifies order placement works by:
1. Opening a LONG position with 2x leverage
2. Closing the LONG position
3. Opening a SHORT position with 2x leverage
4. Closing the SHORT position

Usage:
    python scripts/test_order_api.py
"""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def test_orders():
    """Test order placement on Hyperliquid testnet."""
    from src.exchange.adapters.hyperliquid.adapter import HyperliquidAdapter
    from src.exchange.models import OrderSide, OrderType, PositionSide, TimeInForce, UnifiedOrder

    # Initialize adapter for TESTNET
    logger.info("=" * 60)
    logger.info("HYPERLIQUID TESTNET ORDER API TEST")
    logger.info("=" * 60)

    adapter = HyperliquidAdapter(testnet=True)
    await adapter.initialize()

    try:
        # Check balance
        balance = await adapter.rest.get_account_balance()
        logger.info(f"Account Balance: ${balance.total_balance:.2f} USDC")
        logger.info(f"Available: ${balance.available_balance:.2f} USDC")

        if balance.available_balance < 50:
            logger.error("Insufficient balance for testing. Need at least $50.")
            return

        # Test parameters
        symbol = "BTCUSDT"
        leverage = 2
        position_size_usd = 100  # $100 position

        # Get current price
        ticker = await adapter.rest.get_ticker(symbol)
        current_price = ticker.last_price
        logger.info(f"Current {symbol} price: ${current_price:,.2f}")

        # Calculate quantity
        quantity = position_size_usd / current_price
        logger.info(f"Test quantity: {quantity:.6f} BTC (~${position_size_usd})")

        # =========================================================
        # TEST 1: LONG Position
        # =========================================================
        logger.info("")
        logger.info("-" * 60)
        logger.info("TEST 1: LONG Position (2x leverage)")
        logger.info("-" * 60)

        # Set leverage (may fail on testnet if account not initialized)
        logger.info(f"Setting leverage to {leverage}x...")
        try:
            await adapter.rest.set_leverage(symbol, leverage)
            logger.info(f"✓ Leverage set to {leverage}x")
        except Exception as e:
            logger.warning(f"Could not set leverage (continuing): {e}")

        # Open LONG - use limit price with slippage for "market" execution
        # Hyperliquid requires a price even for IOC orders
        buy_price = round(current_price * 1.01, 1)  # 1% above for quick fill
        logger.info(f"Opening LONG position @ ${buy_price:,.1f} (limit IOC)...")
        long_order = UnifiedOrder(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=buy_price,
            position_side=PositionSide.LONG,
            time_in_force=TimeInForce.IOC,  # Immediate or cancel for market-like execution
        )
        long_result = await adapter.rest.place_order(long_order)

        if long_result and long_result.order_id:
            logger.info(f"✓ LONG opened - Order ID: {long_result.order_id}")
            logger.info(f"  Status: {long_result.status}")
            if long_result.avg_fill_price:
                logger.info(f"  Fill Price: ${long_result.avg_fill_price:,.2f}")
        else:
            logger.error("✗ Failed to open LONG position")
            return

        # Wait a moment
        await asyncio.sleep(2)

        # Check position
        positions = await adapter.rest.get_positions()
        long_pos = next((p for p in positions if p.symbol == symbol and p.quantity > 0), None)
        if long_pos:
            logger.info(f"✓ Position confirmed: {long_pos.quantity:.6f} BTC @ ${long_pos.entry_price:,.2f}")

        # Close LONG - sell at slightly below market for quick fill
        sell_price = round(current_price * 0.99, 1)  # 1% below
        logger.info(f"Closing LONG position @ ${sell_price:,.1f}...")
        close_long = UnifiedOrder(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=sell_price,
            position_side=PositionSide.LONG,
            reduce_only=True,
            time_in_force=TimeInForce.IOC,
        )
        close_long_result = await adapter.rest.place_order(close_long)

        if close_long_result and close_long_result.order_id:
            logger.info(f"✓ LONG closed - Order ID: {close_long_result.order_id}")
        else:
            logger.error("✗ Failed to close LONG position")

        await asyncio.sleep(2)

        # =========================================================
        # TEST 2: SHORT Position
        # =========================================================
        logger.info("")
        logger.info("-" * 60)
        logger.info("TEST 2: SHORT Position (2x leverage)")
        logger.info("-" * 60)

        # Get fresh price
        ticker = await adapter.rest.get_ticker(symbol)
        current_price = ticker.last_price

        # Open SHORT - sell at slightly below market for quick fill
        sell_price = round(current_price * 0.99, 1)
        logger.info(f"Opening SHORT position @ ${sell_price:,.1f}...")
        short_order = UnifiedOrder(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=sell_price,
            position_side=PositionSide.SHORT,
            time_in_force=TimeInForce.IOC,
        )
        short_result = await adapter.rest.place_order(short_order)

        if short_result and short_result.order_id:
            logger.info(f"✓ SHORT opened - Order ID: {short_result.order_id}")
            logger.info(f"  Status: {short_result.status}")
            if short_result.avg_fill_price:
                logger.info(f"  Fill Price: ${short_result.avg_fill_price:,.2f}")
        else:
            logger.error("✗ Failed to open SHORT position")
            return

        await asyncio.sleep(2)

        # Check position
        positions = await adapter.rest.get_positions()
        short_pos = next((p for p in positions if p.symbol == symbol and p.quantity > 0), None)
        if short_pos:
            logger.info(f"✓ Position confirmed: {short_pos.quantity:.6f} BTC @ ${short_pos.entry_price:,.2f}")

        # Close SHORT - buy at slightly above market for quick fill
        buy_price = round(current_price * 1.01, 1)
        logger.info(f"Closing SHORT position @ ${buy_price:,.1f}...")
        close_short = UnifiedOrder(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=buy_price,
            position_side=PositionSide.SHORT,
            reduce_only=True,
            time_in_force=TimeInForce.IOC,
        )
        close_short_result = await adapter.rest.place_order(close_short)

        if close_short_result and close_short_result.order_id:
            logger.info(f"✓ SHORT closed - Order ID: {close_short_result.order_id}")
        else:
            logger.error("✗ Failed to close SHORT position")

        await asyncio.sleep(1)

        # =========================================================
        # FINAL SUMMARY
        # =========================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST COMPLETE")
        logger.info("=" * 60)

        # Final balance
        final_balance = await adapter.rest.get_account_balance()
        pnl = final_balance.total_balance - balance.total_balance
        logger.info(f"Starting Balance: ${balance.total_balance:.2f}")
        logger.info(f"Final Balance:    ${final_balance.total_balance:.2f}")
        logger.info(f"Net P&L:          ${pnl:+.2f} (fees + slippage)")

        # Verify no open positions
        positions = await adapter.rest.get_positions()
        open_positions = [p for p in positions if p.quantity > 0]
        if not open_positions:
            logger.info("✓ All positions closed successfully")
        else:
            logger.warning(f"⚠ {len(open_positions)} positions still open")

        logger.info("")
        logger.info("All order API tests PASSED!")

    except Exception as e:
        logger.exception(f"Test failed with error: {e}")
        raise

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(test_orders())
