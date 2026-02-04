#!/usr/bin/env python3
"""
Hyperliquid Testnet Paper Trading - Medallion V2 Strategy

Runs the proven Medallion V2 strategy (67.5% CAGR, 3.9% DD, 3.67 Sharpe)
on Hyperliquid testnet for paper trading validation.

Usage:
    python scripts/hyperliquid_testnet_trader.py
    python scripts/hyperliquid_testnet_trader.py --leverage 5
    python scripts/hyperliquid_testnet_trader.py --dry-run  # Log signals, no orders

Requirements:
    - EXCHANGE_WALLET_PRIVATE_KEY environment variable set
    - Testnet USDC from: https://app.hyperliquid-testnet.xyz/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.medallion_live import MedallionLiveConfig, MedallionLiveEngine


def setup_logging(log_level: str) -> None:
    """Configure logging with structured output."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

    # File handler for trade logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "medallion_live.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Medallion V2 Paper Trading on Hyperliquid Testnet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/hyperliquid_testnet_trader.py                  # Start with defaults
  python scripts/hyperliquid_testnet_trader.py --base-leverage 3  # Use 3x base leverage
  python scripts/hyperliquid_testnet_trader.py --dry-run        # Signals only, no orders
  python scripts/hyperliquid_testnet_trader.py --log-level DEBUG  # Verbose logging

Strategy Settings:
  - 15 assets: BTC, ETH, BNB, XRP, SOL, AVAX, NEAR, APT, AAVE, UNI, LINK, FET, DOGE, PAXG, SPX
  - BULLISH regime only (60%+ confidence)
  - ML tiers: 70+ (Tier 1), 65+ with momentum (Tier 2), 65+ with uptrend (Tier 3)
  - Position: 5% max per asset, 40% max portfolio exposure
  - Exits: 1% stop loss, 4% take profit, 36h time limit
        """,
    )

    parser.add_argument(
        "--base-leverage",
        type=float,
        default=5.0,
        help="Base leverage for dynamic calculation (default: 5.0)",
    )

    parser.add_argument(
        "--min-leverage",
        type=float,
        default=1.0,
        help="Minimum leverage in high-risk conditions (default: 1.0)",
    )

    parser.add_argument(
        "--max-leverage",
        type=float,
        default=10.0,
        help="Maximum leverage in favorable conditions (default: 10.0)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals without placing orders",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="USE WITH CAUTION: Trade on mainnet instead of testnet",
    )

    return parser.parse_args()


def print_banner(config: MedallionLiveConfig, testnet: bool, dry_run: bool) -> None:
    """Print startup banner with configuration."""
    mode = "DRY RUN" if dry_run else "PAPER TRADING"
    network = "TESTNET" if testnet else "⚠️  MAINNET ⚠️"

    leverage_str = f"{config.base_leverage:.0f}x (dynamic: {config.min_leverage:.0f}x-{config.max_leverage:.0f}x)"

    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MEDALLION V2 - HYPERLIQUID {mode:<12}                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Network:     {network:<62} ║
║  Leverage:    {leverage_str:<62}║
║  Assets:      {len(config.assets)} ({', '.join(config.assets[:6])}...){' ' * 26}║
║  Regime:      {config.regime_required} @ {config.regime_confidence_min:.0f}%+ confidence{' ' * 30}║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Dynamic Leverage: auto-adjusts based on volatility, drawdown, win rate{' ' * 5}║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Risk Management:{' ' * 58}║
║    Stop Loss:    {config.stop_loss_pct*100:.0f}%{' ' * 59}║
║    Take Profit:  {config.take_profit_pct*100:.0f}%{' ' * 59}║
║    Time Exit:    {config.time_exit_hours} hours{' ' * 53}║
║    Max Position: {config.max_single_position*100:.0f}% per asset{' ' * 47}║
║    Max Exposure: {config.max_portfolio_exposure*100:.0f}% total{' ' * 50}║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Strategy: LightGBM-only (67.5% CAGR, 3.9% DD, 3.67 Sharpe){' ' * 17}║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main() -> int:
    """Main entry point for the paper trader."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # Validate mainnet usage
    if args.mainnet:
        confirm = input("⚠️  WARNING: You selected MAINNET. Type 'CONFIRM' to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("Aborted. Use --dry-run for safe testing.")
            return 1

    # Create configuration
    config = MedallionLiveConfig(
        base_leverage=args.base_leverage,
        min_leverage=args.min_leverage,
        max_leverage=args.max_leverage,
    )

    testnet = not args.mainnet
    print_banner(config, testnet, args.dry_run)

    # Create engine
    engine = MedallionLiveEngine(
        config=config,
        testnet=testnet,
        dry_run=args.dry_run,
    )

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_requested = False

    def signal_handler() -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown requested, exiting immediately...")
            sys.exit(1)
        shutdown_requested = True
        logger.info("Shutdown signal received, stopping gracefully...")
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Initialize and start
        logger.info("Initializing trading engine...")
        await engine.initialize()

        logger.info("Starting trading engine... (Ctrl+C to stop)")
        await engine.start()

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1

    finally:
        # Ensure cleanup
        if engine._running:
            await engine.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
