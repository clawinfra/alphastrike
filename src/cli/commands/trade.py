"""Trade command - Start live/paper trading."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import click

from src.cli.utils import (
    DEFAULT_ASSETS,
    console,
    print_banner,
    print_error,
    setup_logging,
)


@click.command()
@click.option(
    "--exchange", "-e",
    type=click.Choice(["hyperliquid", "weex"], case_sensitive=False),
    required=True,
    help="Exchange to trade on",
)
@click.option(
    "--testnet",
    is_flag=True,
    default=False,
    help="Use testnet (default for DEX)",
)
@click.option(
    "--mainnet",
    is_flag=True,
    default=False,
    help="Use mainnet (requires confirmation)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Log signals without placing orders",
)
@click.option(
    "--base-leverage",
    type=float,
    default=5.0,
    help="Base leverage (default: 5.0)",
)
@click.option(
    "--min-leverage",
    type=float,
    default=1.0,
    help="Minimum leverage (default: 1.0)",
)
@click.option(
    "--max-leverage",
    type=float,
    default=10.0,
    help="Maximum leverage (default: 10.0)",
)
@click.option(
    "--assets",
    type=str,
    default=None,
    help="Comma-separated assets (default: all)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging verbosity",
)
@click.option(
    "--db",
    "db_backend",
    type=click.Choice(["jsonl", "sqlite"], case_sensitive=False),
    default="jsonl",
    help="Trade storage backend: jsonl (default) or sqlite",
)
def trade(
    exchange: str,
    testnet: bool,
    mainnet: bool,
    dry_run: bool,
    base_leverage: float,
    min_leverage: float,
    max_leverage: float,
    assets: str | None,
    log_level: str,
    db_backend: str,
) -> None:
    """Start live or paper trading."""
    setup_logging(log_level)

    # Determine network
    if mainnet and testnet:
        print_error("Cannot specify both --mainnet and --testnet")
        sys.exit(1)

    # Default to testnet for DEX
    if exchange == "hyperliquid" and not mainnet and not testnet:
        testnet = True

    # Mainnet confirmation for DEX (skip if dry-run since no real orders)
    if exchange == "hyperliquid" and mainnet and not dry_run:
        confirm = click.prompt(
            click.style("WARNING: You selected MAINNET. Type 'CONFIRM' to proceed", fg="yellow"),
            default="",
        )
        if confirm.strip() != "CONFIRM":
            console.print("[yellow]Aborted. Use --testnet for safe paper trading.[/yellow]")
            sys.exit(0)
    elif exchange == "hyperliquid" and mainnet and dry_run:
        console.print("[yellow]MAINNET DRY-RUN: No real orders will be placed[/yellow]")

    # Parse assets
    asset_list = [a.strip().upper() for a in assets.split(",")] if assets else None

    # Print banner
    network = "TESTNET" if testnet else "MAINNET"
    mode = "DRY RUN" if dry_run else "LIVE TRADING"

    print_banner(
        f"ALPHASTRIKE - {exchange.upper()} {mode}",
        subtitle=f"Network: {network}",
        items={
            "Leverage": f"{base_leverage}x (dynamic: {min_leverage}x-{max_leverage}x)",
            "Assets": ", ".join(asset_list[:5]) + "..." if asset_list else f"All {len(DEFAULT_ASSETS)} assets",
            "Mode": mode,
            "Storage": db_backend.upper(),
        },
    )

    # Run the trading engine
    if exchange == "hyperliquid":
        asyncio.run(_run_hyperliquid_trader(
            testnet=testnet,
            dry_run=dry_run,
            base_leverage=base_leverage,
            min_leverage=min_leverage,
            max_leverage=max_leverage,
            assets=asset_list,
            db_backend=db_backend,
        ))
    elif exchange == "weex":
        print_error("WEEX trading not yet implemented in CLI. Use main.py for now.")
        sys.exit(1)


async def _run_hyperliquid_trader(
    testnet: bool,
    dry_run: bool,
    base_leverage: float,
    min_leverage: float,
    max_leverage: float,
    assets: list[str] | None,
    db_backend: str,
) -> None:
    """Run the Hyperliquid trading engine."""
    from typing import Literal

    from src.trading.medallion_live import MedallionLiveConfig, MedallionLiveEngine

    logger = logging.getLogger(__name__)

    # Build config
    config = MedallionLiveConfig(
        base_leverage=base_leverage,
        min_leverage=min_leverage,
        max_leverage=max_leverage,
    )
    if assets:
        config.assets = assets

    # Create engine with selected storage backend
    backend: Literal["jsonl", "sqlite"] = "sqlite" if db_backend == "sqlite" else "jsonl"
    engine = MedallionLiveEngine(
        config=config,
        testnet=testnet,
        dry_run=dry_run,
        db_backend=backend,
    )

    # Set up signal handlers
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
        logger.info("Initializing trading engine...")
        await engine.initialize()

        logger.info("Starting trading engine... (Ctrl+C to stop)")
        await engine.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)

    finally:
        if engine.is_running:
            await engine.stop()
