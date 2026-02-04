"""Status command - Check trader status, positions, P&L."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import click

from src.cli.utils import console, print_error, print_status_box


@click.command()
@click.option(
    "--exchange", "-e",
    type=click.Choice(["hyperliquid", "weex"], case_sensitive=False),
    required=True,
    help="Exchange to check status on",
)
@click.option(
    "--testnet",
    is_flag=True,
    default=False,
    help="Check testnet (default for DEX)",
)
@click.option(
    "--mainnet",
    is_flag=True,
    default=False,
    help="Check mainnet",
)
@click.option(
    "--watch", "-w",
    is_flag=True,
    default=False,
    help="Continuous updates (refresh every 5s)",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output as JSON",
)
def status(
    exchange: str,
    testnet: bool,
    mainnet: bool,
    watch: bool,
    json_output: bool,
) -> None:
    """Check trader status, positions, and P&L."""
    # Determine network
    if mainnet and testnet:
        print_error("Cannot specify both --mainnet and --testnet")
        sys.exit(1)

    # Default to testnet for DEX
    if exchange == "hyperliquid" and not mainnet and not testnet:
        testnet = True

    if exchange == "hyperliquid":
        if watch:
            try:
                while True:
                    try:
                        asyncio.run(_show_hyperliquid_status(testnet, json_output))
                    except Exception as e:
                        print_error(f"Failed to fetch status: {e}")
                    if not json_output:
                        console.print("\n[dim]Refreshing in 5s... (Ctrl+C to stop)[/dim]")
                    time.sleep(5)
                    if not json_output:
                        console.clear()
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped watching[/yellow]")
        else:
            try:
                asyncio.run(_show_hyperliquid_status(testnet, json_output))
            except Exception as e:
                print_error(f"Failed to fetch status: {e}")
                sys.exit(1)
    elif exchange == "weex":
        print_error("WEEX status not yet implemented in CLI.")
        sys.exit(1)


async def _show_hyperliquid_status(testnet: bool, json_output: bool) -> None:
    """Show Hyperliquid account status."""
    from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient

    client = HyperliquidRESTClient(testnet=testnet)
    await client.initialize()

    try:
        # Get account info
        balance_info = await client.get_account_balance()
        positions = await client.get_positions()

        # Filter open positions
        open_positions = [p for p in positions if abs(p.quantity) > 0]

        # Calculate totals
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)
        total_exposure = sum(abs(p.notional_value) for p in open_positions)

        status_data = {
            "network": "testnet" if testnet else "mainnet",
            "balance": {
                "total": balance_info.total_balance,
                "available": balance_info.available_balance,
                "unrealized_pnl": total_unrealized,
            },
            "positions": {
                "count": len(open_positions),
                "total_exposure": total_exposure,
                "details": [
                    {
                        "symbol": p.symbol,
                        "side": "LONG" if p.quantity > 0 else "SHORT",
                        "size": abs(p.quantity),
                        "entry_price": p.entry_price,
                        "mark_price": p.mark_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "pnl_pct": (p.mark_price - p.entry_price) / p.entry_price * 100 if p.entry_price else 0,
                    }
                    for p in open_positions
                ],
            },
        }

        if json_output:
            console.print_json(json.dumps(status_data, indent=2, default=str))
        else:
            network_str = "Testnet" if testnet else "Mainnet"

            # Build sections
            sections = [
                {
                    "header": "Account",
                    "items": {
                        "Balance": f"${balance_info.total_balance:,.2f} USDC",
                        "Available": f"${balance_info.available_balance:,.2f}",
                        "Unrealized P&L": f"${total_unrealized:+,.2f}",
                    },
                },
            ]

            if open_positions:
                position_items = {}
                for p in open_positions:
                    side = "LONG" if p.quantity > 0 else "SHORT"
                    pnl_pct = (p.mark_price - p.entry_price) / p.entry_price * 100 if p.entry_price else 0
                    color = "green" if p.unrealized_pnl >= 0 else "red"
                    position_items[f"{p.symbol} {side}"] = f"[{color}]{pnl_pct:+.2f}%[/{color}] (${p.unrealized_pnl:+,.2f})"

                sections.append({
                    "header": f"Open Positions ({len(open_positions)})",
                    "items": position_items,
                })
            else:
                sections.append({
                    "header": "Positions",
                    "items": {"Status": "No open positions"},
                })

            print_status_box(f"ALPHASTRIKE - Hyperliquid {network_str}", sections)

    finally:
        await client.close()
