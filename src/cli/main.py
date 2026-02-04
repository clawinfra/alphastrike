"""
AlphaStrike CLI - Main Entry Point

Unified CLI for the AlphaStrike trading bot supporting both CEX and DEX exchanges.

Usage:
    alphastrike trade --exchange hyperliquid --testnet
    alphastrike backtest --exchange hyperliquid --days 180
    alphastrike status --exchange hyperliquid
    alphastrike train --exchange hyperliquid
    alphastrike config show
"""

from __future__ import annotations

import click

from src.cli.commands import backtest, config, status, trade, train


@click.group()
@click.version_option(version="2.0.0", prog_name="alphastrike")
def main() -> None:
    """
    AlphaStrike - Autonomous ML-based crypto trading system.

    Supports both CEX (WEEX) and DEX (Hyperliquid) exchanges with
    unified interface for trading, backtesting, and model training.

    \b
    Examples:
      alphastrike trade --exchange hyperliquid --testnet
      alphastrike backtest --exchange hyperliquid --days 180
      alphastrike status --exchange hyperliquid --json
    """
    pass


# Register commands
main.add_command(trade.trade)
main.add_command(backtest.backtest)
main.add_command(train.train)
main.add_command(status.status)
main.add_command(config.config)


if __name__ == "__main__":
    main()
