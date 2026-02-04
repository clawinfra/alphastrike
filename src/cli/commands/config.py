"""Config command - View and manage configuration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click

from src.cli.utils.output import console, print_banner, print_error, print_success


@click.group()
def config() -> None:
    """View and manage configuration."""
    pass


@config.command("show")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output as JSON",
)
def config_show(json_output: bool) -> None:
    """Display current configuration."""
    from src.core.config import get_settings

    settings = get_settings()

    config_data = {
        "exchange": {
            "name": str(settings.exchange.name.value),
            "wallet_address": settings.exchange.wallet_address or "not set",
            "private_key_set": bool(settings.exchange.wallet_private_key),
            "api_key_set": bool(settings.exchange.api_key),
        },
        "trading": {
            "enabled": settings.trading_enabled,
            "paper_mode": settings.paper_trading,
        },
        "risk": {
            "max_leverage": settings.risk.max_leverage,
            "daily_drawdown_limit": settings.risk.daily_drawdown_limit,
            "total_drawdown_limit": settings.risk.total_drawdown_limit,
        },
        "position_sizing": {
            "per_trade_exposure": settings.position.per_trade_exposure,
            "min_notional_value": settings.position.min_notional_value,
        },
        "paths": {
            "database": str(settings.database_path),
            "models": str(settings.models_dir),
        },
        "logging": {
            "level": settings.logging.level,
            "format": settings.logging.format,
        },
    }

    if json_output:
        console.print_json(json.dumps(config_data, indent=2))
    else:
        # Mask sensitive data
        wallet = settings.exchange.wallet_address
        if wallet and len(wallet) > 10:
            wallet = f"{wallet[:6]}...{wallet[-4:]}"

        print_banner(
            "ALPHASTRIKE CONFIGURATION",
            items={
                "Exchange": str(settings.exchange.name.value),
                "Wallet": wallet or "[dim]not set[/dim]",
                "Private Key": "[green]✓ configured[/green]" if settings.exchange.wallet_private_key else "[red]✗ not set[/red]",
                "API Key": "[green]✓ configured[/green]" if settings.exchange.api_key else "[dim]not set[/dim]",
            },
        )

        console.print("\n[bold]Trading[/bold]")
        console.print(f"  Enabled:     {'[green]Yes[/green]' if settings.trading_enabled else '[red]No[/red]'}")
        console.print(f"  Paper Mode:  {'[yellow]Yes[/yellow]' if settings.paper_trading else 'No'}")

        console.print("\n[bold]Risk Management[/bold]")
        console.print(f"  Max Leverage:       {settings.risk.max_leverage}x")
        console.print(f"  Daily DD Limit:     {settings.risk.daily_drawdown_limit:.1%}")
        console.print(f"  Total DD Limit:     {settings.risk.total_drawdown_limit:.1%}")

        console.print("\n[bold]Paths[/bold]")
        console.print(f"  Database:  {settings.database_path}")
        console.print(f"  Models:    {settings.models_dir}")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value in .env file."""
    env_path = Path(".env")

    if not env_path.exists():
        print_error(".env file not found. Create one from .env.example")
        sys.exit(1)

    # Read existing content
    content = env_path.read_text()
    lines = content.split("\n")

    # Find and update or append
    key_upper = key.upper()
    found = False
    new_lines = []

    for line in lines:
        if line.strip().startswith(f"{key_upper}=") or line.strip().startswith(f"# {key_upper}="):
            new_lines.append(f"{key_upper}={value}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key_upper}={value}")

    # Write back
    env_path.write_text("\n".join(new_lines))
    print_success(f"Set {key_upper}={value}")
    console.print("[dim]Restart the trader for changes to take effect[/dim]")


@config.command("edit")
def config_edit() -> None:
    """Open .env in default editor."""
    env_path = Path(".env")

    if not env_path.exists():
        # Create from example
        example_path = Path(".env.example")
        if example_path.exists():
            import shutil
            shutil.copy(example_path, env_path)
            console.print("[yellow]Created .env from .env.example[/yellow]")
        else:
            print_error("No .env or .env.example found")
            sys.exit(1)

    # Get editor
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))

    try:
        subprocess.run([editor, str(env_path)], check=True)
        print_success("Configuration saved")
    except FileNotFoundError:
        print_error(f"Editor '{editor}' not found. Set EDITOR environment variable.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print_error("Editor exited with error")
        sys.exit(1)


@config.command("validate")
def config_validate() -> None:
    """Validate configuration is complete and correct."""
    from src.core.config import get_settings

    errors = []
    warnings = []

    try:
        settings = get_settings()
    except Exception as e:
        print_error(f"Failed to load settings: {e}")
        sys.exit(1)

    # Check exchange config
    exchange = settings.exchange.name.value

    if exchange == "hyperliquid":
        if not settings.exchange.wallet_private_key:
            errors.append("EXCHANGE_WALLET_PRIVATE_KEY is required for Hyperliquid")
        elif not settings.exchange.wallet_private_key.startswith("0x"):
            errors.append("EXCHANGE_WALLET_PRIVATE_KEY must start with '0x'")
    elif exchange == "weex":
        if not settings.exchange.api_key:
            errors.append("EXCHANGE_API_KEY is required for WEEX")
        if not settings.exchange.api_secret:
            errors.append("EXCHANGE_API_SECRET is required for WEEX")
        if not settings.exchange.api_passphrase:
            warnings.append("EXCHANGE_API_PASSPHRASE may be required for WEEX")

    # Check model files
    models_dir = Path(settings.models_dir)
    if not models_dir.exists():
        warnings.append(f"Models directory not found: {models_dir}")
    else:
        model_count = len(list(models_dir.glob(f"lightgbm_{exchange}_*.txt")))
        if model_count == 0:
            warnings.append(f"No models found for {exchange}. Run 'alphastrike train' first.")
        else:
            console.print(f"[green]✓[/green] Found {model_count} models for {exchange}")

    # Check database
    db_path = Path(settings.database_path)
    if not db_path.parent.exists():
        warnings.append(f"Database directory not found: {db_path.parent}")

    # Report results
    if errors:
        console.print("\n[red bold]Errors:[/red bold]")
        for err in errors:
            console.print(f"  [red]✗[/red] {err}")

    if warnings:
        console.print("\n[yellow bold]Warnings:[/yellow bold]")
        for warn in warnings:
            console.print(f"  [yellow]![/yellow] {warn}")

    if not errors and not warnings:
        print_success("Configuration is valid")
    elif errors:
        sys.exit(1)
