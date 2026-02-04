# AlphaStrike CLI Design

## Overview

Unified CLI for the AlphaStrike trading bot supporting both CEX (WEEX) and DEX (Hyperliquid) exchanges.

## Command Structure

```bash
alphastrike <command> [options]

Commands:
  trade      Start live/paper trading
  backtest   Run historical backtest
  train      Train ML models
  status     Check trader status, positions, P&L
  config     View/edit configuration
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Interface | Subcommands | Extensible, industry standard |
| Exchange selection | Per-subcommand `--exchange` flag | LLM-friendly, self-contained commands |
| Executable | `alphastrike` | Matches project name |
| Network | Testnet default, `--mainnet` requires confirmation | Safe defaults |
| Output | Human-readable default, `--json` for LLM | Flexible for both humans and automation |
| Package manager | `uv` only | Fast, modern |
| CLI library | `click` | Battle-tested, clean subcommand handling |

## Commands

### `trade` - Live/Paper Trading

```bash
alphastrike trade [options]

Options:
  --exchange, -e      hyperliquid | weex (required or from .env)
  --testnet           Use testnet (default for DEX)
  --mainnet           Use mainnet (requires confirmation)
  --dry-run           Log signals without placing orders
  --base-leverage     Base leverage (default: 5.0)
  --min-leverage      Minimum leverage (default: 1.0)
  --max-leverage      Maximum leverage (default: 10.0)
  --assets            Comma-separated assets (default: BTC,ETH,SOL...)
  --json              Output status updates as JSON
```

**Examples:**
```bash
# Paper trading on Hyperliquid testnet
alphastrike trade --exchange hyperliquid --testnet

# Dry-run (signals only, no orders)
alphastrike trade --exchange hyperliquid --dry-run

# WEEX with custom leverage
alphastrike trade --exchange weex --base-leverage 3

# Mainnet (prompts for confirmation)
alphastrike trade --exchange hyperliquid --mainnet
```

### `backtest` - Historical Backtest

```bash
alphastrike backtest [options]

Options:
  --exchange, -e      hyperliquid | weex (required or from .env)
  --days              Backtest period in days (default: 180)
  --initial-balance   Starting balance (default: 10000)
  --base-leverage     Base leverage (default: 5.0)
  --assets            Comma-separated assets (default: all 15)
  --output, -o        Save results to file (CSV/JSON based on extension)
  --json              Output results as JSON
```

**Examples:**
```bash
# Standard 180-day backtest
alphastrike backtest --exchange hyperliquid

# Custom period and balance
alphastrike backtest --exchange hyperliquid --days 90 --initial-balance 5000

# Save results to file
alphastrike backtest --exchange hyperliquid --output results/backtest_2024.json

# JSON output for LLM analysis
alphastrike backtest --exchange hyperliquid --json
```

### `train` - Train ML Models

```bash
alphastrike train [options]

Options:
  --exchange, -e      hyperliquid | weex (required or from .env)
  --assets            Comma-separated assets to train (default: all)
  --days              Training data period (default: 365)
  --output-dir        Model output directory (default: models/)
  --force             Overwrite existing models
  --json              Output training metrics as JSON
```

**Examples:**
```bash
# Train all models for Hyperliquid
alphastrike train --exchange hyperliquid

# Train specific assets
alphastrike train --exchange hyperliquid --assets BTC,ETH,SOL

# Custom training period
alphastrike train --exchange hyperliquid --days 730

# Retrain and overwrite existing
alphastrike train --exchange hyperliquid --force
```

### `status` - Check Trader Status

```bash
alphastrike status [options]

Options:
  --exchange, -e      hyperliquid | weex (required or from .env)
  --testnet           Check testnet (default for DEX)
  --mainnet           Check mainnet
  --watch, -w         Continuous updates (refresh every 5s)
  --json              Output as JSON
```

**Examples:**
```bash
# Check current status
alphastrike status --exchange hyperliquid --testnet

# Live monitoring
alphastrike status --exchange hyperliquid --watch

# JSON for LLM
alphastrike status --exchange hyperliquid --json
```

**Output (human-readable):**
```
╔══════════════════════════════════════════════════════════════╗
║  ALPHASTRIKE STATUS - Hyperliquid Testnet                    ║
╠══════════════════════════════════════════════════════════════╣
║  Trader:     RUNNING (uptime: 2h 15m)                        ║
║  Balance:    $999.00 USDC                                    ║
║  Leverage:   5.0x (dynamic)                                  ║
║  Regime:     BULLISH (72% confidence)                        ║
╠══════════════════════════════════════════════════════════════╣
║  OPEN POSITIONS                                              ║
║  BTCUSDT   LONG   +1.2%   $49.95   ml_tier1                 ║
║  ETHUSDT   LONG   -0.3%   $49.95   ml_tier2                 ║
╠══════════════════════════════════════════════════════════════╣
║  TODAY: 3 trades | +$12.50 | Win rate: 66%                  ║
╚══════════════════════════════════════════════════════════════╝
```

### `config` - Configuration Management

```bash
alphastrike config <action> [options]

Actions:
  show        Display current configuration
  set         Set a configuration value
  edit        Open .env in editor
  validate    Check configuration is valid

Options:
  --json      Output as JSON (for show)
```

**Examples:**
```bash
# Show current config
alphastrike config show

# Show as JSON
alphastrike config show --json

# Set a value
alphastrike config set EXCHANGE_NAME hyperliquid

# Open .env in default editor
alphastrike config edit

# Validate config (useful before trading)
alphastrike config validate
```

## File Structure

```
alphastrike_ai/
├── pyproject.toml          # Add CLI entry point
├── src/
│   └── cli/                # NEW - CLI module
│       ├── __init__.py
│       ├── main.py         # Entry point, command routing
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── trade.py    # trade command
│       │   ├── backtest.py # backtest command
│       │   ├── train.py    # train command
│       │   ├── status.py   # status command
│       │   └── config.py   # config command
│       └── utils/
│           ├── __init__.py
│           ├── output.py   # JSON/pretty output formatting
│           └── banner.py   # ASCII banners
```

## pyproject.toml Entry Point

```toml
[project.scripts]
alphastrike = "src.cli.main:main"
```

## Installation

```bash
# Install with uv
uv pip install -e .

# Run
alphastrike trade --exchange hyperliquid --testnet

# Or run directly without installing
uv run alphastrike trade --exchange hyperliquid --testnet
```

## Dependencies

Add to pyproject.toml:
```toml
dependencies = [
    "click>=8.0",
    # ... existing deps
]
```
