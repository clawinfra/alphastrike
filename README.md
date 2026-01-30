# AlphaStrike Trading Bot

Autonomous ML-based crypto trading system for WEEX perpetual futures.

## Features

- **ML Ensemble**: 4-model ensemble (XGBoost, LightGBM, LSTM, Random Forest)
- **Regime Adaptation**: Automatic strategy adjustment based on market conditions
- **Adaptive Risk**: Multi-layer risk management with position scaling
- **Self-Healing**: Automatic model health detection and retraining

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your WEEX API credentials

# Run the bot
uv run python main.py
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System architecture details
- [PRD](docs/PRD.md) - Product requirements document

## Development

```bash
# Run type checking
uv run pyright src/

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check src/
```

## License

MIT
