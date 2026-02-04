"""
Per-Asset Adaptive Configuration

Each trading pair maintains its own optimized parameters that can be
tuned independently based on rolling performance metrics.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml


@dataclass
class AdaptiveAssetConfig:
    """Tunable parameters for a single trading asset."""

    symbol: str

    # === Signal Generation ===
    conviction_threshold: float = 70.0  # 60-85 range, higher = fewer but better signals
    require_daily_trend_for_short: bool = True  # Skip shorts in NEUTRAL daily
    short_conviction_penalty: int = 5  # Extra points required for shorts

    # === Position Sizing ===
    position_size_multiplier: float = 1.0  # 0.5-1.5 range
    max_position_pct: float = 0.02  # Max 2% of portfolio per trade

    # === Risk Management ===
    stop_atr_multiplier: float = 2.0  # 1.5-3.0 range
    take_profit_atr_multiplier: float = 2.5  # 2.0-4.0 range
    trailing_stop_enabled: bool = False
    trailing_stop_activation_atr: float = 1.5  # Activate after 1.5 ATR profit

    # === Feature/Model Weights ===
    ensemble_weight_xgb: float = 0.4
    ensemble_weight_lgb: float = 0.4
    ensemble_weight_nn: float = 0.2

    # === Trading Rules ===
    short_enabled: bool = True
    long_enabled: bool = True
    max_trades_per_day: int = 5
    min_time_between_trades_minutes: int = 60

    # === Quality Thresholds ===
    min_trades_for_tuning: int = 20  # Need 20+ trades before auto-tuning
    min_data_quality_score: float = 0.8  # Don't tune on bad data

    # === Performance Thresholds (Retune Triggers) ===
    win_rate_threshold: float = 0.5  # Retune if win rate drops below
    max_drawdown_threshold: float = 0.05  # Retune if DD exceeds 5%
    rolling_window_days: int = 30  # Performance evaluation window

    # === Metadata ===
    last_tuned: str | None = None  # ISO timestamp
    tune_count: int = 0
    created_at: str | None = None
    notes: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptiveAssetConfig":
        """Create from dictionary."""
        return cls(**data)

    def mark_tuned(self) -> None:
        """Update metadata after tuning."""
        self.last_tuned = datetime.now(UTC).isoformat()
        self.tune_count += 1


def get_config_path(symbol: str, config_dir: Path | None = None) -> Path:
    """Get path to config file for a symbol."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs" / "assets"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / f"{symbol.lower()}.yaml"


def load_asset_config(symbol: str, config_dir: Path | None = None) -> AdaptiveAssetConfig:
    """
    Load asset config from YAML file, or create default if not exists.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        config_dir: Optional custom config directory

    Returns:
        AdaptiveAssetConfig for the symbol
    """
    config_path = get_config_path(symbol, config_dir)

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
            return AdaptiveAssetConfig.from_dict(data)

    # Create default config
    config = AdaptiveAssetConfig(symbol=symbol)
    save_asset_config(config, config_dir)
    return config


def save_asset_config(config: AdaptiveAssetConfig, config_dir: Path | None = None) -> Path:
    """
    Save asset config to YAML file.

    Args:
        config: The config to save
        config_dir: Optional custom config directory

    Returns:
        Path to saved config file
    """
    config_path = get_config_path(config.symbol, config_dir)

    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    return config_path


def load_all_asset_configs(config_dir: Path | None = None) -> dict[str, AdaptiveAssetConfig]:
    """
    Load all asset configs from the config directory.

    Returns:
        Dictionary mapping symbol to config
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs" / "assets"

    configs = {}
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            symbol = config_file.stem.upper()
            configs[symbol] = load_asset_config(symbol, config_dir)

    return configs


# Default configs for major pairs with optimized parameters from backtesting
DEFAULT_CONFIGS = {
    "BTCUSDT": AdaptiveAssetConfig(
        symbol="BTCUSDT",
        conviction_threshold=70,
        stop_atr_multiplier=2.0,
        short_conviction_penalty=5,
        notes="Primary pair, well-optimized from backtests"
    ),
    "ETHUSDT": AdaptiveAssetConfig(
        symbol="ETHUSDT",
        conviction_threshold=70,
        stop_atr_multiplier=1.8,
        short_conviction_penalty=5,
        notes="High win rate in backtests"
    ),
    "SOLUSDT": AdaptiveAssetConfig(
        symbol="SOLUSDT",
        conviction_threshold=80,  # Higher threshold due to lower win rate
        stop_atr_multiplier=2.5,  # Wider stops for volatility
        short_enabled=False,  # Disable shorts due to poor performance
        position_size_multiplier=0.5,  # Smaller positions
        notes="Underperforming in backtests, conservative settings"
    ),
}


def initialize_default_configs(config_dir: Path | None = None) -> None:
    """Create default config files for major pairs if they don't exist."""
    for symbol, config in DEFAULT_CONFIGS.items():
        config_path = get_config_path(symbol, config_dir)
        if not config_path.exists():
            save_asset_config(config, config_dir)
