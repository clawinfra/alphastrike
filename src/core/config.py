"""
AlphaStrike Trading Bot - Core Configuration Module

Centralized configuration using Pydantic Settings with validation.
All trading parameters are type-safe and loaded from environment variables.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class TradingPair(str, Enum):
    """Approved trading pairs for WEEX competition."""
    BTCUSDT = "cmt_btcusdt"
    ETHUSDT = "cmt_ethusdt"
    BNBUSDT = "cmt_bnbusdt"
    SOLUSDT = "cmt_solusdt"
    XRPUSDT = "cmt_xrpusdt"
    DOGEUSDT = "cmt_dogeusdt"
    LTCUSDT = "cmt_ltcusdt"
    ADAUSDT = "cmt_adausdt"


class MarketRegime(str, Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME_VOLATILITY = "extreme_volatility"
    TREND_EXHAUSTION = "trend_exhaustion"


class TradingThresholds(BaseSettings):
    """Signal generation thresholds."""

    model_config = SettingsConfigDict(env_prefix="THRESHOLD_")

    # Signal thresholds (weighted average from ensemble)
    long_threshold: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Minimum weighted average for LONG signal"
    )
    short_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=0.5,
        description="Maximum weighted average for SHORT signal"
    )
    high_confidence: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Minimum confidence for full position sizing"
    )

    # Adaptive threshold ranges
    adaptive_long_min: float = Field(default=0.65, description="Minimum adaptive LONG threshold")
    adaptive_long_max: float = Field(default=0.85, description="Maximum adaptive LONG threshold")
    adaptive_short_min: float = Field(default=0.15, description="Minimum adaptive SHORT threshold")
    adaptive_short_max: float = Field(default=0.35, description="Maximum adaptive SHORT threshold")


class PositionSizingConfig(BaseSettings):
    """Position sizing parameters."""

    model_config = SettingsConfigDict(env_prefix="POSITION_")

    # Minimum notional value for orders
    min_notional_value: float = Field(
        default=150.0,
        gt=0,
        description="Minimum order value in USDT"
    )

    # Exposure limits (base values - adaptive adjustments applied at runtime)
    per_trade_exposure: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Base per-trade exposure (10%)"
    )
    per_trade_exposure_min: float = Field(default=0.05, description="Minimum per-trade exposure")
    per_trade_exposure_max: float = Field(default=0.20, description="Maximum per-trade exposure")

    per_pair_exposure: float = Field(
        default=0.25,
        ge=0.05,
        le=0.60,
        description="Base per-pair exposure (25%)"
    )
    per_pair_exposure_min: float = Field(default=0.15, description="Minimum per-pair exposure")
    per_pair_exposure_max: float = Field(default=0.45, description="Maximum per-pair exposure")

    total_exposure: float = Field(
        default=0.80,
        ge=0.10,
        le=1.0,
        description="Base total portfolio exposure (80%)"
    )
    total_exposure_min: float = Field(default=0.30, description="Minimum total exposure")
    total_exposure_max: float = Field(default=0.95, description="Maximum total exposure")

    # Pyramiding
    pyramiding_enabled: bool = Field(default=True, description="Allow adding to winning positions")
    pyramiding_max_per_pair: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Maximum pyramiding per pair (50%)"
    )


class RiskControlConfig(BaseSettings):
    """Risk management parameters."""

    model_config = SettingsConfigDict(env_prefix="RISK_")

    # Leverage limits
    max_leverage: int = Field(
        default=20,
        ge=1,
        le=125,
        description="Maximum allowed leverage (WEEX competition: 20x)"
    )
    default_leverage: int = Field(default=5, ge=1, le=20, description="Default leverage for trades")

    # Drawdown limits
    daily_drawdown_limit: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Base daily drawdown limit (5%)"
    )
    daily_drawdown_limit_min: float = Field(default=0.05, description="Minimum daily DD limit")
    daily_drawdown_limit_max: float = Field(default=0.07, description="Maximum daily DD limit")

    total_drawdown_limit: float = Field(
        default=0.15,
        ge=0.05,
        le=0.30,
        description="Total drawdown hard stop (15%)"
    )

    # Balance threshold
    min_balance_threshold: float = Field(
        default=9000.0,
        gt=0,
        description="Minimum account balance before shutdown"
    )

    # Stop-loss/Take-profit defaults (ATR multipliers)
    default_stop_loss_atr: float = Field(default=2.0, description="Default SL in ATR units")
    default_take_profit_atr: float = Field(default=1.5, description="Default TP1 in ATR units")
    trailing_stop_atr: float = Field(default=1.5, description="Trailing stop distance in ATR")


class TradeFrequencyConfig(BaseSettings):
    """Trade frequency limits."""

    model_config = SettingsConfigDict(env_prefix="FREQUENCY_")

    # Cooldowns
    trade_cooldown_seconds: int = Field(
        default=3600,
        ge=60,
        description="Global cooldown between trades (1 hour)"
    )
    symbol_cooldown_seconds: int = Field(
        default=7200,
        ge=60,
        description="Per-symbol cooldown (2 hours)"
    )

    # Daily limits
    max_trades_per_day: int = Field(
        default=12,
        ge=1,
        description="Maximum trades per day"
    )
    max_trades_per_symbol_per_day: int = Field(
        default=2,
        ge=1,
        description="Maximum trades per symbol per day"
    )


class MLConfig(BaseSettings):
    """Machine learning configuration."""

    model_config = SettingsConfigDict(env_prefix="ML_")

    # Ensemble weights
    xgboost_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    lightgbm_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    lstm_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    rf_weight: float = Field(default=0.20, ge=0.0, le=1.0)

    # Training parameters
    min_training_samples: int = Field(default=1000, ge=100)
    validation_split: float = Field(default=0.20, ge=0.1, le=0.4)
    label_threshold: float = Field(
        default=0.005,
        description="Price change threshold for LONG/SHORT labels (0.5%)"
    )

    # Retraining
    retraining_interval_min: int = Field(default=13, description="Min retraining interval (minutes)")
    retraining_interval_max: int = Field(default=90, description="Max retraining interval (minutes)")

    # Health checks
    min_healthy_models: int = Field(default=2, description="Minimum healthy models for trading")
    unhealthy_threshold: int = Field(default=50, description="Consecutive unhealthy checks before action")

    @field_validator("xgboost_weight", "lightgbm_weight", "lstm_weight", "rf_weight")
    @classmethod
    def validate_weights_sum(cls, v: float) -> float:
        """Individual weight validation - sum check done in Settings."""
        return v


class ConfidenceFilterConfig(BaseSettings):
    """Prediction confidence filter configuration."""

    model_config = SettingsConfigDict(env_prefix="CONFIDENCE_")

    enabled: bool = Field(default=True, description="Enable confidence filter")
    min_raw_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    min_model_agreement: float = Field(default=0.50, ge=0.0, le=1.0)
    min_threshold_proximity: float = Field(default=0.10, ge=0.0, le=0.5)
    stability_window: int = Field(default=5, ge=1)
    base_composite_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    # Regime-adjusted thresholds
    trending_threshold: float = Field(default=0.55)
    ranging_threshold: float = Field(default=0.70)
    high_volatility_threshold: float = Field(default=0.65)
    extreme_volatility_threshold: float = Field(default=0.75)


class FeatureValidationConfig(BaseSettings):
    """Feature validation and drift detection configuration."""

    model_config = SettingsConfigDict(env_prefix="FEATURE_")

    psi_threshold: float = Field(default=0.25, description="PSI drift threshold")
    ks_p_value_threshold: float = Field(default=0.01, description="KS test p-value threshold")
    cusum_multiplier: float = Field(default=5.0, description="CUSUM detection multiplier (std)")
    z_score_threshold: float = Field(default=3.5, description="Z-score outlier threshold")
    min_health_for_trading: float = Field(default=30.0, ge=0.0, le=100.0)
    reference_update_hours: int = Field(default=168, description="Reference distribution update interval")


class DataGatewayConfig(BaseSettings):
    """Data quality gateway configuration."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    staleness_threshold_seconds: float = Field(default=5.0, gt=0)
    price_range_threshold: float = Field(default=0.50, description="Max deviation from 24h range")
    volume_spike_multiplier: float = Field(default=100.0, gt=1)
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_reset_seconds: float = Field(default=60.0, gt=0)


class ExchangeConfig(BaseSettings):
    """WEEX exchange configuration."""

    model_config = SettingsConfigDict(env_prefix="WEEX_")

    api_key: str = Field(default="", description="WEEX API key")
    api_secret: str = Field(default="", description="WEEX API secret")
    api_passphrase: str = Field(default="", description="WEEX API passphrase")

    rest_url: str = Field(default="https://api.weex.com", description="REST API base URL")
    ws_url: str = Field(default="wss://ws.weex.com", description="WebSocket URL")

    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Max requests per minute")
    order_update_interval: int = Field(default=30, description="Min seconds between order updates")


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    format: Literal["json", "text"] = Field(default="json")
    ai_logs_dir: Path = Field(default=Path("ai_logs"))
    app_logs_dir: Path = Field(default=Path("logs"))


class Settings(BaseSettings):
    """
    Main settings class combining all configuration sections.

    Usage:
        settings = get_settings()
        print(settings.trading.long_threshold)
        print(settings.risk.max_leverage)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurations
    trading: TradingThresholds = Field(default_factory=TradingThresholds)
    position: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    risk: RiskControlConfig = Field(default_factory=RiskControlConfig)
    frequency: TradeFrequencyConfig = Field(default_factory=TradeFrequencyConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    confidence: ConfidenceFilterConfig = Field(default_factory=ConfidenceFilterConfig)
    features: FeatureValidationConfig = Field(default_factory=FeatureValidationConfig)
    data_gateway: DataGatewayConfig = Field(default_factory=DataGatewayConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Global settings
    trading_enabled: bool = Field(default=True, description="Master trading switch")
    paper_trading: bool = Field(default=False, description="Paper trading mode")
    database_path: Path = Field(default=Path("data/alphastrike.db"))
    models_dir: Path = Field(default=Path("models"))

    # Trading pairs
    trading_pairs: list[str] = Field(
        default=[pair.value for pair in TradingPair],
        description="Active trading pairs"
    )

    @field_validator("trading_pairs")
    @classmethod
    def validate_trading_pairs(cls, v: list[str]) -> list[str]:
        """Validate that all pairs are approved."""
        valid_pairs = {pair.value for pair in TradingPair}
        for pair in v:
            if pair not in valid_pairs:
                raise ValueError(f"Invalid trading pair: {pair}. Must be one of {valid_pairs}")
        return v

    def validate_ml_weights(self) -> None:
        """Validate that ML weights sum to 1.0."""
        total = (
            self.ml.xgboost_weight +
            self.ml.lightgbm_weight +
            self.ml.lstm_weight +
            self.ml.rf_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"ML weights must sum to 1.0, got {total}")


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Validated settings instance

    Example:
        settings = get_settings()
        if settings.risk.max_leverage > 10:
            logger.warning("High leverage configured")
    """
    settings = Settings()
    settings.validate_ml_weights()
    logger.info(
        "Settings loaded",
        extra={
            "trading_enabled": settings.trading_enabled,
            "paper_trading": settings.paper_trading,
            "max_leverage": settings.risk.max_leverage,
            "pairs_count": len(settings.trading_pairs),
        }
    )
    return settings


def reload_settings() -> Settings:
    """
    Force reload settings (clears cache).

    Returns:
        Settings: Fresh settings instance
    """
    get_settings.cache_clear()
    return get_settings()
