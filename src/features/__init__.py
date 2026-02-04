"""
AlphaStrike Trading Bot - Features Package

Feature engineering modules for ML models.
"""

# Alternative data signals (Simons-inspired)
from src.features.alternative_signals import (
    AlternativeDataFetcher,
    AlternativeSignalGenerator,
    AlternativeSignals,
    FundingRateData,
    LongShortRatioData,
    OpenInterestData,
    close_alternative_signal_generator,
    get_alternative_signal_generator,
)
from src.features.feature_validator import (
    REFERENCE_SCHEMA,
    DriftDetector,
    DriftResult,
    FeatureValidator,
    ReferenceDistribution,
    ValidationResult,
    get_feature_validator,
    validate_features,
)
from src.features.microstructure import (
    MICROSTRUCTURE_FEATURE_INFO,
    FundingInfo,
    MicrostructureData,
    MicrostructureFeatures,
    OpenInterestInfo,
    OrderbookSnapshot,
    TradeRecord,
    calculate_microstructure_features,
)
from src.features.pipeline import (
    CachedFeatures,
    CrossAssetData,
    CrossAssetFeatureCalculator,
    FeaturePipeline,
    FeaturePipelineConfig,
    FeeConfig,
    FeeFeatureCalculator,
    OrderbookData,
    TickerData,
    TimeFeatureCalculator,
    VolatilityFeatureCalculator,
    calculate_all_features,
    get_feature_pipeline,
)

# Signal decay tracking
from src.features.signal_tracker import (
    SignalRecord,
    SignalStats,
    SignalTracker,
    get_signal_tracker,
)
from src.features.technical import (
    IndicatorConfig,
    OHLCVData,
    TechnicalFeatures,
    calculate_all_indicators,
    get_technical_features,
)

__all__ = [
    # Microstructure features
    "FundingInfo",
    "MicrostructureData",
    "MicrostructureFeatures",
    "OpenInterestInfo",
    "OrderbookSnapshot",
    "TradeRecord",
    "calculate_microstructure_features",
    "MICROSTRUCTURE_FEATURE_INFO",
    # Technical indicators
    "IndicatorConfig",
    "OHLCVData",
    "TechnicalFeatures",
    "calculate_all_indicators",
    "get_technical_features",
    # Feature validation and drift detection
    "DriftDetector",
    "DriftResult",
    "FeatureValidator",
    "ReferenceDistribution",
    "ValidationResult",
    "get_feature_validator",
    "validate_features",
    "REFERENCE_SCHEMA",
    # Feature pipeline
    "CachedFeatures",
    "CrossAssetData",
    "CrossAssetFeatureCalculator",
    "FeeConfig",
    "FeeFeatureCalculator",
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "OrderbookData",
    "TickerData",
    "TimeFeatureCalculator",
    "VolatilityFeatureCalculator",
    "calculate_all_features",
    "get_feature_pipeline",
    # Alternative data signals (Simons-inspired)
    "AlternativeDataFetcher",
    "AlternativeSignalGenerator",
    "AlternativeSignals",
    "FundingRateData",
    "OpenInterestData",
    "LongShortRatioData",
    "get_alternative_signal_generator",
    "close_alternative_signal_generator",
    # Signal decay tracking
    "SignalTracker",
    "SignalStats",
    "SignalRecord",
    "get_signal_tracker",
]
