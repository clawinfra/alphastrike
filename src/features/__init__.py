"""
AlphaStrike Trading Bot - Features Package

Feature engineering modules for ML models.
"""

from src.features.microstructure import (
    FundingInfo,
    MicrostructureData,
    MicrostructureFeatures,
    OpenInterestInfo,
    OrderbookSnapshot,
    TradeRecord,
    calculate_microstructure_features,
    MICROSTRUCTURE_FEATURE_INFO,
)

from src.features.technical import (
    IndicatorConfig,
    OHLCVData,
    TechnicalFeatures,
    calculate_all_indicators,
    get_technical_features,
)

from src.features.feature_validator import (
    DriftDetector,
    DriftResult,
    FeatureValidator,
    ReferenceDistribution,
    ValidationResult,
    get_feature_validator,
    validate_features,
    REFERENCE_SCHEMA,
)

from src.features.pipeline import (
    CachedFeatures,
    CrossAssetData,
    CrossAssetFeatureCalculator,
    FeeConfig,
    FeeFeatureCalculator,
    FeaturePipeline,
    FeaturePipelineConfig,
    OrderbookData,
    TickerData,
    TimeFeatureCalculator,
    VolatilityFeatureCalculator,
    calculate_all_features,
    get_feature_pipeline,
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
]
