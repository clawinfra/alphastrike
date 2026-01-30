"""
AlphaStrike Trading Bot - Feature Validator with Drift Detection (US-009)

Monitors feature distribution drift using multiple statistical methods:
- PSI (Population Stability Index) - threshold > 0.25 indicates drift
- KS Test (Kolmogorov-Smirnov) - p < 0.01 indicates distribution divergence
- CUSUM (Cumulative Sum Control Chart) - > 5*std indicates persistent drift
- Z-score outlier detection - |z| > 3.5 indicates outliers

Health score calculation provides confidence multipliers for trading decisions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.config import FeatureValidationConfig, get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""

    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    cusum_value: float
    z_score_outliers: int
    total_samples: int
    has_psi_drift: bool
    has_ks_drift: bool
    has_cusum_drift: bool
    has_outliers: bool

    @property
    def is_drifted(self) -> bool:
        """Check if any drift method detected drift."""
        return self.has_psi_drift or self.has_ks_drift or self.has_cusum_drift

    @property
    def drift_severity(self) -> str:
        """Get drift severity classification."""
        drift_count = sum([
            self.has_psi_drift,
            self.has_ks_drift,
            self.has_cusum_drift,
            self.has_outliers,
        ])
        if drift_count == 0:
            return "none"
        elif drift_count == 1:
            return "low"
        elif drift_count == 2:
            return "medium"
        else:
            return "high"


@dataclass
class ValidationResult:
    """Complete validation result for all features."""

    health_score: float
    confidence_multiplier: float
    drifted_features: list[str]
    drift_results: dict[str, DriftResult]
    psi_penalty: float
    ks_penalty: float
    cusum_penalty: float
    outlier_penalty: float
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "health_score": self.health_score,
            "confidence_multiplier": self.confidence_multiplier,
            "drifted_features": self.drifted_features,
            "drifted_feature_count": len(self.drifted_features),
            "psi_penalty": self.psi_penalty,
            "ks_penalty": self.ks_penalty,
            "cusum_penalty": self.cusum_penalty,
            "outlier_penalty": self.outlier_penalty,
            "timestamp": self.timestamp,
        }


@dataclass
class ReferenceDistribution:
    """Reference distribution statistics for a single feature."""

    feature_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    percentiles: list[float]  # [p10, p25, p50, p75, p90]
    bin_edges: list[float]  # For PSI calculation
    bin_counts: list[float]  # Normalized bin counts (probabilities)
    sample_count: int
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_name": self.feature_name,
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "percentiles": self.percentiles,
            "bin_edges": self.bin_edges,
            "bin_counts": self.bin_counts,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceDistribution:
        """Create from dictionary."""
        return cls(
            feature_name=data["feature_name"],
            mean=data["mean"],
            std=data["std"],
            min_val=data["min_val"],
            max_val=data["max_val"],
            percentiles=data["percentiles"],
            bin_edges=data["bin_edges"],
            bin_counts=data["bin_counts"],
            sample_count=data["sample_count"],
            created_at=data.get("created_at", 0.0),
        )


# =============================================================================
# Reference Distribution Schema
# =============================================================================


REFERENCE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "title": "ReferenceDistributions",
    "description": "Reference distributions for feature drift detection",
    "properties": {
        "version": {
            "type": "string",
            "description": "Schema version",
        },
        "created_at": {
            "type": "number",
            "description": "Unix timestamp of creation",
        },
        "updated_at": {
            "type": "number",
            "description": "Unix timestamp of last update",
        },
        "symbol": {
            "type": "string",
            "description": "Trading symbol these distributions apply to",
        },
        "features": {
            "type": "object",
            "description": "Map of feature name to distribution statistics",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "feature_name": {"type": "string"},
                    "mean": {"type": "number"},
                    "std": {"type": "number"},
                    "min_val": {"type": "number"},
                    "max_val": {"type": "number"},
                    "percentiles": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 5,
                        "maxItems": 5,
                        "description": "[p10, p25, p50, p75, p90]",
                    },
                    "bin_edges": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Histogram bin edges for PSI",
                    },
                    "bin_counts": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Normalized bin counts (probabilities)",
                    },
                    "sample_count": {"type": "integer"},
                    "created_at": {"type": "number"},
                },
                "required": [
                    "feature_name",
                    "mean",
                    "std",
                    "min_val",
                    "max_val",
                    "percentiles",
                    "bin_edges",
                    "bin_counts",
                    "sample_count",
                ],
            },
        },
    },
    "required": ["version", "features"],
}


# =============================================================================
# Drift Detector
# =============================================================================


@dataclass
class DriftDetector:
    """
    Statistical drift detection using multiple methods.

    Methods:
    - PSI: Population Stability Index (distributional drift)
    - KS: Kolmogorov-Smirnov test (distribution comparison)
    - CUSUM: Cumulative sum control chart (persistent drift)
    - Z-score: Outlier detection
    """

    # Thresholds (from config or defaults)
    psi_threshold: float = 0.25
    ks_p_value_threshold: float = 0.01
    cusum_multiplier: float = 5.0
    z_score_threshold: float = 3.5
    num_bins: int = 10

    def calculate_psi(
        self,
        reference: ReferenceDistribution,
        current: NDArray[np.float64],
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

        Interpretation:
        - PSI < 0.10: No significant drift
        - 0.10 <= PSI < 0.25: Moderate drift, monitoring required
        - PSI >= 0.25: Significant drift, action required

        Args:
            reference: Reference distribution with bin edges and counts
            current: Current feature values

        Returns:
            PSI score
        """
        if len(current) < 10:
            return 0.0

        # Bin current data using reference bin edges
        bin_edges = np.array(reference.bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        # Calculate current bin counts
        current_counts, _ = np.histogram(current, bins=bin_edges)
        current_pct = current_counts / len(current)

        # Reference percentages
        expected_pct = np.array(reference.bin_counts)

        # Ensure same length
        if len(current_pct) != len(expected_pct):
            return 0.0

        # Add small epsilon to avoid log(0) and division by zero
        epsilon = 1e-10
        current_pct = np.clip(current_pct, epsilon, 1.0 - epsilon)
        expected_pct = np.clip(expected_pct, epsilon, 1.0 - epsilon)

        # Calculate PSI
        psi = np.sum((current_pct - expected_pct) * np.log(current_pct / expected_pct))

        return float(max(0.0, psi))

    def calculate_ks_test(
        self,
        reference: ReferenceDistribution,
        current: NDArray[np.float64],
    ) -> tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov test statistic and p-value.

        Tests whether the current distribution differs from reference.

        Args:
            reference: Reference distribution
            current: Current feature values

        Returns:
            Tuple of (KS statistic, p-value)
        """
        if len(current) < 10:
            return 0.0, 1.0

        # Approximate reference distribution from stored statistics
        # Generate synthetic reference samples from percentiles
        n_ref = reference.sample_count
        if n_ref < 10:
            return 0.0, 1.0

        # Create empirical CDF from percentiles
        percentile_points = [0, 10, 25, 50, 75, 90, 100]
        percentile_values = [
            reference.min_val,
            reference.percentiles[0],
            reference.percentiles[1],
            reference.percentiles[2],
            reference.percentiles[3],
            reference.percentiles[4],
            reference.max_val,
        ]

        # Sort current values for CDF
        sorted_current = np.sort(current)
        n_current = len(sorted_current)

        # Calculate empirical CDF of current data
        current_cdf = np.arange(1, n_current + 1) / n_current

        # Interpolate reference CDF at current data points
        ref_cdf_at_current = np.interp(
            sorted_current,
            percentile_values,
            np.array(percentile_points) / 100.0,
        )

        # KS statistic: max absolute difference between CDFs
        ks_statistic = float(np.max(np.abs(current_cdf - ref_cdf_at_current)))

        # Approximate p-value using asymptotic formula
        # p-value = 2 * exp(-2 * n * D^2) for large n
        n_effective = min(n_current, n_ref)
        lambda_val = (np.sqrt(n_effective) + 0.12 + 0.11 / np.sqrt(n_effective)) * ks_statistic

        # Asymptotic p-value approximation
        if lambda_val <= 0:
            p_value = 1.0
        else:
            # Series approximation for p-value
            p_value = 2.0 * np.exp(-2.0 * lambda_val * lambda_val)
            p_value = float(np.clip(p_value, 0.0, 1.0))

        return ks_statistic, p_value

    def calculate_cusum(
        self,
        reference: ReferenceDistribution,
        current: NDArray[np.float64],
    ) -> float:
        """
        Calculate CUSUM (Cumulative Sum Control Chart) value.

        Detects persistent shifts in the mean.

        Args:
            reference: Reference distribution with mean and std
            current: Current feature values

        Returns:
            Maximum CUSUM value (absolute)
        """
        if len(current) < 5:
            return 0.0

        ref_mean = reference.mean
        ref_std = reference.std

        if ref_std <= 0:
            ref_std = 1.0

        # Standardize current values
        standardized = (current - ref_mean) / ref_std

        # Calculate cumulative sum
        cusum_pos = np.zeros(len(standardized) + 1)
        cusum_neg = np.zeros(len(standardized) + 1)

        # Slack parameter (typically 0.5 or 1.0)
        k = 0.5

        for i, val in enumerate(standardized):
            cusum_pos[i + 1] = max(0, cusum_pos[i] + val - k)
            cusum_neg[i + 1] = max(0, cusum_neg[i] - val - k)

        # Return maximum deviation
        max_cusum = max(np.max(cusum_pos), np.max(cusum_neg))

        return float(max_cusum)

    def calculate_z_score_outliers(
        self,
        reference: ReferenceDistribution,
        current: NDArray[np.float64],
    ) -> int:
        """
        Count outliers based on z-score from reference distribution.

        Args:
            reference: Reference distribution with mean and std
            current: Current feature values

        Returns:
            Count of outliers (|z| > threshold)
        """
        if len(current) == 0:
            return 0

        ref_mean = reference.mean
        ref_std = reference.std

        if ref_std <= 0:
            ref_std = 1.0

        # Calculate z-scores relative to reference
        z_scores = np.abs((current - ref_mean) / ref_std)

        # Count outliers
        outlier_count = int(np.sum(z_scores > self.z_score_threshold))

        return outlier_count

    def detect_drift(
        self,
        reference: ReferenceDistribution,
        current: NDArray[np.float64],
    ) -> DriftResult:
        """
        Run all drift detection methods on a single feature.

        Args:
            reference: Reference distribution
            current: Current feature values

        Returns:
            DriftResult with all metrics
        """
        # Calculate all metrics
        psi_score = self.calculate_psi(reference, current)
        ks_stat, ks_p_value = self.calculate_ks_test(reference, current)
        cusum_value = self.calculate_cusum(reference, current)
        z_outliers = self.calculate_z_score_outliers(reference, current)

        # Determine drift flags
        has_psi_drift = psi_score > self.psi_threshold
        has_ks_drift = ks_p_value < self.ks_p_value_threshold
        has_cusum_drift = cusum_value > (self.cusum_multiplier * reference.std if reference.std > 0 else self.cusum_multiplier)
        has_outliers = z_outliers > int(len(current) * 0.05)  # More than 5% outliers

        return DriftResult(
            feature_name=reference.feature_name,
            psi_score=psi_score,
            ks_statistic=ks_stat,
            ks_p_value=ks_p_value,
            cusum_value=cusum_value,
            z_score_outliers=z_outliers,
            total_samples=len(current),
            has_psi_drift=has_psi_drift,
            has_ks_drift=has_ks_drift,
            has_cusum_drift=has_cusum_drift,
            has_outliers=has_outliers,
        )


# =============================================================================
# Feature Validator
# =============================================================================


@dataclass
class FeatureValidator:
    """
    Validates feature distributions and calculates health scores.

    Usage:
        validator = FeatureValidator()
        validator.load_reference_distributions("path/to/reference.json")

        features = {"rsi_14": np.array([...]), "macd": np.array([...])}
        result = validator.validate(features)

        print(f"Health: {result.health_score}")
        print(f"Confidence multiplier: {result.confidence_multiplier}")
    """

    config: FeatureValidationConfig = field(
        default_factory=lambda: get_settings().features
    )
    drift_detector: DriftDetector = field(default_factory=DriftDetector)
    reference_distributions: dict[str, ReferenceDistribution] = field(
        default_factory=dict
    )
    _initialized: bool = False

    def __post_init__(self) -> None:
        """Initialize drift detector with config values."""
        self.drift_detector = DriftDetector(
            psi_threshold=self.config.psi_threshold,
            ks_p_value_threshold=self.config.ks_p_value_threshold,
            cusum_multiplier=self.config.cusum_multiplier,
            z_score_threshold=self.config.z_score_threshold,
        )

    def load_reference_distributions(self, path: str | Path) -> bool:
        """
        Load reference distributions from JSON file.

        Args:
            path: Path to reference distributions JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Reference distributions file not found: {path}")
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            features_data = data.get("features", {})
            self.reference_distributions = {
                name: ReferenceDistribution.from_dict(dist_data)
                for name, dist_data in features_data.items()
            }

            self._initialized = True
            logger.info(
                f"Loaded {len(self.reference_distributions)} reference distributions"
            )
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load reference distributions: {e}")
            return False

    def save_reference_distributions(
        self,
        path: str | Path,
        symbol: str = "default",
    ) -> bool:
        """
        Save current reference distributions to JSON file.

        Args:
            path: Output path for JSON file
            symbol: Trading symbol these distributions apply to

        Returns:
            True if saved successfully
        """
        import time

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "created_at": time.time(),
            "updated_at": time.time(),
            "symbol": symbol,
            "features": {
                name: dist.to_dict()
                for name, dist in self.reference_distributions.items()
            },
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved reference distributions to {path}")
            return True
        except OSError as e:
            logger.error(f"Failed to save reference distributions: {e}")
            return False

    def build_reference_from_data(
        self,
        features: dict[str, NDArray[np.float64]],
        min_samples: int = 1000,
    ) -> bool:
        """
        Build reference distributions from historical feature data.

        Args:
            features: Dictionary of feature name to historical values
            min_samples: Minimum samples required per feature

        Returns:
            True if reference was built successfully
        """
        import time

        for name, values in features.items():
            if len(values) < min_samples:
                logger.warning(
                    f"Insufficient samples for {name}: {len(values)} < {min_samples}"
                )
                continue

            values = np.asarray(values, dtype=np.float64)
            values = values[~np.isnan(values)]  # Remove NaN

            if len(values) < min_samples:
                continue

            # Calculate statistics
            mean = float(np.mean(values))
            std = float(np.std(values))
            min_val = float(np.min(values))
            max_val = float(np.max(values))

            # Calculate percentiles
            percentiles = [
                float(np.percentile(values, p)) for p in [10, 25, 50, 75, 90]
            ]

            # Create histogram bins
            bin_edges = np.histogram_bin_edges(values, bins=self.drift_detector.num_bins)
            bin_counts, _ = np.histogram(values, bins=bin_edges)
            bin_counts_normalized = bin_counts / len(values)

            self.reference_distributions[name] = ReferenceDistribution(
                feature_name=name,
                mean=mean,
                std=std,
                min_val=min_val,
                max_val=max_val,
                percentiles=percentiles,
                bin_edges=bin_edges.tolist(),
                bin_counts=bin_counts_normalized.tolist(),
                sample_count=len(values),
                created_at=time.time(),
            )

        self._initialized = len(self.reference_distributions) > 0
        logger.info(
            f"Built reference distributions for {len(self.reference_distributions)} features"
        )
        return self._initialized

    def validate(
        self,
        features: dict[str, NDArray[np.float64]],
        timestamp: float = 0.0,
    ) -> ValidationResult:
        """
        Validate current features against reference distributions.

        Calculates health score based on drift detection results:
        - health_score = 100 - (psi_penalty + ks_penalty + cusum_penalty + outlier_penalty)
        - Each penalty is 0-25 points

        Args:
            features: Dictionary of feature name to current values
            timestamp: Optional timestamp for the validation

        Returns:
            ValidationResult with health score and confidence multiplier
        """
        import time

        if timestamp <= 0:
            timestamp = time.time()

        if not self._initialized or not self.reference_distributions:
            # Return healthy result if no reference exists
            logger.warning("No reference distributions loaded, returning healthy result")
            return ValidationResult(
                health_score=100.0,
                confidence_multiplier=1.0,
                drifted_features=[],
                drift_results={},
                psi_penalty=0.0,
                ks_penalty=0.0,
                cusum_penalty=0.0,
                outlier_penalty=0.0,
                timestamp=timestamp,
            )

        drift_results: dict[str, DriftResult] = {}
        drifted_features: list[str] = []

        # Track penalty components
        psi_drift_count = 0
        ks_drift_count = 0
        cusum_drift_count = 0
        outlier_drift_count = 0
        total_features = 0

        for name, values in features.items():
            if name not in self.reference_distributions:
                continue

            reference = self.reference_distributions[name]
            values = np.asarray(values, dtype=np.float64)
            values = values[~np.isnan(values)]

            if len(values) < 5:
                continue

            total_features += 1

            # Detect drift
            result = self.drift_detector.detect_drift(reference, values)
            drift_results[name] = result

            if result.is_drifted:
                drifted_features.append(name)

            # Count drift types
            if result.has_psi_drift:
                psi_drift_count += 1
            if result.has_ks_drift:
                ks_drift_count += 1
            if result.has_cusum_drift:
                cusum_drift_count += 1
            if result.has_outliers:
                outlier_drift_count += 1

        # Calculate penalties (0-25 each, based on fraction of features with drift)
        if total_features > 0:
            psi_penalty = 25.0 * (psi_drift_count / total_features)
            ks_penalty = 25.0 * (ks_drift_count / total_features)
            cusum_penalty = 25.0 * (cusum_drift_count / total_features)
            outlier_penalty = 25.0 * (outlier_drift_count / total_features)
        else:
            psi_penalty = ks_penalty = cusum_penalty = outlier_penalty = 0.0

        # Calculate health score
        health_score = 100.0 - (psi_penalty + ks_penalty + cusum_penalty + outlier_penalty)
        health_score = max(0.0, min(100.0, health_score))

        # Calculate confidence multiplier
        confidence_multiplier = self._calculate_confidence_multiplier(health_score)

        return ValidationResult(
            health_score=health_score,
            confidence_multiplier=confidence_multiplier,
            drifted_features=drifted_features,
            drift_results=drift_results,
            psi_penalty=psi_penalty,
            ks_penalty=ks_penalty,
            cusum_penalty=cusum_penalty,
            outlier_penalty=outlier_penalty,
            timestamp=timestamp,
        )

    def _calculate_confidence_multiplier(self, health_score: float) -> float:
        """
        Calculate confidence multiplier based on health score.

        health >= 70: multiplier = 1.0
        health 50-70: multiplier = 0.7 + (health-50) * 0.015
        health 30-50: multiplier = 0.4 + (health-30) * 0.015
        health < 30: multiplier = 0.1

        Args:
            health_score: Health score (0-100)

        Returns:
            Confidence multiplier (0.1 to 1.0)
        """
        if health_score >= 70:
            return 1.0
        elif health_score >= 50:
            return 0.7 + (health_score - 50) * 0.015
        elif health_score >= 30:
            return 0.4 + (health_score - 30) * 0.015
        else:
            return 0.1

    def is_healthy(self, health_score: float | None = None) -> bool:
        """
        Check if feature health meets minimum threshold for trading.

        Args:
            health_score: Health score to check, or None to use last validation

        Returns:
            True if healthy enough for trading
        """
        if health_score is None:
            return True  # Default to healthy if no validation done

        return health_score >= self.config.min_health_for_trading

    def get_feature_names(self) -> list[str]:
        """Get list of monitored feature names."""
        return list(self.reference_distributions.keys())

    def get_reference_schema(self) -> dict[str, Any]:
        """Get JSON schema for reference distributions file."""
        return REFERENCE_SCHEMA.copy()


# =============================================================================
# Module-level convenience functions
# =============================================================================


_default_validator: FeatureValidator | None = None


def get_feature_validator() -> FeatureValidator:
    """
    Get the default FeatureValidator singleton instance.

    Returns:
        FeatureValidator instance
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = FeatureValidator()
    return _default_validator


def validate_features(
    features: dict[str, NDArray[np.float64]],
) -> ValidationResult:
    """
    Convenience function to validate features using default validator.

    Args:
        features: Dictionary of feature name to values

    Returns:
        ValidationResult with health score and drift info
    """
    return get_feature_validator().validate(features)
