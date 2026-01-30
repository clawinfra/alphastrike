"""
AlphaStrike Trading Bot - Drift Detector Module

Utility module for detecting distribution drift and statistical anomalies
in feature distributions. Used for model health monitoring and data quality.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class DriftDetector:
    """
    Wrapper around statistical methods for drift detection.

    Provides methods for:
    - Population Stability Index (PSI) for distribution comparison
    - Kolmogorov-Smirnov test for distribution similarity
    - CUSUM test for mean shift detection
    - Z-score calculation for outlier detection

    Example:
        detector = DriftDetector()
        reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        psi = detector.calculate_psi(current, reference)
        ks_stat, p_value = detector.ks_test(current, reference)
    """

    def calculate_psi(
        self,
        current: np.ndarray,
        reference: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures how much a distribution has shifted from a reference.
        - PSI < 0.10: No significant shift
        - PSI 0.10-0.25: Moderate shift, monitor closely
        - PSI > 0.25: Significant shift, investigate

        Args:
            current: Current distribution values
            reference: Reference (baseline) distribution values
            bins: Number of bins for histogram comparison

        Returns:
            PSI value (0 = identical distributions)

        Example:
            >>> detector = DriftDetector()
            >>> ref = np.random.normal(0, 1, 1000)
            >>> curr = np.random.normal(0.5, 1, 1000)  # Shifted mean
            >>> psi = detector.calculate_psi(curr, ref)
            >>> psi > 0.1  # Should detect shift
            True
        """
        # Handle edge cases
        if len(current) == 0 or len(reference) == 0:
            return 0.0

        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)

        # Calculate proportions for each distribution
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions with smoothing to avoid division by zero
        epsilon = 1e-10
        ref_proportions = (ref_counts + epsilon) / (len(reference) + bins * epsilon)
        curr_proportions = (curr_counts + epsilon) / (len(current) + bins * epsilon)

        # Calculate PSI
        psi = np.sum(
            (curr_proportions - ref_proportions)
            * np.log(curr_proportions / ref_proportions)
        )

        return float(psi)

    def ks_test(
        self,
        current: np.ndarray,
        reference: np.ndarray,
    ) -> tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov two-sample test.

        Tests whether two samples come from the same distribution.
        A low p-value (< 0.05) suggests the distributions are different.

        Args:
            current: Current distribution values
            reference: Reference (baseline) distribution values

        Returns:
            Tuple of (KS statistic, p-value)
            - statistic: Maximum difference between CDFs (0-1)
            - p_value: Probability that samples come from same distribution

        Example:
            >>> detector = DriftDetector()
            >>> ref = np.random.normal(0, 1, 100)
            >>> curr = np.random.normal(0, 1, 100)  # Same distribution
            >>> stat, p_val = detector.ks_test(curr, ref)
            >>> p_val > 0.05  # Should not reject null hypothesis
            True
        """
        # Handle edge cases
        if len(current) == 0 or len(reference) == 0:
            return (0.0, 1.0)

        result = stats.ks_2samp(current, reference)
        # ks_2samp returns a KstestResult with statistic and pvalue attributes
        # Using getattr for pyright compatibility
        stat: float = getattr(result, "statistic")
        pval: float = getattr(result, "pvalue")
        return (stat, pval)

    def cusum_test(
        self,
        values: list[float],
        target_mean: float,
        std: float,
    ) -> float:
        """
        Calculate CUSUM (Cumulative Sum) statistic for mean shift detection.

        CUSUM accumulates deviations from a target mean, detecting
        persistent shifts that might be missed by point-in-time checks.

        Args:
            values: Sequence of values to test
            target_mean: Expected mean value
            std: Standard deviation for normalization

        Returns:
            Maximum absolute CUSUM value (normalized by std)
            Higher values indicate stronger evidence of mean shift

        Example:
            >>> detector = DriftDetector()
            >>> values = [0.0, 0.1, 0.5, 0.8, 1.2, 1.5]  # Drifting upward
            >>> cusum = detector.cusum_test(values, 0.0, 0.5)
            >>> cusum > 2.0  # Should detect drift
            True
        """
        if not values or std <= 0:
            return 0.0

        # Calculate cumulative sum of deviations
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0

        for value in values:
            deviation = (value - target_mean) / std

            # Two-sided CUSUM
            cusum_pos = max(0.0, cusum_pos + deviation)
            cusum_neg = max(0.0, cusum_neg - deviation)

            max_cusum = max(max_cusum, cusum_pos, cusum_neg)

        return max_cusum

    def z_score(
        self,
        value: float,
        mean: float,
        std: float,
    ) -> float:
        """
        Calculate z-score for a value.

        Z-score indicates how many standard deviations a value is from the mean.
        - |z| < 2: Normal range
        - |z| 2-3: Unusual, worth monitoring
        - |z| > 3: Potential outlier

        Args:
            value: Value to evaluate
            mean: Population mean
            std: Population standard deviation

        Returns:
            Z-score (can be negative)

        Example:
            >>> detector = DriftDetector()
            >>> detector.z_score(3.0, 0.0, 1.0)
            3.0
            >>> detector.z_score(-2.0, 0.0, 1.0)
            -2.0
        """
        if std <= 0:
            return 0.0

        return (value - mean) / std
