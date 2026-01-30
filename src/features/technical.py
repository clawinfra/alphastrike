"""
AlphaStrike Trading Bot - Technical Indicators Feature Module

US-006: Technical analysis indicators for ML feature engineering.
All calculations use numpy for vectorized performance.

Contains 35 technical indicators across 4 categories:
- MOMENTUM: RSI (7, 14, 21), RSI_slope, RSI_divergence, Stochastic (K, D), MACD
- TREND: ADX, +DI, -DI, ADX_ROC, ADX_slope, EMA (9, 21, 50), EMA_crossovers
- VOLATILITY: ATR, ATR_ratio, Bollinger Bands (upper, middle, lower, bandwidth), ATR_percentile
- VOLUME: Volume_ratio, OBV, Volume_profile
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OHLCVData(TypedDict):
    """OHLCV data structure for indicator calculations."""
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]


@dataclass
class IndicatorConfig:
    """Configuration for technical indicator parameters."""
    # RSI periods
    rsi_short: int = 7
    rsi_medium: int = 14
    rsi_long: int = 21
    rsi_slope_period: int = 5
    rsi_divergence_period: int = 10

    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth: int = 3

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ADX
    adx_period: int = 14
    adx_roc_period: int = 5
    adx_slope_period: int = 5

    # EMA
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50

    # ATR
    atr_period: int = 14
    atr_percentile_period: int = 100

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Volume
    volume_ratio_period: int = 20
    volume_profile_bins: int = 10


# =============================================================================
# Helper Functions
# =============================================================================

def _ema(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Input price array
        period: EMA period

    Returns:
        EMA values array
    """
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)

    if len(data) == 0:
        return result

    # Initialize with SMA for first `period` values
    if len(data) >= period:
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    else:
        # Not enough data, use simple average
        result[-1] = np.mean(data) if len(data) > 0 else 0.0

    return result


def _sma(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    Calculate Simple Moving Average.

    Args:
        data: Input price array
        period: SMA period

    Returns:
        SMA values array
    """
    result = np.zeros_like(data)

    if len(data) < period:
        return result

    # Use cumsum for efficient rolling mean
    cumsum = np.cumsum(data)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period

    return result


def _true_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        True Range values
    """
    if len(close) == 0:
        return np.array([], dtype=np.float64)

    prev_close = np.concatenate([[close[0]], close[:-1]])

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    return np.maximum(tr1, np.maximum(tr2, tr3))


def _wilder_smooth(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    Wilder's smoothing method (used in RSI, ATR, ADX).

    Args:
        data: Input data array
        period: Smoothing period

    Returns:
        Smoothed values
    """
    result = np.zeros_like(data)

    if len(data) < period:
        return result

    # First value is simple average
    result[period - 1] = np.mean(data[:period])

    # Subsequent values use Wilder smoothing
    for i in range(period, len(data)):
        result[i] = (result[i - 1] * (period - 1) + data[i]) / period

    return result


# =============================================================================
# Momentum Indicators
# =============================================================================

def calculate_rsi(close: NDArray[np.float64], period: int = 14) -> NDArray[np.float64]:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        close: Closing prices array
        period: RSI period (default 14)

    Returns:
        RSI values (0-100 scale)
    """
    result = np.full_like(close, 50.0)  # Default to neutral

    if len(close) < period + 1:
        return result

    # Calculate price changes
    delta = np.diff(close)

    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Wilder smoothing
    avg_gain = _wilder_smooth(gains, period)
    avg_loss = _wilder_smooth(losses, period)

    # Calculate RS and RSI
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases
    rsi = np.where(np.isnan(rsi) | np.isinf(rsi), 50.0, rsi)

    # Align with original close array (RSI starts at index period)
    result[period:] = rsi[period - 1:]

    return result


def calculate_rsi_slope(rsi: NDArray[np.float64], period: int = 5) -> NDArray[np.float64]:
    """
    Calculate RSI slope (rate of change).

    Args:
        rsi: RSI values
        period: Slope calculation period

    Returns:
        RSI slope values
    """
    result = np.zeros_like(rsi)

    if len(rsi) < period + 1:
        return result

    # Simple linear regression slope over the period
    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, len(rsi)):
        y = rsi[i - period + 1:i + 1]
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_var if x_var != 0 else 0.0
        result[i] = slope

    return result


def calculate_rsi_divergence(
    close: NDArray[np.float64],
    rsi: NDArray[np.float64],
    period: int = 10
) -> NDArray[np.float64]:
    """
    Calculate RSI divergence indicator.

    Positive: Bullish divergence (price down, RSI up)
    Negative: Bearish divergence (price up, RSI down)

    Args:
        close: Closing prices
        rsi: RSI values
        period: Lookback period for divergence

    Returns:
        Divergence values (-1 to 1 normalized)
    """
    result = np.zeros_like(close)

    if len(close) < period:
        return result

    for i in range(period - 1, len(close)):
        # Calculate slopes over the period
        price_change = (close[i] - close[i - period + 1]) / close[i - period + 1] if close[i - period + 1] != 0 else 0
        rsi_change = rsi[i] - rsi[i - period + 1]

        # Divergence: opposite directions
        if price_change > 0 and rsi_change < 0:
            # Bearish divergence
            result[i] = -min(abs(rsi_change) / 50.0, 1.0)
        elif price_change < 0 and rsi_change > 0:
            # Bullish divergence
            result[i] = min(abs(rsi_change) / 50.0, 1.0)
        else:
            result[i] = 0.0

    return result


def calculate_stochastic(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate Stochastic Oscillator (%K and %D).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period
        smooth: Smoothing period for %K

    Returns:
        Tuple of (%K, %D) arrays (0-100 scale)
    """
    stoch_k = np.full_like(close, 50.0)
    stoch_d = np.full_like(close, 50.0)

    if len(close) < k_period:
        return stoch_k, stoch_d

    # Calculate raw %K
    raw_k = np.zeros_like(close)
    for i in range(k_period - 1, len(close)):
        highest_high = np.max(high[i - k_period + 1:i + 1])
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        range_val = highest_high - lowest_low
        if range_val != 0:
            raw_k[i] = ((close[i] - lowest_low) / range_val) * 100.0
        else:
            raw_k[i] = 50.0

    # Smooth %K
    stoch_k = _sma(raw_k, smooth)

    # Calculate %D (SMA of %K)
    stoch_d = _sma(stoch_k, d_period)

    # Fill early values with 50 (neutral)
    stoch_k = np.where(stoch_k == 0, 50.0, stoch_k)
    stoch_d = np.where(stoch_d == 0, 50.0, stoch_d)

    return stoch_k, stoch_d


def calculate_macd(
    close: NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate MACD (line, signal, histogram).

    Args:
        close: Closing prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    # Calculate EMAs
    fast_ema = _ema(close, fast_period)
    slow_ema = _ema(close, slow_period)

    # MACD line
    macd_line = fast_ema - slow_ema

    # Signal line (EMA of MACD)
    signal_line = _ema(macd_line, signal_period)

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_momentum_features(ohlcv: OHLCVData, config: IndicatorConfig) -> dict[str, NDArray[np.float64]]:
    """
    Calculate all momentum indicators.

    Args:
        ohlcv: OHLCV data
        config: Indicator configuration

    Returns:
        Dictionary of momentum indicator arrays
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # RSI at multiple periods
    rsi_7 = calculate_rsi(close, config.rsi_short)
    rsi_14 = calculate_rsi(close, config.rsi_medium)
    rsi_21 = calculate_rsi(close, config.rsi_long)

    # RSI slope and divergence (using RSI-14 as reference)
    rsi_slope = calculate_rsi_slope(rsi_14, config.rsi_slope_period)
    rsi_divergence = calculate_rsi_divergence(close, rsi_14, config.rsi_divergence_period)

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(
        high, low, close,
        config.stoch_k_period, config.stoch_d_period, config.stoch_smooth
    )

    # MACD
    macd_line, macd_signal, macd_histogram = calculate_macd(
        close, config.macd_fast, config.macd_slow, config.macd_signal
    )

    return {
        "rsi_7": rsi_7,
        "rsi_14": rsi_14,
        "rsi_21": rsi_21,
        "rsi_slope": rsi_slope,
        "rsi_divergence": rsi_divergence,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
    }


# =============================================================================
# Trend Indicators
# =============================================================================

def calculate_adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate Average Directional Index (ADX) and +DI/-DI.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period

    Returns:
        Tuple of (ADX, +DI, -DI) arrays
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)

    if n < period + 1:
        return adx, plus_di, minus_di

    # Calculate True Range
    tr = _true_range(high, low, close)

    # Calculate +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

    # Wilder smooth the values
    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    # Calculate +DI and -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = np.where(smoothed_tr != 0, 100.0 * smoothed_plus_dm / smoothed_tr, 0.0)
        minus_di = np.where(smoothed_tr != 0, 100.0 * smoothed_minus_dm / smoothed_tr, 0.0)

    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.where(di_sum != 0, 100.0 * di_diff / di_sum, 0.0)

    # ADX is Wilder smooth of DX
    adx = _wilder_smooth(dx, period)

    # Clean up any NaN/Inf
    adx = np.nan_to_num(adx, nan=0.0, posinf=100.0, neginf=0.0)
    plus_di = np.nan_to_num(plus_di, nan=0.0, posinf=100.0, neginf=0.0)
    minus_di = np.nan_to_num(minus_di, nan=0.0, posinf=100.0, neginf=0.0)

    return adx, plus_di, minus_di


def calculate_adx_roc(adx: NDArray[np.float64], period: int = 5) -> NDArray[np.float64]:
    """
    Calculate ADX Rate of Change.

    Args:
        adx: ADX values
        period: ROC period

    Returns:
        ADX ROC values
    """
    result = np.zeros_like(adx)

    if len(adx) < period + 1:
        return result

    for i in range(period, len(adx)):
        if adx[i - period] != 0:
            result[i] = (adx[i] - adx[i - period]) / adx[i - period] * 100.0
        else:
            result[i] = 0.0

    return result


def calculate_adx_slope(adx: NDArray[np.float64], period: int = 5) -> NDArray[np.float64]:
    """
    Calculate ADX slope (similar to RSI slope).

    Args:
        adx: ADX values
        period: Slope period

    Returns:
        ADX slope values
    """
    return calculate_rsi_slope(adx, period)


def calculate_ema_crossovers(
    close: NDArray[np.float64],
    fast_period: int = 9,
    medium_period: int = 21,
    slow_period: int = 50
) -> dict[str, NDArray[np.float64]]:
    """
    Calculate EMA values and crossover signals.

    Args:
        close: Closing prices
        fast_period: Fast EMA period
        medium_period: Medium EMA period
        slow_period: Slow EMA period

    Returns:
        Dictionary with EMA values and crossover signals
    """
    ema_fast = _ema(close, fast_period)
    ema_medium = _ema(close, medium_period)
    ema_slow = _ema(close, slow_period)

    # Crossover signals: 1 = bullish, -1 = bearish, 0 = no signal
    # Fast crosses medium
    fast_above_medium = (ema_fast > ema_medium).astype(np.float64)
    fast_medium_cross = np.diff(fast_above_medium, prepend=fast_above_medium[0])

    # Medium crosses slow
    medium_above_slow = (ema_medium > ema_slow).astype(np.float64)
    medium_slow_cross = np.diff(medium_above_slow, prepend=medium_above_slow[0])

    # Price position relative to EMAs (normalized distance)
    with np.errstate(divide='ignore', invalid='ignore'):
        price_vs_fast = np.where(ema_fast != 0, (close - ema_fast) / ema_fast, 0.0)
        price_vs_medium = np.where(ema_medium != 0, (close - ema_medium) / ema_medium, 0.0)
        price_vs_slow = np.where(ema_slow != 0, (close - ema_slow) / ema_slow, 0.0)

    return {
        "ema_9": ema_fast,
        "ema_21": ema_medium,
        "ema_50": ema_slow,
        "ema_fast_medium_cross": fast_medium_cross,
        "ema_medium_slow_cross": medium_slow_cross,
        "price_vs_ema_fast": price_vs_fast,
        "price_vs_ema_medium": price_vs_medium,
        "price_vs_ema_slow": price_vs_slow,
    }


def calculate_trend_features(ohlcv: OHLCVData, config: IndicatorConfig) -> dict[str, NDArray[np.float64]]:
    """
    Calculate all trend indicators.

    Args:
        ohlcv: OHLCV data
        config: Indicator configuration

    Returns:
        Dictionary of trend indicator arrays
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # ADX and directional indicators
    adx, plus_di, minus_di = calculate_adx(high, low, close, config.adx_period)
    adx_roc = calculate_adx_roc(adx, config.adx_roc_period)
    adx_slope = calculate_adx_slope(adx, config.adx_slope_period)

    # EMA crossovers
    ema_features = calculate_ema_crossovers(
        close, config.ema_fast, config.ema_medium, config.ema_slow
    )

    result = {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "adx_roc": adx_roc,
        "adx_slope": adx_slope,
    }
    result.update(ema_features)

    return result


# =============================================================================
# Volatility Indicators
# =============================================================================

def calculate_atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """
    Calculate Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR values
    """
    tr = _true_range(high, low, close)
    return _wilder_smooth(tr, period)


def calculate_atr_ratio(
    close: NDArray[np.float64],
    atr: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate ATR as percentage of price (normalized volatility).

    Args:
        close: Closing prices
        atr: ATR values

    Returns:
        ATR ratio (ATR / price * 100)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(close != 0, (atr / close) * 100.0, 0.0)
    return np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)


def calculate_atr_percentile(
    atr: NDArray[np.float64],
    period: int = 100
) -> NDArray[np.float64]:
    """
    Calculate ATR percentile (current ATR's rank in recent history).

    Args:
        atr: ATR values
        period: Lookback period for percentile calculation

    Returns:
        ATR percentile (0-100)
    """
    result = np.full_like(atr, 50.0)  # Default to median

    if len(atr) < period:
        return result

    for i in range(period - 1, len(atr)):
        window = atr[i - period + 1:i + 1]
        # Count how many values are below current
        percentile = (np.sum(window < atr[i]) / len(window)) * 100.0
        result[i] = percentile

    return result


def calculate_bollinger_bands(
    close: NDArray[np.float64],
    period: int = 20,
    std_dev: float = 2.0
) -> dict[str, NDArray[np.float64]]:
    """
    Calculate Bollinger Bands.

    Args:
        close: Closing prices
        period: SMA period
        std_dev: Standard deviation multiplier

    Returns:
        Dictionary with upper, middle, lower bands and bandwidth
    """
    middle = _sma(close, period)

    # Calculate rolling standard deviation
    std = np.zeros_like(close)
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i - period + 1:i + 1], ddof=1)

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    # Bandwidth: (upper - lower) / middle
    with np.errstate(divide='ignore', invalid='ignore'):
        bandwidth = np.where(middle != 0, (upper - lower) / middle, 0.0)
    bandwidth = np.nan_to_num(bandwidth, nan=0.0, posinf=0.0, neginf=0.0)

    # %B: (price - lower) / (upper - lower)
    band_range = upper - lower
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_b = np.where(band_range != 0, (close - lower) / band_range, 0.5)
    percent_b = np.nan_to_num(percent_b, nan=0.5, posinf=1.0, neginf=0.0)

    return {
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "bb_percent_b": percent_b,
    }


def calculate_volatility_features(ohlcv: OHLCVData, config: IndicatorConfig) -> dict[str, NDArray[np.float64]]:
    """
    Calculate all volatility indicators.

    Args:
        ohlcv: OHLCV data
        config: Indicator configuration

    Returns:
        Dictionary of volatility indicator arrays
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # ATR
    atr = calculate_atr(high, low, close, config.atr_period)
    atr_ratio = calculate_atr_ratio(close, atr)
    atr_percentile = calculate_atr_percentile(atr, config.atr_percentile_period)

    # Bollinger Bands
    bb_features = calculate_bollinger_bands(close, config.bb_period, config.bb_std)

    result = {
        "atr": atr,
        "atr_ratio": atr_ratio,
        "atr_percentile": atr_percentile,
    }
    result.update(bb_features)

    return result


# =============================================================================
# Volume Indicators
# =============================================================================

def calculate_volume_ratio(
    volume: NDArray[np.float64],
    period: int = 20
) -> NDArray[np.float64]:
    """
    Calculate volume ratio (current volume / average volume).

    Args:
        volume: Volume data
        period: Average period

    Returns:
        Volume ratio values
    """
    avg_volume = _sma(volume, period)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(avg_volume != 0, volume / avg_volume, 1.0)

    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


def calculate_obv(
    close: NDArray[np.float64],
    volume: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate On-Balance Volume (OBV).

    Args:
        close: Closing prices
        volume: Volume data

    Returns:
        OBV values (normalized to z-score for ML)
    """
    if len(close) == 0:
        return np.array([], dtype=np.float64)

    obv = np.zeros_like(volume)
    obv[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    # Normalize OBV to z-score for ML compatibility
    if len(obv) > 20:
        mean_obv = np.mean(obv[-100:]) if len(obv) >= 100 else np.mean(obv)
        std_obv = np.std(obv[-100:]) if len(obv) >= 100 else np.std(obv)
        if std_obv != 0:
            obv_normalized = (obv - mean_obv) / std_obv
        else:
            obv_normalized = np.zeros_like(obv)
    else:
        obv_normalized = obv

    return obv_normalized


def calculate_volume_profile(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    bins: int = 10
) -> dict[str, NDArray[np.float64]]:
    """
    Calculate volume profile features.

    Args:
        close: Closing prices
        volume: Volume data
        high: High prices
        low: Low prices
        bins: Number of price bins

    Returns:
        Dictionary with volume profile features
    """
    n = len(close)

    # Point of Control (POC) - price level with highest volume
    poc_distance = np.zeros(n, dtype=np.float64)

    # Value Area features
    value_area_high = np.zeros(n, dtype=np.float64)
    value_area_low = np.zeros(n, dtype=np.float64)

    lookback = 50  # Use last 50 candles for profile

    for i in range(lookback - 1, n):
        window_close = close[i - lookback + 1:i + 1]
        window_volume = volume[i - lookback + 1:i + 1]
        window_high = high[i - lookback + 1:i + 1]
        window_low = low[i - lookback + 1:i + 1]

        # Create price bins
        price_min = np.min(window_low)
        price_max = np.max(window_high)

        if price_max == price_min:
            continue

        bin_edges = np.linspace(price_min, price_max, bins + 1)

        # Allocate volume to bins
        vol_profile = np.zeros(bins, dtype=np.float64)
        for j in range(len(window_close)):
            # Find which bin this candle belongs to
            bin_idx = int((window_close[j] - price_min) / (price_max - price_min) * (bins - 1))
            bin_idx = min(max(bin_idx, 0), bins - 1)
            vol_profile[bin_idx] += window_volume[j]

        # Find POC (bin with max volume)
        poc_bin = np.argmax(vol_profile)
        poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2

        # POC distance from current price (normalized)
        poc_distance[i] = (close[i] - poc_price) / close[i] if close[i] != 0 else 0

        # Value Area: 70% of volume
        total_vol = np.sum(vol_profile)
        if total_vol > 0:
            sorted_idx = np.argsort(vol_profile)[::-1]
            cumsum = 0.0
            va_bins: list[int] = []
            for idx in sorted_idx:
                cumsum += vol_profile[idx]
                va_bins.append(idx)
                if cumsum >= total_vol * 0.7:
                    break

            va_bin_min = min(va_bins)
            va_bin_max = max(va_bins)
            value_area_low[i] = bin_edges[va_bin_min]
            value_area_high[i] = bin_edges[va_bin_max + 1]

    # Normalize value area to relative position
    with np.errstate(divide='ignore', invalid='ignore'):
        va_position = np.where(
            (value_area_high != 0) & (value_area_high != value_area_low),
            (close - value_area_low) / (value_area_high - value_area_low),
            0.5
        )
    va_position = np.nan_to_num(va_position, nan=0.5, posinf=1.0, neginf=0.0)

    return {
        "volume_poc_distance": poc_distance,
        "volume_va_position": va_position,
    }


def calculate_volume_features(ohlcv: OHLCVData, config: IndicatorConfig) -> dict[str, NDArray[np.float64]]:
    """
    Calculate all volume indicators.

    Args:
        ohlcv: OHLCV data
        config: Indicator configuration

    Returns:
        Dictionary of volume indicator arrays
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    # Volume ratio
    volume_ratio = calculate_volume_ratio(volume, config.volume_ratio_period)

    # OBV
    obv = calculate_obv(close, volume)

    # Volume profile
    vp_features = calculate_volume_profile(close, volume, high, low, config.volume_profile_bins)

    result = {
        "volume_ratio": volume_ratio,
        "obv": obv,
    }
    result.update(vp_features)

    return result


# =============================================================================
# Main TechnicalFeatures Class
# =============================================================================

class TechnicalFeatures:
    """
    Orchestrates calculation of all technical indicators.

    Usage:
        # Create instance with default config
        tech = TechnicalFeatures()

        # Or with custom config
        config = IndicatorConfig(rsi_medium=10, adx_period=20)
        tech = TechnicalFeatures(config)

        # Calculate features from numpy arrays
        features = tech.calculate(open_arr, high_arr, low_arr, close_arr, volume_arr)

        # Get feature names
        names = tech.feature_names
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """
        Initialize TechnicalFeatures calculator.

        Args:
            config: Indicator configuration (uses defaults if None)
        """
        self.config = config or IndicatorConfig()
        self._feature_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        """Get list of all feature names in order."""
        if self._feature_names is None:
            # Generate feature names by running a dummy calculation
            dummy_data: OHLCVData = {
                "open": np.array([1.0] * 200, dtype=np.float64),
                "high": np.array([1.1] * 200, dtype=np.float64),
                "low": np.array([0.9] * 200, dtype=np.float64),
                "close": np.array([1.0] * 200, dtype=np.float64),
                "volume": np.array([100.0] * 200, dtype=np.float64),
            }
            features = self._calculate_all(dummy_data)
            self._feature_names = list(features.keys())
        return self._feature_names

    def calculate(
        self,
        open_prices: NDArray[np.float64],
        high_prices: NDArray[np.float64],
        low_prices: NDArray[np.float64],
        close_prices: NDArray[np.float64],
        volume: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """
        Calculate all technical indicators from OHLCV arrays.

        Args:
            open_prices: Open prices array
            high_prices: High prices array
            low_prices: Low prices array
            close_prices: Close prices array
            volume: Volume array

        Returns:
            Dictionary mapping feature names to numpy arrays
        """
        ohlcv: OHLCVData = {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
        return self._calculate_all(ohlcv)

    def _calculate_all(self, ohlcv: OHLCVData) -> dict[str, NDArray[np.float64]]:
        """
        Calculate all features from OHLCV data.

        Args:
            ohlcv: OHLCV data structure

        Returns:
            Dictionary of all features
        """
        features: dict[str, NDArray[np.float64]] = {}

        # Momentum indicators
        momentum = calculate_momentum_features(ohlcv, self.config)
        features.update(momentum)

        # Trend indicators
        trend = calculate_trend_features(ohlcv, self.config)
        features.update(trend)

        # Volatility indicators
        volatility = calculate_volatility_features(ohlcv, self.config)
        features.update(volatility)

        # Volume indicators
        volume_features = calculate_volume_features(ohlcv, self.config)
        features.update(volume_features)

        logger.debug(f"Calculated {len(features)} technical features")

        return features

    def calculate_latest(
        self,
        open_prices: NDArray[np.float64],
        high_prices: NDArray[np.float64],
        low_prices: NDArray[np.float64],
        close_prices: NDArray[np.float64],
        volume: NDArray[np.float64]
    ) -> dict[str, float]:
        """
        Calculate features and return only the latest values.

        Convenience method for real-time prediction.

        Args:
            open_prices: Open prices array
            high_prices: High prices array
            low_prices: Low prices array
            close_prices: Close prices array
            volume: Volume array

        Returns:
            Dictionary mapping feature names to latest scalar values
        """
        all_features = self.calculate(open_prices, high_prices, low_prices, close_prices, volume)

        return {name: float(values[-1]) for name, values in all_features.items()}

    def get_feature_count(self) -> int:
        """Get total number of features calculated."""
        return len(self.feature_names)


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_tech: TechnicalFeatures | None = None


def get_technical_features() -> TechnicalFeatures:
    """
    Get the default TechnicalFeatures singleton instance.

    Returns:
        TechnicalFeatures instance with default configuration
    """
    global _default_tech
    if _default_tech is None:
        _default_tech = TechnicalFeatures()
    return _default_tech


def calculate_all_indicators(
    open_prices: NDArray[np.float64],
    high_prices: NDArray[np.float64],
    low_prices: NDArray[np.float64],
    close_prices: NDArray[np.float64],
    volume: NDArray[np.float64]
) -> dict[str, NDArray[np.float64]]:
    """
    Convenience function to calculate all technical indicators.

    Args:
        open_prices: Open prices array
        high_prices: High prices array
        low_prices: Low prices array
        close_prices: Close prices array
        volume: Volume array

    Returns:
        Dictionary of all technical indicator arrays
    """
    return get_technical_features().calculate(
        open_prices, high_prices, low_prices, close_prices, volume
    )
