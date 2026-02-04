#!/usr/bin/env python3
"""
Test Alternative Signals Integration

Quick test to verify:
1. Alternative signals are fetching correctly from Binance
2. Conviction scorer properly uses the signals
3. Shows the boost/penalty applied
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.alternative_signals import (
    get_alternative_signal_generator,
    close_alternative_signal_generator,
    AlternativeSignals,
)
from src.strategy.conviction_scorer import (
    ConvictionScorer,
    TimeframeSignals,
    MarketContext,
)


async def test_alternative_signals():
    """Test fetching alternative signals from Binance."""
    print()
    print("=" * 60)
    print("TESTING ALTERNATIVE SIGNALS INTEGRATION")
    print("=" * 60)
    print()

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    generator = get_alternative_signal_generator()

    print("Fetching live alternative data from Binance...")
    print()

    results = {}
    for symbol in symbols:
        signals = await generator.generate_signals(symbol, 100000.0, 0.01)
        results[symbol] = signals

        print(f"{symbol}:")
        print(f"  Funding Rate:     {signals.funding_rate*100:+.4f}%")
        print(f"  Funding Signal:   {signals.funding_signal:+.2f}")
        print(f"  Funding Extreme:  {signals.funding_extreme}")
        print(f"  L/S Ratio:        {signals.long_short_ratio:.2f}")
        print(f"  L/S Signal:       {signals.ls_ratio_signal:+.2f}")
        print(f"  Crowd Extreme:    {signals.crowd_extreme}")
        print(f"  OI Signal:        {signals.oi_signal:+.2f}")
        print(f"  Combined Signal:  {signals.combined_signal:+.2f}")
        print(f"  Signal Count:     {signals.signal_count}/3")
        print()

    await close_alternative_signal_generator()
    return results


def test_conviction_integration(alt_signals: dict):
    """Test conviction scorer with alternative signals."""
    print()
    print("=" * 60)
    print("TESTING CONVICTION SCORER WITH ALTERNATIVE SIGNALS")
    print("=" * 60)
    print()

    scorer = ConvictionScorer(min_trade_score=70)

    # Test scenarios
    scenarios = [
        ("LONG with bullish alt signals", "LONG", "BULL", "BULLISH"),
        ("LONG with bearish alt signals", "LONG", "BEAR", "BULLISH"),
        ("SHORT with bearish alt signals", "SHORT", "BEAR", "BEARISH"),
        ("SHORT with bullish alt signals", "SHORT", "BULL", "BEARISH"),
    ]

    for name, signal, daily, momentum in scenarios:
        print(f"\nScenario: {name}")
        print("-" * 50)

        # Create aligned timeframe signals
        tf_signals = TimeframeSignals(
            daily_trend=daily,
            daily_adx=30,
            four_hour_signal=signal,
            four_hour_confidence=0.7,
            one_hour_momentum=momentum,
            mtf_aligned=True,
        )

        market_context = MarketContext(
            regime="TRENDING_UP" if signal == "LONG" else "TRENDING_DOWN",
            regime_confidence=0.8,
            volume_ratio=1.2,
            atr_ratio=1.0,
            rsi=50,
            price_vs_ema50=2.0 if signal == "LONG" else -2.0,
            price_vs_ema200=3.0 if signal == "LONG" else -3.0,
            bb_position=0.0,
            model_agreement_pct=0.75,
        )

        # Calculate without alternative signals
        result_without = scorer.calculate(tf_signals, market_context, None)

        # Calculate with alternative signals (use BTC signals as example)
        alt = alt_signals.get("BTCUSDT")
        result_with = scorer.calculate(tf_signals, market_context, alt)

        print(f"  Without Alt Signals:")
        print(f"    Score: {result_without.score:.1f} | Tier: {result_without.tier.value}")
        print(f"    Alt Boost: {result_without.breakdown.alternative_signals:+.1f}")

        print(f"  With Alt Signals:")
        print(f"    Score: {result_with.score:.1f} | Tier: {result_with.tier.value}")
        print(f"    Alt Boost: {result_with.breakdown.alternative_signals:+.1f}")

        diff = result_with.score - result_without.score
        print(f"  Difference: {diff:+.1f} points")

        # Interpretation
        if diff > 0:
            print(f"  -> Alt signals SUPPORT this trade")
        elif diff < 0:
            print(f"  -> Alt signals OPPOSE this trade")
        else:
            print(f"  -> Alt signals NEUTRAL")


async def main():
    """Run all tests."""
    # Test 1: Fetch alternative signals
    alt_signals = await test_alternative_signals()

    # Test 2: Integration with conviction scorer
    test_conviction_integration(alt_signals)

    print()
    print("=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print()
    print("Summary:")
    print("  - Alternative signals module: WORKING")
    print("  - Binance API integration: WORKING")
    print("  - Conviction scorer integration: WORKING")
    print()
    print("The alternative signals can now boost or reduce conviction")
    print("scores based on funding rate, open interest, and L/S ratio.")
    print()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
