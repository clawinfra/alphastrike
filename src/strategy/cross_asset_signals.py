"""
Cross-Asset Signal Engine for Medallion-Style Trading

Exploits cross-asset relationships:
1. BTC leads alts by 1-4 hours (lead-lag)
2. PAXG inverse to crypto (risk-off indicator)
3. SPX correlation varies by regime
4. Sector rotation patterns

This is a key component of achieving Medallion-style returns through
diversification and cross-asset intelligence.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.exchange.models import UnifiedCandle

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetSignal:
    """Cross-asset signal for a symbol."""
    leader_momentum: float = 0.0      # Signal from leading assets (BTC/ETH)
    risk_regime: float = 0.0          # Risk-on/risk-off from PAXG, SPX
    sector_momentum: float = 0.0       # Momentum of the asset's sector
    combined_signal: float = 0.0       # Weighted combination
    confidence: float = 0.0            # Signal confidence (0-1)
    description: str = ""


@dataclass
class SectorScore:
    """Sector momentum score."""
    sector: str
    score: float  # -1 to +1
    assets: list[str]
    leading: bool = False  # Is this sector leading?


class CrossAssetSignalEngine:
    """
    Exploits cross-asset relationships for enhanced signal generation.

    Philosophy (Jim Simons-inspired):
    - Markets are interconnected - use this information
    - BTC leads alts by 1-4 hours
    - Gold (PAXG) is a risk-off indicator
    - Sector rotation provides timing signals
    """

    # Lead-lag relationships (leader -> followers)
    LEAD_LAG_MAP = {
        "BTC": ["ETH", "SOL", "BNB", "XRP", "AVAX", "NEAR", "APT"],  # BTC leads majors
        "ETH": ["AAVE", "UNI", "LINK", "ARB", "OP"],  # ETH leads DeFi/L2
        "SPX": ["BTC", "ETH"],  # TradFi leads crypto in risk-on periods
    }

    # Inverse relationships (risk indicators)
    INVERSE_PAIRS = [
        ("PAXG", "BTC"),   # Gold vs crypto
        ("PAXG", "ETH"),
        ("PAXG", "SOL"),
    ]

    # Sector definitions
    SECTORS = {
        "crypto_major": ["BTC", "ETH", "BNB", "XRP"],
        "layer1": ["SOL", "AVAX", "NEAR", "APT", "SUI"],
        "defi": ["AAVE", "UNI", "LINK"],
        "ai": ["RNDR", "FET"],
        "meme": ["DOGE", "HYPE"],
        "traditional": ["PAXG", "SPX"],
    }

    # Signal weights
    LEADER_WEIGHT = 0.40
    RISK_REGIME_WEIGHT = 0.30
    SECTOR_WEIGHT = 0.30

    def __init__(self, lookback_short: int = 6, lookback_long: int = 24):
        """
        Initialize cross-asset signal engine.

        Args:
            lookback_short: Short-term lookback for momentum (candles)
            lookback_long: Long-term lookback for trend (candles)
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def calculate_signal(
        self,
        symbol: str,
        direction: str,  # "LONG" or "SHORT"
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> CrossAssetSignal:
        """
        Calculate cross-asset signal for a given symbol and direction.

        Args:
            symbol: Target symbol (e.g., "SOLUSDT")
            direction: Trade direction ("LONG" or "SHORT")
            all_candles: Dict of symbol -> candle history for all assets

        Returns:
            CrossAssetSignal with scores for each factor
        """
        # Normalize symbol (remove USDT suffix for matching)
        coin = symbol.replace("USDT", "")

        signals = CrossAssetSignal()
        reasons = []

        # 1. Leader momentum
        leader_signal = self._calculate_leader_signal(coin, all_candles)
        signals.leader_momentum = leader_signal

        if abs(leader_signal) > 0.02:
            if (direction == "LONG" and leader_signal > 0) or \
               (direction == "SHORT" and leader_signal < 0):
                reasons.append(f"leader aligned ({leader_signal:+.2%})")
            else:
                reasons.append(f"leader opposed ({leader_signal:+.2%})")

        # 2. Risk regime from PAXG/SPX
        risk_signal = self._calculate_risk_regime(coin, all_candles)
        signals.risk_regime = risk_signal

        if abs(risk_signal) > 0.01:
            if risk_signal > 0:
                reasons.append("risk-on")
            else:
                reasons.append("risk-off")

        # 3. Sector momentum
        sector = self._get_sector(coin)
        sector_signal = self._calculate_sector_momentum(sector, all_candles)
        signals.sector_momentum = sector_signal

        if abs(sector_signal) > 0.3:
            reasons.append(f"{sector} {'leading' if sector_signal > 0 else 'lagging'}")

        # Calculate combined signal
        # For LONG: positive signals are good
        # For SHORT: negative signals are good
        raw_combined = (
            signals.leader_momentum * self.LEADER_WEIGHT +
            signals.risk_regime * self.RISK_REGIME_WEIGHT +
            signals.sector_momentum * self.SECTOR_WEIGHT
        )

        if direction == "LONG":
            signals.combined_signal = raw_combined
        else:
            signals.combined_signal = -raw_combined

        # Calculate confidence based on signal agreement
        signal_signs = [
            np.sign(signals.leader_momentum) if abs(signals.leader_momentum) > 0.01 else 0,
            np.sign(signals.risk_regime) if abs(signals.risk_regime) > 0.01 else 0,
            np.sign(signals.sector_momentum) if abs(signals.sector_momentum) > 0.2 else 0,
        ]
        non_zero = [s for s in signal_signs if s != 0]
        if non_zero:
            agreement = abs(sum(non_zero)) / len(non_zero)
            signals.confidence = agreement
        else:
            signals.confidence = 0.5  # Neutral

        signals.description = ", ".join(reasons) if reasons else "neutral"

        return signals

    def _calculate_leader_signal(
        self,
        coin: str,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> float:
        """
        Calculate signal from market leaders (BTC, ETH).

        Returns momentum of leaders that this coin follows.
        """
        if coin in ("BTC", "ETH"):
            return 0.0  # Leaders don't follow themselves

        # Find which leaders this coin follows
        leaders = []
        for leader, followers in self.LEAD_LAG_MAP.items():
            if coin in followers:
                leaders.append(leader)

        if not leaders:
            leaders = ["BTC"]  # Default to BTC as leader

        # Calculate average leader momentum
        momentums = []
        for leader in leaders:
            leader_key = f"{leader}USDT"
            if leader_key in all_candles and len(all_candles[leader_key]) > self.lookback_short:
                mom = self._calc_momentum(all_candles[leader_key], self.lookback_short)
                momentums.append(mom)

        if momentums:
            return float(np.mean(momentums))
        return 0.0

    def _calculate_risk_regime(
        self,
        coin: str,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> float:
        """
        Calculate risk regime from PAXG (gold) and SPX.

        Strong PAXG (gold) = risk-off = bearish for crypto
        Strong SPX = risk-on = bullish for crypto

        Returns positive for risk-on, negative for risk-off.
        """
        # Traditional assets are not affected by this
        if coin in ("PAXG", "SPX", "GOLD", "EUR"):
            return 0.0

        risk_signals = []

        # PAXG (gold) - inverse relationship
        if "PAXGUSDT" in all_candles and len(all_candles["PAXGUSDT"]) > self.lookback_short:
            paxg_mom = self._calc_momentum(all_candles["PAXGUSDT"], self.lookback_short)
            # Strong gold = risk-off = negative for crypto
            risk_signals.append(-paxg_mom * 2)  # Weight gold signal

        # SPX - positive correlation in risk-on
        if "SPXUSDT" in all_candles and len(all_candles["SPXUSDT"]) > self.lookback_short:
            spx_mom = self._calc_momentum(all_candles["SPXUSDT"], self.lookback_short)
            # Strong SPX = risk-on = positive for crypto
            risk_signals.append(spx_mom * 1.5)

        if risk_signals:
            return float(np.mean(risk_signals))
        return 0.0

    def _calculate_sector_momentum(
        self,
        sector: str,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> float:
        """
        Calculate momentum for a sector.

        Returns normalized score (-1 to +1).
        """
        if sector not in self.SECTORS:
            return 0.0

        sector_assets = self.SECTORS[sector]
        momentums = []

        for asset in sector_assets:
            asset_key = f"{asset}USDT"
            if asset_key in all_candles and len(all_candles[asset_key]) > self.lookback_short:
                mom = self._calc_momentum(all_candles[asset_key], self.lookback_short)
                momentums.append(mom)

        if momentums:
            # Normalize to -1 to +1 range
            avg_mom = float(np.mean(momentums))
            return float(np.clip(avg_mom * 20, -1, 1))  # 5% move = full score
        return 0.0

    def calculate_all_sector_scores(
        self,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> list[SectorScore]:
        """
        Calculate momentum scores for all sectors.

        Returns list of SectorScore sorted by score (leaders first).
        """
        scores = []

        for sector, assets in self.SECTORS.items():
            score = self._calculate_sector_momentum(sector, all_candles)
            available_assets = [a for a in assets if f"{a}USDT" in all_candles]

            scores.append(SectorScore(
                sector=sector,
                score=score,
                assets=available_assets,
            ))

        # Sort by score (highest first)
        scores.sort(key=lambda x: x.score, reverse=True)

        # Mark top sector as leading
        if scores and scores[0].score > 0.3:
            scores[0].leading = True

        return scores

    def get_sector_allocation_bias(
        self,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> dict[str, float]:
        """
        Get allocation bias per sector based on momentum.

        Leading sectors get +20% allocation, lagging get -20%.

        Returns dict of sector -> bias multiplier.
        """
        sector_scores = self.calculate_all_sector_scores(all_candles)

        bias = {}
        n = len(sector_scores)

        for i, ss in enumerate(sector_scores):
            # Top sectors get positive bias, bottom get negative
            if n > 1:
                rank_position = (n - 1 - 2*i) / (n - 1)  # Ranges from 1 to -1
            else:
                rank_position = 0
            bias[ss.sector] = 1.0 + rank_position * 0.20  # 0.8 to 1.2 range

        return bias

    def _get_sector(self, coin: str) -> str:
        """Get sector for a coin."""
        for sector, assets in self.SECTORS.items():
            if coin in assets:
                return sector
        return "crypto_major"  # Default

    def _calc_momentum(
        self,
        candles: list[UnifiedCandle],
        lookback: int,
    ) -> float:
        """Calculate price momentum (rate of change)."""
        if len(candles) < lookback + 1:
            return 0.0
        current = candles[-1].close
        past = candles[-lookback-1].close
        if past == 0:
            return 0.0
        return (current - past) / past

    def get_conviction_bonus(
        self,
        symbol: str,
        direction: str,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> tuple[float, str]:
        """
        Calculate conviction bonus points from cross-asset signals.

        Returns (bonus_points, reason) where bonus_points is 0-15.
        """
        signal = self.calculate_signal(symbol, direction, all_candles)

        bonus = 0.0
        reasons = []

        # Leader alignment: up to 6 points
        if signal.combined_signal > 0:
            if signal.leader_momentum > 0.02:
                pts = min(6.0, signal.leader_momentum * 150)
                bonus += pts
                reasons.append(f"leader +{pts:.1f}")

        # Risk regime: up to 4 points
        if direction == "LONG" and signal.risk_regime > 0.01:
            pts = min(4.0, signal.risk_regime * 100)
            bonus += pts
            reasons.append(f"risk-on +{pts:.1f}")
        elif direction == "SHORT" and signal.risk_regime < -0.01:
            pts = min(4.0, abs(signal.risk_regime) * 100)
            bonus += pts
            reasons.append(f"risk-off +{pts:.1f}")

        # Sector momentum: up to 5 points
        if signal.sector_momentum > 0.3 and direction == "LONG":
            pts = min(5.0, signal.sector_momentum * 5)
            bonus += pts
            reasons.append(f"sector +{pts:.1f}")
        elif signal.sector_momentum < -0.3 and direction == "SHORT":
            pts = min(5.0, abs(signal.sector_momentum) * 5)
            bonus += pts
            reasons.append(f"sector +{pts:.1f}")

        reason = ", ".join(reasons) if reasons else "no cross-asset bonus"
        return bonus, reason


class SectorRotationEngine:
    """
    Detects sector rotation and provides allocation guidance.

    Medallion insight: Capital flows from sector to sector.
    Early detection of rotation = edge.
    """

    def __init__(self, signal_engine: CrossAssetSignalEngine):
        self.signal_engine = signal_engine
        self._rotation_history: list[tuple[str, float]] = []  # (leading_sector, timestamp)

    def detect_rotation(
        self,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> str | None:
        """
        Detect if sector rotation is occurring.

        Returns the new leading sector if rotation detected, None otherwise.
        """
        scores = self.signal_engine.calculate_all_sector_scores(all_candles)

        if not scores:
            return None

        current_leader = scores[0].sector if scores[0].score > 0.3 else None

        # Check if leader changed
        if self._rotation_history:
            prev_leader = self._rotation_history[-1][0]
            if current_leader and current_leader != prev_leader:
                logger.info(f"Sector rotation detected: {prev_leader} -> {current_leader}")
                return current_leader

        if current_leader:
            import time
            self._rotation_history.append((current_leader, time.time()))
            # Keep only last 10 rotations
            self._rotation_history = self._rotation_history[-10:]

        return None

    def get_overweight_sectors(
        self,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> list[str]:
        """Get sectors to overweight based on momentum."""
        scores = self.signal_engine.calculate_all_sector_scores(all_candles)
        return [s.sector for s in scores if s.score > 0.3]

    def get_underweight_sectors(
        self,
        all_candles: dict[str, list[UnifiedCandle]],
    ) -> list[str]:
        """Get sectors to underweight based on momentum."""
        scores = self.signal_engine.calculate_all_sector_scores(all_candles)
        return [s.sector for s in scores if s.score < -0.3]
