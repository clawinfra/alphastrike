"""
AlphaStrike Trading Bot - AI Logger for Trade Explanations (US-026)

WEEX competition compliance module for logging AI trade decisions.
Generates human-readable explanations for each trade with model outputs,
regime analysis, and risk check details.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.data.database import AILogEntry, Database, get_database

logger = logging.getLogger(__name__)


@dataclass
class AIExplanation:
    """
    AI explanation for a trade decision.

    Stores all relevant information about why a trade was executed,
    including model outputs, market regime, and risk check results.

    Attributes:
        order_id: Unique order identifier
        symbol: Trading pair symbol (e.g., "cmt_btcusdt")
        signal: Trade signal ("LONG", "SHORT", "HOLD")
        confidence: Confidence level in the signal (0-1)
        weighted_average: Weighted average of model predictions
        model_outputs: Dictionary of individual model predictions
        regime: Current market regime
        risk_checks: List of risk checks that were passed
        reasoning: Human-readable explanation of the trade
        timestamp: When the decision was made
    """

    order_id: str
    symbol: str
    signal: str
    confidence: float
    weighted_average: float
    model_outputs: dict[str, float]
    regime: str
    risk_checks: list[str]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "weighted_average": self.weighted_average,
            "model_outputs": self.model_outputs,
            "regime": self.regime,
            "risk_checks": self.risk_checks,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_ai_log_entry(self) -> AILogEntry:
        """Convert to AILogEntry for database storage."""
        return AILogEntry(
            id=str(uuid.uuid4()),
            order_id=self.order_id,
            symbol=self.symbol,
            signal=self.signal,
            confidence=self.confidence,
            weighted_average=self.weighted_average,
            model_outputs=json.dumps(self.model_outputs),
            regime=self.regime,
            risk_checks=json.dumps(self.risk_checks),
            reasoning=self.reasoning,
            timestamp=self.timestamp,
            uploaded=False,
        )

    @classmethod
    def from_dict(cls, data: dict) -> AIExplanation:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            order_id=data["order_id"],
            symbol=data["symbol"],
            signal=data["signal"],
            confidence=data["confidence"],
            weighted_average=data["weighted_average"],
            model_outputs=data["model_outputs"],
            regime=data["regime"],
            risk_checks=data["risk_checks"],
            reasoning=data["reasoning"],
            timestamp=timestamp,
        )


class AILogger:
    """
    AI Logger for WEEX competition compliance.

    Generates and persists human-readable explanations for each trade decision.
    Logs include model outputs, market regime analysis, and risk check details.

    Example:
        ai_logger = AILogger()
        explanation = ai_logger.log_trade_decision(
            order_id="order_123",
            symbol="cmt_btcusdt",
            signal="LONG",
            confidence=0.78,
            weighted_avg=0.81,
            model_outputs={"xgboost": 0.82, "lightgbm": 0.79, "lstm": 0.85, "rf": 0.78},
            regime="trending_up",
            risk_checks=["max_position", "daily_loss_limit", "leverage_check"],
        )
        await ai_logger.save_to_file(explanation)
        await ai_logger.save_to_database(explanation)
    """

    # Default log directory
    DEFAULT_LOG_DIR = Path("ai_logs")

    def __init__(
        self,
        log_dir: Path | None = None,
        database: Database | None = None,
    ) -> None:
        """
        Initialize AILogger.

        Args:
            log_dir: Directory for log files. Defaults to ai_logs/
            database: Database instance. If None, uses singleton from get_database()
        """
        self.log_dir = log_dir or self.DEFAULT_LOG_DIR
        self._database = database
        logger.info(f"AILogger initialized with log_dir={self.log_dir}")

    async def _get_database(self) -> Database:
        """Get database instance, using singleton if not explicitly set."""
        if self._database is not None:
            return self._database
        return await get_database()

    def log_trade_decision(
        self,
        order_id: str,
        symbol: str,
        signal: str,
        confidence: float,
        weighted_avg: float,
        model_outputs: dict[str, float],
        regime: str,
        risk_checks: list[str],
        features: dict[str, float] | None = None,
    ) -> AIExplanation:
        """
        Log a trade decision with full explanation.

        Creates a comprehensive AI explanation for the trade decision,
        including human-readable reasoning.

        Args:
            order_id: Unique order identifier
            symbol: Trading pair symbol
            signal: Trade signal ("LONG", "SHORT", "HOLD")
            confidence: Confidence level (0-1)
            weighted_avg: Weighted average of model predictions
            model_outputs: Dictionary of individual model predictions
            regime: Current market regime
            risk_checks: List of passed risk checks
            features: Optional dictionary of technical features for reasoning

        Returns:
            AIExplanation with all trade decision details
        """
        reasoning = self.generate_reasoning(
            signal=signal,
            confidence=confidence,
            regime=regime,
            model_outputs=model_outputs,
            features=features or {},
        )

        explanation = AIExplanation(
            order_id=order_id,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            weighted_average=weighted_avg,
            model_outputs=model_outputs,
            regime=regime,
            risk_checks=risk_checks,
            reasoning=reasoning,
        )

        logger.info(
            "Trade decision logged",
            extra={
                "order_id": order_id,
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
            },
        )

        return explanation

    def generate_reasoning(
        self,
        signal: str,
        confidence: float,
        regime: str,
        model_outputs: dict[str, float],
        features: dict[str, float],
    ) -> str:
        """
        Generate human-readable reasoning for a trade decision.

        Creates a detailed explanation combining model predictions,
        market regime, and technical indicators.

        Args:
            signal: Trade signal ("LONG", "SHORT", "HOLD")
            confidence: Confidence level (0-1)
            regime: Current market regime
            model_outputs: Dictionary of individual model predictions
            features: Dictionary of technical features

        Returns:
            Human-readable explanation string
        """
        # Format confidence as percentage
        confidence_pct = int(confidence * 100)

        # Format regime for display
        regime_display = regime.replace("_", " ")

        # Build model outputs summary
        model_parts: list[str] = []
        model_name_map = {
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "lstm": "LSTM",
            "rf": "RF",
            "random_forest": "RF",
            "gradient_boosting": "GradientBoost",
        }

        for model_name, prediction in model_outputs.items():
            display_name = model_name_map.get(model_name.lower(), model_name)
            model_parts.append(f"{display_name} predicts {prediction:.2f}")

        model_summary = ", ".join(model_parts) if model_parts else "No model outputs"

        # Build feature summary
        feature_parts: list[str] = []

        # ADX indicator
        adx = features.get("adx")
        if adx is not None:
            if adx >= 25:
                feature_parts.append(f"ADX at {adx:.0f}")
            else:
                feature_parts.append(f"ADX weak at {adx:.0f}")

        # RSI indicator
        rsi = features.get("rsi")
        if rsi is not None:
            if rsi > 70:
                feature_parts.append(f"RSI overbought at {rsi:.0f}")
            elif rsi < 30:
                feature_parts.append(f"RSI oversold at {rsi:.0f}")
            else:
                feature_parts.append(f"RSI at {rsi:.0f}")

        # EMA relationship
        price = features.get("close") or features.get("price")
        ema20 = features.get("ema_20") or features.get("ema20")
        if price is not None and ema20 is not None:
            if price > ema20:
                feature_parts.append("price above EMA20")
            else:
                feature_parts.append("price below EMA20")

        # MACD
        macd = features.get("macd")
        macd_signal = features.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                feature_parts.append("MACD bullish crossover")
            else:
                feature_parts.append("MACD bearish")

        # Momentum description
        momentum = features.get("momentum")
        if momentum is not None:
            if signal == "LONG" and momentum > 0:
                feature_parts.insert(0, "Strong bullish momentum")
            elif signal == "SHORT" and momentum < 0:
                feature_parts.insert(0, "Strong bearish momentum")

        # If no momentum but we have ADX for strength
        if not momentum and adx is not None and adx >= 30:
            if signal == "LONG":
                feature_parts.insert(0, "Strong bullish momentum")
            elif signal == "SHORT":
                feature_parts.insert(0, "Strong bearish momentum")

        feature_summary = " with " + " and ".join(feature_parts) if feature_parts else ""

        # Build the reasoning string
        reasoning = (
            f"{signal} signal generated with {confidence_pct}% confidence. "
            f"Market regime: {regime_display}. "
            f"{model_summary}."
        )

        if feature_summary:
            reasoning += feature_summary + "."

        return reasoning

    async def save_to_file(self, explanation: AIExplanation) -> Path:
        """
        Save AI explanation to a JSON file.

        Creates a file in the ai_logs directory with the order_id as filename.

        Args:
            explanation: AIExplanation to save

        Returns:
            Path to the saved file
        """
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp and order_id
        timestamp_str = explanation.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{explanation.order_id}.json"
        file_path = self.log_dir / filename

        # Write to file
        with open(file_path, "w") as f:
            json.dump(explanation.to_dict(), f, indent=2)

        logger.info(f"AI explanation saved to {file_path}")
        return file_path

    async def save_to_database(self, explanation: AIExplanation) -> None:
        """
        Save AI explanation to the database.

        Stores the explanation in the ai_log_uploads table for
        WEEX competition compliance tracking.

        Args:
            explanation: AIExplanation to save
        """
        db = await self._get_database()
        log_entry = explanation.to_ai_log_entry()
        await db.save_ai_log(log_entry)

        logger.info(
            "AI explanation saved to database",
            extra={"order_id": explanation.order_id},
        )

    async def get_pending_uploads(self) -> list[AIExplanation]:
        """
        Get AI explanations that have not yet been uploaded.

        Returns logs from the database that are marked as not uploaded,
        for batch upload to WEEX competition API.

        Returns:
            List of AIExplanation objects pending upload
        """
        db = await self._get_database()
        pending_logs = await db.get_pending_ai_logs()

        explanations: list[AIExplanation] = []
        for log in pending_logs:
            try:
                # Parse JSON fields
                model_outputs = json.loads(log["model_outputs"])
                risk_checks = json.loads(log["risk_checks"])

                explanation = AIExplanation(
                    order_id=log["order_id"],
                    symbol=log["symbol"],
                    signal=log["signal"],
                    confidence=log["confidence"],
                    weighted_average=log["weighted_average"],
                    model_outputs=model_outputs,
                    regime=log["regime"],
                    risk_checks=risk_checks,
                    reasoning=log["reasoning"],
                    timestamp=datetime.fromisoformat(log["timestamp"]),
                )
                explanations.append(explanation)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse AI log: {e}", extra={"log_id": log.get("id")})

        logger.info(f"Retrieved {len(explanations)} pending AI log uploads")
        return explanations
