"""
AlphaStrike Trading Bot - LSTM Model Wrapper

PyTorch-based LSTM model for time series prediction in trading.
Implements sliding window sequence generation and GPU acceleration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""

    input_size: int  # Number of features
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 20
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001


@dataclass
class TrainingResult:
    """Result of model training."""

    epochs_completed: int
    final_loss: float
    best_loss: float
    training_time_seconds: float
    samples_used: int
    history: dict[str, list[float]] = field(default_factory=dict)


class LSTMNetwork(nn.Module):
    """
    LSTM neural network for sequence prediction.

    Architecture:
    - LSTM layers with dropout
    - Linear output layer
    - Sigmoid activation for probability output
    """

    def __init__(self, config: LSTMConfig) -> None:
        """
        Initialize LSTM network.

        Args:
            config: LSTM configuration parameters
        """
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch, 1) with probabilities
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the last time step output
        last_output = lstm_out[:, -1, :]

        # Pass through linear layer and sigmoid
        out = self.fc(last_output)
        out = self.sigmoid(out)

        return out


class LSTMModel:
    """
    LSTM model wrapper for trading predictions.

    Handles training, prediction, serialization, and health checks.
    Automatically uses GPU if available.
    """

    def __init__(self, config: LSTMConfig) -> None:
        """
        Initialize LSTM model.

        Args:
            config: LSTM configuration parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: LSTMNetwork | None = None
        self._last_predictions: np.ndarray | None = None
        self._is_trained = False

        logger.info(
            "LSTMModel initialized",
            extra={
                "device": str(self.device),
                "input_size": config.input_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
            },
        )

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Create sequences using sliding window.

        Args:
            X: Feature array of shape (samples, features)
            y: Target array of shape (samples,), optional

        Returns:
            Tuple of (sequences, targets) where sequences has shape
            (num_sequences, sequence_length, features)
        """
        seq_length = self.config.sequence_length
        num_samples = len(X)

        if num_samples < seq_length:
            raise ValueError(
                f"Not enough samples ({num_samples}) for sequence length ({seq_length})"
            )

        num_sequences = num_samples - seq_length + 1
        sequences = np.zeros(
            (num_sequences, seq_length, X.shape[1]), dtype=np.float32
        )

        for i in range(num_sequences):
            sequences[i] = X[i : i + seq_length]

        targets = None
        if y is not None:
            # Target corresponds to the last element of each sequence
            targets = y[seq_length - 1 :].astype(np.float32)

        return sequences, targets

    def train(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Train the LSTM model.

        Args:
            X: Feature array of shape (samples, features)
            y: Target array of shape (samples,) with values 0 or 1

        Returns:
            TrainingResult with training metrics
        """
        import time

        start_time = time.time()

        # Validate input dimensions
        if X.shape[1] != self.config.input_size:
            raise ValueError(
                f"Input size mismatch: expected {self.config.input_size}, "
                f"got {X.shape[1]}"
            )

        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        if y_seq is None:
            raise ValueError("Target array is required for training")

        logger.info(
            "Training sequences created",
            extra={
                "num_sequences": len(X_seq),
                "sequence_shape": X_seq.shape,
            },
        )

        # Convert to tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).reshape(-1, 1).to(
            self.device
        )

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Initialize network
        self.network = LSTMNetwork(self.config).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )

        # Training loop
        history: dict[str, list[float]] = {"loss": []}
        best_loss = float("inf")

        for epoch in range(self.config.epochs):
            self.network.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history["loss"].append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"Epoch {epoch + 1}/{self.config.epochs}, Loss: {avg_loss:.6f}"
                )

        training_time = time.time() - start_time
        self._is_trained = True

        logger.info(
            "Training completed",
            extra={
                "epochs": self.config.epochs,
                "final_loss": history["loss"][-1],
                "best_loss": best_loss,
                "training_time": f"{training_time:.2f}s",
            },
        )

        return TrainingResult(
            epochs_completed=self.config.epochs,
            final_loss=history["loss"][-1],
            best_loss=best_loss,
            training_time_seconds=training_time,
            samples_used=len(X_seq),
            history=history,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.

        Args:
            X: Feature array of shape (samples, features)

        Returns:
            Array of probabilities (0-1) of shape (num_predictions,)
        """
        if self.network is None or not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Validate input dimensions
        if X.shape[1] != self.config.input_size:
            raise ValueError(
                f"Input size mismatch: expected {self.config.input_size}, "
                f"got {X.shape[1]}"
            )

        # Create sequences
        X_seq, _ = self._create_sequences(X)

        # Convert to tensor
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        # Predict
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(X_tensor)
            predictions = predictions.cpu().numpy().flatten()

        # Store for health check
        self._last_predictions = predictions

        return predictions

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model file
        """
        if self.network is None:
            raise RuntimeError("No model to save - train first")

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and config
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.network.state_dict(),
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "sequence_length": self.config.sequence_length,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
            },
            "is_trained": self._is_trained,
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to the model file
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint: dict[str, Any] = torch.load(
            path, map_location=self.device, weights_only=False
        )

        # Verify config compatibility
        saved_config = checkpoint["config"]
        if saved_config["input_size"] != self.config.input_size:
            raise ValueError(
                f"Input size mismatch: model has {saved_config['input_size']}, "
                f"config has {self.config.input_size}"
            )

        # Update config from checkpoint
        self.config = LSTMConfig(
            input_size=saved_config["input_size"],
            hidden_size=saved_config["hidden_size"],
            num_layers=saved_config["num_layers"],
            dropout=saved_config["dropout"],
            sequence_length=saved_config["sequence_length"],
            batch_size=saved_config["batch_size"],
            epochs=saved_config["epochs"],
            learning_rate=saved_config["learning_rate"],
        )

        # Create and load network
        self.network = LSTMNetwork(self.config).to(self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self._is_trained = checkpoint.get("is_trained", True)

        logger.info(f"Model loaded from {path}")

    def health_check(self) -> bool:
        """
        Check model health.

        Detects:
        - Low prediction variance (< 0.01)
        - Constant output

        Returns:
            True if model is healthy, False otherwise
        """
        if self.network is None or not self._is_trained:
            logger.warning("Health check failed: model not trained")
            return False

        if self._last_predictions is None or len(self._last_predictions) == 0:
            logger.warning("Health check failed: no predictions available")
            return False

        predictions = self._last_predictions

        # Check for low variance
        variance = float(np.var(predictions))
        if variance < 0.01:
            logger.warning(
                f"Health check failed: low prediction variance ({variance:.6f})"
            )
            return False

        # Check for constant output
        unique_values = np.unique(predictions)
        if len(unique_values) == 1:
            logger.warning(
                f"Health check failed: constant output ({unique_values[0]:.6f})"
            )
            return False

        # Check for extreme values (all near 0 or all near 1)
        mean_pred = float(np.mean(predictions))
        if mean_pred < 0.05 or mean_pred > 0.95:
            logger.warning(
                f"Health check warning: extreme mean prediction ({mean_pred:.4f})"
            )
            # Not failing, just warning - model might be strongly biased

        logger.debug(
            "Health check passed",
            extra={
                "variance": f"{variance:.6f}",
                "unique_values": len(unique_values),
                "mean": f"{mean_pred:.4f}",
            },
        )

        return True
