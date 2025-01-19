# src/models/config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the narrative classification model."""

    # Base model configuration
    model_name: str = "xlm-roberta-base"
    max_length: int = 512

    # Model architecture
    dropout_prob: float = 0.1
    hidden_size: int = 768  # Default for xlm-roberta-base

    # Training configuration
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 10
    batch_size: int = 16

    # Multi-task learning weights
    narrative_loss_weight: float = 1.0
    subnarrative_loss_weight: float = 1.0

    # Thresholds for prediction
    narrative_threshold: float = 0.5
    subnarrative_threshold: float = 0.5

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dropout_prob < 0 or self.dropout_prob > 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")
        if self.max_epochs <= 0:
            raise ValueError("Max epochs must be positive")