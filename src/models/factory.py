import torch
from typing import Optional
from .classifier import NarrativeClassifier
from ..utils.config import Config

def create_model(config: Config, device: str = "cuda", checkpoint_path: Optional[str] = None) -> NarrativeClassifier:
    """
    Create or load a NarrativeClassifier model.
    Args:
        config: Model configuration
        device: Device to load the model on
        checkpoint_path: Optional path to model checkpoint
    Returns:
        Initialized NarrativeClassifier
    """
    # Create model instance
    model = NarrativeClassifier(config)

    # Load checkpoint if provided
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    # Move model to device
    model = model.to(device)

    return model
