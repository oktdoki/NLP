# src/models/factory.py

from typing import Optional
import torch
from .config import ModelConfig
from .classifier import NarrativeClassifier

class ModelFactory:
    """Factory class for creating and managing narrative classifiers."""
    
    @staticmethod
    def create_model(
        num_narratives: int,
        num_subnarratives: int,
        config: Optional[ModelConfig] = None
    ) -> NarrativeClassifier:
        """
        Create a new narrative classifier.
        
        Args:
            num_narratives: Number of narrative classes
            num_subnarratives: Number of subnarrative classes
            config: Optional model configuration
            
        Returns:
            NarrativeClassifier: Initialized model
        """
        # Use default config if none provided
        if config is None:
            config = ModelConfig()
            
        # Create and return model
        model = NarrativeClassifier(
            config=config,
            num_narratives=num_narratives,
            num_subnarratives=num_subnarratives
        )
        
        return model
    
    @staticmethod
    def save_model(model: NarrativeClassifier, path: str):
        """
        Save model to disk.
        
        Args:
            model: Model to save
            path: Path to save model to
        """
        torch.save({
            'state_dict': model.state_dict(),
            'config': model.config
        }, path)
    
    @staticmethod
    def load_model(path: str) -> NarrativeClassifier:
        """
        Load model from checkpoint.

        Args:
            path: Path to load model from

        Returns:
            NarrativeClassifier: Loaded model
        """
        # Load checkpoint
        try:
            checkpoint = torch.load(path)

            # Handle different checkpoint formats
            state_dict = checkpoint.get('state_dict', checkpoint)
            config = checkpoint.get('config', ModelConfig())

            # Get dimensions from the state dict
            narrative_weight = state_dict['narrative_classifier.4.weight']
            subnarrative_weight = state_dict['subnarrative_classifier.4.weight']

            num_narratives = narrative_weight.size(0)
            num_subnarratives = subnarrative_weight.size(0)

            # Create model
            model = NarrativeClassifier(
                config=config,
                num_narratives=num_narratives,
                num_subnarratives=num_subnarratives
            )

            # Load state dict
            model.load_state_dict(state_dict)

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")