# src/models/factory.py

from typing import Optional, Dict, List
import torch
from .config import ModelConfig
from .classifier import NarrativeClassifier

class ModelFactory:
    """Factory class for creating and managing narrative classifiers."""
    
    @staticmethod
    def create_model(
        num_narratives: int,
        num_subnarratives: int,
        narrative_to_idx: Optional[Dict[str, int]] = None,
        subnarrative_to_idx: Optional[Dict[str, int]] = None,
        config: Optional[ModelConfig] = None,
        narrative_to_subnarratives: Optional[Dict[int, List[int]]] = None,
    ) -> NarrativeClassifier:
        """Create a new narrative classifier."""
        if config is None:
            config = ModelConfig()

        narrative_to_idx = narrative_to_idx or {}
        subnarrative_to_idx = subnarrative_to_idx or {}

        model = NarrativeClassifier(
            config=config,
            num_narratives=num_narratives,
            num_subnarratives=num_subnarratives,
            narrative_to_idx=narrative_to_idx,
            subnarrative_to_idx=subnarrative_to_idx,
        )

        return model

    
    @staticmethod
    def save_model(model: NarrativeClassifier, path: str):
        """Save model with only essential components."""
        save_dict = {
            'state_dict': model.state_dict(),
            'config': model.config,
            'num_narratives': model.narrative_classifier[4].weight.size(0),
            'num_subnarratives': model.subnarrative_classifier[4].weight.size(0)
        }

        # Save with compression to reduce file size
        torch.save(save_dict, path, _use_new_zipfile_serialization=True)
    
    @staticmethod
    def load_model(path: str) -> NarrativeClassifier:
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            config = checkpoint.get('config', ModelConfig())

            # Get dimensions from state dict
            narrative_weight = state_dict['narrative_classifier.4.weight']
            subnarrative_weight = state_dict['subnarrative_classifier.4.weight']

            num_narratives = narrative_weight.size(0)
            num_subnarratives = subnarrative_weight.size(0)

            # Load label mappings if they exist in checkpoint
            narrative_to_idx = checkpoint.get('narrative_to_idx', {})
            subnarrative_to_idx = checkpoint.get('subnarrative_to_idx', {})

            model = NarrativeClassifier(
                config=config,
                num_narratives=num_narratives,
                num_subnarratives=num_subnarratives,
                narrative_to_idx=narrative_to_idx,
                subnarrative_to_idx=subnarrative_to_idx
            )

            model.load_state_dict(state_dict)
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")