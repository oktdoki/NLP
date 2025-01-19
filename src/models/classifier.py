# src/models/classifier.py

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple, Optional

from .config import ModelConfig
from .loss import NarrativeClassificationLoss

class NarrativeClassifier(nn.Module):
    """Multi-label classifier for narratives and subnarratives."""

    def __init__(
        self,
        config: ModelConfig,
        num_narratives: int,
        num_subnarratives: int
    ):
        """
        Initialize the classifier.

        Args:
            config: Model configuration
            num_narratives: Number of narrative classes
            num_subnarratives: Number of subnarrative classes
        """
        super().__init__()

        # Load base model
        self.base_model = AutoModel.from_pretrained(config.model_name)

        # Freeze base model if specified
        if hasattr(config, 'freeze_base') and config.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Classification heads
        self.narrative_classifier = nn.Sequential(
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size, num_narratives)
        )

        self.subnarrative_classifier = nn.Sequential(
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size, num_subnarratives)
        )

        # Loss function
        self.loss_fn = NarrativeClassificationLoss(
            narrative_weight=config.narrative_loss_weight,
            subnarrative_weight=config.subnarrative_loss_weight
        )

        # Save configuration
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        narrative_labels: Optional[torch.Tensor] = None,
        subnarrative_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            narrative_labels: Optional narrative labels for training
            subnarrative_labels: Optional subnarrative labels for training

        Returns:
            Dictionary containing model outputs and optionally losses
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Get logits from classification heads
        narrative_logits = self.narrative_classifier(pooled_output)
        subnarrative_logits = self.subnarrative_classifier(pooled_output)

        # Prepare output dictionary
        output_dict = {
            'narrative_logits': narrative_logits,
            'subnarrative_logits': subnarrative_logits
        }

        # Calculate loss if labels are provided
        if narrative_labels is not None and subnarrative_labels is not None:
            loss, loss_dict = self.loss_fn(
                narrative_logits,
                subnarrative_logits,
                narrative_labels,
                subnarrative_labels
            )
            output_dict.update(loss_dict)

            # Calculate metrics
            metrics = self.loss_fn.calculate_metrics(
                narrative_logits,
                subnarrative_logits,
                narrative_labels,
                subnarrative_labels,
                threshold=self.config.narrative_threshold
            )
            output_dict.update(metrics)

        return output_dict

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for inference.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of narrative and subnarrative predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)

            narrative_preds = torch.sigmoid(outputs['narrative_logits'])
            subnarrative_preds = torch.sigmoid(outputs['subnarrative_logits'])

            # Apply thresholds
            narrative_preds = (narrative_preds > self.config.narrative_threshold).float()
            subnarrative_preds = (subnarrative_preds > self.config.subnarrative_threshold).float()

        return narrative_preds, subnarrative_preds