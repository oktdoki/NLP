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
        num_subnarratives: int,
        narrative_to_idx: Optional[Dict[str, int]] = None,  # Add this
        subnarrative_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the classifier.

        Args:
            config: Model configuration
            num_narratives: Number of narrative classes
            num_subnarratives: Number of subnarrative classes
            narrative_to_idx: Optional mapping from narrative labels to indices
            subnarrative_to_idx: Optional mapping from subnarrative labels to indices
        """
        super().__init__()

        # Store label mappings
        self.narrative_to_idx = narrative_to_idx or {}
        self.subnarrative_to_idx = subnarrative_to_idx or {}
        # Load base model
        self.base_model = AutoModel.from_pretrained(config.model_name)

        self.narrative_to_subnarratives = {}
        for subnarr, subidx in subnarrative_to_idx.items():
            if ':' in subnarr:
                narr = subnarr.split(':')[0].strip()
                if narr in narrative_to_idx:
                    narr_idx = narrative_to_idx[narr]
                    if narr_idx not in self.narrative_to_subnarratives:
                        self.narrative_to_subnarratives[narr_idx] = []
                    self.narrative_to_subnarratives[narr_idx].append(subidx)

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
            subnarrative_weight=config.subnarrative_loss_weight,
            narrative_to_subnarratives=self.narrative_to_subnarratives
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
        """Make predictions with refined thresholding."""
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)

            # Get probabilities
            narrative_probs = torch.sigmoid(outputs['narrative_logits'])
            subnarrative_probs = torch.sigmoid(outputs['subnarrative_logits'])

            # Dynamic thresholding for narratives
            # Take only the top probability if it's above threshold
            max_narrative_probs, max_narrative_indices = narrative_probs.max(dim=1)
            narrative_preds = torch.zeros_like(narrative_probs)
            above_thresh = max_narrative_probs > self.config.narrative_threshold
            narrative_preds[above_thresh, max_narrative_indices[above_thresh]] = 1

            # Add "Other" only if no other narrative is predicted
            other_idx = self.narrative_to_idx.get("Other", -1)
            if other_idx != -1:
                no_predictions = (narrative_preds.sum(dim=1) == 0)
                narrative_preds[no_predictions, other_idx] = 1

            subnarrative_preds = torch.zeros_like(subnarrative_probs)

            for i in range(narrative_preds.size(0)):
                for narr_idx in torch.nonzero(narrative_preds[i]).squeeze(1):
                    if narr_idx in self.narrative_to_subnarratives:
                        valid_subnarrs = self.narrative_to_subnarratives[narr_idx]
                        valid_probs = subnarrative_probs[i, valid_subnarrs]
                        valid_preds = valid_probs > self.config.subnarrative_threshold
                        subnarrative_preds[i, valid_subnarrs] = valid_preds

                        if valid_preds.sum() == 0:
                            best_subnarr = valid_subnarrs[valid_probs.argmax()]
                            subnarrative_preds[i, best_subnarr] = 1

        return narrative_preds, subnarrative_preds