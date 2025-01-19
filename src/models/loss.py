# src/models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class NarrativeClassificationLoss(nn.Module):
    """Loss function for multi-label narrative classification."""

    def __init__(
        self,
        narrative_weight: float = 1.0,
        subnarrative_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize loss function.

        Args:
            narrative_weight: Weight for narrative classification loss
            subnarrative_weight: Weight for subnarrative classification loss
            reduction: Reduction method ('mean' or 'sum')
        """
        super().__init__()
        self.narrative_weight = narrative_weight
        self.subnarrative_weight = subnarrative_weight
        self.reduction = reduction

    def forward(
        self,
        narrative_logits: torch.Tensor,
        subnarrative_logits: torch.Tensor,
        narrative_labels: torch.Tensor,
        subnarrative_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate the weighted loss for both narrative and subnarrative predictions.

        Args:
            narrative_logits: Predicted narrative logits (batch_size, num_narratives)
            subnarrative_logits: Predicted subnarrative logits (batch_size, num_subnarratives)
            narrative_labels: True narrative labels (batch_size, num_narratives)
            subnarrative_labels: True subnarrative labels (batch_size, num_subnarratives)

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual losses
        """
        # Calculate BCE loss for narratives
        narrative_loss = F.binary_cross_entropy_with_logits(
            narrative_logits,
            narrative_labels.float(),
            reduction=self.reduction
        )

        # Calculate BCE loss for subnarratives
        subnarrative_loss = F.binary_cross_entropy_with_logits(
            subnarrative_logits,
            subnarrative_labels.float(),
            reduction=self.reduction
        )

        # Combine losses with weights
        total_loss = (
            self.narrative_weight * narrative_loss +
            self.subnarrative_weight * subnarrative_loss
        )

        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            'narrative_loss': narrative_loss,
            'subnarrative_loss': subnarrative_loss
        }

        return total_loss, loss_dict

    def calculate_metrics(
        self,
        narrative_logits: torch.Tensor,
        subnarrative_logits: torch.Tensor,
        narrative_labels: torch.Tensor,
        subnarrative_labels: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate various metrics for model evaluation.

        Args:
            narrative_logits: Predicted narrative logits
            subnarrative_logits: Predicted subnarrative logits
            narrative_labels: True narrative labels
            subnarrative_labels: True subnarrative labels
            threshold: Decision threshold for binary classification

        Returns:
            Dictionary containing various metrics
        """
        # Convert logits to predictions
        narrative_preds = (torch.sigmoid(narrative_logits) > threshold).float()
        subnarrative_preds = (torch.sigmoid(subnarrative_logits) > threshold).float()

        # Calculate metrics for narratives
        narrative_accuracy = (narrative_preds == narrative_labels).float().mean()
        narrative_f1 = self._calculate_f1(narrative_preds, narrative_labels)

        # Calculate metrics for subnarratives
        subnarrative_accuracy = (subnarrative_preds == subnarrative_labels).float().mean()
        subnarrative_f1 = self._calculate_f1(subnarrative_preds, subnarrative_labels)

        return {
            'narrative_accuracy': narrative_accuracy.item(),
            'narrative_f1': narrative_f1.item(),
            'subnarrative_accuracy': subnarrative_accuracy.item(),
            'subnarrative_f1': subnarrative_f1.item()
        }

    @staticmethod
    def _calculate_f1(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate F1 score for multi-label classification."""
        # Calculate true positives, false positives, false negatives
        true_positives = (preds * labels).sum(dim=1)
        false_positives = (preds * (1 - labels)).sum(dim=1)
        false_negatives = ((1 - preds) * labels).sum(dim=1)

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return f1.mean()