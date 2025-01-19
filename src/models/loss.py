# src/models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class NarrativeClassificationLoss(nn.Module):
    """Loss function for multi-label narrative classification with class balancing and hierarchy."""

    def __init__(
        self,
        narrative_weight: float = 1.0,
        subnarrative_weight: float = 1.0,
        hierarchy_weight: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        Initialize loss function.

        Args:
            narrative_weight: Weight for narrative classification loss
            subnarrative_weight: Weight for subnarrative classification loss
            hierarchy_weight: Weight for hierarchical consistency loss
            reduction: Reduction method ('mean' or 'sum')
        """
        super().__init__()
        self.narrative_weight = narrative_weight
        self.subnarrative_weight = subnarrative_weight
        self.hierarchy_weight = hierarchy_weight
        self.reduction = reduction

    def forward(
        self,
        narrative_logits: torch.Tensor,
        subnarrative_logits: torch.Tensor,
        narrative_labels: torch.Tensor,
        subnarrative_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate the weighted loss with class balancing and hierarchical consistency.

        Args:
            narrative_logits: Predicted narrative logits (batch_size, num_narratives)
            subnarrative_logits: Predicted subnarrative logits (batch_size, num_subnarratives)
            narrative_labels: True narrative labels (batch_size, num_narratives)
            subnarrative_labels: True subnarrative labels (batch_size, num_subnarratives)

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual losses
        """
        # Calculate class weights based on inverse frequency
        narrative_pos_weight = (1 - narrative_labels.float()).sum(0) / (narrative_labels.float().sum(0) + 1e-6)
        subnarrative_pos_weight = (1 - subnarrative_labels.float()).sum(0) / (subnarrative_labels.float().sum(0) + 1e-6)

        # Handle edge cases where a class might not appear in the batch
        narrative_pos_weight = torch.clamp(narrative_pos_weight, 1.0, 50.0)
        subnarrative_pos_weight = torch.clamp(subnarrative_pos_weight, 1.0, 50.0)

        # Calculate BCE loss for narratives with class weights
        narrative_loss = F.binary_cross_entropy_with_logits(
            narrative_logits,
            narrative_labels.float(),
            pos_weight=narrative_pos_weight.to(narrative_logits.device),
            reduction=self.reduction
        )

        # Calculate BCE loss for subnarratives with class weights
        subnarrative_loss = F.binary_cross_entropy_with_logits(
            subnarrative_logits,
            subnarrative_labels.float(),
            pos_weight=subnarrative_pos_weight.to(subnarrative_logits.device),
            reduction=self.reduction
        )

        # Calculate hierarchical consistency loss
        narrative_probs = torch.sigmoid(narrative_logits)
        subnarrative_probs = torch.sigmoid(subnarrative_logits)

        # Each subnarrative should not have higher probability than its parent narrative
        max_subnarrative_probs = torch.max(subnarrative_probs, dim=1)[0]
        hierarchy_loss = F.relu(max_subnarrative_probs - narrative_probs.max(dim=1)[0]).mean()

        # Add focal loss term to handle extreme class imbalance
        gamma = 2.0  # focusing parameter
        narrative_focal = ((1 - narrative_probs) ** gamma * narrative_labels.float() * F.logsigmoid(narrative_logits)).mean()
        subnarrative_focal = ((1 - subnarrative_probs) ** gamma * subnarrative_labels.float() * F.logsigmoid(subnarrative_logits)).mean()

        # Combine all losses
        total_loss = (
            self.narrative_weight * (narrative_loss - 0.1 * narrative_focal) +
            self.subnarrative_weight * (subnarrative_loss - 0.1 * subnarrative_focal) +
            self.hierarchy_weight * hierarchy_loss
        )

        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            'narrative_loss': narrative_loss,
            'subnarrative_loss': subnarrative_loss,
            'hierarchy_loss': hierarchy_loss,
            'narrative_focal_term': narrative_focal,
            'subnarrative_focal_term': subnarrative_focal
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
        # Get probabilities
        narrative_probs = torch.sigmoid(narrative_logits)
        subnarrative_probs = torch.sigmoid(subnarrative_logits)

        # Dynamic thresholding for narratives
        k_narratives = 2  # Based on average narratives per article
        narrative_thresh = narrative_probs.topk(k_narratives, dim=1)[0][:, -1].unsqueeze(-1)
        narrative_thresh = torch.maximum(
            narrative_thresh,
            torch.tensor(threshold).to(narrative_thresh.device)
        )

        # Convert to predictions
        narrative_preds = (narrative_probs >= narrative_thresh).float()
        subnarrative_preds = (subnarrative_probs >= threshold).float()

        # Ensure at least one narrative is predicted
        zero_narratives = (narrative_preds.sum(dim=1) == 0)
        if zero_narratives.any():
            top_narratives = narrative_probs[zero_narratives].argmax(dim=1)
            narrative_preds[zero_narratives, top_narratives] = 1

        # Calculate metrics
        narrative_accuracy = (narrative_preds == narrative_labels).float().mean()
        narrative_f1 = self._calculate_f1(narrative_preds, narrative_labels)

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