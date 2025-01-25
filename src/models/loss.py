# src/models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class NarrativeClassificationLoss(nn.Module):
    """Loss function for multi-label narrative classification with class balancing and hierarchy."""

    def __init__(
            self,
            narrative_weight: float = 1.0,
            subnarrative_weight: float = 1.0,
            hierarchy_weight: float = 0.5,
            reduction: str = 'mean',
            narrative_to_subnarratives: Optional[Dict[int, List[int]]] = None
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
        self.narrative_to_subnarratives = narrative_to_subnarratives or {}

    def forward(
            self,
            narrative_logits: torch.Tensor,
            subnarrative_logits: torch.Tensor,
            narrative_labels: torch.Tensor,
            subnarrative_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # Basis loss Berechnung bleibt gleich
        narrative_pos_weight = (1 - narrative_labels.float()).sum(0) / (narrative_labels.float().sum(0) + 1e-6)
        subnarrative_pos_weight = (1 - subnarrative_labels.float()).sum(0) / (subnarrative_labels.float().sum(0) + 1e-6)

        narrative_pos_weight = torch.clamp(narrative_pos_weight, 1.0, 50.0)
        subnarrative_pos_weight = torch.clamp(subnarrative_pos_weight, 1.0, 50.0)

        narrative_loss = F.binary_cross_entropy_with_logits(
            narrative_logits,
            narrative_labels.float(),
            pos_weight=narrative_pos_weight.to(narrative_logits.device),
            reduction=self.reduction
        )

        # Subnarrative loss nur für valide Subnarrative des jeweiligen Narrativs
        batch_size = narrative_logits.size(0)
        subnarrative_loss = 0
        valid_count = 0

        for i in range(batch_size):
            for narr_idx in torch.nonzero(narrative_labels[i]).squeeze(1):
                if narr_idx in self.narrative_to_subnarratives:
                    valid_subnarrs = self.narrative_to_subnarratives[narr_idx]
                    subnarr_logits = subnarrative_logits[i, valid_subnarrs]
                    subnarr_labels = subnarrative_labels[i, valid_subnarrs]
                    subnarr_weights = subnarrative_pos_weight[valid_subnarrs]

                    if len(valid_subnarrs) > 0:
                        subnarrative_loss += F.binary_cross_entropy_with_logits(
                            subnarr_logits,
                            subnarr_labels.float(),
                            pos_weight=subnarr_weights.to(subnarr_logits.device),
                            reduction=self.reduction
                        )
                        valid_count += 1

        if valid_count > 0:
            subnarrative_loss = subnarrative_loss / valid_count

        # Hierarchische Loss-Berechnung
        narrative_probs = torch.sigmoid(narrative_logits)
        subnarrative_probs = torch.sigmoid(subnarrative_logits)

        hierarchy_loss = 0
        for i in range(batch_size):
            for narr_idx in torch.nonzero(narrative_labels[i]).squeeze(1):
                if narr_idx in self.narrative_to_subnarratives:
                    valid_subnarrs = self.narrative_to_subnarratives[narr_idx]
                    subnarr_probs = subnarrative_probs[i, valid_subnarrs]
                    narr_prob = narrative_probs[i, narr_idx]
                    hierarchy_loss += F.relu(subnarr_probs.max() - narr_prob)

        hierarchy_loss = hierarchy_loss / batch_size

        # Focal Loss bleibt gleich
        gamma = 2.0
        narrative_focal = (
                    (1 - narrative_probs) ** gamma * narrative_labels.float() * F.logsigmoid(narrative_logits)).mean()
        subnarrative_focal = ((1 - subnarrative_probs) ** gamma * subnarrative_labels.float() * F.logsigmoid(
            subnarrative_logits)).mean()

        total_loss = (
                self.narrative_weight * (narrative_loss - 0.1 * narrative_focal) +
                self.subnarrative_weight * (subnarrative_loss - 0.1 * subnarrative_focal) +
                self.hierarchy_weight * hierarchy_loss
        )

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

        narrative_probs = torch.sigmoid(narrative_logits)
        subnarrative_probs = torch.sigmoid(subnarrative_logits)

        # Narrative predictions wie bisher
        k_narratives = 2
        narrative_thresh = narrative_probs.topk(k_narratives, dim=1)[0][:, -1].unsqueeze(-1)
        narrative_thresh = torch.maximum(
            narrative_thresh,
            torch.tensor(threshold).to(narrative_thresh.device)
        )
        narrative_preds = (narrative_probs >= narrative_thresh).float()

        # Subnarrative predictions basierend auf Narrative
        subnarrative_preds = torch.zeros_like(subnarrative_probs)

        for i in range(narrative_preds.size(0)):
            for narr_idx in torch.nonzero(narrative_preds[i]).squeeze(1):
                if narr_idx in self.narrative_to_subnarratives:
                    valid_subnarrs = self.narrative_to_subnarratives[narr_idx]
                    valid_probs = subnarrative_probs[i, valid_subnarrs]
                    valid_preds = valid_probs > threshold
                    subnarrative_preds[i, valid_subnarrs] = valid_preds

                    if valid_preds.sum() == 0:
                        best_subnarr = valid_subnarrs[valid_probs.argmax()]
                        subnarrative_preds[i, best_subnarr] = 1

        # Metrik-Berechnung
        narrative_accuracy = (narrative_preds == narrative_labels).float().mean()
        narrative_f1 = self._calculate_f1(narrative_preds, narrative_labels)

        subnarrative_accuracy = (subnarrative_preds == subnarrative_labels).float().mean()
        subnarrative_f1 = self._calculate_f1(subnarrative_preds, subnarrative_labels)

        # Hierarchie-Metrik
        hierarchy_accuracy = self._calculate_hierarchy_accuracy(
            narrative_preds, subnarrative_preds, narrative_labels, subnarrative_labels
        )

        return {
            'narrative_accuracy': narrative_accuracy.item(),
            'narrative_f1': narrative_f1.item(),
            'subnarrative_accuracy': subnarrative_accuracy.item(),
            'subnarrative_f1': subnarrative_f1.item(),
            'hierarchy_accuracy': hierarchy_accuracy.item()
        }

    def _calculate_hierarchy_accuracy(
            self,
            narrative_preds: torch.Tensor,
            subnarrative_preds: torch.Tensor,
            narrative_labels: torch.Tensor,
            subnarrative_labels: torch.Tensor
    ) -> torch.Tensor:
        """Berechnet wie gut die hierarchischen Beziehungen eingehalten werden"""

        batch_size = narrative_preds.size(0)
        correct = 0
        total = 0

        for i in range(batch_size):
            for narr_idx in torch.nonzero(narrative_labels[i]).squeeze(1):
                if narr_idx in self.narrative_to_subnarratives:
                    valid_subnarrs = self.narrative_to_subnarratives[narr_idx]
                    pred_match = (subnarrative_preds[i, valid_subnarrs].sum() > 0) == narrative_preds[i, narr_idx]
                    correct += pred_match.item()
                    total += 1

        return torch.tensor(correct / max(total, 1))

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
