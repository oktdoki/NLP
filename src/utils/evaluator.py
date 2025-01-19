# src/utils/evaluator.py

import torch
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ..models.classifier import NarrativeClassifier

class ModelEvaluator:
    """Class for evaluating narrative classification models."""

    def __init__(
        self,
        model: NarrativeClassifier,
        test_loader: DataLoader,
        narrative_idx_to_label: Dict[int, str],
        subnarrative_idx_to_label: Dict[int, str],
        device: str = None
    ):
        self.model = model
        self.test_loader = test_loader
        self.narrative_idx_to_label = narrative_idx_to_label
        self.subnarrative_idx_to_label = subnarrative_idx_to_label
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Initialize metrics dictionaries
        self.narrative_class_metrics = {}
        self.subnarrative_class_metrics = {}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        self.model.eval()
        all_narrative_preds = []
        all_narrative_labels = []
        all_subnarrative_preds = []
        all_subnarrative_labels = []

        # Collect predictions
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                narrative_preds, subnarrative_preds = self.model.predict(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                all_narrative_preds.append(narrative_preds.cpu())
                all_narrative_labels.append(batch['narrative_labels'].cpu())
                all_subnarrative_preds.append(subnarrative_preds.cpu())
                all_subnarrative_labels.append(batch['subnarrative_labels'].cpu())

        # Concatenate predictions and labels
        narrative_preds = torch.cat(all_narrative_preds, dim=0).numpy()
        narrative_labels = torch.cat(all_narrative_labels, dim=0).numpy()
        subnarrative_preds = torch.cat(all_subnarrative_preds, dim=0).numpy()
        subnarrative_labels = torch.cat(all_subnarrative_labels, dim=0).numpy()

        # Calculate overall metrics
        metrics = {}

        # Calculate narrative metrics
        narrative_metrics = self._calculate_metrics(
            narrative_labels,
            narrative_preds,
            "narrative"
        )
        metrics.update(narrative_metrics)

        # Calculate subnarrative metrics
        subnarrative_metrics = self._calculate_metrics(
            subnarrative_labels,
            subnarrative_preds,
            "subnarrative"
        )
        metrics.update(subnarrative_metrics)

        return metrics

    def _calculate_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """Calculate metrics for a set of predictions."""
        # Overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )

        metrics = {
            f'{prefix}_precision': precision,
            f'{prefix}_recall': recall,
            f'{prefix}_f1': f1
        }

        # Per-class metrics
        per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        # Store per-class metrics
        class_metrics = {}
        idx_to_label = (self.narrative_idx_to_label if prefix == "narrative"
                       else self.subnarrative_idx_to_label)

        for idx, (p, r, f) in enumerate(zip(per_class_p, per_class_r, per_class_f1)):
            label = idx_to_label[idx]
            class_metrics[label] = {
                'precision': p,
                'recall': r,
                'f1': f,
                'support': labels[:, idx].sum()  # Number of true instances
            }

        # Store in instance variables
        if prefix == "narrative":
            self.narrative_class_metrics = class_metrics
        else:
            self.subnarrative_class_metrics = class_metrics

        return metrics

    def get_error_analysis(self, num_examples: int = 5) -> List[Dict]:
        """Get examples of prediction errors."""
        self.model.eval()
        errors = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                narrative_preds, subnarrative_preds = self.model.predict(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                # Find misclassifications
                narrative_errors = (narrative_preds != batch['narrative_labels']).any(dim=1)
                subnarrative_errors = (subnarrative_preds != batch['subnarrative_labels']).any(dim=1)

                for i in range(len(batch['input_ids'])):
                    if narrative_errors[i] or subnarrative_errors[i]:
                        error = self._format_error_example(
                            batch, i, narrative_preds[i], subnarrative_preds[i]
                        )
                        errors.append(error)

                        if len(errors) >= num_examples:
                            return errors

        return errors

    def _format_error_example(
        self,
        batch: Dict[str, torch.Tensor],
        idx: int,
        narrative_pred: torch.Tensor,
        subnarrative_pred: torch.Tensor
    ) -> Dict:
        """Format an error example with detailed information."""
        # Get predictions and true labels
        true_narratives = [
            self.narrative_idx_to_label[i]
            for i, val in enumerate(batch['narrative_labels'][idx])
            if val == 1
        ]

        pred_narratives = [
            self.narrative_idx_to_label[i]
            for i, val in enumerate(narrative_pred)
            if val == 1
        ]

        true_subnarratives = [
            self.subnarrative_idx_to_label[i]
            for i, val in enumerate(batch['subnarrative_labels'][idx])
            if val == 1
        ]

        pred_subnarratives = [
            self.subnarrative_idx_to_label[i]
            for i, val in enumerate(subnarrative_pred)
            if val == 1
        ]

        return {
            'true_narratives': true_narratives,
            'predicted_narratives': pred_narratives,
            'true_subnarratives': true_subnarratives,
            'predicted_subnarratives': pred_subnarratives,
            'is_narrative_error': len(set(true_narratives) ^ set(pred_narratives)) > 0,
            'is_subnarrative_error': len(set(true_subnarratives) ^ set(pred_subnarratives)) > 0
        }