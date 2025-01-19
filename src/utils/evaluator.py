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
        """
        Initialize the evaluator.

        Args:
            model: Model to evaluate
            test_loader: DataLoader for test data
            narrative_idx_to_label: Mapping from narrative indices to labels
            subnarrative_idx_to_label: Mapping from subnarrative indices to labels
            device: Device to use for evaluation
        """
        self.model = model
        self.test_loader = test_loader
        self.narrative_idx_to_label = narrative_idx_to_label
        self.subnarrative_idx_to_label = subnarrative_idx_to_label
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_narrative_preds = []
        all_narrative_labels = []
        all_subnarrative_preds = []
        all_subnarrative_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get predictions
                narrative_preds, subnarrative_preds = self.model.predict(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                # Collect predictions and labels
                all_narrative_preds.append(narrative_preds.cpu())
                all_narrative_labels.append(batch['narrative_labels'].cpu())
                all_subnarrative_preds.append(subnarrative_preds.cpu())
                all_subnarrative_labels.append(batch['subnarrative_labels'].cpu())

        # Concatenate all predictions and labels
        narrative_preds = torch.cat(all_narrative_preds, dim=0).numpy()
        narrative_labels = torch.cat(all_narrative_labels, dim=0).numpy()
        subnarrative_preds = torch.cat(all_subnarrative_preds, dim=0).numpy()
        subnarrative_labels = torch.cat(all_subnarrative_labels, dim=0).numpy()

        # Calculate metrics
        metrics = {}

        # Narrative metrics
        n_precision, n_recall, n_f1, _ = precision_recall_fscore_support(
            narrative_labels, narrative_preds, average='weighted'
        )
        metrics['narrative_precision'] = n_precision
        metrics['narrative_recall'] = n_recall
        metrics['narrative_f1'] = n_f1

        # Subnarrative metrics
        s_precision, s_recall, s_f1, _ = precision_recall_fscore_support(
            subnarrative_labels, subnarrative_preds, average='weighted'
        )
        metrics['subnarrative_precision'] = s_precision
        metrics['subnarrative_recall'] = s_recall
        metrics['subnarrative_f1'] = s_f1

        # Per-class metrics
        self.narrative_class_metrics = self._calculate_per_class_metrics(
            narrative_labels, narrative_preds, self.narrative_idx_to_label
        )

        self.subnarrative_class_metrics = self._calculate_per_class_metrics(
            subnarrative_labels, subnarrative_preds, self.subnarrative_idx_to_label
        )

        return metrics

    def _calculate_per_class_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        idx_to_label: Dict[int, str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and F1 for each class."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )

        class_metrics = {}
        for idx, (p, r, f) in enumerate(zip(precision, recall, f1)):
            label = idx_to_label[idx]
            class_metrics[label] = {
                'precision': p,
                'recall': r,
                'f1': f
            }

        return class_metrics

    def get_error_analysis(self, num_examples: int = 5) -> List[Dict]:
        """
        Get examples of prediction errors for analysis.

        Args:
            num_examples: Number of error examples to return

        Returns:
            List of dictionaries containing error examples
        """
        self.model.eval()
        errors = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get predictions
                narrative_preds, subnarrative_preds = self.model.predict(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                # Find errors
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
        """Format a single error example."""
        # Get true and predicted labels
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
            'predicted_subnarratives': pred_subnarratives
        }