# src/models/classifier.py

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Dict, Optional, Tuple

class NarrativeClassifier(nn.Module):
    def __init__(self, config):
        """
        Initialize the narrative classifier.
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()

        # Load XLM-RoBERTa model
        self.xlmr = XLMRobertaModel.from_pretrained(
            config.model.name,
            hidden_dropout_prob=config.model.hidden_dropout_prob,
            attention_probs_dropout_prob=config.model.attention_dropout_prob
        )

        # Get hidden size from model config
        hidden_size = self.xlmr.config.hidden_size

        # Classification heads for narratives and subnarratives
        self.narrative_classifier = nn.Linear(hidden_size, len(config.data.label_categories["narratives"]))
        self.subnarrative_classifier = nn.Linear(hidden_size, len(config.data.label_categories["subnarratives"]))

        # Dropout for regularization
        self.dropout = nn.Dropout(config.model.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, labels=None):
        # Ensure we're using the right output attribs from xlmr
        outputs = self.xlmr(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get pooled output more explicitly
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        narrative_logits = self.narrative_classifier(pooled_output)
        subnarrative_logits = self.subnarrative_classifier(pooled_output)

        result = {
            "narrative_logits": narrative_logits,
            "subnarrative_logits": subnarrative_logits
        }

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            narrative_labels = labels["narratives"].float()
            subnarrative_labels = labels["subnarratives"].float()

            # Calculate losses
            narrative_loss = loss_fct(narrative_logits, narrative_labels)
            subnarrative_loss = loss_fct(subnarrative_logits, subnarrative_labels)

            # Combine losses with equal weighting
            total_loss = narrative_loss + subnarrative_loss
            result["loss"] = total_loss

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for narratives and subnarratives.
        Args:
            input_ids: Tensor of input token IDs
            attention_mask: Tensor of attention mask
            threshold: Probability threshold for binary predictions
        Returns:
            Tuple of narrative and subnarrative predictions
        """
        # Get logits
        outputs = self.forward(input_ids, attention_mask)

        # Apply sigmoid and threshold
        narrative_preds = torch.sigmoid(outputs["narrative_logits"]) > threshold
        subnarrative_preds = torch.sigmoid(outputs["subnarrative_logits"]) > threshold

        return narrative_preds, subnarrative_preds