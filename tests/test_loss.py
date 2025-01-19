import torch
from src.models.loss import NarrativeClassificationLoss

def test_narrative_classification_loss():
    # Initialize the loss function
    loss_fn = NarrativeClassificationLoss(narrative_weight=1.0, subnarrative_weight=1.0)

    # Fake logits and labels for narratives and subnarratives
    batch_size = 4
    narrative_logits = torch.randn(batch_size, 21)  # Example: 21 narrative classes
    subnarrative_logits = torch.randn(batch_size, 69)  # Example: 69 subnarrative classes
    narrative_labels = torch.randint(0, 2, (batch_size, 21)).float()
    subnarrative_labels = torch.randint(0, 2, (batch_size, 69)).float()

    # Compute loss
    loss, components = loss_fn(
        narrative_logits=narrative_logits,
        subnarrative_logits=subnarrative_logits,
        narrative_labels=narrative_labels,
        subnarrative_labels=subnarrative_labels
    )

    # Print outputs for inspection
    print("Total Loss:", components['total_loss'].item())
    print("Narrative Loss:", components['narrative_loss'].item())
    print("Subnarrative Loss:", components['subnarrative_loss'].item())

    # Assertions to verify loss values are computed correctly
    assert loss > 0, "Total loss should be greater than 0"
    assert components['narrative_loss'] > 0, "Narrative loss should be greater than 0"
    assert components['subnarrative_loss'] > 0, "Subnarrative loss should be greater than 0"

if __name__ == "__main__":
    test_narrative_classification_loss()
