# tests/test_model.py

import torch
import pytest
from src.utils.config import load_config
from src.models.factory import create_model

@pytest.fixture
def config():
    return load_config('tests/test_config.yaml')

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def test_model_creation(config, device):
    model = create_model(config, device)
    assert model is not None
    assert next(model.parameters()).device.type == device

def test_model_forward(config, device):
    model = create_model(config, device)

    # Create dummy batch
    batch_size = 2
    seq_length = 32
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(device),
        'attention_mask': torch.ones(batch_size, seq_length).to(device),
    }

    # Test forward pass without labels
    outputs = model(**batch)
    assert 'narrative_logits' in outputs
    assert 'subnarrative_logits' in outputs

    # Check output shapes
    assert outputs['narrative_logits'].shape == (batch_size, len(config.data.label_categories['narratives']))
    assert outputs['subnarrative_logits'].shape == (batch_size, len(config.data.label_categories['subnarratives']))

def test_model_training(config, device):
    model = create_model(config, device)
    model.train()  # Put model in training mode

    batch_size = 2
    seq_length = 32
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(device),
        'attention_mask': torch.ones(batch_size, seq_length).to(device),
        'labels': {
            'narratives': torch.randint(0, 2, (batch_size, len(config.data.label_categories['narratives']))).float().to(device),
            'subnarratives': torch.randint(0, 2, (batch_size, len(config.data.label_categories['subnarratives']))).float().to(device)
        }
    }

    # Zero gradients
    model.zero_grad()

    # Forward pass
    outputs = model(**batch)
    assert 'loss' in outputs
    assert outputs['loss'].requires_grad

    # Backward pass
    outputs['loss'].backward()

    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_grad = True
                break
    assert has_grad, "No gradients were computed"

