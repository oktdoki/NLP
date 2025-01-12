# tests/test_trainer.py

import torch
from torch.utils.data import TensorDataset, DataLoader
import pytest
from src.utils.config import load_config
from src.models.factory import create_model
from src.training.trainer import Trainer

@pytest.fixture
def config():
    return load_config('tests/test_config.yaml')

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture
def dummy_data(config, device):
    # Create dummy dataset
    batch_size = 2
    num_samples = 10
    seq_length = 32

    # Create random data
    input_ids = torch.randint(0, 1000, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length)
    narrative_labels = torch.randint(0, 2, (num_samples, len(config.data.label_categories['narratives'])))
    subnarrative_labels = torch.randint(0, 2, (num_samples, len(config.data.label_categories['subnarratives'])))

    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, narrative_labels, subnarrative_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_trainer_initialization(config, device, dummy_data):
    model = create_model(config, device)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dummy_data,
        val_loader=dummy_data,
        device=device
    )

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None

@pytest.mark.skip(reason="Full training test takes too long")
def test_training_loop(config, device, dummy_data):
    model = create_model(config, device)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dummy_data,
        val_loader=dummy_data,
        device=device
    )

    # Run one epoch
    trainer.train()

    # Check that model parameters were updated
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

def test_single_batch_step(config, device, dummy_data):
    model = create_model(config, device)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dummy_data,
        val_loader=dummy_data,
        device=device
    )

    # Get a single batch
    batch = next(iter(dummy_data))
    batch = {
        'input_ids': batch[0].to(device),
        'attention_mask': batch[1].to(device),
        'narrative_labels': batch[2].to(device),
        'subnarrative_labels': batch[3].to(device)
    }

    # Process batch
    model.train()
    model.zero_grad()  # Explicitly zero gradients

    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels={
            'narratives': batch['narrative_labels'],
            'subnarratives': batch['subnarrative_labels']
        }
    )

    assert 'loss' in outputs
    loss = outputs['loss']
    loss.backward()

    # Check at least some parameters have gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients were computed"