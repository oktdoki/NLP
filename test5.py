# test_training.py

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataset import NarrativeDataset
from src.models.factory import ModelFactory
from src.models.config import ModelConfig
from src.training.trainer import Trainer

def test_training():
    print("Testing training functionality...")

    # Use small configuration for testing
    test_config = ModelConfig(
        model_name='xlm-roberta-base',
        max_length=512,
        batch_size=2,  # Small batch size for testing
        max_epochs=2,  # Just 2 epochs for testing
        learning_rate=2e-5,
        warmup_steps=2
    )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(test_config.model_name)

    # Load small subset of data
    print("\nLoading datasets...")
    train_dataset = NarrativeDataset(
        data_dir='data_set/target_4_December_JSON',
        language='EN',
        split='train',
        tokenizer=tokenizer
    )

    val_dataset = NarrativeDataset(
        data_dir='data_set/target_4_December_JSON',
        language='EN',
        split='val',
        tokenizer=tokenizer
    )

    # Create minimal data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=test_config.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=test_config.batch_size,
        shuffle=False,
        num_workers=0  # Use 0 for testing
    )

    print("\nCreating model...")
    model = ModelFactory.create_model(
        num_narratives=len(train_dataset.narrative_to_idx),
        num_subnarratives=len(train_dataset.subnarrative_to_idx),
        config=test_config
    )

    # Test trainer initialization
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=test_config
    )

    # Test single training epoch
    print("\nTesting single training epoch...")
    try:
        train_metrics = trainer.train_epoch()
        print(f"Training metrics: {train_metrics}")
        print("Training epoch test passed!")
    except Exception as e:
        print(f"Training epoch test failed with error: {str(e)}")
        return

    # Test validation
    print("\nTesting validation...")
    try:
        val_metrics = trainer.validate()
        print(f"Validation metrics: {val_metrics}")
        print("Validation test passed!")
    except Exception as e:
        print(f"Validation test failed with error: {str(e)}")
        return

    # Test full training loop
    print("\nTesting full training loop...")
    try:
        trainer.train(checkpoint_dir='test_checkpoints')
        print("Full training loop test passed!")
    except Exception as e:
        print(f"Full training loop test failed with error: {str(e)}")
        return

    # Test model loading from checkpoint
    print("\nTesting checkpoint loading...")
    try:
        checkpoint_path = 'test_checkpoints/best_model.pt'
        checkpoint = torch.load(checkpoint_path)
        print("Checkpoint contains:")
        print(f"- Epoch: {checkpoint['epoch']}")
        print(f"- Metrics: {checkpoint['metrics']}")
        print("Checkpoint loading test passed!")
    except Exception as e:
        print(f"Checkpoint loading test failed with error: {str(e)}")
        return

    print("\nAll training tests completed successfully!")

if __name__ == "__main__":
    test_training()