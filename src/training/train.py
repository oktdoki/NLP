# train.py

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from src.models.factory import ModelFactory
from src.models.config import ModelConfig
from src.training.trainer import Trainer
from src.preprocessing.dataset import NarrativeDataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Configuration
    data_dir = 'data_set/target_4_December_JSON'
    model_name = 'xlm-roberta-base'
    batch_size = 8
    max_epochs = 10

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = NarrativeDataset(
        data_dir=data_dir,
        language='EN',
        split='train',
        tokenizer=tokenizer
    )

    val_dataset = NarrativeDataset(
        data_dir=data_dir,
        language='EN',
        split='val',
        tokenizer=tokenizer
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Create model configuration
    config = ModelConfig(
        model_name=model_name,
        max_epochs=max_epochs,
        batch_size=batch_size
    )

    # Create model
    logger.info("Creating model...")
    model = ModelFactory.create_model(
        num_narratives=len(train_dataset.narrative_to_idx),
        num_subnarratives=len(train_dataset.subnarrative_to_idx),
        config=config
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Start training
    logger.info("Starting training...")
    trainer.train(checkpoint_dir=str(checkpoint_dir))

if __name__ == "__main__":
    main()