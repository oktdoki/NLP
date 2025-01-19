from typing import Dict, Tuple
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from src.preprocessing.dataset import NarrativeDataset
from src.utils.constants import SUPPORTED_LANGUAGES

def get_tokenizer(model_name: str = 'xlm-roberta-base'):
    """
    Initialize the tokenizer.

    Args:
        model_name (str): Name of the pretrained model

    Returns:
        PreTrainedTokenizer: HuggingFace tokenizer
    """
    return AutoTokenizer.from_pretrained(model_name)

def create_dataloaders(
    data_dir: str,
    language: str = 'EN',
    model_name: str = 'xlm-roberta-base',
    batch_size: int = 16,
    val_split: float = 0.1,
    max_length: int = 512,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir (str): Path to data directory
        language (str): Language code
        model_name (str): Name of pretrained model for tokenizer
        batch_size (int): Batch size for dataloaders
        val_split (float): Fraction of data to use for validation
        max_length (int): Maximum sequence length
        num_workers (int): Number of workers for dataloaders

    Returns:
        Dict containing train and validation dataloaders
    """
    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language {language} not supported. Choose from {SUPPORTED_LANGUAGES}")

    # Initialize tokenizer
    tokenizer = get_tokenizer(model_name)

    # Create dataset
    full_dataset = NarrativeDataset(
        data_dir=data_dir,
        language=language,
        split='train',  # We'll split this into train/val
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {
        'train': train_loader,
        'val': val_loader
    }

def get_label_info(dataset: NarrativeDataset) -> Dict[str, int]:
    """
    Get information about the labels in the dataset.

    Args:
        dataset (NarrativeDataset): The dataset

    Returns:
        Dict containing number of narratives and subnarratives
    """
    return {
        'num_narratives': len(dataset.narrative_to_idx),
        'num_subnarratives': len(dataset.subnarrative_to_idx)
    }