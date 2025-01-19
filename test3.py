# test_splitting.py

from transformers import AutoTokenizer
from src.preprocessing.dataset import NarrativeDataset
from pathlib import Path

def test_data_splits():
    """Test data splitting functionality."""
    print("Testing data splitting...")

    # Create datasets for different splits
    splits = ['train', 'val', 'test']
    datasets = {}

    for split in splits:
        print(f"\nCreating {split} dataset...")
        datasets[split] = NarrativeDataset(
            data_dir='data_set/target_4_December_JSON',
            language='EN',
            split=split,
            tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
            verbose=True
        )

    # Test consistency across splits
    print("\nTesting label consistency across splits...")

    # Check narrative labels
    train_narratives = set(datasets['train'].narrative_to_idx.keys())
    val_narratives = set(datasets['val'].narrative_to_idx.keys())
    test_narratives = set(datasets['test'].narrative_to_idx.keys())

    print("\nUnique narratives in each split:")
    print(f"Train: {len(train_narratives)}")
    print(f"Val: {len(val_narratives)}")
    print(f"Test: {len(test_narratives)}")

    # Check label overlap
    narratives_overlap = train_narratives & val_narratives & test_narratives
    print(f"\nNarratives present in all splits: {len(narratives_overlap)}")

    # Test loading a few samples from each split
    print("\nTesting sample loading from each split...")
    for split_name, dataset in datasets.items():
        print(f"\n{split_name.upper()} Split Sample:")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Input shape: {sample['input_ids'].shape}")
            print(f"Number of narratives: {sample['narrative_labels'].sum().item()}")
            print(f"Number of subnarratives: {sample['subnarrative_labels'].sum().item()}")
        else:
            print("No samples in split!")

if __name__ == "__main__":
    test_data_splits()