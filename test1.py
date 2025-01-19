# test_implementation.py

from transformers import AutoTokenizer
from src.preprocessing.dataset import NarrativeDataset
import torch
from pprint import pprint

def main():
    print("Testing dataset implementation...")

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # Create dataset
    print("\nCreating dataset...")
    dataset = NarrativeDataset(
        data_dir='data_set/target_4_December_JSON',
        language='EN',
        split='train',
        tokenizer=tokenizer,
        verbose=True  # This will print the dataset statistics
    )

    print("\nNarrative categories:")
    pprint(dataset.narrative_to_idx)

    print("\nExamining first 5 articles:")
    for i in range(min(5, len(dataset))):
        print(f"\n--- Article {i+1} ---")

        # Get item
        item = dataset[i]

        # Get original labels
        original_narratives = [dataset.idx_to_narrative[idx]
                             for idx, val in enumerate(item['narrative_labels'])
                             if val == 1]

        original_subnarratives = [dataset.idx_to_subnarrative[idx]
                                for idx, val in enumerate(item['subnarrative_labels'])
                                if val == 1]

        # Print information
        print(f"Text length (tokens): {len(item['input_ids'])}")
        print("\nNarratives:")
        pprint(original_narratives)
        print("\nSubnarratives:")
        pprint(original_subnarratives)

        # Print first 50 tokens of text
        article_text = dataset.articles[i].text
        decoded_text = tokenizer.decode(item['input_ids'][:50])
        print(f"\nFirst 50 tokens of text:")
        print(decoded_text + "...")

if __name__ == "__main__":
    main()