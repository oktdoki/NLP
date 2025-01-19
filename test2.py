# test_cleaning.py

from transformers import AutoTokenizer
from src.preprocessing.dataset import NarrativeDataset
import torch
from pprint import pprint

def test_text_cleaning():
    # Create dataset instance
    dataset = NarrativeDataset(
        data_dir='data_set/target_4_December_JSON',
        language='EN',
        split='train',
        verbose=False  # Disable verbose output for cleaner test output
    )

    # Test cases for text cleaning
    test_cases = [
        "Normal text with   extra   spaces",
        "Text with URL: http://example.com",
        "Text with email: user@example.com",
        "Text with [metadata] inside brackets",
        "Text with HTML &amp; entities",
        "Text with\nmultiple\n\nNewlines",
        "Text with special chars: @#$%^",
        """Multi-line
        text with irregular
        spacing"""
    ]

    print("Testing text cleaning with various cases:")
    print("-" * 50)

    for i, test_text in enumerate(test_cases, 1):
        cleaned = dataset._clean_text(test_text)
        print(f"\nTest case {i}:")
        print("Original:", test_text)
        print("Cleaned:", cleaned)
        print("-" * 50)

    # Test actual data
    print("\nTesting first 3 actual articles:")
    dataset = NarrativeDataset(
        data_dir='data_set/target_4_December_JSON',
        language='EN',
        split='train',
        tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
        verbose=False
    )

    for i in range(min(3, len(dataset))):
        article = dataset.articles[i]
        print(f"\nArticle {i+1} sample:")
        print(article.text[:200] + "...")  # First 200 characters
        print(f"Text length: {len(article.text)} characters")
        print(f"Category: {article.category}")
        print(f"Article ID: {article.article_id}")
        print("-" * 50)

if __name__ == "__main__":
    test_text_cleaning()