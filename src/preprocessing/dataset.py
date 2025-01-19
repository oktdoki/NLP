# src/preprocessing/dataset.py

import re
import html
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from pathlib import Path
from .dataset_validator import DatasetValidator
from src.models.data_models import Article
from src.utils.data_splitting import DataSplitter

class NarrativeDataset(Dataset):
    """Dataset class for narrative classification."""

    def __init__(
        self,
        data_dir: str,
        language: str = "EN",
        split: str = "train",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        verbose: bool = True,
        split_file: Optional[str] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Base directory containing the JSON files
            language (str): Language code (EN, BG, PT, HI, RU)
            split (str): Data split (train, dev, test)
            tokenizer: HuggingFace tokenizer for text encoding
            max_length (int): Maximum sequence length for tokenization
            verbose (bool): Whether to print validation statistics
            split_file (str, optional): Path to existing split file
        """
        self.data_dir = Path(data_dir)
        self.language = language
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.verbose = verbose

        # Initialize validator
        self.validator = DatasetValidator(self.data_dir, language)

        # Load and validate all data first
        try:
            # Get all valid articles
            all_articles = self.validator.validate_dataset()

            if not all_articles:
                raise RuntimeError("No valid articles found in dataset")

            # Clean texts
            for article in all_articles:
                article.text = self._clean_text(article.text)

            # Handle data splitting
            splitter = DataSplitter()
            if split_file:
                # Load existing splits
                splits = splitter.load_splits(Path(split_file), all_articles)
            else:
                # Create new splits
                splits = splitter.split_data(all_articles)
                # Save splits if in verbose mode
                if verbose:
                    output_dir = self.data_dir / 'splits'
                    splitter.save_splits(splits, output_dir, language)

            # Select appropriate split
            if split not in splits:
                raise ValueError(f"Invalid split name: {split}")
            self.articles = splits[split]

            if verbose:
                # Print split statistics
                stats = splitter.get_split_stats(splits)
                print(f"\nSplit Statistics:")
                for split_name, split_stats in stats.items():
                    print(f"\n{split_name.upper()}:")
                    print(f"Total articles: {split_stats['total_articles']}")
                    print(f"Categories: CC={split_stats['categories']['CC']}, "
                          f"URW={split_stats['categories']['URW']}")
                    print(f"Average text length: {split_stats['avg_text_length']:.1f}")
                    print(f"Average narratives per article: "
                          f"{split_stats['avg_narratives']:.1f}")

            # Create label mappings from all articles to ensure consistency across splits
            self._create_label_mappings_from_articles(all_articles)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize dataset: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean article text."""
        text = html.unescape(text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = ' '.join(text.split())
        text = re.sub(r'[^\w\s.,!?;:\"\'()\[\]{}\-]', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()

    def _create_label_mappings_from_articles(self, articles: List[Article]):
        """Create label mappings using all articles to ensure consistency."""
        # Get unique labels from all articles
        narratives = {
            narrative
            for article in articles
            for narrative in article.narratives
        }
        subnarratives = {
            subnarrative
            for article in articles
            for subnarrative in article.subnarratives
        }

        # Create mappings
        self.narrative_to_idx = {
            label: idx for idx, label in enumerate(sorted(narratives))
        }
        self.subnarrative_to_idx = {
            label: idx for idx, label in enumerate(sorted(subnarratives))
        }

        # Create reverse mappings
        self.idx_to_narrative = {
            idx: label for label, idx in self.narrative_to_idx.items()
        }
        self.idx_to_subnarrative = {
            idx: label for label, idx in self.subnarrative_to_idx.items()
        }

    def _create_label_mappings(self):
        """Create mappings between label strings and indices."""
        # Get unique labels
        narratives = {
            narrative
            for article in self.articles
            for narrative in article.narratives
        }
        subnarratives = {
            subnarrative
            for article in self.articles
            for subnarrative in article.subnarratives
        }

        # Create mappings
        self.narrative_to_idx = {
            label: idx for idx, label in enumerate(sorted(narratives))
        }
        self.subnarrative_to_idx = {
            label: idx for idx, label in enumerate(sorted(subnarratives))
        }

        # Create reverse mappings
        self.idx_to_narrative = {
            idx: label for label, idx in self.narrative_to_idx.items()
        }
        self.idx_to_subnarrative = {
            idx: label for label, idx in self.subnarrative_to_idx.items()
        }

    def encode_labels(
        self,
        narratives: List[str],
        subnarratives: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert labels to multi-hot encoded tensors."""
        narrative_tensor = torch.zeros(len(self.narrative_to_idx))
        subnarrative_tensor = torch.zeros(len(self.subnarrative_to_idx))

        for narrative in narratives:
            narrative_tensor[self.narrative_to_idx[narrative]] = 1

        for subnarrative in subnarratives:
            subnarrative_tensor[self.subnarrative_to_idx[subnarrative]] = 1

        return narrative_tensor, subnarrative_tensor

    def __len__(self) -> int:
        """Return the number of articles."""
        return len(self.articles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single dataset item."""
        article = self.articles[idx]

        # Tokenize text if tokenizer is provided
        if self.tokenizer:
            encoding = self.tokenizer(
                article.text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            input_ids = None
            attention_mask = None

        # Encode labels
        narrative_labels, subnarrative_labels = self.encode_labels(
            article.narratives,
            article.subnarratives
        )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'narrative_labels': narrative_labels,
            'subnarrative_labels': subnarrative_labels
        }