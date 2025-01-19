# src/utils/data_splitting.py

import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from src.models.data_models import Article

class DataSplitter:
    """Handles dataset splitting into train/validation/test sets."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the data splitter.

        Args:
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def _get_stratification_labels(self, articles: List[Article]) -> List[str]:
        """
        Create stratification labels based on article categories.

        Args:
            articles: List of articles to stratify

        Returns:
            List of stratification labels
        """
        return [article.category for article in articles]

    def split_data(
        self,
        articles: List[Article]
    ) -> Dict[str, List[Article]]:
        """
        Split articles into train/validation/test sets.

        Args:
            articles: List of articles to split

        Returns:
            Dictionary containing train, val, and test splits
        """
        if not articles:
            raise ValueError("No articles provided for splitting")

        # Get stratification labels
        stratify_labels = self._get_stratification_labels(articles)

        # First split: separate test set
        train_val, test = train_test_split(
            articles,
            test_size=self.test_size,
            stratify=stratify_labels,
            random_state=self.random_state
        )

        # Update stratification labels for train/val split
        train_val_labels = [article.category for article in train_val]

        # Second split: separate train and validation
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val_labels,
            random_state=self.random_state
        )

        return {
            'train': train,
            'val': val,
            'test': test
        }

    def save_splits(
        self,
        splits: Dict[str, List[Article]],
        output_dir: Path,
        language: str
    ):
        """
        Save split information to disk.

        Args:
            splits: Dictionary containing data splits
            output_dir: Directory to save split information
            language: Language code
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save split indices
        split_info = {
            split_name: [article.article_id for article in articles]
            for split_name, articles in splits.items()
        }

        # Save to JSON
        with open(output_dir / f"{language}_splits.json", 'w') as f:
            json.dump(split_info, f, indent=2)

    def load_splits(
        self,
        split_file: Path,
        articles: List[Article]
    ) -> Dict[str, List[Article]]:
        """
        Load split information from disk.

        Args:
            split_file: Path to split information JSON file
            articles: List of all articles

        Returns:
            Dictionary containing data splits
        """
        # Create article_id to article mapping
        article_map = {article.article_id: article for article in articles}

        # Load split information
        with open(split_file, 'r') as f:
            split_info = json.load(f)

        # Create splits
        splits = {}
        for split_name, article_ids in split_info.items():
            splits[split_name] = [
                article_map[article_id]
                for article_id in article_ids
                if article_id in article_map
            ]

        return splits

    def get_split_stats(self, splits: Dict[str, List[Article]]) -> Dict[str, Dict]:
        """
        Get statistics about the splits.

        Args:
            splits: Dictionary containing data splits

        Returns:
            Dictionary containing split statistics
        """
        stats = {}

        for split_name, articles in splits.items():
            split_stats = {
                'total_articles': len(articles),
                'categories': {
                    'CC': sum(1 for a in articles if a.category == 'CC'),
                    'URW': sum(1 for a in articles if a.category == 'URW')
                },
                'avg_text_length': np.mean([len(a.text) for a in articles]),
                'avg_narratives': np.mean([len(a.narratives) for a in articles])
            }
            stats[split_name] = split_stats

        return stats