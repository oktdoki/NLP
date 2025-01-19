from dataclasses import dataclass
from typing import List, Literal, Optional
from pathlib import Path

@dataclass
class Article:
    """Data model for a single article."""
    text: str
    article_id: str
    category: Literal["CC", "URW"]  # Only allow these two values
    narratives: List[str]
    subnarratives: List[str]

    def __post_init__(self):
        """Validate the article after initialization."""
        if len(self.narratives) != len(self.subnarratives):
            raise ValueError("Number of narratives and subnarratives must match")
        if not self.text.strip():
            raise ValueError("Article text cannot be empty")
        if not self.article_id.strip():
            raise ValueError("Article ID cannot be empty")

@dataclass
class DatasetMetrics:
    """Data model for dataset statistics."""
    total_files: int = 0
    valid_files: int = 0
    avg_text_length: float = 0.0
    cc_articles: int = 0
    urw_articles: int = 0
    unique_narratives: int = 0
    unique_subnarratives: int = 0
    errors: List[str] = None

    def __post_init__(self):
        """Initialize empty error list if None."""
        if self.errors is None:
            self.errors = []

    def add_error(self, error: str):
        """Add an error message to the metrics."""
        self.errors.append(error)

    def print_summary(self):
        """Print a formatted summary of the metrics."""
        print("\nDataset Statistics:")
        print(f"Total files processed: {self.total_files}")
        print(f"Valid files: {self.valid_files}")
        print(f"Average text length: {self.avg_text_length:.2f} characters")
        print(f"\nCategory Distribution:")
        print(f"Climate Change (CC): {self.cc_articles}")
        print(f"Ukraine-Russia War (URW): {self.urw_articles}")
        print(f"\nLabel Statistics:")
        print(f"Unique narratives: {self.unique_narratives}")
        print(f"Unique subnarratives: {self.unique_subnarratives}")

        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"- {error}")