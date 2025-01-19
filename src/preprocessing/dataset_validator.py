# src/utils/validation.py

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..models.data_models import Article, DatasetMetrics

class DatasetValidator:
    """Class for validating dataset files and structure."""

    def __init__(self, data_dir: Path, language: str):
        self.data_dir = Path(data_dir)
        self.language = language
        self.metrics = DatasetMetrics()

    def validate_directory(self) -> bool:
        """
        Validate the dataset directory structure.

        Returns:
            bool: True if directory structure is valid
        """
        lang_dir = self.data_dir / self.language

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if not lang_dir.exists():
            raise FileNotFoundError(f"Language directory not found: {lang_dir}")

        json_files = list(lang_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {lang_dir}")

        return True

    def load_and_validate_file(self, file_path: Path) -> Optional[Article]:
        """
        Load and validate a single JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Optional[Article]: Validated Article object or None if invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Create Article object (validation happens in __post_init__)
            article = Article(**data)

            # Update metrics
            self.metrics.total_files += 1
            self.metrics.valid_files += 1
            self.metrics.avg_text_length += len(article.text)

            if article.category == "CC":
                self.metrics.cc_articles += 1
            else:
                self.metrics.urw_articles += 1

            return article

        except json.JSONDecodeError:
            self.metrics.add_error(f"Invalid JSON format in {file_path}")
        except TypeError as e:
            self.metrics.add_error(f"Missing required fields in {file_path}: {str(e)}")
        except ValueError as e:
            self.metrics.add_error(f"Validation error in {file_path}: {str(e)}")
        except Exception as e:
            self.metrics.add_error(f"Unexpected error in {file_path}: {str(e)}")

        self.metrics.total_files += 1
        return None

    def validate_dataset(self) -> List[Article]:
        """
        Validate entire dataset.

        Returns:
            List[Article]: List of valid Article objects
        """
        # First validate directory structure
        self.validate_directory()

        # Process all files
        valid_articles = []
        lang_dir = self.data_dir / self.language

        for json_file in lang_dir.glob("*.json"):
            article = self.load_and_validate_file(json_file)
            if article:
                valid_articles.append(article)

        # Update final metrics
        if valid_articles:
            self.metrics.avg_text_length /= self.metrics.valid_files
            self.metrics.unique_narratives = len({
                narrative
                for article in valid_articles
                for narrative in article.narratives
            })
            self.metrics.unique_subnarratives = len({
                subnarrative
                for article in valid_articles
                for subnarrative in article.subnarratives
            })

        return valid_articles