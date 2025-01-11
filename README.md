# SemEval 2025 Task 10: Narrative Classification

This repository contains the implementation for Subtask 2 of SemEval 2025 Task 10 - Multilingual Classification of Narratives from Online News.

## Project Structure

```
├── Data set/                          # Dataset provided by SemEval
│   └── training_data_16_October_release/
│       ├── BG/                        # Bulgarian articles
│       ├── EN/                        # English articles
│       ├── HI/                        # Hindi articles
│       └── PT/                        # Portuguese articles
├── src/
│   ├── models/                        # Model implementations
│   │   ├── classifier.py              # XLM-R classifier
│   │   └── config.py                  # Model configuration
│   ├── preprocessing/                 # Data preprocessing scripts
│   ├── training/                     # Training scripts
│   └── utils/                        # Utility functions
├── config/                           # Configuration files
├── notebooks/                        # Analysis notebooks
├── results/                         
│   ├── models/                       # Saved models
│   └── predictions/                  # Model predictions
└── requirements.txt                  # Python dependencies
```

## Data Format Requirements

### Input Data Format
The preprocessing module should provide data in the following format for training:

```json
{
    "text": "Article content including title...",
    "article_id": "EN_CC_100000.txt",
    "narratives": ["URW", "CC"],
    "subnarratives": ["URW: Blaming Others", "CC: Climate Change Denial"]
}
```

### Raw Data Structure
- Each article is a .txt file with:
    - Title on first line
    - Empty second line
    - Content from third line onwards
- Labels are in subtask-2-annotations.txt files with format:
    - `article_id    narrative_1;...;narrative_N    subnarrative_1;...;subnarrative_N`

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Expected preprocessed data format:
- Clean text content
- Properly formatted labels
- Consistent handling across all languages (BG, EN, HI, PT)
- "Other" labels properly handled when no specific narrative/subnarrative applies

## Model Overview

The implementation uses XLM-RoBERTa for multilingual narrative classification:
- Multi-label classification
- Supports all four languages (BG, EN, HI, PT)
- Handles hierarchical narrative structure (main narratives and subnarratives)

## Evaluation

The official evaluation measure is averaged (over test documents) samples F1 computed for entire narrative_x:subnarrative_x labels. The model will be evaluated on:
- Narrative classification (URW, CC)
- Subnarrative classification (e.g., "URW: Blaming Others")

## Dependencies

Core dependencies include:
- torch
- transformers
- pandas
- numpy
- scikit-learn
- wandb (for experiment tracking)

See requirements.txt for complete list and versions.

## Notes for Preprocessing

The preprocessing module should handle:
1. Text cleaning
    - Remove unnecessary whitespace
    - Handle special characters
    - Preserve title and content structure

2. Label processing
    - Parse annotation files correctly
    - Handle multi-label cases
    - Process "Other" labels appropriately

