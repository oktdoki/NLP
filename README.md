# SemEval 2025 Task 10: Narrative Classification

This repository contains the implementation for Subtask 2 of SemEval 2025 Task 10 - Multilingual Classification of Narratives from Online News.

## Project Structure

```
├── config/                          
│   └── model_config.yaml/             # all basic config for the model and training
├── Data set/                          # Dataset provided by SemEval
|   └── balanced_data/                 # articles with similar amount of each narrative
│       ├── EN_balanced/               # English articles
│       └── PT_balanced/                # Portuguese articles
|   └── cleaned_data/                  # articles with no duplicate narratives
│       ├── EN_cleaned/                # English articles
│       └── PT_cleaned/                # Portuguese articles
|   └── target_4_December_JSON/        # articles in format for training
│       ├── BG/                        # Bulgarian articles
│       ├── EN/                        # English articles
│       ├── HI/                        # Hindi articles
|       ├── RU/                        # Russian articles
│       └── PT/                        # Portuguese articles
|   └── target_4_December_release/             #updated data set
│       ├── BG/                        # Bulgarian articles
│       ├── EN/                        # English articles
│       ├── HI/                        # Hindi articles
|       ├── RU/                        # Russian articles
│       └── PT/                        # Portuguese articles
│   └── training_data_16_October_release/
│       ├── BG/                        # Bulgarian articles
│       ├── EN/                        # English articles
│       ├── HI/                        # Hindi articles
│       └── PT/                        # Portuguese articles
│   └── training_data_16_October_release/
│   └── training_data_16_October_release/
├── evaluation_results/                # test results for both models and languages
├── src
│   ├── models                         # basic classes for the model and the loss
│   │   ├── classifier.py
│   │   ├── config.py
│   │   ├── data_models.py
│   │   ├── factory.py
│   │   └── loss.py
│   ├── preprocessing                  # loading, preprocessing and checking of articles
│   │   ├── dataset.py
│   │   └── dataset_validator.py
│   ├── training                       # main training class and trainer definition
│   │   ├── train.py
│   │   └── trainer.py
│   └── utils                          # evaluation and other utils
│       ├── config.py
│       ├── constants.py
│       ├── data_splitting.py
│       ├── data_utils.py
│       ├── evaluate.py
│       ├── evaluator.py
│       ├── file_utils.py
│       └── metrics.py
├── tests/
|   ├── test_config.yaml/             # Test configuration
│   ├── test_loss.py/                 # Test loss
│   ├── test_model.py/                # Test model
│   └── test_trainer.py/              # Test trainer
└── requirements.txt                  # Python dependencies
```

## Data Format Requirements

### Input Data Format
The preprocessing module should provide data in the following format for training:

```json
{
  "text": "Ativistas climáticos ou idiotas úteis?\n\n“Pela libertação da...",
  "article_id": "PT_01.txt",
  "category": "CC",
  "narratives": [
    "Hidden plots by secret schemes of powerful groups"
  ],
  "subnarratives": [
    "Climate agenda has hidden motives"
  ]
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
- Consistent handling across all languages (EN,PT)
- "Other" labels properly handled when no specific narrative/subnarrative applies

## Model Overview

The implementation uses XLM-RoBERTa and mBert for multilingual narrative classification:
- Multi-label classification
- Supports two languages (EN,PT)
- Handles hierarchical narrative structure (main narratives and subnarratives)

## Evaluation

The official evaluation measure is averaged (over test documents) samples F1 computed for entire narrative_x:subnarrative_x labels. The model will be evaluated on:
- Narrative classification 
- Subnarrative classification

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

