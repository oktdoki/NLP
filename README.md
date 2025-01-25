# SemEval 2025 Task 10: Multilingual Narrative Classification

This repository implements a multilingual narrative classification system for subtask 2 of SemEval 2025 Task 10, focusing on characterizing narratives from online news across multiple languages.

## Project Structure
```
├── checkpoints/                     # Model checkpoints
├── config/                          # Configuration files
│   └── model_config.yaml
├── data_set/                        # Dataset directories
│   ├── target_4_December_JSON/      # Latest dataset version
│   ├── dev-documents_4_December/    # Development set
│   └── training_data_16_October/    # Training data
├── evaluation_results/              # Evaluation outputs
├── src/
│   ├── models/                      # Model implementations
│   │   ├── classifier.py           # Narrative classifier
│   │   ├── factory.py             # Model creation utilities
│   │   ├── config.py              # Model configuration
│   │   └── loss.py                # Custom loss functions
│   ├── preprocessing/              # Data preprocessing 
│   │   ├── dataset.py             # Dataset handling
│   │   └── dataset_validator.py    # Data validation
│   ├── training/                   # Training logic
│   │   ├── trainer.py             # Training loop
│   │   └── train.py               # Training entry point
│   └── utils/                      # Utility functions
│       ├── evaluator.py           # Evaluation metrics
│       ├── data_splitting.py      # Dataset splitting
│       └── data_utils.py          # Data processing utilities
└── tests/                          # Unit tests
```

## Features
- XLM-RoBERTa-based multilingual classifier
- Hierarchical multi-label classification (narratives and subnarratives)
- Support for Bulgarian, English, Hindi, and Portuguese
- Custom loss function with hierarchy consistency enforcement
- Robust evaluation metrics including F1 score and hierarchy accuracy

## Setup

```bash
pip install -r Requirements.txt
```

## Data Format
### Input JSON Structure
```json
{
    "text": "Article content",
    "article_id": "LANG_CATEGORY_ID.txt",
    "category": "CC|URW",  
    "narratives": ["narrative1", "narrative2"],
    "subnarratives": ["narrative1: subnarrative1", "narrative2: subnarrative2"]
}
```

### Training Data Organization
- Articles stored in language-specific directories (BG, EN, HI, PT)
- Raw text files with title on first line
- Annotation files containing narrative/subnarrative labels
- Development and test sets provided separately

## Training
```bash
python src/training/train.py
```

## Evaluation
```bash
python evaluate.py
```

Key metrics:
- Narrative classification F1
- Subnarrative classification F1
- Hierarchy consistency
- Per-class precision/recall

## Model Architecture
- Base: XLM-RoBERTa
- Dual classification heads for narratives and subnarratives
- Custom hierarchical loss function
- Dynamic thresholding for predictions
- Support for multi-label classification

## Dependencies
Core requirements:
- torch>=2.0.0
- transformers>=4.20.0
- scikit-learn>=1.0.0
- pandas>=1.3.0
- numpy>=1.21.0

See Requirements.txt for complete list.