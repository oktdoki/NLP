# evaluate.py

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import json
from pathlib import Path
from pprint import pprint

from src.utils.evaluator import ModelEvaluator
from src.models.factory import ModelFactory
from src.preprocessing.dataset import NarrativeDataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def convert_to_json_serializable(obj):
    """Convert values to JSON serializable format."""
    if hasattr(obj, 'item'):  # Handle numpy numbers and torch tensors
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(x) for x in obj]
    return obj

def main():
    print("Starting model evaluation...")

    checkpoint_path = "checkpoints/best_model.pt"
    data_dir = "data_set/target_4_December_JSON"
    output_dir = "evaluation_results"
    batch_size = 8

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    print("\nLoading test dataset...")
    test_dataset = NarrativeDataset(
        data_dir=data_dir,
        language='EN',
        split='test',
        tokenizer=tokenizer
    )
    print("\nLoading model from checkpoint...")
    model = ModelFactory.load_model(checkpoint_path)

    # Set the label mappings
    model.narrative_to_idx = test_dataset.narrative_to_idx
    model.subnarrative_to_idx = test_dataset.subnarrative_to_idx

    print("\nDataset statistics:")
    print(f"Number of narratives: {len(test_dataset.idx_to_narrative)}")
    print(f"Narrative labels:", list(test_dataset.idx_to_narrative.items())[:5])
    print(f"\nNumber of subnarratives: {len(test_dataset.idx_to_subnarrative)}")
    print(f"Subnarrative labels:", list(test_dataset.idx_to_subnarrative.items())[:5])

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print("\nLoading model from checkpoint...")
    model = ModelFactory.load_model(checkpoint_path)

    #Add model dimension check here, after model loading
    narrative_weights = model.narrative_classifier[4].weight
    subnarrative_weights = model.subnarrative_classifier[4].weight
    print(f"\nModel dimensions:")
    print(f"Narrative classifier output: {narrative_weights.shape}")
    print(f"Subnarrative classifier output: {subnarrative_weights.shape}")


    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        narrative_idx_to_label=test_dataset.idx_to_narrative,
        subnarrative_idx_to_label=test_dataset.idx_to_subnarrative
    )

    print("\nRunning evaluation...")
    metrics = evaluator.evaluate()

    print("\nOverall Metrics:")
    print("Narrative Classification:")
    print(f"  Precision: {metrics['narrative_precision']:.4f}")
    print(f"  Recall: {metrics['narrative_recall']:.4f}")
    print(f"  F1: {metrics['narrative_f1']:.4f}")

    print("\nSubnarrative Classification:")
    print(f"  Precision: {metrics['subnarrative_precision']:.4f}")
    print(f"  Recall: {metrics['subnarrative_recall']:.4f}")
    print(f"  F1: {metrics['subnarrative_f1']:.4f}")

    print("\nGetting error analysis...")
    errors = evaluator.get_error_analysis(num_examples=5)

    # Save detailed results
    results = {
        'overall_metrics': metrics,
        'narrative_class_metrics': {
            label: {
                'metrics': convert_to_json_serializable(metrics),
                'support': convert_to_json_serializable(metrics['support'])
            }
            for label, metrics in evaluator.narrative_class_metrics.items()
        },
        'subnarrative_class_metrics': {
            label: {
                'metrics': convert_to_json_serializable(metrics),
                'support': convert_to_json_serializable(metrics['support'])
            }
            for label, metrics in evaluator.subnarrative_class_metrics.items()
        },
        'error_examples': convert_to_json_serializable(errors)
    }

    output_path = Path(output_dir) / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print detailed analysis
    print("\nTop 5 Best Performing Narrative Categories:")
    narrative_f1s = [(label, metrics['f1'])
                     for label, metrics in evaluator.narrative_class_metrics.items()]
    for label, f1 in sorted(narrative_f1s, key=lambda x: x[1], reverse=True)[:5]:
        support = evaluator.narrative_class_metrics[label]['support']
        print(f"{label}: F1={f1:.4f} (Support: {support})")

    print("\nExample Errors:")
    for i, error in enumerate(errors[:3], 1):
        print(f"\nError Example {i}:")
        print("True Narratives:", error['true_narratives'])
        print("Predicted Narratives:", error['predicted_narratives'])
        print("True Subnarratives:", error['true_subnarratives'])
        print("Predicted Subnarratives:", error['predicted_subnarratives'])
        if error['is_narrative_error']:
            print("⚠️ Narrative Prediction Error")
        if error['is_subnarrative_error']:
            print("⚠️ Subnarrative Prediction Error")

if __name__ == "__main__":
    main()