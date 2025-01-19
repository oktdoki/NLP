# evaluate.py

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import json
from pathlib import Path

from src.utils.evaluator import ModelEvaluator
from src.models.factory import ModelFactory
from src.preprocessing.dataset import NarrativeDataset

def main():
    print("Starting model evaluation...")

    # Configuration
    checkpoint_path = "checkpoints/best_model.pt"
    data_dir = "data_set/target_4_December_JSON"
    output_dir = "evaluation_results"
    batch_size = 8

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = NarrativeDataset(
        data_dir=data_dir,
        language='EN',
        split='test',
        tokenizer=tokenizer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Load model from checkpoint
    print("\nLoading model from checkpoint...")
    model = ModelFactory.load_model(checkpoint_path)

    print(model)
    breakpoint()
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        narrative_idx_to_label=test_dataset.idx_to_narrative,
        subnarrative_idx_to_label=test_dataset.idx_to_subnarrative
    )

    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluator.evaluate()

    # Print overall metrics
    print("\nOverall Metrics:")
    print("Narrative Classification:")
    print(f"  Precision: {metrics['narrative_precision']:.4f}")
    print(f"  Recall: {metrics['narrative_recall']:.4f}")
    print(f"  F1: {metrics['narrative_f1']:.4f}")

    print("\nSubnarrative Classification:")
    print(f"  Precision: {metrics['subnarrative_precision']:.4f}")
    print(f"  Recall: {metrics['subnarrative_recall']:.4f}")
    print(f"  F1: {metrics['subnarrative_f1']:.4f}")

    # Get error analysis
    print("\nGetting error analysis...")
    errors = evaluator.get_error_analysis(num_examples=5)

    # Save results
    results = {
        'overall_metrics': metrics,
        'narrative_class_metrics': evaluator.narrative_class_metrics,
        'subnarrative_class_metrics': evaluator.subnarrative_class_metrics,
        'error_examples': errors
    }

    output_path = Path(output_dir) / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print per-class highlights
    print("\nHighlights from per-class metrics:")

    # Best and worst performing narratives
    narrative_f1s = [(label, metrics['f1'])
                     for label, metrics in evaluator.narrative_class_metrics.items()]
    best_narrative = max(narrative_f1s, key=lambda x: x[1])
    worst_narrative = min(narrative_f1s, key=lambda x: x[1])

    print("\nNarrative Categories:")
    print(f"Best performing: {best_narrative[0]} (F1: {best_narrative[1]:.4f})")
    print(f"Worst performing: {worst_narrative[0]} (F1: {worst_narrative[1]:.4f})")

    # Print some error examples
    print("\nExample Errors:")
    for i, error in enumerate(errors[:3], 1):
        print(f"\nError Example {i}:")
        print("True Narratives:", error['true_narratives'])
        print("Predicted Narratives:", error['predicted_narratives'])
        print("True Subnarratives:", error['true_subnarratives'])
        print("Predicted Subnarratives:", error['predicted_subnarratives'])

if __name__ == "__main__":
    main()