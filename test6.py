# test_evaluation.py

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.utils.evaluator import ModelEvaluator
from src.models.factory import ModelFactory
from src.models.config import ModelConfig
from src.preprocessing.dataset import NarrativeDataset

def test_evaluation():
    print("Testing evaluation components...")

    # Create minimal test configuration
    test_config = ModelConfig(
        model_name='xlm-roberta-base',
        max_length=512,
        batch_size=2,  # Small batch size for testing
    )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(test_config.model_name)

    # Load small test dataset
    print("\nLoading test dataset...")
    test_dataset = NarrativeDataset(
        data_dir='data_set/target_4_December_JSON',
        language='EN',
        split='test',
        tokenizer=tokenizer,
        verbose=False  # Disable verbose output for cleaner testing
    )

    # Create test dataloader with minimal settings
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_config.batch_size,
        shuffle=False,
        num_workers=0  # Use 0 for testing
    )

    # Create model
    print("\nCreating test model...")
    model = ModelFactory.create_model(
        num_narratives=len(test_dataset.narrative_to_idx),
        num_subnarratives=len(test_dataset.subnarrative_to_idx),
        config=test_config
    )

    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        narrative_idx_to_label=test_dataset.idx_to_narrative,
        subnarrative_idx_to_label=test_dataset.idx_to_subnarrative
    )

    # Test main evaluation
    print("\nTesting main evaluation...")
    try:
        metrics = evaluator.evaluate()
        print("\nEvaluation metrics:")
        print("Narrative Classification:")
        print(f"  Precision: {metrics['narrative_precision']:.4f}")
        print(f"  Recall: {metrics['narrative_recall']:.4f}")
        print(f"  F1: {metrics['narrative_f1']:.4f}")
        print("Main evaluation test passed!")
    except Exception as e:
        print(f"Main evaluation test failed with error: {str(e)}")
        return

    # Test per-class metrics
    print("\nTesting per-class metrics...")
    try:
        print("\nNarrative class metrics sample:")
        for label, metrics in list(evaluator.narrative_class_metrics.items())[:2]:
            print(f"{label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
        print("Per-class metrics test passed!")
    except Exception as e:
        print(f"Per-class metrics test failed with error: {str(e)}")
        return

    # Test error analysis
    print("\nTesting error analysis...")
    try:
        errors = evaluator.get_error_analysis(num_examples=2)
        print("\nError analysis examples:")
        for i, error in enumerate(errors, 1):
            print(f"\nError Example {i}:")
            print("True Narratives:", error['true_narratives'])
            print("Predicted Narratives:", error['predicted_narratives'])
        print("Error analysis test passed!")
    except Exception as e:
        print(f"Error analysis test failed with error: {str(e)}")
        return

    # Test evaluator with different batch sizes
    print("\nTesting with different batch size...")
    try:
        test_loader_batch1 = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        evaluator_batch1 = ModelEvaluator(
            model=model,
            test_loader=test_loader_batch1,
            narrative_idx_to_label=test_dataset.idx_to_narrative,
            subnarrative_idx_to_label=test_dataset.idx_to_subnarrative
        )
        metrics_batch1 = evaluator_batch1.evaluate()
        print("Different batch size test passed!")
    except Exception as e:
        print(f"Different batch size test failed with error: {str(e)}")
        return

    print("\nAll evaluation tests completed successfully!")

    # Print test summary
    print("\nTest Summary:")
    print("✓ Main evaluation")
    print("✓ Per-class metrics")
    print("✓ Error analysis")
    print("✓ Different batch sizes")

if __name__ == "__main__":
    test_evaluation()