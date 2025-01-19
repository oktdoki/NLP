# test_model.py

import torch
from transformers import AutoTokenizer
from src.models.factory import ModelFactory
from src.models.config import ModelConfig
from src.preprocessing.dataset import NarrativeDataset

def test_model():
    print("Testing narrative classification model...")

    # First load a small part of the dataset to get dimensions
    print("\nLoading dataset for dimensions...")
    dataset = NarrativeDataset(
        data_dir='data_set/target_4_December_JSON',
        language='EN',
        split='train',
        tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base')
    )

    # Get dimensions from dataset
    num_narratives = len(dataset.narrative_to_idx)
    num_subnarratives = len(dataset.subnarrative_to_idx)

    print(f"\nFound {num_narratives} narratives and {num_subnarratives} subnarratives")

    # Create model configuration
    print("\nCreating model configuration...")
    config = ModelConfig(
        model_name="xlm-roberta-base",
        max_length=512,
        batch_size=4,  # Small batch size for testing
        learning_rate=2e-5
    )

    # Create model
    print("\nCreating model...")
    model = ModelFactory.create_model(
        num_narratives=num_narratives,
        num_subnarratives=num_subnarratives,
        config=config
    )

    print("\nModel created successfully!")

    # Test forward pass
    print("\nTesting forward pass...")

    # Get a batch from dataset
    batch = dataset[0]  # Get first item

    # Add batch dimension
    input_ids = batch['input_ids'].unsqueeze(0)  # Shape: [1, seq_len]
    attention_mask = batch['attention_mask'].unsqueeze(0)
    narrative_labels = batch['narrative_labels'].unsqueeze(0)
    subnarrative_labels = batch['subnarrative_labels'].unsqueeze(0)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        narrative_labels=narrative_labels,
        subnarrative_labels=subnarrative_labels
    )

    # Print output shapes and values
    print("\nOutput shapes:")
    print(f"Narrative logits: {outputs['narrative_logits'].shape}")
    print(f"Subnarrative logits: {outputs['subnarrative_logits'].shape}")

    print("\nLoss values:")
    print(f"Total loss: {outputs['total_loss'].item():.4f}")
    print(f"Narrative loss: {outputs['narrative_loss'].item():.4f}")
    print(f"Subnarrative loss: {outputs['subnarrative_loss'].item():.4f}")

    # Test prediction mode
    print("\nTesting prediction mode...")
    narrative_preds, subnarrative_preds = model.predict(input_ids, attention_mask)

    print("\nPrediction shapes:")
    print(f"Narrative predictions: {narrative_preds.shape}")
    print(f"Subnarrative predictions: {subnarrative_preds.shape}")

    # Test save and load
    print("\nTesting save and load functionality...")

    # Save model
    save_path = "test_model.pt"
    print(f"Saving model to {save_path}")
    ModelFactory.save_model(model, save_path)

    # Load model
    print("Loading model...")
    loaded_model = ModelFactory.load_model(save_path)

    # Verify loaded model
    print("\nVerifying loaded model...")
    loaded_outputs = loaded_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        narrative_labels=narrative_labels,
        subnarrative_labels=subnarrative_labels
    )

    # Compare outputs
    original_loss = outputs['total_loss'].item()
    loaded_loss = loaded_outputs['total_loss'].item()

    print(f"\nOriginal model loss: {original_loss:.4f}")
    print(f"Loaded model loss: {loaded_loss:.4f}")
    print(f"Loss difference: {abs(original_loss - loaded_loss):.8f}")

    threshold = original_loss * 0.001  # 0.1% difference allowed

    if abs(original_loss - loaded_loss) < threshold:
        print(f"\nModel save/load test passed! (Difference within {threshold:.8f})")
    else:
        print("\nWarning: Model save/load test failed!")

    print("\nTest completed!")

if __name__ == "__main__":
    test_model()