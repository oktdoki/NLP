# checkpoint_inspector.py
import torch
import os

def inspect_checkpoint(checkpoint_path):
    print(f"\nInspecting checkpoint at: {checkpoint_path}")
    print(f"File size: {os.path.getsize(checkpoint_path) / (1024*1024*1024):.2f} GB")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print("\nCheckpoint type:", type(checkpoint))

        if isinstance(checkpoint, dict):
            print("\nTop level keys:", checkpoint.keys())

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("\nState dict keys:")
                for key in sorted(state_dict.keys()):
                    if isinstance(state_dict[key], torch.Tensor):
                        print(f"{key}: {state_dict[key].shape}")
                    else:
                        print(f"{key}: {type(state_dict[key])}")

            if 'config' in checkpoint:
                print("\nConfig:")
                print(checkpoint['config'])
        else:
            print("\nCheckpoint content structure:")
            print(checkpoint)

    except Exception as e:
        print(f"\nError loading checkpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    checkpoint_path = "checkpoints/best_model.pt"
    inspect_checkpoint(checkpoint_path)