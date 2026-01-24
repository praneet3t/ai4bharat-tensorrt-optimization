import os
from onnxruntime.transformers.optimizer import optimize_model

def optimize_fusions():
    # Relative paths from the script's location to project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(root, "models", "onnx", "model.onnx")
    output_path = os.path.join(root, "models", "onnx", "fused_model.onnx")

    print(f"Loading raw ONNX from: {input_path}")
    
    # Wav2Vec2 is basically BERT but for audio, so 'bert' type works best here.
    # We ignore the shape inference errors—they're annoying but non-fatal.
    optimizer = optimize_model(
        input_path,
        model_type='bert',
        num_heads=12,
        hidden_size=768
    )
    
    optimizer.save_model_to_file(output_path)
    print(f"✅ Fused model saved to: {output_path}")

if __name__ == "__main__":
    optimize_fusions()