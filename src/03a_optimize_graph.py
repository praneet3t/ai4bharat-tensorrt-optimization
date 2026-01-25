import os
from onnxruntime.transformers import optimizer

def run_task_3a():
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    input_path = os.path.join(project_root, "models", "onnx", "model.onnx")
    output_path = os.path.join(project_root, "models", "onnx", "optimized_model.onnx")
    
    print(f"Starting Graph Optimization on {input_path}...")

    # optimize_model handles the Multi-Head Attention fusions
    # model_type='bert' is a safe common denominator for Wav2Vec2 architectures
    optimized_model = optimizer.optimize_model(
        input_path,
        model_type='bert', 
        num_heads=16,      # Wav2Vec2-Large typically has 16 heads
        hidden_size=1024   # Wav2Vec2-Large hidden size
    )

    # Save with external data format because it's still > 2GB
    optimized_model.save_model_to_file(output_path, use_external_data_format=True)
    print(f"✅ Optimization Complete! Saved to {output_path}")

if __name__ == "__main__":
    run_task_3a()