import os
import torch
from transformers import Wav2Vec2ForCTC

def export_to_onnx():
    # 1. Absolute Path Setup
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    model_dir = os.path.join(project_root, "models", "original_hindi")
    output_dir = os.path.join(project_root, "models", "onnx")
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load the Local Model
    print(f"Loading model from {model_dir}...")
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    model.eval()

    # 3. Create Dummy Input (1 second of audio @ 16kHz)
    # Shape: [batch_size, sequence_length]
    dummy_input = torch.randn(1, 16000)

    # 4. Export with Dynamic Axes
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # High compatibility with TensorRT
        do_constant_folding=True,
        input_names=['input_values'],
        output_names=['logits'],
        # Allow variable batch size and audio length
        dynamic_axes={
            'input_values': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print(f"Export Successful! File saved at: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()