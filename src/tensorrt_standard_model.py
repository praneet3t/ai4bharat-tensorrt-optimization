import torch
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config

def export_local_to_standard_onnx():
    # 1. Setup paths
    # Using your actual local directory path
    local_dir = r"C:\Users\apran\Videos\Cin\LIBRARY\ai4bharat-tensorrt-optimization\models\original_hindi"
    onnx_output_path = os.path.join(local_dir, "..", "onnx", "standard_model.onnx")
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)

    print(f"Loading model from {local_dir}...")
    
    # 2. Load with mismatch handling
    # ignore_mismatched_sizes=True allows it to overwrite the 768 default with the 1024 from your bin
    try:
        model = Wav2Vec2ForCTC.from_pretrained(
            local_dir, 
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Prepare Dummy Input (1 batch, 5 seconds @ 16kHz)
    dummy_input = torch.randn(1, 80000)

    print("Exporting to standard ONNX (Opset 17)...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_values'],
        output_names=['logits'],
        dynamic_axes={
            'input_values': {1: 'sequence_length'},
            'logits': {1: 'sequence_length'}
        }
    )
    
    print(f"✅ Export Complete: {onnx_output_path}")

if __name__ == "__main__":
    export_local_to_standard_onnx()