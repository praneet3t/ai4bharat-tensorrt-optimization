import os

# 1. FIX: Prevent "OMP: Error #15" crash on Windows/Linux
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import Wav2Vec2ForCTC

def export_to_onnx():
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    # Paths
    model_dir = os.path.join(project_root, "models", "original_hindi")
    output_dir = os.path.join(project_root, "models", "onnx")
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {model_dir}")
    try:
        # Load model to CPU
        model = Wav2Vec2ForCTC.from_pretrained(model_dir, low_cpu_mem_usage=False).to("cpu")
    except OSError:
        print(f"❌ Error: Model not found at {model_dir}")
        return

    model.eval()

    # Dummy input (1 sample, 1 second)
    dummy_input = torch.randn(1, 16000) 

    print(f"Exporting to ONNX (Opset 17)...")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            # CRITICAL FIX: Opset 17 prevents the "Garbage Text" bug later in TensorRT
            opset_version=17, 
            do_constant_folding=True,
            input_names=['input_values'],
            output_names=['logits'],
            dynamic_axes={
                'input_values': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
            # REMOVED: dynamo=False (This causes TypeError)
        )
    
    # Verification
    if os.path.exists(onnx_path):
        main_size = os.path.getsize(onnx_path) / (1024**2)
        print(f"✅ Export Successful! Main graph: {onnx_path} ({main_size:.2f} MB)")
        
        # Check for external weights (common with Wav2Vec2)
        data_file = onnx_path + ".data"
        if os.path.exists(data_file):
            data_size = os.path.getsize(data_file) / (1024**3)
            print(f"📦 Weights saved externally: {data_file} ({data_size:.2f} GB)")
            print("👉 NOTE: When moving to Colab, you MUST upload both .onnx and .onnx.data")
        else:
            print("👉 Weights embedded in the main file.")
    else:
        print("❌ Export failed.")

if __name__ == "__main__":
    export_to_onnx()