import os
import torch
from transformers import Wav2Vec2ForCTC

def export_to_onnx():
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    model_dir = os.path.join(project_root, "models", "original_hindi")
    output_dir = os.path.join(project_root, "models", "onnx")
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model for Legacy Stable Export...")
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, low_cpu_mem_usage=False)
    model.eval()

    dummy_input = torch.randn(1, 16000) 

    print(f"Exporting to ONNX Opset 14 (Legacy Path)...")
    
    # We use dynamo=False to explicitly tell PyTorch NOT to use the new exporter
    # This avoids the version converter and kaxes crash entirely.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14, # Opset 14 is the most stable for TensorRT 8.x
            do_constant_folding=True,
            input_names=['input_values'],
            output_names=['logits'],
            dynamic_axes={
                'input_values': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            # THIS IS THE KEY: Disable the new engine
            dynamo=False 
        )
    
    # Verification
    if os.path.exists(onnx_path):
        # Note: Large models (>2GB) put weights in model.onnx.data
        main_size = os.path.getsize(onnx_path) / (1024**2)
        print(f"✅ Export Successful! Main graph: {onnx_path} ({main_size:.2f} MB)")
        
        # Check for external weights file
        data_file = onnx_path + ".data"
        if os.path.exists(data_file):
            data_size = os.path.getsize(data_file) / (1024**3)
            print(f"📦 Weights saved externally to {data_file} ({data_size:.2f} GB)")
        else:
            print("Note: Weights are embedded in the main .onnx file.")
    else:
        print("❌ Export failed.")

if __name__ == "__main__":
    export_to_onnx()