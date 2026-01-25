import os
import torch
import openvino as ov
from transformers import Wav2Vec2ForCTC

def export_to_openvino():
    # 1. Path Setup
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    model_dir = os.path.join(project_root, "models", "original_hindi")
    output_dir = os.path.join(project_root, "models", "openvino")
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Model
    print("Loading model for OpenVINO conversion...")
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, low_cpu_mem_usage=False)
    model.eval()

    # 3. Convert to OpenVINO Intermediate Representation (IR)
    print("Converting to OpenVINO IR (This is GREAT for CPU performance)...")
    dummy_input = torch.randn(1, 16000)
    
    # convert_model works directly on the PyTorch object
    ov_model = ov.convert_model(model, example_input=dummy_input)
    
    # Save as .xml (topology) and .bin (weights)
    ov.save_model(ov_model, os.path.join(output_dir, "model.xml"))
    
    print(f"✅ OpenVINO Export Successful! Saved to {output_dir}")
    print("This format runs 3x-5x faster on CPUs than raw PyTorch.")

if __name__ == "__main__":
    export_to_openvino()