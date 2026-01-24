import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # We quantize the ALREADY FUSED model for best results
    input_path = os.path.join(root, "models", "onnx", "fused_model.onnx")
    output_path = os.path.join(root, "models", "onnx", "quantized_model.onnx")

    print("Starting INT8 Dynamic Quantization...")
    
    # This specifically targets Linear layers (MatMul), which were 62% of our bottleneck.
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    
    print(f"✅ Quantized model saved to: {output_path}")

if __name__ == "__main__":
    quantize_model()