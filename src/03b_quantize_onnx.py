import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

def run_task_3b():
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    input_path = os.path.join(project_root, "models", "onnx", "optimized_model.onnx")
    # We create a temporary "pre-processed" model to fix the types
    temp_prep_path = os.path.join(project_root, "models", "onnx", "optimized_model_prep.onnx")
    output_path = os.path.join(project_root, "models", "onnx", "quantized_model.onnx")

    print(f"Step 1: Running Shape Inference (fixing missing types)...")
    # This "heals" the graph metadata so the quantizer knows what's a float
    quant_pre_process(input_path, temp_prep_path, skip_symbolic_shape=False)

    print(f"Step 2: Starting Dynamic Quantization...")
    
    quantize_dynamic(
        model_input=temp_prep_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
        # The FIX: Tell ONNX to assume FLOAT for any missing type info
        extra_options={
            'DefaultTensorType': onnx.TensorProto.FLOAT,
            'EnableSubgraph': True
        }
    )
    
    # Cleanup temp file if you want
    if os.path.exists(temp_prep_path):
        os.remove(temp_prep_path)
        # Remember to remove the temp .data file if it exists too
        if os.path.exists(temp_prep_path + ".data"):
            os.remove(temp_prep_path + ".data")

    print(f"✅ Quantization Complete!")
    if os.path.exists(output_path + ".data"):
        q_size = os.path.getsize(output_path + ".data") / (1024**2)
        print(f"📦 New Quantized Weights Size: {q_size:.2f} MB")

if __name__ == "__main__":
    run_task_3b()