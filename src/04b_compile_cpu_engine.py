import os
import openvino as ov

def compile_for_cpu():
    # 1. Path Setup
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_xml = os.path.join(root, "models", "openvino", "model.xml")
    
    if not os.path.exists(model_xml):
        print("❌ Error: OpenVINO model.xml not found. Run 02b first.")
        return

    # 2. Initialize Core
    core = ov.Core()
    
    # 3. Read and Compile
    print("Reading OpenVINO IR...")
    model = core.read_model(model=model_xml)
    
    print("Compiling model for CPU... (Optimizing for your local architecture)")
    # This step is the CPU equivalent of creating a TensorRT engine
    compiled_model = core.compile_model(model=model, device_name="CPU")
    
    # 4. Verification
    print("✅ CPU Engine/Compiled Model is ready.")
    print(f"Inference Precision: {core.get_property('CPU', 'OPTIMIZATION_CAPABILITIES')}")

if __name__ == "__main__":
    compile_for_cpu()