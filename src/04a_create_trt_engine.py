import os
import tensorrt as trt

def build_engine():
    # 1. Path Setup
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(root, "models", "onnx", "model.onnx")
    engine_path = os.path.join(root, "models", "tensorrt", "model.engine")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    # 2. Initialize Builder and Config
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    # 3. Parse ONNX Model
    print(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 4. Set Optimizations (Step 5 requirements)
    # Enable FP16 precision for a 2x speedup on supported GPUs
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 optimization enabled.")
    
    # Set memory limit for the build process (e.g., 2GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    # 5. Build and Save
    print("Building TensorRT engine... (This can take 5-10 minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✅ TensorRT Engine created: {engine_path}")

if __name__ == "__main__":
    build_engine()