import tensorrt as trt
import os
import torch

# Global Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine():
    # Use your absolute path
    onnx_file = "/content/quantized_model.onnx"
    engine_file = "/content/models/trt/model.engine"

    # Defensive Hardware Check
    if not torch.cuda.is_available():
        return

    # Initialize Builder (TensorRT 10.x style)
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Workspace for large speech models (1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    print(f"Parsing ONNX from: {onnx_file}")
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Optimization Profile (Wav2Vec2 dynamic length)
    profile = builder.create_optimization_profile()
    # Name MUST match your ONNX input name exactly
    profile.set_shape("input_values", (1, 16000), (1, 80000), (1, 480000))
    config.add_optimization_profile(profile)

    print("Building TensorRT Engine... (approx 3-5 minutes)")
    plan = builder.build_serialized_network(network, config)
    
    with open(engine_file, 'wb') as f:
        f.write(plan)
    print(f"✅ Success! Engine saved to {engine_file}")

if __name__ == "__main__":
    build_engine()