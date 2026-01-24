import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoprimaryctx # Better than autoinit for TRT 10.x
import numpy as np
import time
import os

def run_benchmark():
    engine_path = "/content/models/trt/model.engine"
    
    # 1. Deserialize Engine
    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 2. Setup Input (5 seconds of audio)
    input_shape = (1, 80000)
    context.set_input_shape("input_values", input_shape)
    
    # Get dynamic output shape (Logits)
    output_shape = context.get_tensor_shape("logits")
    
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    # Allocate GPU Memory
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    print("Benchmarking 100 iterations...")
    start = time.time()
    for _ in range(100):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # execute_async_v3 is the 2026 standard for dynamic shapes
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    
    avg_latency = (time.time() - start) / 100 * 1000
    print(f"\n🚀 TENSORRT PERFORMANCE")
    print(f"Avg Latency for 5s audio: {avg_latency:.2f} ms")
    print(f"Speedup: {5000 / avg_latency:.1f}x faster than real-time")

if __name__ == "__main__":
    run_benchmark()