import os
import time
import numpy as np
import openvino as ov

def run_optimized_inference():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_xml = os.path.join(root, "models", "openvino", "model.xml")
    
    core = ov.Core()
    model = core.read_model(model_xml)

    # STEP 5 OPTIMIZATION: Set performance hints for CPU
    # 'THROUGHPUT' tells OpenVINO to use all CPU cores to process multiple audios at once
    config = {"PERFORMANCE_HINT": "THROUGHPUT"}
    compiled_model = core.compile_model(model, "CPU", config)
    
    # Create a "dummy" batch of 4 audio snippets (1 sec each)
    batch_size = 4
    input_data = np.random.randn(batch_size, 16000).astype(np.float32)

    print(f"Running optimized batch inference (Batch Size: {batch_size})...")
    start_time = time.time()
    
    # Run inference
    results = compiled_model(input_data)
    
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    print(f"✅ Batch Inference Complete!")
    print(f"Total Time for {batch_size} samples: {total_time:.2f} ms")
    print(f"Latency per sample: {total_time/batch_size:.2f} ms")

if __name__ == "__main__":
    run_optimized_inference()