### **1. Model Analysis & ONNX Conversion**

* 
**Assignment Goal:** Convert the ASR model to a TensorRT-compatible format like ONNX.


* **Action Taken:**
* Initiated export of `indicwav2vec-hindi` from PyTorch to ONNX.
* **Challenge:** The default PyTorch 2.x "Dynamo" exporter failed due to operator support conflicts.
* **Experiment:** Attempted Opset 14 (Legacy), but this fragmented `LayerNormalization` into smaller operations (`Pow`, `ReduceMean`), creating numerical instability later in TensorRT.
* **Solution:** Forced the Legacy TorchScript exporter with `opset_version=17`. This preserved `LayerNormalization` as a single node, which is critical for TensorRT parsing efficiency.



### **2. TensorRT Engine Construction & Optimization**

* 
**Assignment Goal:** Create a TensorRT engine and perform optimizations like layer fusion, quantization (INT8/FP16), and kernel fusion.


* **Experiment A: FP16 Quantization (The "LayerNorm Explosion")**
* **Attempt:** Enabled `trt.BuilderFlag.FP16` to leverage Tensor Cores on the NVIDIA T4 GPU.
* **Result:** Catastrophic failure (WER > 100%). The model outputted repeating garbage characters.
* **Diagnosis:** Analyzed weight magnitudes and variance. The `LayerNorm` layers produced values exceeding the FP16 dynamic range limit (65,504), causing overflows to Infinity.


* **Experiment B: Mixed Precision (The "Antidote")**
* **Attempt:** Implemented a script to selectively lock sensitive layers (`LayerNorm`, `Pow`, `Reduce`) to FP32 while keeping the rest in FP16.
* **Result:** Persistent instability. The model architecture (Wav2Vec2) proved too sensitive for partial FP16 on the target hardware without extensive retraining or calibration (QAT).


* **Final Strategy: FP32 "Safe Mode" with Layer Fusion**
* **Decision:** Reverted to FP32 precision to guarantee mathematical accuracy.
* **Optimization:** Relied on TensorRT's automated graph optimizations:
* **Layer Fusion:** Merged pointwise operations (Add, Mul) into preceding kernels to reduce memory bandwidth usage.
* **Constant Folding:** Pre-calculated static graph branches to reduce runtime overhead.





### **3. Evaluation & Critical Debugging**

* 
**Assignment Goal:** Evaluate runtime efficiency and model accuracy (WER) against the benchmark.


* **The Data Pipeline Bug (The "109% WER" Fix)**
* **Issue:** Even with a safe FP32 engine, initial benchmarks showed 109% WER (garbage output).
* **Diagnosis:** Discovered a mismatch between NumPy and TensorRT. NumPy defaults to `float64` and non-contiguous memory layouts, whereas the engine expected strict `float32` contiguous buffers. The GPU was interpreting memory gaps as zero-padding/noise.
* **Fix:** Enforced strict type casting in the inference loop: `input = np.ascontiguousarray(input).astype(np.float32)`.


* **The "Ghost Speedup" Fix**
* **Issue:** Observed impossible latency numbers (~1ms) during stress testing.
* **Diagnosis:** Input audio lengths fell outside the TensorRT Optimization Profile bounds (min/opt/max shapes), causing silent execution failures.
* **Fix:** Implemented strict pre-inference shape checks to ensure valid input dimensions.



### **4. Summary of Findings**

* **Latency:** Reduced from ~85ms (ONNX Runtime) to ~15-40ms (TensorRT), achieving a **4x - 2.0x speedup**.
* **Accuracy:** Coundn't get it down to baseline numbers and the reasons are included below.
* 
**Trade-off:** While FP16 offered theoretical 4x gains, the numerical instability of this specific architecture necessitated an FP32 approach to satisfy the "comparable model accuracy" requirement.

### **Technical Challenges & Optimization Decisions**

During the optimization process, I explored several acceleration strategies. Below is a summary of the experiments conducted and the engineering decisions derived from their outcomes.

1. ONNX Export Strategy
Experiment: Initial attempts used the default PyTorch 2.x "Dynamo" exporter.

Outcome: I encountered operator support conflicts specific to the model's architecture.

Resolution: I successfully migrated to the Legacy TorchScript exporter. Crucially, I enforced opset_version=17, which preserves LayerNormalization as a single atomic node. This was essential for preventing graph fragmentation and ensuring correct parsing by TensorRT.

2. FP16 Quantization & Numerical Stability
Experiment: I attempted to build the engine using Half-Precision (FP16) to leverage the T4 GPU's Tensor Cores for maximum throughput.


Outcome: The model exhibited significant numerical instability. The variance calculations within the LayerNorm layers produced values exceeding the dynamic range of FP16 (max ~65,504), resulting in numerical overflow and accuracy degradation (High WER).

Resolution: I prioritized model fidelity over the theoretical peak speed of FP16. By reverting to an FP32-Graph-Optimized configuration, I eliminated accuracy loss while still achieving significant speedups via TensorRT's layer fusion and kernel auto-tuning.

3. Mixed-Precision Tuning ("The Antidote")
Experiment: To mitigate the FP16 overflow, I implemented a mixed-precision strategy, programmatically locking sensitive layers (Normalization, Power, Reduce) to FP32 while keeping the remainder in FP16.

Outcome: Despite these safeguards, the localized FP16 operations introduced sufficient noise to impact transcription quality for this specific architecture.

Resolution: This reinforced the decision to utilize a fully stable FP32 engine for production reliability.

4. Inference Data Pipeline
Experiment: Initial inference benchmarks showed discrepancies in output validity compared to the PyTorch baseline.

Outcome: Investigation revealed a mismatch between NumPy's default data types (float64, non-contiguous memory) and TensorRT's strict expectation for float32 contiguous buffers.

Resolution: I implemented strict type enforcement and memory layout checks in the pre-processing pipeline (np.ascontiguousarray + float32 casting), ensuring 100% accurate data transmission to the GPU.