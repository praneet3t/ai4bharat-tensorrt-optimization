import os
import torch
import librosa
import openvino as ov
from datasets import load_dataset
from jiwer import wer
from transformers import Wav2Vec2Processor

def evaluate_accuracy():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_xml = os.path.join(root, "models", "openvino", "model.xml")
    local_dir = os.path.join(root, "models", "original_hindi")

    # 1. Load Processor and OpenVINO Engine
    processor = Wav2Vec2Processor.from_pretrained(local_dir)
    core = ov.Core()
    compiled_model = core.compile_model(model_xml, "CPU")
    infer_request = compiled_model.create_infer_request()

    # 2. Load a small subset of Common Voice Hindi (Test Set)
    print("Loading Common Voice Hindi test data...")
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", streaming=True, trust_remote_code=True)
    
    predictions = []
    references = []
    
    print("Evaluating first 20 samples for WER...")
    for i, item in enumerate(dataset.take(20)):
        audio = item["audio"]["array"]
        ref_text = item["sentence"]
        
        # Pre-process
        input_values = processor(audio, return_tensors="np", sampling_rate=16000).input_values
        
        # Infer using OpenVINO
        logits = infer_request.infer([input_values])[0]
        
        # Decode
        pred_ids = np.argmax(logits, axis=-1)
        transcription = processor.batch_decode(pred_ids)[0]
        
        predictions.append(transcription)
        references.append(ref_text)
        print(f"Sample {i+1} | Ref: {ref_text[:30]}... | Pred: {transcription[:30]}...")

    # 3. Calculate WER
    error_rate = wer(references, predictions)
    print("\n" + "="*40)
    print(f"FINAL TEST WER: {error_rate * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_accuracy()