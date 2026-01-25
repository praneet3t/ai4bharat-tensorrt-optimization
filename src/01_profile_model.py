import os
import torch
import librosa
from huggingface_hub import snapshot_download, login
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.profiler import profile, record_function, ProfilerActivity

def run_step_1():
    model_id = "ai4bharat/indicwav2vec-hindi"
    
    # Path setup
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    local_dir = os.path.join(project_root, "models", "original_hindi")
    audio_path = os.path.join(project_root, "src", "data", "sample_audio", "sample_hindi.wav")
    report_dir = os.path.join(project_root, "reports", "profiler_plots")

    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 1. AUTHENTICATION & FULL DOWNLOAD
    print("\n--- AUTHENTICATION & SYNC ---")
    token = input("Paste your Hugging Face token here: ").strip()
    
    try:
        login(token=token)
        print(f"Syncing full repository to {local_dir}...")
        # snapshot_download ensures config.json, vocab.json, and weights are all present
        snapshot_download(
            repo_id=model_id, 
            local_dir=local_dir, 
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "gradio_queue.db*"] # Skip unnecessary files
        )
    except Exception as e:
        print(f"❌ DOWNLOAD FAILED: {e}")
        return

    # 2. LOAD MODEL (The "Safe" Way)
    print("\nLoading model components...")
    processor = Wav2Vec2Processor.from_pretrained(local_dir)
    # low_cpu_mem_usage=False avoids the 'meta-device' trap you hit
    model = Wav2Vec2ForCTC.from_pretrained(local_dir, low_cpu_mem_usage=False)
    model.eval()

    # 3. PROCESS AUDIO
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio not found at {audio_path}")
        return
    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, return_tensors="pt", sampling_rate=16000).input_values

    # 4. PROFILE (CPU ONLY)
    print("Profiling CPU inference operations...")
    with profile(
        activities=[ProfilerActivity.CPU], 
        record_shapes=True,
        profile_memory=True # Added to check for model size/memory walls
    ) as prof:
        with record_function("wav2vec2_inference"):
            with torch.no_grad():
                model(inputs)

    # 5. REPORTING
    print("\n" + "="*70)
    print("                TASK 1: PYTORCH CPU PROFILER TABLE")
    print("="*70)
    
    # We sort by 'self_cpu_time_total' to see exactly which Op is the culprit
    table_str = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    print(table_str)

    report_file = os.path.join(report_dir, "profiler_report.txt")
    with open(report_file, "w") as f:
        f.write(table_str)
    
    prof.export_chrome_trace(os.path.join(report_dir, "pytorch_trace.json"))
    print(f"\n✅ TASK COMPLETE. Files synced and report saved to {report_file}")

if __name__ == "__main__":
    run_step_1()