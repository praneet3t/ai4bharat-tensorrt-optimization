import os
import torch
import librosa
from huggingface_hub import hf_hub_download, login
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.profiler import profile, record_function, ProfilerActivity

def run_step_1():
    model_id = "ai4bharat/indicwav2vec-hindi"
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    
    local_dir = os.path.join(project_root, "models", "original_hindi")
    audio_path = os.path.join(project_root, "src", "data", "sample_audio", "sample_hindi.wav")
    report_dir = os.path.join(project_root, "reports", "profiler_plots")

    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print("\n--- AUTHENTICATION ---")
    token = input("Paste your Hugging Face token here: ").strip()
    
    try:
        login(token=token)
        print("Auth Success! Verifying gated files...")
        
        files = ["config.json", "preprocessor_config.json", "pytorch_model.bin", "vocab.json", "tokenizer_config.json"]
        for f in files:
            print(f"Syncing {f}...")
            hf_hub_download(repo_id=model_id, filename=f, local_dir=local_dir, token=token)
            
    except Exception as e:
        print(f"❌ AUTH FAILED: {e}")
        return

    # 3. LOAD MODEL
    print("\nLoading model from local folder...")
    processor = Wav2Vec2Processor.from_pretrained(local_dir)
    model = Wav2Vec2ForCTC.from_pretrained(local_dir)
    model.eval()

    # 4. PROCESS AUDIO
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio not found at {audio_path}")
        return
    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, return_tensors="pt", sampling_rate=16000).input_values

    # 5. PROFILE
    print("Profiling inference operations...")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("wav2vec2_inference"):
            with torch.no_grad():
                model(inputs)

   # 6. REPORT
    print("\n" + "="*70)
    print("                TASK 1: PYTORCH CPU PROFILER TABLE")
    print("="*70)
    
    # 1. Generate the table string
    table_str = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
    
    # 2. Print it to terminal so you can see it
    print(table_str)

    # 3. SAVE TO FILE (Add this exactly here)
    report_file = os.path.join(report_dir, "profiler_report.txt")
    with open(report_file, "w") as f:
        f.write(table_str)
    
    # 4. Save the Chrome trace file
    prof.export_chrome_trace(os.path.join(report_dir, "pytorch_trace.json"))
    print(f"\n✅ TASK COMPLETE. Report saved to {report_file}")

if __name__ == "__main__":
    run_step_1()