import librosa
import soundfile as sf
import os

def convert_to_wav(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading {input_path}...")
    # sr=16000 automatically resamples it
    # mono=True ensures it's single-channel
    audio, sr = librosa.load(input_path, sr=16000, mono=True)
    
    print(f"Saving to {output_path}...")
    sf.write(output_path, audio, sr)
    print("Conversion successful!")

if __name__ == "__main__":
    input_file = "C:\\Users\\apran\\Videos\\Cin\\LIBRARY\\ai4bharat-asr-tensorrt-optimization\\src\\data\\sample_audio\\common_voice_hi_23795244.mp3" 
    output_file = "data/sample_audio/sample_hindi.wav"
    
    if os.path.exists(input_file):
        convert_to_wav(input_file, output_file)
    else:
        print(f"Error: Could not find {input_file}. Please check the filename.")