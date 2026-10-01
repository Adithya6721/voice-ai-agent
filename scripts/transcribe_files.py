import sys
import os
import glob
import time
import click
import soundfile as sf
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voice_ai.inference.device import get_device
from voice_ai.asr.whisper import WhisperASR

@click.command()
@click.option("--model", default="base", help="Model size: tiny, base, small, medium, distil-small.en, etc.")
@click.option("--dir", "directory", default="data/benchmark", help="Directory containing wav files")
def main(model: str, directory: str):
    logger.remove()
    logger.add(sys.stderr, level="ERROR") # Hide pipeline logs for clean output
    
    device = get_device(preference="cpu")
    print(f"\n🚀 Loading model: {model} on {device.type.value}...")
    
    asr = WhisperASR(
        device=device,
        model_size=model,
        compute_type="auto"
    )
    
    # Warmup
    print("Warming up model...")
    dummy = np.zeros(16000, dtype=np.float32)
    asr.transcribe(dummy)
    
    files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    if not files:
        print(f"No .wav files found in {directory}")
        return
        
    print(f"\n--- Transcriptions for {model} ---")
    for f in files:
        fname = os.path.basename(f)
        audio, sr = sf.read(f)
        
        # Ensure mono float32 16kHz
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        start = time.time()
        result = asr.transcribe(audio)
        rtf = (time.time() - start) / (len(audio) / sr)
        
        print(f"[{fname}] {result.text.strip()} (RTF: {rtf:.2f})")
        
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
