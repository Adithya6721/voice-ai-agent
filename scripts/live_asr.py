#!/usr/bin/env python3
"""
scripts/live_asr.py - Run the VAD + ASR (Parakeet) pipeline live from the microphone.
SLM post-processing is disabled.
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voice_ai.pipeline.audio_pipeline import AudioPipeline
from voice_ai.inference.device import get_device
from voice_ai.utils.logging import setup_logging

def main():
    setup_logging(level="INFO")
    
    print("Initializing Live VAD + ASR Pipeline (Parakeet, No SLM)...")
    
    pipeline = AudioPipeline(
        device=get_device(),
        asr_engine="parakeet",
        whisper_model="parakeet-tdt-v2", # Ensures parakeet is used
        enable_slm=False,                # Disable SLM as requested
        enable_enhancement=True,         # Enables HPF (80Hz) + Noise Suppression + Gentle Normalization
        vad_threshold=0.85
    )
    
    pipeline.run()

if __name__ == "__main__":
    main()
