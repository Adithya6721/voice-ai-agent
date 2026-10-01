import sys
import os
import queue
import time
import sounddevice as sd
import soundfile as sf
import numpy as np

# Ensure data/benchmark directory exists
os.makedirs("data/benchmark", exist_ok=True)

SENTENCES = [
    ("s1", "I think we should schedule a meeting with Rahul tomorrow at ten in the morning."),
    ("s2", "Uh, I actually wanted to reschedule the meeting because I have another class at that time."),
    ("s3", "Write a Python function that finds the maximum subarray sum using Kadane’s algorithm."),
    ("s4", "Please summarize this document and give me the three most important points as bullet points."),
    ("s5", "The weather in Chennai is quite hot today, but I still want to go out in the evening."),
    ("s6", "Can you remind me to submit the machine learning assignment before Friday and send the report to my professor?"),
]

SAMPLE_RATE = 16000
CHANNELS = 1

def record_audio(filename, sentence):
    print("\n" + "="*80)
    print(f"Sentence [{filename}]:\n")
    print(f"  {sentence}\n")
    print("="*80)
    
    input("Press Enter to START recording...")
    print("\n🔴 RECORDING... (Press Ctrl+C to STOP)")
    
    q = queue.Queue()
    
    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        # We record as float32 to match Whisper's expected input
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⏹️ Recording stopped.")
    
    print("Saving to file...")
    
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())
    
    if len(audio_data) == 0:
        print("No audio recorded! Try again.")
        return False
        
    audio_np = np.concatenate(audio_data, axis=0)
    
    # Save as WAV file (16kHz, mono, float32 or int16)
    sf.write(f"data/benchmark/{filename}.wav", audio_np, SAMPLE_RATE, subtype='PCM_16')
    print(f"✅ Saved to data/benchmark/{filename}.wav")
    return True

def main():
    print("Welcome to the Benchmark Recorder!")
    print("You will be prompted to read 6 sentences.")
    print("For each sentence:")
    print(" 1. Press Enter to start")
    print(" 2. Read the sentence")
    print(" 3. Press Ctrl+C to stop recording\n")
    
    for filename, sentence in SENTENCES:
        success = False
        while not success:
            success = record_audio(filename, sentence)
            if not success:
                retry = input("Try again? (y/n): ")
                if retry.lower() != 'y':
                    break
        time.sleep(1)
        
    print("\n🎉 All sentences recorded successfully!")
    print("You can find them in the 'data/benchmark/' directory.")

if __name__ == "__main__":
    main()
