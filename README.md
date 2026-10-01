# local-voice-ai

A fully local, privacy-first voice AI assistant that runs entirely on your machine.
No API keys. No cloud. No data leaves your device.

## How it works

```
Microphone → 16 kHz PCM → Silero VAD (CPU) → ASR (GPU/CPU) → Raw Transcript → SLM → Text Insertion
```

## First Functional Slice

The current build covers:

- Audio capture from microphone at 16 kHz
- Real-time Voice Activity Detection (Silero VAD, CPU-only)
- Automatic Speech Recognition (Whisper via faster-whisper)
- Console output of raw transcripts

## Models Used

| Component | Model | Backend | Device |
|-----------|-------|---------|--------|
| VAD | Silero VAD | ONNX | CPU only (intentional) |
| ASR | Whisper small | faster-whisper / CTranslate2 | CUDA → CPU auto-fallback |

## Quick Start

```powershell
# 1. Download models once (~246 MB Whisper small + ~2 MB Silero VAD)
python scripts/download_models.py

# 2. Run the audio pipeline — speak into mic, see transcripts printed live
python scripts/record_sample.py

# 3. Use the tiny model for faster CPU-only inference
python scripts/record_sample.py --model tiny

# 4. Auto-detect spoken language instead of assuming English
python scripts/record_sample.py --language auto

# 5. List available microphone devices
python scripts/record_sample.py --list-devices

# 6. Benchmark ASR accuracy and latency
python scripts/benchmark_asr.py --audio data/samples/test.wav --reference "your ground truth"

# 7. Benchmark VAD latency (P50/P95/P99)
python scripts/benchmark_vad.py --audio data/samples/test.wav
```

## Installed Package Versions

| Package | Version |
|---------|---------|
| torch | 2.9.1 |
| torchaudio | 2.9.1 |
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.8.2 |
| silero-vad | 6.2.2 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| sounddevice | 0.5.6 |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (optional — auto-detects and falls back to CPU seamlessly)
- Microphone

## Running Tests

```powershell
python -m pytest tests/ -v
# 23 tests passing
```

## Repository Layout

```
local-voice-ai/
├── config/             # YAML configuration (pipeline, models, logging)
├── models/             # Model weight directories (auto-downloaded)
├── scripts/            # CLI entry points (record, benchmark, download)
├── src/voice_ai/
│   ├── audio/          # Mic capture, resampling, audio utilities
│   ├── vad/            # Silero VAD + speech segment detector
│   ├── asr/            # Whisper + Parakeet/Moonshine stubs + transcriber
│   ├── inference/      # Device detection (CUDA → CPU auto-fallback)
│   ├── pipeline/       # AudioPipeline orchestrator + event system
│   ├── slm/            # SLM post-processing (future)
│   ├── system/         # OS hotkey + text insertion (future)
│   └── api/            # FastAPI + WebSocket server (future)
├── tests/              # pytest test suite (23 tests passing)
└── data/samples/       # Place test audio files here for benchmarking
```

## License

MIT
