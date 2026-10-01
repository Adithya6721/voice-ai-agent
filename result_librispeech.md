# LibriSpeech `test.clean` Benchmark & Extrapolation

**Hardware Device:** CPU

**Sample Slice Tested:** 1000 samples (~7505.9s audio)

**Full Dataset Projection:** 2,620 utterances (~5.4 hours audio)

## 1. Standalone ASR Performance & Estimated Full Runtime

| Model | Sample WER (%) | Sample RTF | Avg Latency (ms) | RAM (MB) | Full 2.6k Libri Est Time |
|---|---|---|---|---|---|
| **Parakeet TDT v2** | 2.5% | 0.46 | 3456ms | ~225 | **2.49 hours** (~149 mins) |
| **Whisper Small** | 5.5% | 0.78 | 5827ms | ~383 | **4.19 hours** (~252 mins) |
| **Distil-Whisper** | 5.6% | 1.75 | 13152ms | ~279 | **9.46 hours** (~568 mins) |
| **Whisper Base** | 7.4% | 0.28 | 2116ms | ~138 | **1.52 hours** (~91 mins) |

## 2. VAD + ASR Pipeline Impact

| Model | Standalone WER (%) | VAD+ASR WER (%) | VAD Overhead (RTF) | Full Pipeline Est Time |
|---|---|---|---|---|
| **Parakeet TDT v2** | 2.5% | 11.6% | +0.25 RTF | 3.81 hours |
| **Whisper Small** | 5.5% | 13.3% | +0.44 RTF | 6.55 hours |
| **Distil-Whisper** | 5.6% | 13.4% | +-0.77 RTF | 5.29 hours |
| **Whisper Base** | 7.4% | 15.2% | +0.06 RTF | 1.85 hours |
