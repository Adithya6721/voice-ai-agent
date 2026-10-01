"""
voice_ai.asr.parakeet - NVIDIA Parakeet (TDT) integration via sherpa-onnx.

sherpa-onnx provides highly optimized edge ASR.
Parakeet TDT (FastConformer) provides excellent accuracy with very fast RTF.
"""

from __future__ import annotations

import os
import time
import urllib.request
import tarfile
import numpy as np
from typing import Optional
from loguru import logger

from voice_ai.asr.base import BaseASR, TranscriptionResult
from voice_ai.inference.device import DeviceInfo, get_device
from voice_ai.audio.audio_utils import SAMPLE_RATE

# See: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
MODEL_URLS = {
    "parakeet-tdt-v2": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
    "parakeet-tdt-v3": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
}
MODEL_DIR_NAMES = {
    "parakeet-tdt-v2": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    "parakeet-tdt-v3": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
}

class ParakeetASR(BaseASR):
    def __init__(
        self,
        device: Optional[DeviceInfo] = None,
        model_size: str = "parakeet-tdt-v2",
        download_root: Optional[str] = None,
    ) -> None:
        if device is None:
            device = get_device()
        self._device = device
        self._model_size = model_size
        
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("Please install sherpa-onnx: `pip install sherpa-onnx`")

        if download_root is None:
            download_root = os.path.expanduser("~/.cache/sherpa-onnx")
            
        os.makedirs(download_root, exist_ok=True)
        model_key = model_size if model_size in MODEL_URLS else "parakeet-tdt-v2"
        model_path = os.path.join(download_root, MODEL_DIR_NAMES[model_key])
        
        if not os.path.exists(model_path):
            self._download_model(model_key, download_root)

        logger.info(f"Loading Parakeet {model_size} via sherpa-onnx | device={device.type.value}")
        load_start = time.perf_counter()
        
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(model_path, "encoder.int8.onnx"),
            decoder=os.path.join(model_path, "decoder.int8.onnx"),
            joiner=os.path.join(model_path, "joiner.int8.onnx"),
            tokens=os.path.join(model_path, "tokens.txt"),
            num_threads=4,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            model_type="nemo_transducer",
            provider="cuda" if device.is_cuda else "cpu"
        )
            
        load_ms = (time.perf_counter() - load_start) * 1000
        logger.info(f"Parakeet loaded in {load_ms:.0f}ms.")

    def _download_model(self, model_key: str, download_root: str) -> None:
        import tarfile
        url = MODEL_URLS[model_key]
        tar_path = os.path.join(download_root, "parakeet.tar.bz2")
        logger.info(f"Downloading Parakeet model to {tar_path}...")
        urllib.request.urlretrieve(url, tar_path)
        
        logger.info("Extracting Parakeet model...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=download_root)
        os.remove(tar_path)
        logger.info("Parakeet model ready.")

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        import sherpa_onnx
        
        start_time = time.perf_counter()
        duration = len(audio) / SAMPLE_RATE

        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.lower()
        except Exception as exc:
            logger.error(f"Parakeet transcription failed: {exc}")
            text = ""

        latency = (time.perf_counter() - start_time) * 1000

        return TranscriptionResult(
            text=text,
            language="en",
            duration_s=duration,
            latency_ms=latency,
            model_name=self._model_size,
        )
