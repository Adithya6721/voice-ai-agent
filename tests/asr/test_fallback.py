import pytest
import numpy as np
from voice_ai.asr.factory import FallbackASR, DEFAULT_FALLBACK_CHAIN
from voice_ai.asr.base import BaseASR, TranscriptionResult

class MockFailingASR(BaseASR):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Simulated load failure")

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        return TranscriptionResult(text="")

class MockSuccessASR(BaseASR):
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        return TranscriptionResult(text="hello world", model_name="MockSuccessASR")

def test_fallback_chain_initialization(monkeypatch):
    monkeypatch.setattr("voice_ai.asr.factory.create_single_model", MockSuccessASR)
    
    fallback = FallbackASR()
    assert fallback.active_model is not None
    assert "MockSuccessASR" in fallback.name

def test_fallback_chain_stepping(monkeypatch):
    call_count = 0
    
    def mock_create(engine, model_size, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Primary model missing")
        return MockSuccessASR()

    monkeypatch.setattr("voice_ai.asr.factory.create_single_model", mock_create)
    
    chain = [
        {"engine": "parakeet", "model_size": "parakeet-tdt-v2"},
        {"engine": "whisper", "model_size": "small"},
    ]
    fallback = FallbackASR(chain=chain)
    assert call_count == 2
    assert "MockSuccessASR" in fallback.name
