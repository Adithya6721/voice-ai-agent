"""
voice_ai.audio.enhancer — Audio preprocessing & DSP enhancement module.

Implements a 3-stage DSP pipeline for microphone audio streams:
  1. High-Pass Filter (HPF ~80 Hz): removes DC offset, sub-bass AC hums, and handling noise.
  2. Spectral Noise Suppression: reduces stationary background noise using spectral gating.
  3. Gentle Normalization: peak/RMS scaling with noise floor threshold to prevent amplifying silence.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from loguru import logger

from voice_ai.audio.audio_utils import SAMPLE_RATE


class AudioEnhancer:
    """
    DSP Audio Enhancement pipeline for live speech streams.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        enable_hpf: bool = True,
        hpf_cutoff_hz: float = 80.0,
        enable_noise_suppression: bool = True,
        enable_normalization: bool = True,
        target_peak_db: float = -3.0,
        max_gain_db: float = 6.0,
        noise_floor_rms: float = 0.005,
    ) -> None:
        self.sample_rate = sample_rate
        self.enable_hpf = enable_hpf
        self.hpf_cutoff_hz = hpf_cutoff_hz
        self.enable_noise_suppression = enable_noise_suppression
        self.enable_normalization = enable_normalization
        self.target_peak = 10 ** (target_peak_db / 20.0)  # e.g., -3 dB -> ~0.707
        self.max_gain = 10 ** (max_gain_db / 20.0)        # e.g., +6 dB -> ~2.0
        self.noise_floor_rms = noise_floor_rms

        # Design Butterworth High-Pass Filter SOS (Second-Order Sections for numerical stability)
        nyquist = 0.5 * self.sample_rate
        norm_cutoff = self.hpf_cutoff_hz / nyquist
        self._sos = signal.butter(4, norm_cutoff, btype="highpass", output="sos")
        
        # Filter state for real-time streaming
        self._zi = signal.sosfilt_zi(self._sos)

    def high_pass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply 4th-order Butterworth High-Pass Filter to remove sub-bass rumble (< 80 Hz)."""
        if len(audio) == 0:
            return audio
        
        if self._zi is not None:
            # Use streaming filter with state retention across chunks
            filtered, self._zi = signal.sosfilt(self._sos, audio, zi=self._zi)
            return filtered.astype(np.float32)
        else:
            return signal.sosfilt(self._sos, audio).astype(np.float32)

    def suppress_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Lightweight spectral subtraction / soft noise gate for stationary background noise.
        Computes short-term energy and applies soft thresholding on low-energy noise frames.
        """
        if len(audio) == 0:
            return audio

        # Calculate RMS energy of chunk
        rms = np.sqrt(np.mean(audio**2))
        
        # Soft noise gate: if below noise floor, smoothly attenuate audio
        if rms < self.noise_floor_rms:
            # Calculate attenuation factor (0.0 to 1.0)
            attenuation = (rms / self.noise_floor_rms) ** 2
            return (audio * attenuation).astype(np.float32)

        return audio

    def gentle_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Gentle peak normalization with maximum gain cap to avoid amplifying background noise floor.
        """
        if len(audio) == 0:
            return audio

        max_val = np.max(np.abs(audio))
        if max_val < self.noise_floor_rms:
            # Do not amplify audio if it is effectively silent
            return audio

        # Calculate desired gain to reach target peak
        gain = self.target_peak / max_val
        # Cap gain to prevent over-amplification
        gain = min(gain, self.max_gain)
        
        # Apply gentle gain scaling
        if gain > 1.0:
            audio = audio * gain

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Run full 3-stage audio enhancement pipeline on audio array."""
        if len(audio) == 0:
            return audio

        # Stage 1: High-pass filter
        if self.enable_hpf:
            audio = self.high_pass_filter(audio)

        # Stage 2: Noise suppression / noise gate
        if self.enable_noise_suppression:
            audio = self.suppress_noise(audio)

        # Stage 3: Gentle normalization
        if self.enable_normalization:
            audio = self.gentle_normalize(audio)

        return audio

    def reset_state(self) -> None:
        """Reset internal filter states for new audio streams."""
        self._zi = signal.sosfilt_zi(self._sos)
