"""
voice_ai.inference.device — device detection and selection.

Handles auto-detection of CUDA vs CPU and provides a single
source of truth for which compute device is available.

Design:
- Call `get_device()` once at startup.
- Pass the returned `DeviceInfo` to model constructors.
- No model code should import torch directly for device checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable description of the selected inference device."""

    type: DeviceType
    index: int = 0          # CUDA device index (ignored for CPU)
    name: str = "CPU"       # Human-readable device name
    vram_mb: int = 0        # Available VRAM (0 for CPU)

    @property
    def is_cuda(self) -> bool:
        return self.type == DeviceType.CUDA

    @property
    def torch_device(self) -> str:
        """Return a torch-compatible device string e.g. 'cuda:0' or 'cpu'."""
        if self.type == DeviceType.CUDA:
            return f"cuda:{self.index}"
        return "cpu"

    @property
    def ct2_device(self) -> str:
        """Return a CTranslate2-compatible device string for faster-whisper."""
        return self.type.value   # 'cuda' or 'cpu'

    def __str__(self) -> str:
        if self.is_cuda:
            return f"CUDA:{self.index} ({self.name}, {self.vram_mb} MB VRAM)"
        return "CPU"


def _probe_cuda(device_index: int) -> DeviceInfo | None:
    """
    Try to detect a CUDA-capable GPU at the given index.
    Returns DeviceInfo if found, None otherwise.
    """
    try:
        import torch  # type: ignore[import-untyped]

        if not torch.cuda.is_available():
            return None

        if device_index >= torch.cuda.device_count():
            logger.warning(
                f"CUDA device index {device_index} requested but only "
                f"{torch.cuda.device_count()} device(s) found. Falling back to index 0."
            )
            device_index = 0

        props = torch.cuda.get_device_properties(device_index)
        vram_mb = props.total_memory // (1024 * 1024)

        return DeviceInfo(
            type=DeviceType.CUDA,
            index=device_index,
            name=props.name,
            vram_mb=vram_mb,
        )
    except ImportError:
        logger.debug("torch not installed — skipping CUDA probe.")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"CUDA probe failed: {exc}")
        return None


def get_device(preference: str = "auto", cuda_index: int = 0) -> DeviceInfo:
    """
    Resolve the compute device to use.

    Args:
        preference: 'auto' | 'cuda' | 'cpu'
            - 'auto': use CUDA if available, else CPU
            - 'cuda': require CUDA (raises if not available)
            - 'cpu':  always use CPU

    Returns:
        DeviceInfo describing the selected device.

    Raises:
        RuntimeError: if 'cuda' is requested but not available.
    """
    # Allow environment variable override
    preference = os.environ.get("DEVICE", preference).lower()
    cuda_index = int(os.environ.get("CUDA_DEVICE_INDEX", cuda_index))

    if preference == "cpu":
        logger.info("Device: CPU (forced by configuration)")
        return DeviceInfo(type=DeviceType.CPU)

    cuda = _probe_cuda(cuda_index)

    if preference == "cuda":
        if cuda is None:
            raise RuntimeError(
                "CUDA device requested but not available. "
                "Install CUDA-enabled PyTorch or set DEVICE=cpu."
            )
        logger.info(f"Device: {cuda}")
        return cuda

    # preference == "auto"
    if cuda is not None:
        logger.info(f"Device: {cuda} (auto-selected)")
        return cuda

    logger.info("Device: CPU (no CUDA GPU detected — auto-selected)")
    return DeviceInfo(type=DeviceType.CPU)


def whisper_compute_type(device: DeviceInfo, requested: str = "auto") -> str:
    """
    Resolve the best CTranslate2 compute type for faster-whisper.

    auto behaviour:
      - CUDA  -> float16  (fast, accurate, uses Tensor Cores)
      - CPU   -> int8     (fastest on CPU, minimal accuracy loss)

    Fallback:
      - If 'float16' is requested on CPU, automatically fallback to 'int8'
        because CTranslate2 CPU engine does not support float16 execution.
    """
    if requested == "auto":
        return "float16" if device.is_cuda else "int8"
    if not device.is_cuda and requested == "float16":
        logger.warning(
            "float16 compute_type requested on CPU, but CTranslate2 CPU does not "
            "support float16 execution. Falling back to 'int8'."
        )
        return "int8"
    return requested
