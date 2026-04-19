"""Lightweight microphone DSP for capture path (laptop/speaker setups).

True acoustic echo cancellation needs a reference signal from playback; this module
applies streaming high-pass filtering and an optional noise gate to reduce
low-frequency rumble and steady background hiss — helpful before VAD/STT.

For production-grade AEC, integrate a WebRTC-based stack or OS/driver AEC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.models import AudioChunk

logger = logging.getLogger(__name__)

try:
    from scipy.signal import butter, lfilter, lfilter_zi
except ImportError:  # pragma: no cover
    butter = lfilter = lfilter_zi = None  # type: ignore


class InputDSPMode(str, Enum):
    NONE = "none"
    HIGHPASS = "highpass"
    NOISE_GATE = "noise_gate"
    HIGHPASS_NOISE_GATE = "highpass_noise_gate"


@dataclass
class _StreamingHighpass:
    """One-pole high-pass via scipy lfilter with streaming state."""

    def __init__(self, sample_rate: int, cutoff_hz: float = 80.0, order: int = 2) -> None:
        if butter is None or lfilter is None or lfilter_zi is None:
            raise RuntimeError("scipy is required for highpass DSP mode")
        self._b, self._a = butter(order, cutoff_hz / (sample_rate / 2.0), btype="high")
        self._zi = lfilter_zi(self._b, self._a)

    def process(self, samples_int16: np.ndarray) -> np.ndarray:
        x = samples_int16.astype(np.float32) / 32768.0
        y, self._zi = lfilter(self._b, self._a, x, zi=self._zi)
        out = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
        return out


def _noise_gate(samples_int16: np.ndarray, floor_rms: float = 800.0, attenuation: float = 0.25) -> np.ndarray:
    """Attenuate frames with very low RMS (simple gate; use conservative floor)."""
    x = samples_int16.astype(np.float32)
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    if rms < floor_rms:
        return (x * attenuation).astype(np.int16)
    return samples_int16


class MicDSPPipeline:
    """Stateful pipeline applied per AudioChunk."""

    def __init__(
        self,
        mode: InputDSPMode = InputDSPMode.NONE,
        sample_rate: int = 16000,
        highpass_cutoff_hz: float = 80.0,
        noise_gate_floor_rms: float = 800.0,
    ) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._hp: Optional[_StreamingHighpass] = None
        self._noise_gate_floor = noise_gate_floor_rms

        if mode in (InputDSPMode.HIGHPASS, InputDSPMode.HIGHPASS_NOISE_GATE):
            if butter is None:
                logger.warning("scipy not installed; input_dsp highpass disabled")
                self._mode = (
                    InputDSPMode.NOISE_GATE
                    if mode == InputDSPMode.HIGHPASS_NOISE_GATE
                    else InputDSPMode.NONE
                )
            else:
                self._hp = _StreamingHighpass(sample_rate, cutoff_hz=highpass_cutoff_hz)

    def process(self, chunk: AudioChunk) -> AudioChunk:
        if self._mode == InputDSPMode.NONE:
            return chunk

        data = chunk.data
        if self._hp is not None:
            data = self._hp.process(data)

        if self._mode in (InputDSPMode.NOISE_GATE, InputDSPMode.HIGHPASS_NOISE_GATE):
            data = _noise_gate(data, floor_rms=self._noise_gate_floor)

        return AudioChunk(
            data=data,
            sample_rate=chunk.sample_rate,
            timestamp=chunk.timestamp,
            duration_ms=chunk.duration_ms,
        )


def build_mic_dsp(
    mode_str: str,
    sample_rate: int,
    highpass_cutoff_hz: float = 80.0,
    noise_gate_floor_rms: float = 800.0,
) -> MicDSPPipeline:
    try:
        mode = InputDSPMode(mode_str.lower())
    except ValueError:
        logger.warning(f"Unknown input_dsp mode '{mode_str}', using none")
        mode = InputDSPMode.NONE
    return MicDSPPipeline(
        mode=mode,
        sample_rate=sample_rate,
        highpass_cutoff_hz=highpass_cutoff_hz,
        noise_gate_floor_rms=noise_gate_floor_rms,
    )
