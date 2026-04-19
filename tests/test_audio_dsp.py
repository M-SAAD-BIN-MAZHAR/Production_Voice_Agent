"""Tests for microphone DSP pipeline."""

import numpy as np

from src.audio_dsp import build_mic_dsp
from src.models import AudioChunk


def test_dsp_none_passthrough():
    dsp = build_mic_dsp("none", 16000)
    data = np.random.randint(-1000, 1000, 320, dtype=np.int16)
    ch = AudioChunk(data=data, sample_rate=16000, timestamp=0.0, duration_ms=20)
    out = dsp.process(ch)
    assert np.array_equal(out.data, data)


def test_noise_gate_only():
    dsp = build_mic_dsp("noise_gate", 16000, noise_gate_floor_rms=5000.0)
    # Very quiet frame should be attenuated
    quiet = np.zeros(320, dtype=np.int16)
    ch = AudioChunk(data=quiet, sample_rate=16000, timestamp=0.0, duration_ms=20)
    out = dsp.process(ch)
    assert float(np.sqrt(np.mean(out.data.astype(np.float64) ** 2))) < 1.0
