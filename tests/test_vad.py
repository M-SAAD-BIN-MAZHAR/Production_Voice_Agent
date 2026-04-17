"""Unit tests for VADModule."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vad import VADModule
from models import AudioChunk, VADEvent, VADEventType


def create_audio_chunk(duration_ms: int = 100, sample_rate: int = 16000, 
                       amplitude: float = 0.1) -> AudioChunk:
    """Helper to create test audio chunks."""
    num_samples = int(sample_rate * duration_ms / 1000)
    # Generate audio with specified amplitude
    audio_data = (np.random.randn(num_samples) * amplitude * 32767).astype(np.int16)
    
    return AudioChunk(
        data=audio_data,
        sample_rate=sample_rate,
        timestamp=time.time(),
        duration_ms=duration_ms
    )


@pytest.mark.asyncio
async def test_vad_initialization():
    """Test VADModule initialization with default parameters."""
    vad = VADModule(threshold=0.5, silence_duration_ms=700)
    
    assert vad.threshold == 0.5
    assert vad.silence_duration_ms == 700
    assert vad._current_state == VADEventType.SILENCE


@pytest.mark.asyncio
async def test_vad_initialization_custom_params():
    """Test VADModule initialization with custom parameters."""
    vad = VADModule(threshold=0.7, silence_duration_ms=500)
    
    assert vad.threshold == 0.7
    assert vad.silence_duration_ms == 500


@pytest.mark.asyncio
async def test_vad_configure():
    """Test VAD configuration update."""
    vad = VADModule(threshold=0.5, silence_duration_ms=700)
    
    vad.configure(threshold=0.6, silence_duration_ms=800)
    
    assert vad.threshold == 0.6
    assert vad.silence_duration_ms == 800


@pytest.mark.asyncio
async def test_vad_silence_detection():
    """Test VAD detects silence correctly."""
    vad = VADModule(threshold=0.5, silence_duration_ms=700)
    
    # Create silent audio chunk (low amplitude)
    silent_chunk = create_audio_chunk(duration_ms=100, amplitude=0.01)
    
    event = await vad.process_audio(silent_chunk)
    
    assert event.event_type == VADEventType.SILENCE
    assert 0.0 <= event.confidence <= 1.0


@pytest.mark.asyncio
async def test_vad_speech_start_detection():
    """Test VAD detects speech start."""
    vad = VADModule(threshold=0.3, silence_duration_ms=700)
    
    # First, process silence
    silent_chunk = create_audio_chunk(duration_ms=100, amplitude=0.01)
    await vad.process_audio(silent_chunk)
    
    # Then, process speech (high amplitude)
    speech_chunk = create_audio_chunk(duration_ms=100, amplitude=0.5)
    event = await vad.process_audio(speech_chunk)
    
    assert event.event_type == VADEventType.SPEECH_START


@pytest.mark.asyncio
async def test_vad_speech_continue():
    """Test VAD transitions to SPEECH_CONTINUE."""
    vad = VADModule(threshold=0.3, silence_duration_ms=700)
    
    # Process silence
    silent_chunk = create_audio_chunk(duration_ms=100, amplitude=0.01)
    await vad.process_audio(silent_chunk)
    
    # Process speech start
    speech_chunk = create_audio_chunk(duration_ms=100, amplitude=0.5)
    await vad.process_audio(speech_chunk)
    
    # Process continuing speech
    event = await vad.process_audio(speech_chunk)
    
    assert event.event_type == VADEventType.SPEECH_CONTINUE


@pytest.mark.asyncio
async def test_vad_speech_end_after_silence():
    """Test VAD detects speech end after silence duration."""
    vad = VADModule(threshold=0.3, silence_duration_ms=300)  # Short duration for testing
    
    # Process silence
    silent_chunk = create_audio_chunk(duration_ms=100, amplitude=0.01)
    await vad.process_audio(silent_chunk)
    
    # Process speech start
    speech_chunk = create_audio_chunk(duration_ms=100, amplitude=0.5)
    await vad.process_audio(speech_chunk)
    
    # Process continuing speech
    await vad.process_audio(speech_chunk)
    
    # Process silence for required duration
    for _ in range(4):  # 4 * 100ms = 400ms > 300ms threshold
        event = await vad.process_audio(silent_chunk)
        await asyncio.sleep(0.1)  # Simulate time passing
    
    # Should eventually get SPEECH_END
    assert event.event_type in [VADEventType.SPEECH_END, VADEventType.SILENCE]


@pytest.mark.asyncio
async def test_vad_state_machine_transitions():
    """Test VAD state machine transitions correctly."""
    vad = VADModule(threshold=0.3, silence_duration_ms=700)
    
    # Start in SILENCE
    assert vad._current_state == VADEventType.SILENCE
    
    # Transition to SPEECH_START
    speech_chunk = create_audio_chunk(duration_ms=100, amplitude=0.5)
    event = await vad.process_audio(speech_chunk)
    assert event.event_type == VADEventType.SPEECH_START
    
    # Transition to SPEECH_CONTINUE
    event = await vad.process_audio(speech_chunk)
    assert event.event_type == VADEventType.SPEECH_CONTINUE


@pytest.mark.asyncio
async def test_vad_simple_fallback():
    """Test VAD uses simple energy-based fallback when Silero not available."""
    vad = VADModule(threshold=0.3, silence_duration_ms=700)
    
    # Force simple VAD by not loading model
    vad._model = None
    vad._model_loaded = True
    
    # Test with speech-like audio
    speech_chunk = create_audio_chunk(duration_ms=100, amplitude=0.5)
    speech_prob = vad._simple_vad(speech_chunk)
    
    assert 0.0 <= speech_prob <= 1.0
    
    # Test with silence
    silent_chunk = create_audio_chunk(duration_ms=100, amplitude=0.01)
    silence_prob = vad._simple_vad(silent_chunk)
    
    assert silence_prob < speech_prob  # Speech should have higher probability


@pytest.mark.asyncio
async def test_vad_handles_different_sample_rates():
    """Test VAD handles different sample rates."""
    vad = VADModule(threshold=0.5, silence_duration_ms=700)
    
    # Test with 24kHz audio
    chunk_24k = create_audio_chunk(duration_ms=100, sample_rate=24000, amplitude=0.1)
    event = await vad.process_audio(chunk_24k)
    
    assert isinstance(event, VADEvent)
    assert event.event_type in [VADEventType.SILENCE, VADEventType.SPEECH_START]
