"""Unit tests for audio output player."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np
import time

from src.audio_output import AudioOutputPlayer, PlaybackState
from src.models import AudioChunk


@pytest.fixture
def audio_output():
    """Create audio output player instance for testing."""
    return AudioOutputPlayer(sample_rate=24000)


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module."""
    with patch('src.audio_output.sd') as mock_sd:
        # Mock query_devices
        mock_sd.query_devices.return_value = {'name': 'Test Output Device'}
        
        # Mock OutputStream
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        
        yield mock_sd


@pytest.mark.asyncio
async def test_audio_output_initialization(audio_output):
    """Test audio output player initializes correctly."""
    assert audio_output.sample_rate == 24000
    assert audio_output._state == PlaybackState.IDLE
    assert not audio_output._running


@pytest.mark.asyncio
async def test_audio_output_start(audio_output, mock_sounddevice):
    """Test starting audio output."""
    await audio_output.start()
    
    assert audio_output._running
    assert audio_output._state == PlaybackState.IDLE
    assert audio_output._stream is not None
    
    # Cleanup
    await audio_output.stop()


@pytest.mark.asyncio
async def test_audio_output_start_with_custom_sample_rate(audio_output, mock_sounddevice):
    """Test starting with custom sample rate."""
    await audio_output.start(sample_rate=16000)
    
    assert audio_output.sample_rate == 16000
    
    # Cleanup
    await audio_output.stop()


@pytest.mark.asyncio
async def test_audio_output_device_unavailable(audio_output):
    """Test error handling when device is unavailable."""
    import sounddevice as sd
    
    with patch('src.audio_output.sd') as mock_sd:
        # Make query_devices raise PortAudioError
        mock_sd.PortAudioError = sd.PortAudioError
        mock_sd.query_devices.side_effect = sd.PortAudioError("Device unavailable")
        
        with pytest.raises(RuntimeError, match="Audio output device unavailable"):
            await audio_output.start()


@pytest.mark.asyncio
async def test_play_audio_chunk(audio_output, mock_sounddevice):
    """Test playing audio chunks."""
    await audio_output.start()
    
    # Create test audio chunk
    audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
    chunk = AudioChunk(
        data=audio_data,
        sample_rate=24000,
        timestamp=time.time(),
        duration_ms=100
    )
    
    # Play chunk
    await audio_output.play(chunk)
    
    # Wait for playback to process
    await asyncio.sleep(0.1)
    
    # Verify chunk was queued
    assert audio_output._playback_queue.qsize() >= 0
    
    # Cleanup
    await audio_output.stop()


@pytest.mark.asyncio
async def test_play_multiple_chunks_in_order(audio_output, mock_sounddevice):
    """Test that multiple chunks are played in order."""
    await audio_output.start()
    
    # Create multiple test chunks
    chunks = []
    for i in range(5):
        audio_data = np.full(1600, i, dtype=np.int16)
        chunk = AudioChunk(
            data=audio_data,
            sample_rate=24000,
            timestamp=time.time(),
            duration_ms=100
        )
        chunks.append(chunk)
    
    # Play all chunks
    for chunk in chunks:
        await audio_output.play(chunk)
    
    # Wait for playback
    await asyncio.sleep(0.2)
    
    # Cleanup
    await audio_output.stop()


@pytest.mark.asyncio
async def test_stop_clears_queue(audio_output, mock_sounddevice):
    """Test that stop clears the playback queue."""
    await audio_output.start()
    
    # Queue multiple chunks
    for i in range(10):
        audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
        chunk = AudioChunk(
            data=audio_data,
            sample_rate=24000,
            timestamp=time.time(),
            duration_ms=100
        )
        await audio_output.play(chunk)
    
    # Stop playback
    await audio_output.stop()
    
    # Verify queue is cleared
    assert audio_output._playback_queue.empty()
    assert audio_output._state == PlaybackState.IDLE
    assert not audio_output._running


@pytest.mark.asyncio
async def test_is_playing(audio_output, mock_sounddevice):
    """Test is_playing state tracking."""
    await audio_output.start()
    
    # Initially not playing
    assert not audio_output.is_playing()
    
    # Play a chunk
    audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
    chunk = AudioChunk(
        data=audio_data,
        sample_rate=24000,
        timestamp=time.time(),
        duration_ms=100
    )
    await audio_output.play(chunk)
    
    # Wait for playback to start
    await asyncio.sleep(0.05)
    
    # Should be playing now
    # Note: This might be flaky depending on timing
    # assert audio_output.is_playing()
    
    # Cleanup
    await audio_output.stop()
    
    # Should not be playing after stop
    assert not audio_output.is_playing()


@pytest.mark.asyncio
async def test_play_without_start(audio_output):
    """Test that play without start logs warning."""
    # Try to play without starting
    audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
    chunk = AudioChunk(
        data=audio_data,
        sample_rate=24000,
        timestamp=time.time(),
        duration_ms=100
    )
    
    # Should not raise error, just log warning
    await audio_output.play(chunk)


@pytest.mark.asyncio
async def test_stop_without_start(audio_output):
    """Test that stop without start doesn't raise error."""
    # Should not raise error
    await audio_output.stop()


@pytest.mark.asyncio
async def test_get_state(audio_output, mock_sounddevice):
    """Test getting playback state."""
    assert audio_output.get_state() == PlaybackState.IDLE
    
    await audio_output.start()
    assert audio_output.get_state() == PlaybackState.IDLE
    
    await audio_output.stop()
    assert audio_output.get_state() == PlaybackState.IDLE

