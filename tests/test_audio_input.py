"""Unit tests for AudioInputCapture."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sounddevice as sd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_input import AudioInputCapture
from src.models import AudioChunk


@pytest.mark.asyncio
async def test_audio_input_initialization():
    """Test AudioInputCapture initialization with default parameters."""
    capture = AudioInputCapture(sample_rate=16000, chunk_duration_ms=100)
    
    assert capture.sample_rate == 16000
    assert capture.chunk_duration_ms == 100
    assert capture.chunk_size == 1600  # 16000 * 100 / 1000
    assert not capture.is_running()


@pytest.mark.asyncio
async def test_audio_input_initialization_custom_params():
    """Test AudioInputCapture initialization with custom parameters."""
    capture = AudioInputCapture(sample_rate=24000, chunk_duration_ms=150)
    
    assert capture.sample_rate == 24000
    assert capture.chunk_duration_ms == 150
    assert capture.chunk_size == 3600  # 24000 * 150 / 1000


@pytest.mark.asyncio
@patch('sounddevice.InputStream')
@patch('sounddevice.query_devices')
async def test_audio_input_start(mock_query_devices, mock_input_stream):
    """Test starting audio input capture."""
    # Mock device query to return dict when called with kind='input'
    mock_query_devices.return_value = {'name': 'Test Microphone', 'max_input_channels': 1}
    
    # Mock input stream
    mock_stream = MagicMock()
    mock_input_stream.return_value = mock_stream
    
    capture = AudioInputCapture(sample_rate=16000, chunk_duration_ms=100)
    await capture.start()
    
    assert capture.is_running()
    mock_input_stream.assert_called_once()
    mock_stream.start.assert_called_once()


@pytest.mark.asyncio
@patch('sounddevice.InputStream')
@patch('sounddevice.query_devices')
async def test_audio_input_stop(mock_query_devices, mock_input_stream):
    """Test stopping audio input capture."""
    # Mock device query to return dict when called with kind='input'
    mock_query_devices.return_value = {'name': 'Test Microphone', 'max_input_channels': 1}
    
    # Mock input stream
    mock_stream = MagicMock()
    mock_input_stream.return_value = mock_stream
    
    capture = AudioInputCapture(sample_rate=16000, chunk_duration_ms=100)
    await capture.start()
    await capture.stop()
    
    assert not capture.is_running()
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()


@pytest.mark.asyncio
@patch('sounddevice.query_devices')
@patch('sounddevice.InputStream')
async def test_audio_input_device_unavailable(mock_input_stream, mock_query_devices):
    """Test error handling when microphone is unavailable."""
    # Mock device query to return dict when called with kind='input'
    mock_query_devices.return_value = {'name': 'Test Microphone', 'max_input_channels': 1}
    
    # Simulate device unavailable error
    mock_input_stream.side_effect = sd.PortAudioError("Device unavailable")
    
    capture = AudioInputCapture(sample_rate=16000, chunk_duration_ms=100)
    
    with pytest.raises(RuntimeError) as exc_info:
        await capture.start()
    
    assert "Microphone unavailable" in str(exc_info.value)


@pytest.mark.asyncio
@patch('sounddevice.InputStream')
@patch('sounddevice.query_devices')
async def test_audio_callback_creates_chunks(mock_query_devices, mock_input_stream):
    """Test that audio callback creates AudioChunk objects."""
    # Mock device query to return dict when called with kind='input'
    mock_query_devices.return_value = {'name': 'Test Microphone', 'max_input_channels': 1}
    
    # Capture the callback function
    callback_func = None
    
    def capture_callback(*args, **kwargs):
        nonlocal callback_func
        callback_func = kwargs.get('callback')
        mock_stream = MagicMock()
        return mock_stream
    
    mock_input_stream.side_effect = capture_callback
    
    capture = AudioInputCapture(sample_rate=16000, chunk_duration_ms=100)
    await capture.start()
    
    # Simulate audio callback with test data
    test_audio = np.random.randn(1600, 1).astype(np.float32)
    callback_func(test_audio, 1600, None, None)
    
    # Get the chunk from queue
    chunk = await asyncio.wait_for(capture.get_audio_chunk(), timeout=1.0)
    
    assert isinstance(chunk, AudioChunk)
    assert chunk.sample_rate == 16000
    assert chunk.duration_ms == 100
    assert len(chunk.data) == 1600
    
    await capture.stop()


@pytest.mark.asyncio
async def test_audio_input_graceful_shutdown():
    """Test graceful shutdown clears queue."""
    with patch('sounddevice.InputStream'), patch('sounddevice.query_devices'):
        capture = AudioInputCapture(sample_rate=16000, chunk_duration_ms=100)
        await capture.start()
        
        # Add some dummy chunks to queue
        for _ in range(5):
            await capture._audio_queue.put(
                AudioChunk(
                    data=np.zeros(1600, dtype=np.int16),
                    sample_rate=16000,
                    timestamp=0.0,
                    duration_ms=100
                )
            )
        
        await capture.stop()
        
        # Queue should be empty after stop
        assert capture._audio_queue.empty()
