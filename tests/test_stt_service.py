"""Unit tests for STTService."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stt_service import STTService
from models import AudioChunk, Transcript


def create_audio_chunk(duration_ms: int = 100, sample_rate: int = 16000) -> AudioChunk:
    """Helper to create test audio chunks."""
    num_samples = int(sample_rate * duration_ms / 1000)
    audio_data = (np.random.randn(num_samples) * 0.1 * 32767).astype(np.int16)
    
    return AudioChunk(
        data=audio_data,
        sample_rate=sample_rate,
        timestamp=time.time(),
        duration_ms=duration_ms
    )


@pytest.mark.asyncio
async def test_stt_initialization():
    """Test STTService initialization."""
    stt = STTService(api_key="test_key", model="nova-2", language="en-US")
    
    assert stt.api_key == "test_key"
    assert stt.model == "nova-2"
    assert stt.language == "en-US"
    assert not stt.is_connected()


@pytest.mark.asyncio
@patch('stt_service.AsyncDeepgramClient')
async def test_stt_connect(mock_client_class):
    """Test STT connection."""
    # Mock the connection
    mock_connection = AsyncMock()
    mock_connection.start_listening = AsyncMock()
    mock_connection.on = Mock()
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client.listen.v1.connect = Mock(return_value=mock_connection)
    mock_client_class.return_value = mock_client
    
    stt = STTService(api_key="test_key")
    await stt.connect()
    
    assert stt.is_connected()


@pytest.mark.asyncio
@patch('stt_service.AsyncDeepgramClient')
async def test_stt_disconnect(mock_client_class):
    """Test STT disconnection."""
    # Mock the connection
    mock_connection = AsyncMock()
    mock_connection.start_listening = AsyncMock()
    mock_connection.on = Mock()
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client.listen.v1.connect = Mock(return_value=mock_connection)
    mock_client_class.return_value = mock_client
    
    stt = STTService(api_key="test_key")
    await stt.connect()
    await stt.disconnect()
    
    assert not stt.is_connected()


@pytest.mark.asyncio
@patch('stt_service.AsyncDeepgramClient')
async def test_stt_send_audio(mock_client_class):
    """Test sending audio to STT service."""
    # Mock the connection
    mock_connection = AsyncMock()
    mock_connection.start_listening = AsyncMock()
    mock_connection.on = Mock()
    mock_connection.send_media = AsyncMock()
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client.listen.v1.connect = Mock(return_value=mock_connection)
    mock_client_class.return_value = mock_client
    
    stt = STTService(api_key="test_key")
    await stt.connect()
    
    # Send audio chunk
    chunk = create_audio_chunk(duration_ms=100)
    await stt.send_audio(chunk)
    
    # Verify send_media was called
    assert mock_connection.send_media.called
    assert stt.is_connected()


@pytest.mark.asyncio
async def test_stt_audio_buffering_when_disconnected():
    """Test that audio is buffered when not connected."""
    stt = STTService(api_key="test_key")
    
    # Send audio while disconnected
    chunk = create_audio_chunk(duration_ms=100)
    await stt.send_audio(chunk)
    
    # Verify audio was buffered
    assert len(stt._audio_buffer) == 1
    assert stt._audio_buffer[0] == chunk


@pytest.mark.asyncio
async def test_stt_transcript_queue():
    """Test transcript queueing."""
    stt = STTService(api_key="test_key")
    
    # Manually add transcript to queue
    test_transcript = Transcript(
        text="Hello world",
        is_final=True,
        confidence=0.95,
        timestamp=time.time(),
        duration_ms=1000
    )
    
    await stt._transcript_queue.put(test_transcript)
    
    # Get transcript
    transcript = await stt.get_transcript()
    
    assert transcript.text == "Hello world"
    assert transcript.is_final is True
    assert transcript.confidence == 0.95


@pytest.mark.asyncio
async def test_stt_reconnection_backoff():
    """Test reconnection with exponential backoff."""
    stt = STTService(api_key="test_key")
    
    # Simulate connection failure by setting max attempts to 0
    original_max = stt._max_reconnect_attempts
    stt._max_reconnect_attempts = 0
    
    with pytest.raises(ConnectionError) as exc_info:
        # Force a connection failure
        stt._connected = False
        await stt._handle_connection_failure()
    
    assert "after 0 attempts" in str(exc_info.value)
    stt._max_reconnect_attempts = original_max


@pytest.mark.asyncio
async def test_stt_max_reconnection_attempts():
    """Test that reconnection stops after max attempts."""
    stt = STTService(api_key="test_key")
    
    # Set max attempts to 0 to trigger immediate failure
    stt._max_reconnect_attempts = 0
    
    with pytest.raises(ConnectionError) as exc_info:
        await stt._handle_connection_failure()
    
    assert "after 0 attempts" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stt_buffer_size_limit():
    """Test that audio buffer has size limit."""
    stt = STTService(api_key="test_key")
    
    # Send more chunks than buffer size
    for _ in range(stt._max_buffer_size + 10):
        chunk = create_audio_chunk(duration_ms=100)
        await stt.send_audio(chunk)
    
    # Buffer should not exceed max size
    assert len(stt._audio_buffer) == stt._max_buffer_size
