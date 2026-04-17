"""Unit tests for TTS service."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import time

from src.tts_service import TTSService
from src.models import AudioChunk


@pytest.fixture
def tts_service_openai():
    """Create OpenAI TTS service instance for testing."""
    return TTSService(
        provider="openai",
        api_key="test-api-key",
        voice="alloy",
        model="tts-1"
    )


# Edge-TTS test skipped - package not installed
# @pytest.fixture
# def tts_service_edge():
#     """Create Edge-TTS service instance for testing."""
#     return TTSService(
#         provider="edge-tts",
#         voice="en-US-AriaNeural"
#     )


@pytest.mark.asyncio
async def test_tts_service_initialization_openai(tts_service_openai):
    """Test OpenAI TTS service initializes correctly."""
    assert tts_service_openai.provider == "openai"
    assert tts_service_openai.voice == "alloy"
    assert tts_service_openai.model == "tts-1"
    assert tts_service_openai.client is not None


# @pytest.mark.asyncio
# async def test_tts_service_initialization_edge(tts_service_edge):
#     """Test Edge-TTS service initializes correctly."""
#     assert tts_service_edge.provider == "edge-tts"
#     assert tts_service_edge.voice == "en-US-AriaNeural"
#     assert tts_service_edge.client is None


@pytest.mark.asyncio
async def test_tts_service_requires_api_key_for_openai():
    """Test that OpenAI provider requires API key."""
    with pytest.raises(ValueError, match="API key required"):
        TTSService(provider="openai", api_key=None)


@pytest.mark.asyncio
async def test_sentence_boundary_detection(tts_service_openai):
    """Test sentence boundary detection."""
    # Test with period and space
    sentence, remaining = tts_service_openai._detect_sentence_boundary("Hello world. How are you?")
    assert sentence == "Hello world."
    assert remaining == "How are you?"
    
    # Test with question mark
    sentence, remaining = tts_service_openai._detect_sentence_boundary("What time is it? ")
    assert sentence == "What time is it?"
    assert remaining == ""
    
    # Test with exclamation
    sentence, remaining = tts_service_openai._detect_sentence_boundary("Great! Let's go.")
    assert sentence == "Great!"
    assert remaining == "Let's go."
    
    # Test with semicolon
    sentence, remaining = tts_service_openai._detect_sentence_boundary("First part; second part")
    assert sentence == "First part;"
    assert remaining == "second part"
    
    # Test with no boundary
    sentence, remaining = tts_service_openai._detect_sentence_boundary("Incomplete sentence")
    assert sentence is None
    assert remaining == "Incomplete sentence"
    
    # Test with boundary at end
    sentence, remaining = tts_service_openai._detect_sentence_boundary("Complete sentence.")
    assert sentence == "Complete sentence."
    assert remaining == ""


@pytest.mark.asyncio
async def test_synthesize_openai(tts_service_openai):
    """Test OpenAI synthesis."""
    # Mock the OpenAI client
    mock_response = AsyncMock()
    
    async def mock_iter_bytes(chunk_size):
        # Simulate audio chunks
        for i in range(3):
            await asyncio.sleep(0.01)
            # Generate fake audio data
            audio_bytes = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()
            yield audio_bytes
    
    mock_response.iter_bytes = mock_iter_bytes
    tts_service_openai.client.audio.speech.create = AsyncMock(return_value=mock_response)
    
    # Synthesize text
    text = "Hello, world!"
    chunks = []
    
    async for chunk in tts_service_openai.synthesize(text):
        chunks.append(chunk)
    
    # Verify chunks were generated
    assert len(chunks) == 3
    for chunk in chunks:
        assert isinstance(chunk, AudioChunk)
        assert chunk.sample_rate == 24000
        assert len(chunk.data) > 0


@pytest.mark.asyncio
async def test_synthesize_stream_buffers_sentences(tts_service_openai):
    """Test that token stream is buffered until sentence boundaries."""
    # Mock synthesis
    synthesized_texts = []
    
    async def mock_synthesize(text):
        synthesized_texts.append(text)
        # Return fake audio chunk
        audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
        yield AudioChunk(
            data=audio_data,
            sample_rate=24000,
            timestamp=time.time(),
            duration_ms=100
        )
    
    tts_service_openai.synthesize = mock_synthesize
    
    # Create token stream
    async def token_stream():
        tokens = ["Hello", " ", "world", ".", " ", "How", " ", "are", " ", "you", "?"]
        for token in tokens:
            await asyncio.sleep(0.01)
            yield token
    
    # Synthesize stream
    chunks = []
    async for chunk in tts_service_openai.synthesize_stream(token_stream()):
        chunks.append(chunk)
    
    # Verify sentences were synthesized
    assert len(synthesized_texts) == 2
    assert "Hello world." in synthesized_texts[0]
    assert "How are you?" in synthesized_texts[1]


@pytest.mark.asyncio
async def test_synthesize_stream_handles_incomplete_sentence(tts_service_openai):
    """Test that incomplete sentences at end are synthesized."""
    synthesized_texts = []
    
    async def mock_synthesize(text):
        synthesized_texts.append(text)
        audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
        yield AudioChunk(
            data=audio_data,
            sample_rate=24000,
            timestamp=time.time(),
            duration_ms=100
        )
    
    tts_service_openai.synthesize = mock_synthesize
    
    # Create token stream with incomplete sentence
    async def token_stream():
        tokens = ["Hello", " ", "world", ".", " ", "Incomplete"]
        for token in tokens:
            yield token
    
    # Synthesize stream
    chunks = []
    async for chunk in tts_service_openai.synthesize_stream(token_stream()):
        chunks.append(chunk)
    
    # Verify both complete and incomplete sentences were synthesized
    assert len(synthesized_texts) == 2
    assert "Hello world." in synthesized_texts[0]
    assert "Incomplete" in synthesized_texts[1]


@pytest.mark.asyncio
async def test_cancel_synthesis(tts_service_openai):
    """Test that synthesis can be cancelled."""
    # Mock a long-running synthesis
    async def mock_synthesize(text):
        for i in range(100):
            await asyncio.sleep(0.1)
            audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
            yield AudioChunk(
                data=audio_data,
                sample_rate=24000,
                timestamp=time.time(),
                duration_ms=100
            )
    
    tts_service_openai.synthesize = mock_synthesize
    
    # Start synthesis
    chunks = []
    
    async def synthesize():
        async for chunk in tts_service_openai.synthesize("Long text"):
            chunks.append(chunk)
    
    # Start synthesis task
    task = asyncio.create_task(synthesize())
    
    # Wait a bit then cancel
    await asyncio.sleep(0.15)
    await tts_service_openai.cancel_synthesis()
    
    # Wait for task to complete
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        pass
    
    # Verify synthesis was stopped early
    assert len(chunks) < 100


@pytest.mark.asyncio
async def test_synthesize_empty_text(tts_service_openai):
    """Test handling of empty text."""
    chunks = []
    async for chunk in tts_service_openai.synthesize(""):
        chunks.append(chunk)
    
    # Should not generate any chunks
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_punctuation_preservation(tts_service_openai):
    """Test that punctuation is preserved in synthesis."""
    synthesized_text = None
    
    async def mock_create(*args, **kwargs):
        nonlocal synthesized_text
        synthesized_text = kwargs.get("input")
        
        # Return mock response
        mock_response = AsyncMock()
        async def mock_iter_bytes(chunk_size):
            audio_bytes = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()
            yield audio_bytes
        mock_response.iter_bytes = mock_iter_bytes
        return mock_response
    
    tts_service_openai.client.audio.speech.create = mock_create
    
    # Synthesize text with punctuation
    text = "Hello, world! How are you?"
    async for _ in tts_service_openai.synthesize(text):
        pass
    
    # Verify punctuation was preserved
    assert synthesized_text == text


@pytest.mark.asyncio
async def test_voice_consistency(tts_service_openai):
    """Test that voice parameter remains constant."""
    voice_used = []
    
    async def mock_create(*args, **kwargs):
        voice_used.append(kwargs.get("voice"))
        
        mock_response = AsyncMock()
        async def mock_iter_bytes(chunk_size):
            audio_bytes = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()
            yield audio_bytes
        mock_response.iter_bytes = mock_iter_bytes
        return mock_response
    
    tts_service_openai.client.audio.speech.create = mock_create
    
    # Synthesize multiple texts
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    for text in texts:
        async for _ in tts_service_openai.synthesize(text):
            pass
    
    # Verify voice was consistent
    assert len(voice_used) == 3
    assert all(v == "alloy" for v in voice_used)

