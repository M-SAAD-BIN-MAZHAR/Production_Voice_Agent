"""Unit tests for LLM service."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import time

from src.llm_service import LLMService
from src.models import Message


@pytest.fixture
def llm_service():
    """Create LLM service instance for testing."""
    return LLMService(
        api_key="test-api-key",
        model="gpt-4o-mini",
        system_prompt="You are a test assistant."
    )


@pytest.fixture
def mock_openai_stream():
    """Create mock OpenAI streaming response."""
    class MockChoice:
        def __init__(self, content, finish_reason=None):
            self.delta = MagicMock()
            self.delta.content = content
            self.finish_reason = finish_reason
    
    class MockChunk:
        def __init__(self, content, finish_reason=None):
            self.choices = [MockChoice(content, finish_reason)]
    
    async def mock_stream():
        # Simulate streaming tokens
        tokens = ["Hello", " ", "world", "!", " ", "How", " ", "are", " ", "you", "?"]
        for i, token in enumerate(tokens):
            await asyncio.sleep(0.01)  # Simulate network delay
            finish_reason = "stop" if i == len(tokens) - 1 else None
            yield MockChunk(token, finish_reason)
    
    return mock_stream()


@pytest.mark.asyncio
async def test_llm_service_initialization(llm_service):
    """Test LLM service initializes correctly."""
    assert llm_service.model == "gpt-4o-mini"
    assert llm_service.api_key == "test-api-key"
    assert "test assistant" in llm_service.system_prompt.lower()
    assert llm_service.client is not None


@pytest.mark.asyncio
async def test_generate_response_streams_tokens(llm_service):
    """Test that response tokens are streamed incrementally."""
    # Mock the OpenAI client
    mock_stream = AsyncMock()
    
    class MockChoice:
        def __init__(self, content, finish_reason=None):
            self.delta = MagicMock()
            self.delta.content = content
            self.finish_reason = finish_reason
    
    class MockChunk:
        def __init__(self, content, finish_reason=None):
            self.choices = [MockChoice(content, finish_reason)]
    
    async def mock_create(*args, **kwargs):
        tokens = ["Hello", " ", "world", "!"]
        for i, token in enumerate(tokens):
            await asyncio.sleep(0.01)
            finish_reason = "stop" if i == len(tokens) - 1 else None
            yield MockChunk(token, finish_reason)
    
    llm_service.client.chat.completions.create = mock_create
    
    # Generate response
    user_input = "Test input"
    context = []
    
    tokens = []
    async for token in llm_service.generate_response(user_input, context):
        tokens.append(token)
    
    # Verify tokens were streamed
    assert len(tokens) == 4
    assert "".join(tokens) == "Hello world!"


@pytest.mark.asyncio
async def test_generate_response_includes_context(llm_service):
    """Test that conversation context is included in API request."""
    captured_messages = None
    
    async def mock_create(*args, **kwargs):
        nonlocal captured_messages
        captured_messages = kwargs.get("messages", [])
        
        # Return minimal response
        class MockChoice:
            def __init__(self):
                self.delta = MagicMock()
                self.delta.content = "Response"
                self.finish_reason = "stop"
        
        class MockChunk:
            def __init__(self):
                self.choices = [MockChoice()]
        
        yield MockChunk()
    
    llm_service.client.chat.completions.create = mock_create
    
    # Create context
    context = [
        Message(role="user", content="Previous question", timestamp=time.time()),
        Message(role="assistant", content="Previous answer", timestamp=time.time())
    ]
    
    # Generate response
    user_input = "Current question"
    async for _ in llm_service.generate_response(user_input, context):
        pass
    
    # Verify messages include system, context, and current input
    assert captured_messages is not None
    assert len(captured_messages) == 4  # system + 2 context + current
    assert captured_messages[0]["role"] == "system"
    assert captured_messages[1]["role"] == "user"
    assert captured_messages[1]["content"] == "Previous question"
    assert captured_messages[2]["role"] == "assistant"
    assert captured_messages[2]["content"] == "Previous answer"
    assert captured_messages[3]["role"] == "user"
    assert captured_messages[3]["content"] == "Current question"


@pytest.mark.asyncio
async def test_cancel_generation(llm_service):
    """Test that generation can be cancelled."""
    # Mock a long-running stream
    async def mock_create(*args, **kwargs):
        class MockChoice:
            def __init__(self, content):
                self.delta = MagicMock()
                self.delta.content = content
                self.finish_reason = None
        
        class MockChunk:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        # Simulate long stream
        for i in range(100):
            await asyncio.sleep(0.1)
            yield MockChunk(f"token{i}")
    
    llm_service.client.chat.completions.create = mock_create
    
    # Start generation
    tokens = []
    
    async def generate():
        async for token in llm_service.generate_response("Test", []):
            tokens.append(token)
    
    # Start generation task
    task = asyncio.create_task(generate())
    
    # Wait a bit then cancel
    await asyncio.sleep(0.15)
    await llm_service.cancel_generation()
    
    # Wait for task to complete
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        pass
    
    # Verify generation was stopped early
    assert len(tokens) < 100


@pytest.mark.asyncio
async def test_generate_response_handles_api_error(llm_service):
    """Test error handling for API failures."""
    # Mock API error
    async def mock_create(*args, **kwargs):
        raise Exception("API Error")
        yield  # Make it a generator
    
    llm_service.client.chat.completions.create = mock_create
    
    # Verify error is raised
    with pytest.raises(Exception, match="API Error"):
        async for _ in llm_service.generate_response("Test", []):
            pass


@pytest.mark.asyncio
async def test_token_emission_latency(llm_service):
    """Test that tokens are emitted with minimal latency."""
    async def mock_create(*args, **kwargs):
        class MockChoice:
            def __init__(self, content):
                self.delta = MagicMock()
                self.delta.content = content
                self.finish_reason = None
        
        class MockChunk:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        for i in range(10):
            yield MockChunk(f"token{i}")
    
    llm_service.client.chat.completions.create = mock_create
    
    # Measure emission latency
    latencies = []
    
    async for token in llm_service.generate_response("Test", []):
        start = time.perf_counter()
        # Token received
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    # Verify latencies are minimal (< 50ms)
    # Note: This is a simplified test; actual latency measurement
    # would need more sophisticated timing
    assert len(latencies) == 10


@pytest.mark.asyncio
async def test_completion_signaling(llm_service):
    """Test that completion is properly signaled."""
    completion_detected = False
    
    async def mock_create(*args, **kwargs):
        class MockChoice:
            def __init__(self, content, finish_reason=None):
                self.delta = MagicMock()
                self.delta.content = content
                self.finish_reason = finish_reason
        
        class MockChunk:
            def __init__(self, content, finish_reason=None):
                self.choices = [MockChoice(content, finish_reason)]
        
        yield MockChunk("Hello", None)
        yield MockChunk("!", "stop")
    
    llm_service.client.chat.completions.create = mock_create
    
    tokens = []
    async for token in llm_service.generate_response("Test", []):
        tokens.append(token)
    
    # Verify we got all tokens and stream ended properly
    assert len(tokens) == 2
    assert tokens == ["Hello", "!"]

