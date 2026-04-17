"""Large Language Model service using OpenAI."""

import asyncio
import logging
from typing import AsyncIterator, List, Optional
import time

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from src.models import Message


logger = logging.getLogger(__name__)


class LLMService:
    """LLM service using OpenAI Chat Completions API with streaming."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLM service.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini, also supports: gpt-4o)
            system_prompt: System prompt defining assistant personality
        """
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a helpful voice assistant. Provide concise, natural responses "
            "suitable for speech output. Keep responses brief and conversational."
        )
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Cancellation state
        self._current_task: Optional[asyncio.Task] = None
        self._cancelled = False
        
        logger.info(f"LLMService initialized: model={model}")
    
    async def generate_response(
        self, 
        user_input: str, 
        conversation_context: List[Message]
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from LLM.
        
        Args:
            user_input: Current user input text
            conversation_context: Previous conversation messages
            
        Yields:
            str: Response tokens as they are generated
        """
        # Reset cancellation flag
        self._cancelled = False
        
        # Format messages for OpenAI API
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation context
        for msg in conversation_context:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        logger.info(f"Generating LLM response for input: {user_input[:50]}...")
        logger.debug(f"Context messages: {len(conversation_context)}")
        
        try:
            # Create streaming completion
            # Note: OpenAI SDK returns a coroutine that resolves to an async generator
            # For testing, mocks can return async generators directly
            stream_coro = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=500
            )
            
            # Check if it's a coroutine (real SDK) or async generator (mock)
            import inspect
            if inspect.iscoroutine(stream_coro):
                stream = await stream_coro
            else:
                stream = stream_coro
            
            token_count = 0
            start_time = time.time()
            
            # Stream tokens
            async for chunk in stream:
                # Check for cancellation
                if self._cancelled:
                    logger.info("LLM generation cancelled")
                    break
                
                # Extract token from chunk
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        token = delta.content
                        token_count += 1
                        
                        # Emit token with minimal delay
                        receive_time = time.time()
                        yield token
                        emit_time = time.time()
                        
                        # Log if emission latency exceeds threshold
                        latency_ms = (emit_time - receive_time) * 1000
                        if latency_ms > 50:
                            logger.warning(
                                f"Token emission latency: {latency_ms:.1f}ms (> 50ms)"
                            )
                    
                    # Check for completion
                    finish_reason = chunk.choices[0].finish_reason
                    if finish_reason:
                        logger.info(
                            f"LLM generation complete: {token_count} tokens, "
                            f"finish_reason={finish_reason}, "
                            f"duration={time.time() - start_time:.2f}s"
                        )
                        break
            
        except asyncio.CancelledError:
            logger.info("LLM generation task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            raise
    
    async def cancel_generation(self) -> None:
        """Cancel ongoing response generation."""
        logger.info("Cancelling LLM generation...")
        self._cancelled = True
        
        # If there's a current task, cancel it
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        
        logger.info("LLM generation cancelled")
    
    def is_generating(self) -> bool:
        """Check if currently generating a response."""
        return self._current_task is not None and not self._current_task.done()

