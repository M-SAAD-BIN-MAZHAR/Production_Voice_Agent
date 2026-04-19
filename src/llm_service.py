"""Large Language Model service supporting OpenAI, Hugging Face, Pollinations, and Mistral.

Handles conversation generation using multiple LLM providers.
Supports streaming responses for real-time token-by-token output.

Key Features:
- Multi-provider support (OpenAI, Hugging Face, Pollinations, Mistral)
- Streaming token generation for low latency
- Conversation context management
- Cancellation support for interrupts
- Automatic fallback between providers
"""

import asyncio
import logging
from typing import AsyncIterator, Callable, List, Optional
import time
import json

from openai import AsyncOpenAI
import aiohttp

from src.models import Message


logger = logging.getLogger(__name__)


class LLMService:
    """LLM service supporting OpenAI, Hugging Face, Pollinations, and Mistral with streaming."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        provider: str = "openai",
        fallback_provider: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        fallback_model: Optional[str] = None
    ):
        """
        Initialize LLM service.
        
        Args:
            api_key: API key (required for OpenAI, Hugging Face, Mistral; not needed for Pollinations)
            model: Model name 
                   OpenAI: "gpt-4o-mini", "gpt-4o"
                   Hugging Face: "meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", etc.
                   Pollinations: "openai", "mistral", "llama", etc.
                   Mistral: "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large-latest"
            system_prompt: System prompt defining assistant personality
            provider: "openai", "huggingface", "pollinations", or "mistral"
            fallback_provider: Optional fallback provider
            fallback_api_key: API key for fallback provider
            fallback_model: Model for fallback provider
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.system_prompt = system_prompt or (
            "You are a helpful voice assistant. Provide concise, natural responses "
            "suitable for speech output. Keep responses brief and conversational."
        )
        
        # Initialize provider-specific client
        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=api_key)
        elif self.provider == "huggingface":
            self.hf_api_url = f"https://api-inference.huggingface.co/models/{model}"
            self.client = None
        elif self.provider == "pollinations":
            self.pollinations_api_url = "https://text.pollinations.ai/"
            self.client = None
        elif self.provider == "mistral":
            # Mistral uses OpenAI-compatible API
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'huggingface', 'pollinations', or 'mistral'")
        
        # Initialize fallback service
        self._fallback_service: Optional["LLMService"] = None
        if fallback_provider and fallback_provider != provider:
            try:
                self._fallback_service = LLMService(
                    api_key=fallback_api_key or "",
                    model=fallback_model or model,
                    system_prompt=system_prompt,
                    provider=fallback_provider
                )
                logger.info(f"LLM fallback enabled: {provider} -> {fallback_provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM fallback provider '{fallback_provider}': {e}")
                self._fallback_service = None
        
        # Cancellation state
        self._current_task: Optional[asyncio.Task] = None
        self._cancelled = False
        
        logger.info(f"LLMService initialized: provider={self.provider}, model={model}")
    
    async def generate_response(
        self, 
        user_input: str, 
        conversation_context: List[Message],
        on_first_token: Optional[Callable[[], None]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from LLM.
        
        Args:
            user_input: Current user input text
            conversation_context: Previous conversation messages
            on_first_token: Optional callback to fire when first token arrives
            
        Yields:
            str: Response tokens as they are generated
        """
        try:
            if self.provider == "openai":
                async for token in self._generate_openai(user_input, conversation_context, on_first_token):
                    yield token
            elif self.provider == "huggingface":
                async for token in self._generate_huggingface(user_input, conversation_context, on_first_token):
                    yield token
            elif self.provider == "pollinations":
                async for token in self._generate_pollinations(user_input, conversation_context, on_first_token):
                    yield token
            elif self.provider == "mistral":
                async for token in self._generate_mistral(user_input, conversation_context, on_first_token):
                    yield token
        except Exception as e:
            logger.error(f"Error generating response with {self.provider}: {e}", exc_info=True)
            if self._fallback_service:
                logger.warning(f"Falling back LLM provider to {self._fallback_service.provider}")
                async for token in self._fallback_service.generate_response(user_input, conversation_context, on_first_token):
                    yield token
                return
            raise
    
    async def _generate_openai(
        self,
        user_input: str,
        conversation_context: List[Message],
        on_first_token: Optional[Callable[[], None]] = None,
    ) -> AsyncIterator[str]:
        """Generate response using OpenAI API."""
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
        
        self._current_task = asyncio.current_task()
        
        try:
            # Create streaming completion
            # 🤖 LLM PARAMETERS:
            # - model: "gpt-4o-mini" (fast, cheap) or "gpt-4o" (smarter, slower, expensive)
            # - temperature: 0.0-2.0 (0.7 = balanced creativity)
            #   * Lower (0.3-0.5) = more focused, deterministic
            #   * Higher (0.8-1.0) = more creative, varied
            # - max_tokens: Maximum response length (500 = ~375 words)
            #   * Shorter = faster responses, lower cost
            #   * Longer = more detailed responses
            # Note: OpenAI SDK returns a coroutine that resolves to an async generator
            # For testing, mocks can return async generators directly
            stream_coro = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,  # 🤖 LLM: Creativity level (0.0-2.0)
                max_tokens=500    # 🤖 LLM: Max response length
            )
            
            # Check if it's a coroutine (real SDK) or async generator (mock)
            import inspect
            if inspect.iscoroutine(stream_coro):
                stream = await stream_coro
            else:
                stream = stream_coro
            
            token_count = 0
            start_time = time.time()
            first_token_hook_fired = False
            
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
                        
                        if not first_token_hook_fired and on_first_token is not None:
                            first_token_hook_fired = True
                            try:
                                on_first_token()
                            except Exception as hook_err:
                                logger.debug("on_first_token hook error: %s", hook_err)
                        
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
        finally:
            if self._current_task is asyncio.current_task():
                self._current_task = None
    
    async def _generate_huggingface(
        self,
        user_input: str,
        conversation_context: List[Message],
        on_first_token: Optional[Callable[[], None]] = None,
    ) -> AsyncIterator[str]:
        """Generate response using Hugging Face Inference API."""
        # Reset cancellation flag
        self._cancelled = False
        
        # Format prompt for Hugging Face (most models use chat template format)
        prompt = f"{self.system_prompt}\n\n"
        
        # Add conversation context
        for msg in conversation_context:
            if msg.role == "user":
                prompt += f"User: {msg.content}\n"
            else:
                prompt += f"Assistant: {msg.content}\n"
        
        # Add current user input
        prompt += f"User: {user_input}\nAssistant:"
        
        logger.info(f"Generating HF response for input: {user_input[:50]}...")
        logger.debug(f"Context messages: {len(conversation_context)}")
        
        self._current_task = asyncio.current_task()
        
        try:
            # Prepare request payload
            # 🤖 LLM PARAMETERS for Hugging Face:
            # - temperature: 0.0-2.0 (0.7 = balanced creativity)
            # - max_new_tokens: Maximum response length (500 tokens)
            # - top_p: Nucleus sampling (0.95 = diverse but coherent)
            # - repetition_penalty: Prevent repetition (1.1 = slight penalty)
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,  # 🤖 LLM: Max response length
                    "temperature": 0.7,      # 🤖 LLM: Creativity level
                    "top_p": 0.95,
                    "repetition_penalty": 1.1,
                    "return_full_text": False,
                    "do_sample": True
                },
                "options": {
                    "use_cache": False,
                    "wait_for_model": True
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            token_count = 0
            start_time = time.time()
            first_token_hook_fired = False
            
            # Make streaming request to Hugging Face
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.hf_api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Hugging Face API error: {response.status} - {error_text}")
                    
                    # Read response
                    response_text = await response.text()
                    
                    try:
                        result = json.loads(response_text)
                        
                        # Handle different response formats
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                        elif isinstance(result, dict):
                            generated_text = result.get("generated_text", result.get("text", ""))
                        else:
                            generated_text = str(result)
                        
                        # Stream tokens word by word for real-time feel
                        words = generated_text.split()
                        for word in words:
                            if self._cancelled:
                                logger.info("LLM generation cancelled")
                                break
                            
                            token_count += 1
                            
                            if not first_token_hook_fired and on_first_token is not None:
                                first_token_hook_fired = True
                                try:
                                    on_first_token()
                                except Exception as hook_err:
                                    logger.debug("on_first_token hook error: %s", hook_err)
                            
                            yield word + " "
                            await asyncio.sleep(0.01)  # Small delay for streaming effect
                        
                        logger.info(
                            f"HF generation complete: {token_count} tokens, "
                            f"duration={time.time() - start_time:.2f}s"
                        )
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse HF response: {e}")
                        logger.error(f"Response text: {response_text[:200]}")
                        raise
            
        except asyncio.CancelledError:
            logger.info("LLM generation task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error generating HF response: {e}", exc_info=True)
            raise
        finally:
            if self._current_task is asyncio.current_task():
                self._current_task = None
    
    async def _generate_pollinations(
        self,
        user_input: str,
        conversation_context: List[Message],
        on_first_token: Optional[Callable[[], None]] = None,
    ) -> AsyncIterator[str]:
        """Generate response using Pollinations AI (free, no API key required)."""
        # Reset cancellation flag
        self._cancelled = False
        
        # Format prompt for Pollinations
        prompt = f"{self.system_prompt}\n\n"
        
        # Add conversation context
        for msg in conversation_context:
            if msg.role == "user":
                prompt += f"User: {msg.content}\n"
            else:
                prompt += f"Assistant: {msg.content}\n"
        
        # Add current user input
        prompt += f"User: {user_input}\nAssistant:"
        
        logger.info(f"Generating Pollinations response for input: {user_input[:50]}...")
        logger.debug(f"Context messages: {len(conversation_context)}")
        
        self._current_task = asyncio.current_task()
        
        try:
            # Pollinations API parameters
            # 🤖 LLM PARAMETERS for Pollinations:
            # - model: "openai" (GPT-like), "mistral", "llama", etc.
            # - seed: Random seed for reproducibility
            # - jsonMode: false for text generation
            params = {
                "model": self.model,
                "seed": int(time.time()),
                "jsonMode": "false"
            }
            
            token_count = 0
            start_time = time.time()
            first_token_hook_fired = False
            
            # Make request to Pollinations API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.pollinations_api_url,
                    params=params,
                    data=prompt.encode('utf-8'),
                    headers={"Content-Type": "text/plain"},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Pollinations API error: {response.status} - {error_text}")
                    
                    # Read response text
                    response_text = await response.text()
                    
                    # Stream tokens word by word for real-time feel
                    words = response_text.split()
                    for word in words:
                        if self._cancelled:
                            logger.info("LLM generation cancelled")
                            break
                        
                        token_count += 1
                        
                        if not first_token_hook_fired and on_first_token is not None:
                            first_token_hook_fired = True
                            try:
                                on_first_token()
                            except Exception as hook_err:
                                logger.debug("on_first_token hook error: %s", hook_err)
                        
                        yield word + " "
                        await asyncio.sleep(0.01)  # Small delay for streaming effect
                    
                    logger.info(
                        f"Pollinations generation complete: {token_count} tokens, "
                        f"duration={time.time() - start_time:.2f}s"
                    )
            
        except asyncio.CancelledError:
            logger.info("LLM generation task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error generating Pollinations response: {e}", exc_info=True)
            raise
        finally:
            if self._current_task is asyncio.current_task():
                self._current_task = None
    
    async def _generate_mistral(
        self,
        user_input: str,
        conversation_context: List[Message],
        on_first_token: Optional[Callable[[], None]] = None,
    ) -> AsyncIterator[str]:
        """Generate response using Mistral AI (OpenAI-compatible API)."""
        # Reset cancellation flag
        self._cancelled = False
        
        # Format messages for Mistral API (same as OpenAI)
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
        
        logger.info(f"Generating Mistral response for input: {user_input[:50]}...")
        logger.debug(f"Context messages: {len(conversation_context)}")
        
        self._current_task = asyncio.current_task()
        
        try:
            # Create streaming completion (Mistral uses OpenAI-compatible API)
            # 🤖 LLM PARAMETERS for Mistral:
            # - model: "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large-latest"
            # - temperature: 0.0-1.0 (0.7 = balanced creativity)
            # - max_tokens: Maximum response length (500 tokens)
            stream_coro = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,  # 🤖 LLM: Creativity level (0.0-1.0)
                max_tokens=500    # 🤖 LLM: Max response length
            )
            
            # Check if it's a coroutine (real SDK) or async generator (mock)
            import inspect
            if inspect.iscoroutine(stream_coro):
                stream = await stream_coro
            else:
                stream = stream_coro
            
            token_count = 0
            start_time = time.time()
            first_token_hook_fired = False
            
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
                        
                        if not first_token_hook_fired and on_first_token is not None:
                            first_token_hook_fired = True
                            try:
                                on_first_token()
                            except Exception as hook_err:
                                logger.debug("on_first_token hook error: %s", hook_err)
                        
                        yield token
                    
                    # Check for completion
                    finish_reason = chunk.choices[0].finish_reason
                    if finish_reason:
                        logger.info(
                            f"Mistral generation complete: {token_count} tokens, "
                            f"finish_reason={finish_reason}, "
                            f"duration={time.time() - start_time:.2f}s"
                        )
                        break
            
        except asyncio.CancelledError:
            logger.info("LLM generation task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error generating Mistral response: {e}", exc_info=True)
            raise
        finally:
            if self._current_task is asyncio.current_task():
                self._current_task = None
    
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

