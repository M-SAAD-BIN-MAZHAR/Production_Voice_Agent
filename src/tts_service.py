"""Text-to-Speech service using OpenAI, ElevenLabs, Edge-TTS, or Pollinations."""

import asyncio
import logging
from typing import AsyncIterator, Optional
import time
import re
import numpy as np
import aiohttp

from openai import AsyncOpenAI

from src.models import AudioChunk

# Optional imports
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False


logger = logging.getLogger(__name__)


class TTSService:
    """TTS service with sentence buffering for streaming synthesis."""
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        voice: str = "alloy",
        model: str = "tts-1",
        fallback_provider: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        fallback_voice: Optional[str] = None,
        fallback_model: Optional[str] = None
    ):
        """
        Initialize TTS service.
        
        Args:
            provider: TTS provider ("openai", "elevenlabs", "edge-tts", or "pollinations")
            api_key: API key (required for openai and elevenlabs, not needed for pollinations)
            voice: Voice name 
                - openai: alloy/echo/fable/onyx/nova/shimmer
                - elevenlabs: bella/rachel/adam/arnold/charlotte/clyde/etc
                - edge-tts: en-US-AriaNeural
                - pollinations: alloy/echo/fable/onyx/nova/shimmer (same as OpenAI)
            model: Model name (openai: tts-1 or tts-1-hd)
        """
        self.provider = provider
        self.voice = voice
        self.model = model
        self._fallback_service: Optional["TTSService"] = None
        
        # ElevenLabs voice ID mapping
        self.elevenlabs_voice_ids = {
            "bella": "EXAVITQu4vr4xnSDxMaL",
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "adam": "pNInz6obpgDQGcFmaJgB",
            "arnold": "VR6AewLHbXG4fxvB1xwl",
            "charlotte": "XB0fDUnXU5powFXDhCwa",
            "clyde": "2EiwWnXFnvU5JabPnv94",
            "george": "JBFqnCBsd6RMkjW3MqDe",
            "jessica": "aZe922EeXeKc9MZ0xBZO",
            "michael": "IB3nSCWiQThq3daIHI30",
            "sam": "yoZ06aMxZJJ28mfd3POQ",
        }
        
        # Initialize provider client
        if provider == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI TTS")
            self.client = AsyncOpenAI(api_key=api_key)
        elif provider == "elevenlabs":
            if not ELEVENLABS_AVAILABLE:
                raise ValueError("elevenlabs package not installed. Install with: pip install elevenlabs")
            if not api_key:
                raise ValueError("API key required for ElevenLabs TTS")
            self.client = ElevenLabs(api_key=api_key)
            # Map voice name to voice ID
            if voice in self.elevenlabs_voice_ids:
                self.voice = self.elevenlabs_voice_ids[voice]
            else:
                logger.warning(f"Unknown ElevenLabs voice '{voice}', using as voice ID directly")
        elif provider == "edge-tts":
            if not EDGE_TTS_AVAILABLE:
                raise ValueError("edge-tts package not installed. Install with: pip install edge-tts")
            self.client = None  # Edge-TTS doesn't need a client
            if voice == "alloy":  # Default OpenAI voice, switch to Edge-TTS default
                self.voice = "en-US-AriaNeural"
        elif provider == "pollinations":
            self.client = None  # Pollinations doesn't need a client
            self.pollinations_api_url = "https://text.pollinations.ai/openai"
        else:
            raise ValueError(f"Unsupported TTS provider: {provider}")
        
        # Optional fallback provider initialization.
        if fallback_provider and fallback_provider != provider:
            try:
                self._fallback_service = TTSService(
                    provider=fallback_provider,
                    api_key=fallback_api_key,
                    voice=fallback_voice or ("alloy" if fallback_provider == "openai" else voice),
                    model=fallback_model or ("tts-1" if fallback_provider == "openai" else model)
                )
                logger.info(f"TTS fallback enabled: {provider} -> {fallback_provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize TTS fallback provider '{fallback_provider}': {e}")
                self._fallback_service = None
        
        # Cancellation state
        self._cancelled = False
        self._pending_tasks: list[asyncio.Task] = []
        
        logger.info(f"TTSService initialized: provider={provider}, voice={voice}")
    
    def _detect_sentence_boundary(self, text: str) -> tuple[Optional[str], str]:
        """
        Detect sentence boundary in text buffer.
        
        Args:
            text: Text buffer to analyze
            
        Returns:
            Tuple of (complete_sentence, remaining_text)
            If no boundary found, returns (None, text)
        """
        # Sentence boundary patterns: . ? ! ; followed by space or end
        pattern = r'([.?!;])\s+'
        
        match = re.search(pattern, text)
        if match:
            # Found boundary
            end_pos = match.end()
            sentence = text[:end_pos].strip()
            remaining = text[end_pos:]
            return sentence, remaining
        
        # Check for boundary at end of text
        if text and text[-1] in '.?!;':
            return text.strip(), ""
        
        # No boundary found
        return None, text
    
    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Convert text to audio stream.
        
        Args:
            text: Text to synthesize
            
        Yields:
            AudioChunk: Audio chunks as they are generated
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return
        
        logger.info(f"Synthesizing text: {text[:50]}...")
        start_time = time.time()
        
        # Reset cancellation flag for this synthesis
        self._cancelled = False
        
        try:
            if self.provider == "openai":
                async for chunk in self._synthesize_openai(text):
                    if self._cancelled:
                        logger.info("TTS synthesis cancelled")
                        break
                    yield chunk
            elif self.provider == "elevenlabs":
                async for chunk in self._synthesize_elevenlabs(text):
                    if self._cancelled:
                        logger.info("TTS synthesis cancelled")
                        break
                    yield chunk
            elif self.provider == "edge-tts":
                async for chunk in self._synthesize_edge_tts(text):
                    if self._cancelled:
                        logger.info("TTS synthesis cancelled")
                        break
                    yield chunk
            elif self.provider == "pollinations":
                async for chunk in self._synthesize_pollinations(text):
                    if self._cancelled:
                        logger.info("TTS synthesis cancelled")
                        break
                    yield chunk
            
            duration = time.time() - start_time
            logger.info(f"TTS synthesis complete: duration={duration:.2f}s")
            
        except asyncio.CancelledError:
            logger.info("TTS synthesis task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error synthesizing speech with {self.provider}: {e}", exc_info=True)
            if self._fallback_service:
                logger.warning(f"Falling back TTS provider to {self._fallback_service.provider}")
                async for chunk in self._fallback_service.synthesize(text):
                    if self._cancelled:
                        break
                    yield chunk
                return
            raise
    
    async def _synthesize_openai(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize using OpenAI TTS API.
        
        Args:
            text: Text to synthesize
            
        Yields:
            AudioChunk: Audio chunks
        """
        try:
            start_time = time.time()
            
            # Create speech synthesis request
            # 🔊 SOUND QUALITY PARAMETERS:
            # - model: "tts-1" (faster) vs "tts-1-hd" (higher quality, slower)
            # - voice: "alloy/echo/fable/onyx/nova/shimmer" - affects tone and gender
            # - speed: 0.25-4.0 (default 1.0) - lower = slower/clearer, higher = faster
            #   * 0.85 = 15% slower for better clarity
            #   * 1.0 = normal speed (CURRENT - balanced)
            #   * 1.2 = 20% faster
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="pcm",  # Raw PCM for streaming
                speed=1.0  # 🔊 SOUND QUALITY: Normal speed (0.25 to 4.0)
            )
            
            # Get the audio content
            audio_content = response.content
            
            # 🔊 SOUND QUALITY PARAMETERS:
            # - chunk_size: Larger = smoother playback but higher latency
            #   * 4096 (4KB) = low latency, may be choppy
            #   * 8192 (8KB) = balanced
            #   * 16384 (16KB) = smooth playback, slightly higher latency
            # - sample_rate: 24000 Hz (OpenAI default, don't change)
            # Stream audio chunks - larger chunks for smoother playback
            chunk_size = 16384  # 🔊 SOUND QUALITY: 16KB chunks for smoother playback
            sample_rate = 24000  # 🔊 SOUND QUALITY: OpenAI TTS outputs 24kHz (fixed)
            
            first_chunk = True
            first_chunk_time = None
            
            # Split audio into chunks and yield immediately
            for i in range(0, len(audio_content), chunk_size):
                if first_chunk:
                    first_chunk_time = time.time()
                    first_chunk = False
                
                audio_bytes = audio_content[i:i + chunk_size]
                
                # Convert bytes to int16 numpy array
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # 🔊 SOUND QUALITY: Volume amplification
                # Multiply factor controls loudness:
                # - 1.0 = original volume
                # - 1.5 = 50% louder (current setting)
                # - 2.0 = 100% louder (may cause distortion)
                # - 0.8 = 20% quieter
                # np.clip prevents distortion by limiting values to int16 range (-32768 to 32767)
                audio_data = np.clip(audio_data * 1.5, -32768, 32767).astype(np.int16)
                
                # Calculate duration
                duration_ms = int(len(audio_data) / sample_rate * 1000)
                
                chunk = AudioChunk(
                    data=audio_data,
                    sample_rate=sample_rate,
                    timestamp=time.time(),
                    duration_ms=duration_ms
                )
                
                yield chunk
            
            # Log first chunk latency
            if first_chunk_time:
                latency_ms = (first_chunk_time - start_time) * 1000
                if latency_ms > 500:
                    logger.warning(
                        f"First audio chunk latency: {latency_ms:.1f}ms (> 500ms)"
                    )
                    
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}", exc_info=True)
            raise
    
    async def _synthesize_elevenlabs(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize using ElevenLabs TTS with high quality settings.
        
        Args:
            text: Text to synthesize
            
        Yields:
            AudioChunk: Audio chunks
        """
        try:
            start_time = time.time()
            
            # Generate speech using ElevenLabs with high quality settings
            # Use the correct API: text_to_speech.convert()
            audio_stream = self.client.text_to_speech.convert(
                voice_id=self.voice,
                text=text,
                model_id="eleven_multilingual_v2",  # Better quality model
                output_format="pcm_24000",  # High quality PCM format
                voice_settings={
                    "stability": 0.75,  # Higher stability for cleaner audio
                    "similarity_boost": 0.85  # Higher similarity for better voice quality
                }
            )
            
            # Stream audio chunks directly to reduce first-byte latency.
            chunk_size = 4096  # 4KB chunks
            sample_rate = 24000  # ElevenLabs outputs 24kHz
            
            first_chunk = True
            first_chunk_time = None
            byte_buffer = b""
            
            for raw_chunk in audio_stream:
                if self._cancelled:
                    break
                if not raw_chunk:
                    continue
                
                byte_buffer += raw_chunk
                
                while len(byte_buffer) >= chunk_size:
                    if first_chunk:
                        first_chunk_time = time.time()
                        first_chunk = False
                    
                    audio_bytes = byte_buffer[:chunk_size]
                    byte_buffer = byte_buffer[chunk_size:]
                    
                    if len(audio_bytes) % 2 != 0:
                        audio_bytes = audio_bytes[:-1]
                    if not audio_bytes:
                        continue
                    
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    duration_ms = int(len(audio_array) / sample_rate * 1000)
                    yield AudioChunk(
                        data=audio_array,
                        sample_rate=sample_rate,
                        timestamp=time.time(),
                        duration_ms=duration_ms
                    )
            
            # Flush residual bytes at stream end.
            if byte_buffer and not self._cancelled:
                if len(byte_buffer) % 2 != 0:
                    byte_buffer = byte_buffer[:-1]
                if byte_buffer:
                    if first_chunk:
                        first_chunk_time = time.time()
                    audio_array = np.frombuffer(byte_buffer, dtype=np.int16)
                    duration_ms = int(len(audio_array) / sample_rate * 1000)
                    yield AudioChunk(
                        data=audio_array,
                        sample_rate=sample_rate,
                        timestamp=time.time(),
                        duration_ms=duration_ms
                    )
            
            # Log first chunk latency
            if first_chunk_time:
                latency_ms = (first_chunk_time - start_time) * 1000
                if latency_ms > 500:
                    logger.warning(
                        f"First audio chunk latency: {latency_ms:.1f}ms (> 500ms)"
                    )
                    
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}", exc_info=True)
            raise
    
    async def _synthesize_edge_tts(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize using Edge-TTS.
        
        Args:
            text: Text to synthesize
            
        Yields:
            AudioChunk: Audio chunks
        """
        try:
            start_time = time.time()
            
            # Create Edge-TTS communicator
            communicate = edge_tts.Communicate(text, self.voice)
            
            sample_rate = 24000  # Edge-TTS outputs 24kHz
            first_chunk = True
            first_chunk_time = None
            
            # Stream audio chunks
            async for chunk_data in communicate.stream():
                if chunk_data["type"] == "audio":
                    if first_chunk:
                        first_chunk_time = time.time()
                        first_chunk = False
                    
                    # Convert bytes to int16 numpy array
                    audio_bytes = chunk_data["data"]
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Calculate duration
                    duration_ms = int(len(audio_data) / sample_rate * 1000)
                    
                    chunk = AudioChunk(
                        data=audio_data,
                        sample_rate=sample_rate,
                        timestamp=time.time(),
                        duration_ms=duration_ms
                    )
                    
                    yield chunk
            
            # Log first chunk latency
            if first_chunk_time:
                latency_ms = (first_chunk_time - start_time) * 1000
                if latency_ms > 500:
                    logger.warning(
                        f"First audio chunk latency: {latency_ms:.1f}ms (> 500ms)"
                    )
                    
        except Exception as e:
            logger.error(f"Edge-TTS error: {e}", exc_info=True)
            raise
    
    async def _synthesize_pollinations(self, text: str) -> AsyncIterator[AudioChunk]:
        """
        Synthesize using Pollinations AI TTS (free, no API key required).
        
        Args:
            text: Text to synthesize
            
        Yields:
            AudioChunk: Audio chunks
        """
        try:
            start_time = time.time()
            
            # Pollinations TTS API endpoint
            # Uses OpenAI-compatible voices
            url = f"https://text.pollinations.ai/openai?voice={self.voice}"
            
            # Make request to Pollinations TTS API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=text.encode('utf-8'),
                    headers={"Content-Type": "text/plain"},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Pollinations TTS API error: {response.status} - {error_text}")
                    
                    # Get the audio content
                    audio_content = await response.read()
                    
                    # Stream audio chunks
                    chunk_size = 16384  # 16KB chunks for smooth playback
                    sample_rate = 24000  # Pollinations outputs 24kHz
                    
                    first_chunk = True
                    first_chunk_time = None
                    
                    # Split audio into chunks and yield immediately
                    for i in range(0, len(audio_content), chunk_size):
                        if first_chunk:
                            first_chunk_time = time.time()
                            first_chunk = False
                        
                        audio_bytes = audio_content[i:i + chunk_size]
                        
                        # Convert bytes to int16 numpy array
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                        
                        # Volume amplification (same as OpenAI)
                        audio_data = np.clip(audio_data * 1.5, -32768, 32767).astype(np.int16)
                        
                        # Calculate duration
                        duration_ms = int(len(audio_data) / sample_rate * 1000)
                        
                        chunk = AudioChunk(
                            data=audio_data,
                            sample_rate=sample_rate,
                            timestamp=time.time(),
                            duration_ms=duration_ms
                        )
                        
                        yield chunk
                    
                    # Log first chunk latency
                    if first_chunk_time:
                        latency_ms = (first_chunk_time - start_time) * 1000
                        if latency_ms > 500:
                            logger.warning(
                                f"First audio chunk latency: {latency_ms:.1f}ms (> 500ms)"
                            )
                        
        except Exception as e:
            logger.error(f"Pollinations TTS error: {e}", exc_info=True)
            raise
    
    async def synthesize_stream(
        self, 
        token_stream: AsyncIterator[str]
    ) -> AsyncIterator[AudioChunk]:
        """
        Buffer tokens until sentence boundaries and synthesize in parallel.
        
        This enables parallel processing: while synthesizing sentence N,
        we're already collecting tokens for sentence N+1.
        
        Args:
            token_stream: Stream of response tokens from LLM
            
        Yields:
            AudioChunk: Audio chunks as they are generated
        """
        # Reset cancellation flag for this synthesis
        self._cancelled = False
        
        buffer = ""
        synthesis_tasks: dict[int, asyncio.Task] = {}
        next_task_idx = 0
        next_emit_idx = 0
        self._pending_tasks.clear()
        
        async def _emit_ready_tasks() -> AsyncIterator[AudioChunk]:
            nonlocal next_emit_idx
            while next_emit_idx in synthesis_tasks and synthesis_tasks[next_emit_idx].done():
                task = synthesis_tasks.pop(next_emit_idx)
                self._pending_tasks = [t for t in self._pending_tasks if t is not task]
                try:
                    audio_chunks = task.result()
                    for chunk in audio_chunks:
                        if self._cancelled:
                            return
                        yield chunk
                except Exception as e:
                    logger.error(f"Error in sentence synthesis: {e}")
                next_emit_idx += 1
        
        try:
            async for token in token_stream:
                if self._cancelled:
                    logger.info("TTS stream synthesis cancelled")
                    break
                
                # Add token to buffer
                buffer += token
                
                # Check for sentence boundary
                sentence, remaining = self._detect_sentence_boundary(buffer)
                
                if sentence:
                    # Found complete sentence, start synthesizing it in parallel
                    logger.debug(f"Synthesizing sentence: {sentence[:50]}...")
                    
                    # Create task for this sentence
                    task = asyncio.create_task(self._synthesize_sentence_to_list(sentence))
                    synthesis_tasks[next_task_idx] = task
                    self._pending_tasks.append(task)
                    next_task_idx += 1
                    
                    # Keep remaining text in buffer
                    buffer = remaining
                    
                    # Emit finished tasks immediately for lower latency.
                    async for ready_chunk in _emit_ready_tasks():
                        yield ready_chunk
            
            # Synthesize any remaining text
            if buffer.strip() and not self._cancelled:
                logger.debug(f"Synthesizing remaining text: {buffer[:50]}...")
                task = asyncio.create_task(self._synthesize_sentence_to_list(buffer))
                synthesis_tasks[next_task_idx] = task
                self._pending_tasks.append(task)
            
            # Emit any tasks that already completed before final drain.
            async for ready_chunk in _emit_ready_tasks():
                yield ready_chunk
            
            # Yield audio chunks from completed synthesis tasks in order
            while next_emit_idx in synthesis_tasks:
                if self._cancelled:
                    break
                
                task = synthesis_tasks.pop(next_emit_idx)
                self._pending_tasks = [t for t in self._pending_tasks if t is not task]
                try:
                    audio_chunks = await task
                    for chunk in audio_chunks:
                        yield chunk
                except Exception as e:
                    logger.error(f"Error in sentence synthesis: {e}")
                next_emit_idx += 1
                    
        except asyncio.CancelledError:
            logger.info("TTS stream synthesis task cancelled")
            # Cancel all pending synthesis tasks
            for task in synthesis_tasks.values():
                if not task.done():
                    task.cancel()
            raise
        except Exception as e:
            logger.error(f"Error in stream synthesis: {e}", exc_info=True)
            raise
        finally:
            self._pending_tasks.clear()
    
    async def _synthesize_sentence_to_list(self, text: str) -> list:
        """
        Synthesize a sentence and collect all audio chunks into a list.
        
        Args:
            text: Text to synthesize
            
        Returns:
            List of AudioChunk objects
        """
        chunks = []
        async for chunk in self.synthesize(text):
            chunks.append(chunk)
        return chunks
    
    async def cancel_synthesis(self) -> None:
        """Cancel pending synthesis requests."""
        logger.info("Cancelling TTS synthesis...")
        self._cancelled = True
        
        # Cancel all pending tasks
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._pending_tasks.clear()
        logger.info("TTS synthesis cancelled")
    
    def is_synthesizing(self) -> bool:
        """Check if currently synthesizing."""
        return any(not task.done() for task in self._pending_tasks)

