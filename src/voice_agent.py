"""Voice agent orchestrator - main async loop coordinating all components.

This is the main orchestrator that coordinates all voice agent components:
- Audio Input: Captures microphone audio
- VAD: Detects when user is speaking
- STT: Converts speech to text (Deepgram)
- LLM: Generates intelligent responses (OpenAI)
- TTS: Converts text to speech (OpenAI)
- Audio Output: Plays synthesized speech
- Interrupt Manager: Handles barge-in when user interrupts
- Memory Manager: Maintains conversation context
- Response Cache: Caches responses for instant playback

The orchestrator uses async/await with queue-based communication between components.
"""

import asyncio
import logging
from typing import Optional, List
import time

from src.models import (
    AudioChunk, VADEvent, VADEventType, Transcript, Message, SystemState
)
from src.audio_input import AudioInputCapture
from src.vad import VADModule
from src.stt_service import STTService
from src.llm_service import LLMService
from src.tts_service import TTSService
from src.audio_output import AudioOutputPlayer
from src.interrupt_manager import InterruptManager
from src.memory_manager import MemoryManager
from src.terminal_ui import TerminalUI
from src.config_manager import Config
from src.response_cache import ResponseCache


logger = logging.getLogger(__name__)


class VoiceAgent:
    """Main orchestrator for the real-time voice agent system."""
    
    def __init__(self, config: Config):
        """
        Initialize voice agent with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        self.audio_input = AudioInputCapture(
            sample_rate=config.sample_rate,
            chunk_duration_ms=config.chunk_duration_ms
        )
        
        self.vad = VADModule(
            threshold=config.vad_threshold,
            silence_duration_ms=config.vad_silence_duration_ms
        )
        
        self.stt = STTService(
            api_key=config.deepgram_api_key or "",
            model=config.stt_model,
            language=config.stt_language
        )
        
        self.llm = LLMService(
            api_key=config.openai_api_key or "",
            model=config.llm_model,
            system_prompt=config.system_prompt
        )
        
        self.tts = TTSService(
            provider=config.tts_provider,
            api_key=config.elevenlabs_api_key if config.tts_provider == "elevenlabs" else config.openai_api_key,
            voice=config.tts_voice,
            model=config.tts_model
        )
        
        self.audio_output = AudioOutputPlayer(
            sample_rate=24000  # 🔊 SOUND QUALITY: TTS output sample rate (must match TTS service)
        )
        
        self.interrupt_manager = InterruptManager()
        
        self.memory = MemoryManager(
            openai_api_key=config.openai_api_key or "",
            max_turns=config.memory_max_turns,
            similarity_top_k=config.memory_similarity_top_k,
            recent_turns=config.memory_recent_turns,
            vector_db_path=config.vector_db_path
        )
        
        self.ui = TerminalUI()
        
        # Response cache for common queries
        # 💾 PERFORMANCE: Caches text and audio responses for 24 hours
        # Provides instant playback for repeated queries
        self.cache = ResponseCache(
            cache_dir="./data/response_cache",
            max_age_seconds=86400  # 24 hours
        )
        
        # Inter-component queues
        self._audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._vad_queue: asyncio.Queue[VADEvent] = asyncio.Queue()
        self._transcript_queue: asyncio.Queue[Transcript] = asyncio.Queue()
        
        # Component tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        # Current conversation state
        self._current_user_input = ""
        self._current_response_tokens: List[str] = []
        self._is_speaking = False
        
        logger.info("VoiceAgent initialized")
    
    async def start(self) -> None:
        """Start the voice agent and all components."""
        if self._running:
            logger.warning("VoiceAgent already running")
            return
        
        logger.info("Starting VoiceAgent...")
        self._running = True
        
        try:
            # Start UI first
            await self.ui.start()
            await self.ui.update_status(SystemState.INITIALIZING)
            
            # Start audio components
            await self.audio_input.start()
            await self.audio_output.start()
            
            # Connect STT service with timeout
            try:
                await asyncio.wait_for(self.stt.connect(), timeout=15.0)
            except asyncio.TimeoutError:
                logger.error("STT connection timed out")
                raise ConnectionError("STT service connection timed out after 15 seconds")
            
            # Register interrupt callbacks
            self.interrupt_manager.register_interrupt_callback(self._handle_interrupt)
            
            # Create component tasks
            try:
                self._tasks = [
                    asyncio.create_task(self._audio_input_task(), name="audio_input"),
                    asyncio.create_task(self._vad_processing_task(), name="vad_processing"),
                    asyncio.create_task(self._stt_streaming_task(), name="stt_streaming"),
                    asyncio.create_task(self._conversation_task(), name="conversation"),
                    asyncio.create_task(
                        self.interrupt_manager.monitor_interrupts(
                            self._vad_queue,
                            self.audio_output.is_playing
                        ),
                        name="interrupt_monitoring"
                    )
                ]
                logger.info(f"Created {len(self._tasks)} component tasks")
            except Exception as e:
                logger.error(f"Error creating tasks: {e}", exc_info=True)
                raise
            
            # Give tasks a moment to start
            await asyncio.sleep(0.5)
            
            # Update state to listening
            self.interrupt_manager.set_state(SystemState.LISTENING)
            await self.ui.update_status(SystemState.LISTENING)
            
            logger.info("VoiceAgent started successfully")
            
            # Wait for all tasks
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error starting VoiceAgent: {e}", exc_info=True)
            await self.ui.display_error(f"Failed to start: {e}")
            await self.stop()
    
    async def _audio_input_task(self) -> None:
        """Task: Capture audio from microphone."""
        logger.info("Audio input task started")
        
        try:
            while self._running:
                # Get audio chunk
                chunk = await self.audio_input.get_audio_chunk()
                
                # Put in queue for VAD processing
                await self._audio_queue.put(chunk)
                
                # Also send to STT if connected
                if self.stt.is_connected():
                    await self.stt.send_audio(chunk)
                
        except asyncio.CancelledError:
            logger.info("Audio input task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in audio input task: {e}", exc_info=True)
            await self._handle_component_failure("audio_input", e)
    
    async def _vad_processing_task(self) -> None:
        """Task: Process audio for voice activity detection."""
        logger.info("VAD processing task started")
        
        try:
            while self._running:
                # Get audio chunk
                chunk = await self._audio_queue.get()
                
                # Process with VAD
                vad_event = await self.vad.process_audio(chunk)
                
                # Put in queue for interrupt monitoring
                await self._vad_queue.put(vad_event)
                
                # Log speech events (only start and end, not continue)
                if vad_event.event_type == VADEventType.SPEECH_START:
                    logger.info("Speech started")
                elif vad_event.event_type == VADEventType.SPEECH_END:
                    logger.info("Speech ended")
                
        except asyncio.CancelledError:
            logger.info("VAD processing task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in VAD processing task: {e}", exc_info=True)
            await self._handle_component_failure("vad_processing", e)
    
    async def _stt_streaming_task(self) -> None:
        """Task: Stream transcripts from STT service."""
        logger.info("STT streaming task started - waiting for transcripts")
        
        try:
            count = 0
            while self._running:
                count += 1
                if count % 10 == 0:
                    logger.debug(f"STT streaming task alive (attempt {count})")
                
                # Get transcript from STT with timeout
                try:
                    transcript = await asyncio.wait_for(
                        self.stt.get_transcript(),
                        timeout=1.0
                    )
                    logger.debug(f"Got transcript: {transcript.text}")
                    
                    # Update UI with partial or final transcript
                    await self.ui.update_transcript(
                        role="user",
                        content=transcript.text,
                        is_partial=not transcript.is_final
                    )
                    
                    # If final transcript, add to queue for processing
                    if transcript.is_final:
                        await self._transcript_queue.put(transcript)
                        logger.info(f"Final transcript: {transcript.text}")
                        
                except asyncio.TimeoutError:
                    # No transcript received, continue waiting
                    pass
                
        except asyncio.CancelledError:
            logger.info("STT streaming task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in STT streaming task: {e}", exc_info=True)
            await self._handle_component_failure("stt_streaming", e)
    
    async def _conversation_task(self) -> None:
        """Task: Handle conversation flow with parallel LLM+TTS and caching."""
        logger.info("Conversation task started with parallel processing and caching")
        
        try:
            while self._running:
                # Wait for final transcript
                transcript = await self._transcript_queue.get()
                user_input = transcript.text.strip()
                
                if not user_input:
                    continue
                
                self._current_user_input = user_input
                
                # Update state to processing
                self.interrupt_manager.set_state(SystemState.PROCESSING)
                await self.ui.update_status(SystemState.PROCESSING)
                
                # Check cache first for instant responses
                cached_text = self.cache.get_text_response(user_input)
                cached_audio = None
                
                if cached_text:
                    logger.info("Using cached text response")
                    full_response = cached_text
                    cached_audio = self.cache.get_audio_response(cached_text)
                    
                    # Update UI with cached response
                    await self.ui.update_transcript(
                        role="assistant",
                        content=full_response,
                        is_partial=False
                    )
                    
                    if cached_audio:
                        logger.info("Using cached audio response - INSTANT PLAYBACK!")
                        # Update state to speaking
                        self.interrupt_manager.set_state(SystemState.SPEAKING)
                        await self.ui.update_status(SystemState.SPEAKING)
                        self._is_speaking = True
                        
                        # Play cached audio immediately
                        for audio_chunk in cached_audio:
                            await self.audio_output.play(audio_chunk)
                        
                        # Wait for playback to complete
                        while self.audio_output.is_playing():
                            await asyncio.sleep(0.1)
                        
                        self._is_speaking = False
                        
                        # Store in memory
                        await self.memory.add_turn(user_input, full_response)
                        
                        # Update state back to listening
                        self.interrupt_manager.set_state(SystemState.LISTENING)
                        await self.ui.update_status(SystemState.LISTENING)
                        continue  # Skip rate limiting for cached responses
                
                # Check if there are pending queries in the queue
                pending_queries = self._transcript_queue.qsize()
                if pending_queries > 0:
                    logger.info(f"Skipping rate limit delay - {pending_queries} queries pending in queue")
                
                # ⚠️ RATE LIMITING DISABLED
                # The 25-second delay has been removed per user request
                # WARNING: This may cause OpenAI rate limit errors (429 errors)
                # Your OpenAI account limit: 3 requests/min (free tier)
                # To avoid errors, upgrade your OpenAI account or re-enable the delay
                # await asyncio.sleep(25.0)  # Commented out - was preventing rate limits
                
                # Get conversation context from memory
                context = await self.memory.get_context(user_input)
                
                # Generate LLM response (or use cached)
                logger.info(f"Generating response for: {user_input}")
                response_tokens: List[str] = []
                
                try:
                    if cached_text:
                        # Use cached text but need to synthesize audio
                        full_response = cached_text
                    else:
                        # Stream tokens from LLM
                        async for token in self.llm.generate_response(user_input, context):
                            response_tokens.append(token)
                            
                            # Update UI with partial response
                            partial_response = "".join(response_tokens)
                            await self.ui.update_transcript(
                                role="assistant",
                                content=partial_response,
                                is_partial=True
                            )
                        
                        # Complete response received
                        full_response = "".join(response_tokens)
                        logger.info(f"Response generated: {full_response[:100]}...")
                        
                        # Cache the text response
                        self.cache.set_text_response(user_input, full_response)
                        
                        # Update UI with final response
                        await self.ui.update_transcript(
                            role="assistant",
                            content=full_response,
                            is_partial=False
                        )
                    
                    # Update state to speaking
                    self.interrupt_manager.set_state(SystemState.SPEAKING)
                    await self.ui.update_status(SystemState.SPEAKING)
                    self._is_speaking = True
                    
                    # Synthesize and play audio - USE SINGLE REQUEST FOR SPEED
                    audio_chunks_for_cache = []
                    
                    # ⚡ PERFORMANCE: Use single TTS request instead of parallel sentence processing
                    # This reduces API calls from 3+ to 1, avoiding rate limits
                    logger.info("Synthesizing full response in single request for speed")
                    async for audio_chunk in self.tts.synthesize(full_response):
                        audio_chunks_for_cache.append(audio_chunk)
                        await self.audio_output.play(audio_chunk)
                    
                    # Cache audio for future use
                    if audio_chunks_for_cache:
                        self.cache.set_audio_response(full_response, audio_chunks_for_cache)
                    
                    # Wait for playback to complete
                    while self.audio_output.is_playing():
                        await asyncio.sleep(0.1)
                    
                    self._is_speaking = False
                    
                    # Store conversation turn in memory
                    await self.memory.add_turn(user_input, full_response)
                    
                    # Update state back to listening
                    self.interrupt_manager.set_state(SystemState.LISTENING)
                    await self.ui.update_status(SystemState.LISTENING)
                    
                except asyncio.CancelledError:
                    logger.info("Conversation task cancelled during generation")
                    # Don't re-raise, continue to next iteration
                    self._is_speaking = False
                    self.interrupt_manager.set_state(SystemState.LISTENING)
                    await self.ui.update_status(SystemState.LISTENING)
                except Exception as e:
                    logger.error(f"Error generating response: {e}", exc_info=True)
                    await self.ui.display_error(f"Response generation failed: {e}")
                    
                    # Return to listening state and continue
                    self._is_speaking = False
                    self.interrupt_manager.set_state(SystemState.LISTENING)
                    await self.ui.update_status(SystemState.LISTENING)
                
        except asyncio.CancelledError:
            logger.info("Conversation task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in conversation task: {e}", exc_info=True)
            await self._handle_component_failure("conversation", e)
    
    async def _handle_interrupt(self) -> None:
        """Handle interrupt callback - stop all ongoing operations."""
        logger.info("Handling interrupt...")
        
        # Stop audio playback (but keep player ready for next response)
        await self.audio_output.stop()
        
        # Cancel TTS synthesis
        await self.tts.cancel_synthesis()
        
        # Cancel LLM generation
        await self.llm.cancel_generation()
        
        # Clear queues
        self._clear_queue(self._transcript_queue)
        
        self._is_speaking = False
        
        # Restart audio output for next response
        try:
            await self.audio_output.start()
        except Exception as e:
            logger.error(f"Error restarting audio output after interrupt: {e}")
        
        logger.info("Interrupt handled")
    
    def _clear_queue(self, queue: asyncio.Queue) -> None:
        """Clear all items from a queue."""
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break
    
    async def _handle_component_failure(self, component_name: str, error: Exception) -> None:
        """
        Handle component failure with isolation.
        
        Args:
            component_name: Name of failed component
            error: Exception that occurred
        """
        logger.error(f"Component failure: {component_name} - {error}")
        
        # Update UI
        await self.ui.display_error(f"{component_name} failed: {error}")
        
        # Update state
        self.interrupt_manager.set_state(SystemState.ERROR)
        await self.ui.update_status(SystemState.ERROR)
        
        # For critical components, stop the agent
        critical_components = ["audio_input", "audio_output"]
        if component_name in critical_components:
            logger.error(f"Critical component {component_name} failed, stopping agent")
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the voice agent and all components gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping VoiceAgent...")
        self._running = False
        
        # Update state
        self.interrupt_manager.set_state(SystemState.SHUTDOWN)
        await self.ui.update_status(SystemState.SHUTDOWN)
        
        shutdown_start = time.time()
        
        try:
            # Cancel all tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete (with timeout)
            if self._tasks:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=2.0
                )
            
            # Stop interrupt monitoring
            await self.interrupt_manager.stop_monitoring()
            
            # Stop components
            await self.audio_input.stop()
            await self.audio_output.stop()
            await self.stt.disconnect()
            
            # Stop UI last
            await self.ui.stop()
            
            shutdown_duration = time.time() - shutdown_start
            logger.info(f"VoiceAgent stopped in {shutdown_duration:.2f}s")
            
        except asyncio.TimeoutError:
            logger.warning("Graceful shutdown timeout exceeded, forcing stop")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def is_running(self) -> bool:
        """Check if voice agent is running."""
        return self._running
