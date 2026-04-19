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

import numpy as np

from src.audio_dsp import build_mic_dsp
from src.latency_metrics import LatencyMetrics
from src.input_controls import PushToTalkController, WakeWordGate
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
            api_key=(
                config.mistral_api_key if config.llm_provider == "mistral" 
                else config.huggingface_api_key if config.llm_provider == "huggingface" 
                else (config.openai_api_key or "")
            ),
            model=config.llm_model,
            system_prompt=config.system_prompt,
            provider=config.llm_provider,
            fallback_provider=config.llm_fallback_provider if hasattr(config, 'llm_fallback_provider') else None,
            fallback_api_key=(
                config.openai_api_key if hasattr(config, 'llm_fallback_provider') and config.llm_fallback_provider == "openai"
                else config.mistral_api_key if hasattr(config, 'llm_fallback_provider') and config.llm_fallback_provider == "mistral"
                else config.huggingface_api_key if hasattr(config, 'llm_fallback_provider') and config.llm_fallback_provider == "huggingface"
                else ""
            ),
            fallback_model=config.llm_fallback_model if hasattr(config, 'llm_fallback_model') else None
        )
        
        self.tts = TTSService(
            provider=config.tts_provider,
            api_key=config.elevenlabs_api_key if config.tts_provider == "elevenlabs" else config.openai_api_key,
            voice=config.tts_voice,
            model=config.tts_model,
            fallback_provider=config.tts_fallback_provider if hasattr(config, 'tts_fallback_provider') else ("openai" if config.tts_provider == "elevenlabs" else None),
            fallback_api_key=config.openai_api_key if config.tts_provider == "elevenlabs" else None,
            fallback_voice=config.tts_fallback_voice if hasattr(config, 'tts_fallback_voice') else ("alloy" if config.tts_provider == "elevenlabs" else None),
            fallback_model=config.tts_fallback_model if hasattr(config, 'tts_fallback_model') else ("tts-1" if config.tts_provider == "elevenlabs" else None)
        )
        
        self.audio_output = AudioOutputPlayer(
            sample_rate=24000  # 🔊 SOUND QUALITY: TTS output sample rate (must match TTS service)
        )
        
        self.interrupt_manager = InterruptManager(
            interrupt_debounce_ms=config.interrupt_debounce_ms,
            barge_in_min_confidence=config.barge_in_min_confidence,
            barge_in_on=config.barge_in_on,
        )
        
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
        
        self._mic_dsp = build_mic_dsp(
            config.input_dsp,
            config.sample_rate,
            config.input_dsp_highpass_hz,
            config.input_dsp_noise_gate_rms,
        )
        self._metrics = LatencyMetrics(enabled=config.metrics_enabled)
        self._effective_input_mode = "continuous"
        self._ptt: Optional[PushToTalkController] = None
        self._wake: Optional[WakeWordGate] = None
        
        # Inter-component queues
        self._audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=50)
        self._vad_queue: asyncio.Queue[VADEvent] = asyncio.Queue(maxsize=200)
        self._transcript_queue: asyncio.Queue[Transcript] = asyncio.Queue(maxsize=20)
        
        # Component tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        # Current conversation state
        self._current_user_input = ""
        self._current_response_tokens: List[str] = []
        self._is_speaking = False
        
        logger.info("VoiceAgent initialized")
    
    def _setup_voice_input(self) -> None:
        """Configure push-to-talk or wake-word gating from config."""
        self._effective_input_mode = "continuous"
        self._ptt = None
        self._wake = None
        mode = self.config.voice_input_mode
        if mode == "push_to_talk":
            try:
                self._ptt = PushToTalkController(
                    self.config.push_to_talk_key,
                    stdin_arm_seconds=self.config.push_to_talk_stdin_arm_seconds,
                    startup_open_seconds=self.config.push_to_talk_startup_open_seconds,
                )
                self._ptt.start()
                self._effective_input_mode = "push_to_talk"
                logger.info(
                    "Voice input: push_to_talk (hold %s to stream mic to STT/VAD)",
                    self.config.push_to_talk_key,
                )
            except Exception as e:
                logger.error("Push-to-talk failed (%s); using continuous", e)
        elif mode == "wake_word":
            self._wake = WakeWordGate(
                models=list(self.config.wake_word_models),
                sensitivity=self.config.wake_word_sensitivity,
                open_seconds=self.config.wake_word_window_seconds,
            )
            if self._wake.is_available:
                self._effective_input_mode = "wake_word"
                logger.info(
                    "Voice input: wake_word (models=%s); say wake phrase then speak",
                    self.config.wake_word_models,
                )
            else:
                logger.warning(
                    "wake_word mode needs openwakeword (pip install openwakeword); "
                    "using continuous"
                )
                self._wake = None
        else:
            logger.info("Voice input: continuous")
    
    def _is_audio_gate_open(self) -> bool:
        if self._effective_input_mode == "continuous":
            return True
        if self._effective_input_mode == "push_to_talk":
            return self._ptt is not None and self._ptt.is_pressed
        if self._effective_input_mode == "wake_word":
            return self._wake is not None and self._wake.is_open()
        return True

    def _arm_ptt_post_response_window(self) -> None:
        """Re-open push-to-talk gate so the next utterance reaches STT without Enter/Space."""
        if self._ptt is None:
            return
        w = self.config.push_to_talk_after_response_open_seconds
        if w <= 0:
            return
        self._ptt.extend_open_window(w)
        logger.info(
            "PTT: gate open for ~%.0fs — speak your next phrase (no Enter/Space needed during this window).",
            w,
        )

    async def _enter_listening_state(self) -> None:
        self.interrupt_manager.set_state(SystemState.LISTENING)
        await self.ui.update_status(SystemState.LISTENING)
        self._arm_ptt_post_response_window()
    
    @staticmethod
    def _silence_chunk_like(template: AudioChunk) -> AudioChunk:
        """Same shape/rate as template; keeps Deepgram live when mic audio is gated."""
        return AudioChunk(
            data=np.zeros(len(template.data), dtype=np.int16),
            sample_rate=template.sample_rate,
            timestamp=template.timestamp,
            duration_ms=template.duration_ms,
        )
    
    async def _play_tts_chunk(self, chunk: AudioChunk, tts_started: List[bool]) -> None:
        """Play one TTS chunk; mark first-audio latency once."""
        if not tts_started[0]:
            self._metrics.mark_tts_first_chunk()
            tts_started[0] = True
        await self.audio_output.play(chunk)
    
    def _stop_voice_input(self) -> None:
        if self._ptt is not None:
            self._ptt.stop()
            self._ptt = None
        if self._wake is not None:
            self._wake.reset()
            self._wake = None
    
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
            
            # Warm up VAD model during startup to avoid first-utterance stall.
            await self.vad.warmup()
            
            # Connect STT service with timeout
            try:
                await asyncio.wait_for(self.stt.connect(), timeout=15.0)
            except asyncio.TimeoutError:
                logger.error("STT connection timed out")
                raise ConnectionError("STT service connection timed out after 15 seconds")
            
            self._setup_voice_input()
            
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
            await self._enter_listening_state()
            
            logger.info("VoiceAgent started successfully")
            if self._effective_input_mode == "push_to_talk":
                logger.info(
                    "Blocking on audio/STT workers until Ctrl+C (expected). With PTT, few INFO logs until the mic gate is open and speech reaches STT."
                )
            
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
                chunk = self._mic_dsp.process(chunk)
                
                if self._wake is not None:
                    self._wake.process_chunk(chunk.data)
                
                # STT: real speech only when gate is open; silence keepalive when closed
                # (Deepgram net0001 if no bytes for ~10s).
                gate_open = self._is_audio_gate_open()
                if self.stt.is_connected():
                    if gate_open:
                        await self.stt.send_audio(chunk)
                    else:
                        await self.stt.send_audio(self._silence_chunk_like(chunk))
                
                # PTT idle: VAD on silence too — otherwise VAD sees room speech but STT gets
                # silence and you get "Speech started" with no transcripts. While agent speaks,
                # keep real mic on VAD for barge-in even if Space is not held.
                ptt_idle = (
                    self._effective_input_mode == "push_to_talk"
                    and not gate_open
                    and not self._is_speaking
                )
                vad_chunk = self._silence_chunk_like(chunk) if ptt_idle else chunk
                
                # Put in queue for VAD processing. Drop oldest when backpressured.
                if self._audio_queue.full():
                    try:
                        self._audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await self._audio_queue.put(vad_chunk)
                
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
                
                # Preserve recent VAD events under bursty conditions.
                if self._vad_queue.full():
                    try:
                        self._vad_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
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
                    
                    # Ignore STT while agent is speaking to avoid self-transcription loops.
                    if self._is_speaking:
                        logger.debug("Dropping transcript captured during agent speech")
                        continue
                    
                    # Update UI with partial or final transcript
                    await self.ui.update_transcript(
                        role="user",
                        content=transcript.text,
                        is_partial=not transcript.is_final
                    )
                    
                    # If final transcript, add to queue for processing
                    if transcript.is_final:
                        if self._transcript_queue.full():
                            try:
                                self._transcript_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
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
                self._metrics.begin_turn(user_input)
                
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
                        self._metrics.mark_cached_turn()
                        tts_started_cached = [False]
                        for audio_chunk in cached_audio:
                            await self._play_tts_chunk(audio_chunk, tts_started_cached)
                        
                        # Wait for playback to complete
                        while self.audio_output.is_playing():
                            await asyncio.sleep(0.1)
                        
                        self._is_speaking = False
                        self._metrics.mark_playback_end()
                        self._metrics.log_turn_summary()
                        
                        # Store in memory
                        await self.memory.add_turn(user_input, full_response)
                        
                        # Update state back to listening
                        await self._enter_listening_state()
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
                    tts_started = [False]
                    if cached_text:
                        # Use cached text but need to synthesize audio
                        full_response = cached_text
                        
                        # Update state to speaking
                        self.interrupt_manager.set_state(SystemState.SPEAKING)
                        await self.ui.update_status(SystemState.SPEAKING)
                        self._is_speaking = True
                        
                        audio_chunks_for_cache = []
                        async for audio_chunk in self.tts.synthesize(full_response):
                            audio_chunks_for_cache.append(audio_chunk)
                            await self._play_tts_chunk(audio_chunk, tts_started)
                    else:
                        # Provider-aware strategy:
                        # - OpenAI: one TTS request per response (avoids 429 storms on sentence splitting)
                        # - ElevenLabs/Edge: stream sentence-by-sentence for lower first-audio latency
                        use_streaming_tts = self.tts.provider in {"elevenlabs", "edge-tts"}
                        audio_chunks_for_cache = []
                        
                        if use_streaming_tts:
                            token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=256)
                            
                            async def token_stream():
                                while True:
                                    token = await token_queue.get()
                                    if token is None:
                                        break
                                    yield token
                            
                            async def llm_token_producer():
                                try:
                                    async for token in self.llm.generate_response(
                                        user_input,
                                        context,
                                        on_first_token=self._metrics.mark_llm_first_token,
                                    ):
                                        response_tokens.append(token)
                                        
                                        # Update UI with partial response
                                        partial_response = "".join(response_tokens)
                                        await self.ui.update_transcript(
                                            role="assistant",
                                            content=partial_response,
                                            is_partial=True
                                        )
                                        await token_queue.put(token)
                                finally:
                                    self._metrics.mark_llm_end()
                                    await token_queue.put(None)
                            
                            producer_task = asyncio.create_task(llm_token_producer())
                            
                            # Update state to speaking early so barge-in works as soon as audio starts.
                            self.interrupt_manager.set_state(SystemState.SPEAKING)
                            await self.ui.update_status(SystemState.SPEAKING)
                            self._is_speaking = True
                            
                            try:
                                async for audio_chunk in self.tts.synthesize_stream(token_stream()):
                                    audio_chunks_for_cache.append(audio_chunk)
                                    await self._play_tts_chunk(audio_chunk, tts_started)
                                await producer_task
                            except Exception:
                                if not producer_task.done():
                                    producer_task.cancel()
                                raise
                        else:
                            logger.info("Using single-request TTS mode for OpenAI stability")
                            async for token in self.llm.generate_response(
                                user_input,
                                context,
                                on_first_token=self._metrics.mark_llm_first_token,
                            ):
                                response_tokens.append(token)
                                
                                # Update UI with partial response
                                partial_response = "".join(response_tokens)
                                await self.ui.update_transcript(
                                    role="assistant",
                                    content=partial_response,
                                    is_partial=True
                                )
                            self._metrics.mark_llm_end()
                        
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
                        
                        # In single-request mode, synthesize after full text is available.
                        if not use_streaming_tts:
                            self.interrupt_manager.set_state(SystemState.SPEAKING)
                            await self.ui.update_status(SystemState.SPEAKING)
                            self._is_speaking = True
                            async for audio_chunk in self.tts.synthesize(full_response):
                                audio_chunks_for_cache.append(audio_chunk)
                                await self._play_tts_chunk(audio_chunk, tts_started)
                    
                    # Cache audio for future use
                    if audio_chunks_for_cache:
                        self.cache.set_audio_response(full_response, audio_chunks_for_cache)
                    
                    # Wait for playback to complete
                    while self.audio_output.is_playing():
                        await asyncio.sleep(0.1)
                    
                    self._is_speaking = False
                    self._metrics.mark_playback_end()
                    self._metrics.log_turn_summary()
                    
                    # Store conversation turn in memory
                    await self.memory.add_turn(user_input, full_response)
                    
                    # Update state back to listening
                    await self._enter_listening_state()
                    
                except asyncio.CancelledError:
                    logger.info("Conversation task cancelled during generation")
                    # Don't re-raise, continue to next iteration
                    self._is_speaking = False
                    await self._enter_listening_state()
                except Exception as e:
                    logger.error(f"Error generating response: {e}", exc_info=True)
                    await self.ui.display_error(f"Response generation failed: {e}")
                    
                    # Return to listening state and continue
                    self._is_speaking = False
                    await self._enter_listening_state()
                
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
        
        # Do not clear _transcript_queue: a final transcript may already be queued
        # from the user's barge-in phrase; dropping it made the agent look "stuck"
        # waiting for the next line the user thought they already said.
        
        self._is_speaking = False
        
        # Restart audio output for next response
        try:
            await self.audio_output.start()
        except Exception as e:
            logger.error(f"Error restarting audio output after interrupt: {e}")
        
        await self.ui.update_status(SystemState.LISTENING)
        self._arm_ptt_post_response_window()
        logger.info("Interrupt handled")
    
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
            
            self._stop_voice_input()
            
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
