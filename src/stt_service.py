"""Speech-to-Text service using Deepgram."""

import asyncio
import logging
from typing import Optional
import time

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from src.models import AudioChunk, Transcript


logger = logging.getLogger(__name__)


class STTService:
    """Speech-to-text service using Deepgram streaming API."""
    
    def __init__(self, api_key: str, model: str = "nova-2", language: str = "en-US"):
        """
        Initialize STT service.
        
        Args:
            api_key: Deepgram API key
            model: Deepgram model name (default: nova-2)
            language: Language code (default: en-US)
        """
        self.api_key = api_key
        self.model = model
        self.language = language
        
        # Deepgram client and connection
        self._client: Optional[AsyncDeepgramClient] = None
        self._connection = None
        self._connection_context = None  # Store context manager
        self._listening_task: Optional[asyncio.Task] = None  # Background listening task
        
        # Connection state
        self._connected = False
        self._transcript_queue: asyncio.Queue[Transcript] = asyncio.Queue()
        
        # Reconnection state
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._reconnect_backoff = [1, 2, 4]  # seconds
        
        # Audio buffer for reconnection
        self._audio_buffer: list[AudioChunk] = []
        self._max_buffer_size = 50  # Keep last 50 chunks during reconnection
        
        # Track speech timing
        self._speech_start_time: Optional[float] = None
        
        logger.info(f"STTService initialized: model={model}, language={language}")
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection to Deepgram.
        
        Raises:
            ConnectionError: If connection fails after max retries
        """
        if self._connected:
            logger.warning("STT service already connected")
            return
        
        try:
            # Initialize Deepgram async client
            self._client = AsyncDeepgramClient(api_key=self.api_key)
            
            # Create live transcription connection (v1 API)
            # Note: SDK v6 has limited parameter support for live streaming
            # Using minimal working parameters
            options = {
                "model": self.model,
                "language": "en",  # Use 'en' instead of 'en-US' for SDK v6
                "encoding": "linear16",
                "sample_rate": 16000,
                "channels": 1,
            }
            
            logger.debug(f"Connecting with options: {options}")
            
            # Create connection context
            self._connection_context = self._client.listen.v1.connect(**options)
            
            # Enter the context manager
            self._connection = await self._connection_context.__aenter__()
            
            # Register event handlers
            self._connection.on(EventType.OPEN, self._on_open)
            self._connection.on(EventType.MESSAGE, self._on_message)
            self._connection.on(EventType.ERROR, self._on_error)
            self._connection.on(EventType.CLOSE, self._on_close)
            
            # Start listening in a background task
            # This must run continuously to receive messages from Deepgram
            self._listening_task = asyncio.create_task(self._run_listening_loop())
            
            self._connected = True
            self._reconnect_attempts = 0
            logger.info("STT service connected to Deepgram")
                
        except Exception as e:
            logger.error(f"Failed to connect to STT service: {e}")
            await self._handle_connection_failure()
    
    async def _handle_connection_failure(self):
        """Handle connection failure with exponential backoff."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            raise ConnectionError(
                f"Failed to connect to STT service after {self._max_reconnect_attempts} attempts"
            )
        
        backoff_time = self._reconnect_backoff[self._reconnect_attempts]
        self._reconnect_attempts += 1
        
        logger.warning(
            f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} "
            f"in {backoff_time}s..."
        )
        
        await asyncio.sleep(backoff_time)
        await self.connect()
    
    async def _run_listening_loop(self) -> None:
        """Run the listening loop in the background."""
        try:
            logger.debug("Starting listening loop...")
            # Start listening - this will block until connection closes
            await self._connection.start_listening()
            logger.debug("Listening loop ended")
        except asyncio.CancelledError:
            logger.debug("Listening loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in listening loop: {e}", exc_info=True)
            self._connected = False
    
    def _on_open(self, *args, **kwargs):
        """Handle WebSocket connection opened."""
        logger.info("Deepgram WebSocket connection opened")
    
    def _on_message(self, *args, **kwargs):
        """Handle transcript messages from Deepgram."""
        try:
            # Extract message from args
            message = args[0] if args else kwargs.get('message')
            if not message:
                return
            
            # Check if this is an utterance end event
            if hasattr(message, 'type') and message.type == 'UtteranceEnd':
                logger.debug("Utterance end detected by Deepgram")
                # Reset speech timing for next utterance
                self._speech_start_time = None
                return
            
            # Check if this is a transcript result
            if not hasattr(message, 'channel'):
                return
            
            # Parse the transcript
            channel = message.channel
            if not channel or not hasattr(channel, 'alternatives') or not channel.alternatives:
                return
            
            alternative = channel.alternatives[0]
            transcript_text = alternative.transcript
            
            # Skip empty transcripts
            if not transcript_text or not transcript_text.strip():
                return
            
            # Determine if this is a final transcript
            is_final = message.is_final if hasattr(message, 'is_final') else False
            confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0.0
            
            # Track speech timing
            if self._speech_start_time is None:
                self._speech_start_time = time.time()
            
            duration_ms = int((time.time() - self._speech_start_time) * 1000)
            
            # Create transcript object
            transcript = Transcript(
                text=transcript_text,
                is_final=is_final,
                confidence=confidence,
                timestamp=time.time(),
                duration_ms=duration_ms
            )
            
            # Queue the transcript
            self._transcript_queue.put_nowait(transcript)
            
            logger.debug(
                f"Transcript received: '{transcript_text}' "
                f"(final={is_final}, confidence={confidence:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
    
    def _on_utterance_end(self, *args, **kwargs):
        """Handle utterance end event from Deepgram (deprecated - handled in _on_message)."""
        logger.debug("Utterance end detected by Deepgram")
        # Reset speech timing for next utterance
        self._speech_start_time = None
    
    def _on_error(self, *args, **kwargs):
        """Handle WebSocket errors."""
        error = args[0] if args else kwargs.get('error', 'Unknown error')
        logger.error(f"Deepgram WebSocket error: {error}")
        self._connected = False
    
    def _on_close(self, *args, **kwargs):
        """Handle WebSocket connection closed."""
        logger.info("Deepgram WebSocket connection closed")
        self._connected = False
    
    async def send_audio(self, chunk: AudioChunk) -> None:
        """
        Stream audio chunk to STT service.
        
        Args:
            chunk: Audio chunk to send
        """
        if not self._connected or not self._connection:
            # Buffer audio during disconnection
            if len(self._audio_buffer) < self._max_buffer_size:
                self._audio_buffer.append(chunk)
            logger.warning("STT not connected, buffering audio")
            return
        
        try:
            # Send buffered audio first if we were reconnecting
            if self._audio_buffer:
                logger.info(f"Sending {len(self._audio_buffer)} buffered chunks")
                for buffered_chunk in self._audio_buffer:
                    # Convert numpy array to bytes
                    audio_bytes = buffered_chunk.data.tobytes()
                    try:
                        await self._connection.send_media(audio_bytes)
                    except Exception as e:
                        # Ignore "sent X (OK); then received X (OK)" messages
                        if "sent" in str(e) and "(OK)" in str(e):
                            pass  # This is actually a success message
                        else:
                            raise
                self._audio_buffer.clear()
            
            # Send current audio chunk
            audio_bytes = chunk.data.tobytes()
            try:
                await self._connection.send_media(audio_bytes)
            except Exception as e:
                # Ignore "sent X (OK); then received X (OK)" messages
                if "sent" in str(e) and "(OK)" in str(e):
                    pass  # This is actually a success message
                else:
                    raise
                
        except Exception as e:
            logger.error(f"Error sending audio to STT service: {e}")
            self._connected = False
            # Buffer this chunk
            if len(self._audio_buffer) < self._max_buffer_size:
                self._audio_buffer.append(chunk)
    
    async def get_transcript(self, timeout: float = 30.0) -> Optional[Transcript]:
        """
        Receive transcript from STT service (partial or final).
        
        Args:
            timeout: Timeout in seconds (default: 30s)
        
        Returns:
            Transcript: Received transcript, or None if timeout
        """
        try:
            return await asyncio.wait_for(self._transcript_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for transcript from STT service")
            return None
    
    async def disconnect(self) -> None:
        """Close WebSocket connection to STT service."""
        if not self._connected:
            return
        
        logger.info("Disconnecting from STT service...")
        
        try:
            # Cancel listening task first
            if self._listening_task and not self._listening_task.done():
                logger.debug("Cancelling listening task...")
                self._listening_task.cancel()
                try:
                    await self._listening_task
                except asyncio.CancelledError:
                    logger.debug("Listening task cancelled")
            
            # Close Deepgram connection
            if self._connection:
                # Send finalize message to close the stream gracefully
                try:
                    await self._connection.send_finalize()
                except Exception as e:
                    logger.debug(f"Error sending finalize: {e}")
                
                # Exit the context manager
                if self._connection_context:
                    try:
                        await self._connection_context.__aexit__(None, None, None)
                    except Exception as e:
                        logger.debug(f"Error exiting context: {e}")
                
                self._connection = None
                self._connection_context = None
            
            self._connected = False
            self._client = None
            self._listening_task = None
            
            # Clear queues
            while not self._transcript_queue.empty():
                try:
                    self._transcript_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            self._audio_buffer.clear()
            self._speech_start_time = None
            
            logger.info("Disconnected from STT service")
            
        except Exception as e:
            logger.error(f"Error disconnecting from STT service: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to STT service."""
        return self._connected
