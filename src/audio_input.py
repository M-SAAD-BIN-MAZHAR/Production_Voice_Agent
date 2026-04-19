"""Audio input capture module."""

import asyncio
import logging
import numpy as np
import sounddevice as sd
from typing import Optional
import time

from src.models import AudioChunk


logger = logging.getLogger(__name__)


class AudioInputCapture:
    """Captures audio from microphone and streams to downstream components."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        queue_maxsize: int = 50
    ):
        """
        Initialize audio input capture.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            chunk_duration_ms: Duration of each audio chunk in milliseconds (default: 100)
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=queue_maxsize)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        
        logger.info(
            f"AudioInputCapture initialized: sample_rate={sample_rate}Hz, "
            f"chunk_duration={chunk_duration_ms}ms, chunk_size={self.chunk_size} samples"
        )
    
    def _enqueue_audio_chunk(self, chunk: AudioChunk) -> None:
        """Enqueue audio chunk on the event loop thread."""
        try:
            self._audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            # Drop oldest to keep latency low and preserve recency.
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.warning("Audio queue still full, dropping incoming chunk")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Callback function called by sounddevice for each audio chunk.
        
        Args:
            indata: Input audio data as numpy array
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio input status: {status}")
        
        if not self._running:
            return
        
        # 🔊 SOUND QUALITY: Convert float32 to int16 for processing
        # Multiply by 32767 to scale from [-1.0, 1.0] to [-32768, 32767]
        # Convert to int16 and create AudioChunk
        audio_data = (indata[:, 0] * 32767).astype(np.int16)
        chunk = AudioChunk(
            data=audio_data.copy(),
            sample_rate=self.sample_rate,
            timestamp=time.time(),
            duration_ms=self.chunk_duration_ms
        )
        
        # Thread-safe handoff from sounddevice callback thread to event loop.
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._enqueue_audio_chunk, chunk)
    
    async def start(self) -> None:
        """
        Initialize microphone and begin capturing audio.
        
        Raises:
            RuntimeError: If microphone is unavailable or initialization fails
        """
        if self._running:
            logger.warning("AudioInputCapture already running")
            return
        
        try:
            self._loop = asyncio.get_running_loop()
            
            # Check if input device is available
            default_input = sd.query_devices(kind='input')
            logger.info(f"Using input device: {default_input['name']}")
            
            # 🔊 SOUND QUALITY PARAMETERS:
            # - samplerate: Input audio quality (16000 Hz = wideband, good for speech)
            # - channels: 1 (mono) is sufficient for voice input
            # - dtype: np.float32 (internal processing format)
            # - blocksize: Matches chunk_size for consistent processing
            # Create input stream
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,  # 🔊 SOUND QUALITY: 16kHz for speech recognition
                channels=1,  # 🔊 SOUND QUALITY: Mono input
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            
            self._running = True
            self._stream.start()
            logger.info("Audio input capture started")
            
        except sd.PortAudioError as e:
            error_msg = str(e).lower()
            if "device unavailable" in error_msg or "no device" in error_msg:
                raise RuntimeError(
                    f"Microphone unavailable. Please check device connections. Error: {e}"
                )
            else:
                logger.error(f"Audio device error: {e}")
                raise RuntimeError(f"Failed to initialize audio input: {e}")
        except Exception as e:
            logger.error(f"Failed to start audio input: {e}", exc_info=True)
            raise RuntimeError(f"Failed to start audio input: {e}")
    
    async def get_audio_chunk(self) -> AudioChunk:
        """
        Get next audio chunk from capture queue.
        
        Returns:
            AudioChunk: Next audio chunk
        """
        return await self._audio_queue.get()
    
    async def stop(self) -> None:
        """Stop capturing and release microphone."""
        if not self._running:
            return
        
        logger.info("Stopping audio input capture...")
        self._running = False
        self._loop = None
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("Audio input capture stopped")
    
    def is_running(self) -> bool:
        """Check if audio capture is running."""
        return self._running
