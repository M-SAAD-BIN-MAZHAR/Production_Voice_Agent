"""Audio output playback module."""

import asyncio
import logging
import numpy as np
import sounddevice as sd
from typing import Optional
from enum import Enum
import time

from src.models import AudioChunk


logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """Audio playback state."""
    IDLE = "idle"
    PLAYING = "playing"
    STOPPING = "stopping"


class AudioOutputPlayer:
    """Plays audio chunks through speakers in real-time."""
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize audio output player.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 24000)
        """
        self.sample_rate = sample_rate
        
        self._stream: Optional[sd.OutputStream] = None
        self._playback_queue: asyncio.Queue[Optional[AudioChunk]] = asyncio.Queue()
        self._state = PlaybackState.IDLE
        self._running = False
        self._playback_task: Optional[asyncio.Task] = None
        self._current_chunk: Optional[np.ndarray] = None
        self._chunk_position: int = 0
        
        logger.info(f"AudioOutputPlayer initialized: sample_rate={sample_rate}Hz")
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """
        Callback function called by sounddevice for each audio output chunk.
        
        Args:
            outdata: Output audio buffer to fill
            frames: Number of frames requested
            time_info: Time information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio output status: {status}")
        
        # Fill output buffer with audio data or silence
        if self._current_chunk is not None and self._chunk_position < len(self._current_chunk):
            # Copy available audio data
            available = len(self._current_chunk) - self._chunk_position
            to_copy = min(frames, available)
            
            outdata[:to_copy, 0] = self._current_chunk[self._chunk_position:self._chunk_position + to_copy]
            self._chunk_position += to_copy
            
            # Fill remaining with silence if needed
            if to_copy < frames:
                outdata[to_copy:, 0] = 0
        else:
            # No data available, fill with silence
            outdata.fill(0)
    
    async def start(self, sample_rate: Optional[int] = None) -> None:
        """
        Initialize audio output device.
        
        Args:
            sample_rate: Audio sample rate (optional, uses default if not provided)
            
        Raises:
            RuntimeError: If audio output device is unavailable
        """
        if self._running:
            logger.warning("AudioOutputPlayer already running")
            return
        
        if sample_rate:
            self.sample_rate = sample_rate
        
        try:
            # Check if output device is available
            default_output = sd.query_devices(kind='output')
            logger.info(f"Using output device: {default_output['name']}")
            
            # Create output stream with larger buffer
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,  # Mono
                dtype=np.int16,
                callback=self._audio_callback,
                blocksize=2048  # Larger buffer for smoother playback
            )
            
            self._running = True
            self._state = PlaybackState.IDLE
            self._stream.start()
            
            # Start playback task
            self._playback_task = asyncio.create_task(self._playback_loop())
            
            logger.info("Audio output player started")
            
        except sd.PortAudioError as e:
            error_msg = str(e).lower()
            if "device unavailable" in error_msg or "no device" in error_msg:
                raise RuntimeError(
                    f"Audio output device unavailable. Please check device connections. Error: {e}"
                )
            else:
                logger.error(f"Audio device error: {e}")
                raise RuntimeError(f"Failed to initialize audio output: {e}")
        except Exception as e:
            logger.error(f"Failed to start audio output: {e}", exc_info=True)
            raise RuntimeError(f"Failed to start audio output: {e}")
    
    async def _playback_loop(self):
        """Background task that plays audio chunks from the queue."""
        logger.info("Playback loop started")
        
        try:
            while self._running:
                # Get next chunk from queue
                chunk = await self._playback_queue.get()
                
                # None is sentinel value for stop
                if chunk is None:
                    break
                
                # Update state to playing
                if self._state == PlaybackState.IDLE:
                    self._state = PlaybackState.PLAYING
                    logger.debug("Started playing audio")
                
                # Set current chunk for callback to use
                self._current_chunk = chunk.data
                self._chunk_position = 0
                
                # Calculate playback duration
                duration_seconds = len(chunk.data) / chunk.sample_rate
                
                # Wait for chunk to be played
                await asyncio.sleep(duration_seconds + 0.01)
                
                # Mark task as done
                self._playback_queue.task_done()
            
            # Playback stopped
            self._state = PlaybackState.IDLE
            self._current_chunk = None
            logger.info("Playback loop stopped")
            
        except asyncio.CancelledError:
            logger.info("Playback loop cancelled")
            self._state = PlaybackState.IDLE
            raise
        except Exception as e:
            logger.error(f"Error in playback loop: {e}", exc_info=True)
            self._state = PlaybackState.IDLE
    
    async def play(self, chunk: AudioChunk) -> None:
        """
        Queue audio chunk for playback.
        
        Args:
            chunk: Audio chunk to play
        """
        if not self._running:
            logger.warning("AudioOutputPlayer not running, cannot play chunk")
            return
        
        # Add chunk to playback queue
        await self._playback_queue.put(chunk)
        
        # Log first chunk latency
        if self._state == PlaybackState.IDLE:
            start_time = time.time()
            # Wait a bit to see when playback actually starts
            await asyncio.sleep(0.01)
            latency_ms = (time.time() - start_time) * 1000
            if latency_ms > 100:
                logger.warning(
                    f"Audio playback start latency: {latency_ms:.1f}ms (> 100ms)"
                )
    
    async def stop(self) -> None:
        """Stop playback immediately and clear queue."""
        if not self._running:
            return
        
        logger.info("Stopping audio output player...")
        self._state = PlaybackState.STOPPING
        
        # Clear playback queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
                self._playback_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Send sentinel value to stop playback loop
        await self._playback_queue.put(None)
        
        # Wait for playback task to complete
        if self._playback_task and not self._playback_task.done():
            try:
                await asyncio.wait_for(self._playback_task, timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Playback task did not complete in time, cancelling")
                self._playback_task.cancel()
                try:
                    await self._playback_task
                except asyncio.CancelledError:
                    pass
        
        # Stop and close stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        self._running = False
        self._state = PlaybackState.IDLE
        
        logger.info("Audio output player stopped")
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._state == PlaybackState.PLAYING
    
    def get_state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state

