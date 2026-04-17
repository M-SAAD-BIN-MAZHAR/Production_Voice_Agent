"""Voice Activity Detection module.

Detects when the user is speaking using Silero VAD (Voice Activity Detection).
Uses a state machine to track speech start/end events with configurable sensitivity.

Key Features:
- Silero VAD model for accurate speech detection
- Fallback to energy-based VAD if model fails
- Configurable threshold and silence duration
- State machine for reliable speech boundary detection

🔊 SOUND QUALITY IMPACT:
- threshold: Higher = less sensitive, cleaner detection (fewer false positives)
- silence_duration_ms: Longer = won't cut off natural pauses in speech
"""

import asyncio
import logging
import time
from typing import Optional
import numpy as np

from src.models import AudioChunk, VADEvent, VADEventType


logger = logging.getLogger(__name__)


class VADModule:
    """Voice activity detection using Silero VAD."""
    
    def __init__(self, threshold: float = 0.5, silence_duration_ms: int = 700):
        """
        Initialize VAD module.
        
        Args:
            threshold: Speech detection threshold (0.0-1.0, default: 0.5)
                🔊 SOUND QUALITY: Higher threshold = less sensitive
                - 0.5-0.7: More sensitive, may pick up background noise
                - 0.8-0.9: Balanced (recommended)
                - 0.9-0.95: Less sensitive, cleaner detection
            silence_duration_ms: Milliseconds of silence before SPEECH_END (default: 700)
                🔊 SOUND QUALITY: Longer duration = won't cut off pauses
                - 500-1000ms: Faster response, may cut off speech
                - 1000-1500ms: Balanced (recommended)
                - 1500-2000ms: More patient, won't cut off natural pauses
        """
        self.threshold = threshold
        self.silence_duration_ms = silence_duration_ms
        
        # State machine
        self._current_state = VADEventType.SILENCE
        self._silence_start_time: Optional[float] = None
        self._speech_start_time: Optional[float] = None
        
        # Silero VAD model (lazy loaded)
        self._model = None
        self._model_loaded = False
        
        logger.info(
            f"VADModule initialized: threshold={threshold}, "
            f"silence_duration={silence_duration_ms}ms"
        )
    
    def _load_model(self):
        """Load Silero VAD model (lazy loading)."""
        if self._model_loaded:
            return
        
        try:
            import torch
            # Load Silero VAD model
            self._model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self._model_loaded = True
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            # Fall back to simple energy-based VAD
            self._model = None
            self._model_loaded = True
            logger.warning("Using fallback energy-based VAD")
    
    def _simple_vad(self, audio_chunk: AudioChunk) -> float:
        """
        Simple energy-based VAD fallback.
        
        Args:
            audio_chunk: Audio chunk to analyze
            
        Returns:
            Speech probability (0.0-1.0)
        """
        # Calculate RMS energy
        audio_float = audio_chunk.data.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Simple threshold-based detection
        # Normalize RMS to 0-1 range (typical speech RMS is 0.01-0.3)
        normalized_energy = min(rms / 0.3, 1.0)
        
        return normalized_energy
    
    def _silero_vad(self, audio_chunk: AudioChunk) -> float:
        """
        Silero VAD-based speech detection.
        
        Args:
            audio_chunk: Audio chunk to analyze
            
        Returns:
            Speech probability (0.0-1.0)
        """
        if self._model is None:
            return self._simple_vad(audio_chunk)
        
        try:
            import torch
            
            # Convert to float32 and normalize
            audio_float = audio_chunk.data.astype(np.float32) / 32768.0
            
            # Silero VAD expects exactly 512 samples for 16kHz audio
            # Pad or truncate to match expected size
            if len(audio_float) < 512:
                # Pad with zeros
                audio_float = np.pad(audio_float, (0, 512 - len(audio_float)), mode='constant')
            elif len(audio_float) > 512:
                # Truncate to 512 samples
                audio_float = audio_float[:512]
            
            audio_tensor = torch.from_numpy(audio_float)
            
            # Silero VAD expects 16kHz audio
            if audio_chunk.sample_rate != 16000:
                logger.warning(
                    f"Silero VAD expects 16kHz audio, got {audio_chunk.sample_rate}Hz"
                )
            
            # Get speech probability
            speech_prob = self._model(audio_tensor, audio_chunk.sample_rate).item()
            return speech_prob
            
        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return self._simple_vad(audio_chunk)
    
    async def process_audio(self, chunk: AudioChunk) -> VADEvent:
        """
        Analyze audio chunk and return speech detection event.
        
        Args:
            chunk: Audio chunk to analyze
            
        Returns:
            VADEvent: Speech detection event
        """
        # Lazy load model on first use
        if not self._model_loaded:
            self._load_model()
        
        # Get speech probability
        speech_prob = self._silero_vad(chunk)
        
        current_time = time.time()
        is_speech = speech_prob >= self.threshold
        
        # State machine logic
        if self._current_state == VADEventType.SILENCE:
            if is_speech:
                # Transition: SILENCE → SPEECH_START
                self._current_state = VADEventType.SPEECH_START
                self._speech_start_time = current_time
                self._silence_start_time = None
                
                return VADEvent(
                    event_type=VADEventType.SPEECH_START,
                    timestamp=current_time,
                    confidence=speech_prob
                )
            else:
                # Stay in SILENCE
                return VADEvent(
                    event_type=VADEventType.SILENCE,
                    timestamp=current_time,
                    confidence=1.0 - speech_prob
                )
        
        elif self._current_state == VADEventType.SPEECH_START:
            if is_speech:
                # Transition: SPEECH_START → SPEECH_CONTINUE
                self._current_state = VADEventType.SPEECH_CONTINUE
                
                return VADEvent(
                    event_type=VADEventType.SPEECH_CONTINUE,
                    timestamp=current_time,
                    confidence=speech_prob
                )
            else:
                # Start tracking silence
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                
                silence_duration = (current_time - self._silence_start_time) * 1000
                
                if silence_duration >= self.silence_duration_ms:
                    # Transition: SPEECH_START → SPEECH_END → SILENCE
                    self._current_state = VADEventType.SILENCE
                    self._silence_start_time = None
                    self._speech_start_time = None
                    
                    return VADEvent(
                        event_type=VADEventType.SPEECH_END,
                        timestamp=current_time,
                        confidence=1.0 - speech_prob
                    )
                else:
                    # Still in SPEECH_START (short silence)
                    return VADEvent(
                        event_type=VADEventType.SPEECH_START,
                        timestamp=current_time,
                        confidence=speech_prob
                    )
        
        elif self._current_state == VADEventType.SPEECH_CONTINUE:
            if is_speech:
                # Stay in SPEECH_CONTINUE
                self._silence_start_time = None
                
                return VADEvent(
                    event_type=VADEventType.SPEECH_CONTINUE,
                    timestamp=current_time,
                    confidence=speech_prob
                )
            else:
                # Start tracking silence
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                
                silence_duration = (current_time - self._silence_start_time) * 1000
                
                if silence_duration >= self.silence_duration_ms:
                    # Transition: SPEECH_CONTINUE → SPEECH_END → SILENCE
                    self._current_state = VADEventType.SILENCE
                    self._silence_start_time = None
                    self._speech_start_time = None
                    
                    return VADEvent(
                        event_type=VADEventType.SPEECH_END,
                        timestamp=current_time,
                        confidence=1.0 - speech_prob
                    )
                else:
                    # Still in SPEECH_CONTINUE (short silence)
                    return VADEvent(
                        event_type=VADEventType.SPEECH_CONTINUE,
                        timestamp=current_time,
                        confidence=speech_prob
                    )
        
        # Default: return current state
        return VADEvent(
            event_type=self._current_state,
            timestamp=current_time,
            confidence=speech_prob if is_speech else 1.0 - speech_prob
        )
    
    def configure(self, threshold: float = 0.5, silence_duration_ms: int = 700) -> None:
        """
        Configure VAD sensitivity and silence threshold.
        
        Args:
            threshold: Speech detection threshold (0.0-1.0)
            silence_duration_ms: Milliseconds of silence before endpoint
        """
        self.threshold = threshold
        self.silence_duration_ms = silence_duration_ms
        logger.info(
            f"VAD reconfigured: threshold={threshold}, "
            f"silence_duration={silence_duration_ms}ms"
        )
