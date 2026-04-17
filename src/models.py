"""Data models for the real-time voice agent system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    data: np.ndarray  # Audio samples (int16)
    sample_rate: int  # Samples per second
    timestamp: float  # Unix timestamp
    duration_ms: int  # Chunk duration in milliseconds


class VADEventType(Enum):
    """Types of voice activity detection events."""
    SPEECH_START = "speech_start"
    SPEECH_CONTINUE = "speech_continue"
    SPEECH_END = "speech_end"
    SILENCE = "silence"


@dataclass
class VADEvent:
    """Voice activity detection event."""
    event_type: VADEventType
    timestamp: float
    confidence: float  # 0.0-1.0


@dataclass
class Transcript:
    """Speech-to-text transcript."""
    text: str
    is_final: bool
    confidence: float
    timestamp: float
    duration_ms: int  # Time from speech start


@dataclass
class Message:
    """Conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float


class SystemState(Enum):
    """System state enumeration."""
    INITIALIZING = "initializing"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    user_input: str
    assistant_response: str
    timestamp: float
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
