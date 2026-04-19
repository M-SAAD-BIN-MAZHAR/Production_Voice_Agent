"""Configuration manager with validation.

Manages loading, validation, and saving of configuration from multiple sources:
- YAML/JSON configuration files
- Environment variables
- Default values

Priority: Environment variables > Config file > Defaults

Key Features:
- Multi-source configuration loading
- Pydantic validation
- Type conversion
- Required field validation
- Provider-specific validation
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import json
from pydantic import BaseModel, ConfigDict, Field, field_validator
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Configuration for the voice agent system."""
    
    model_config = ConfigDict(extra="ignore")
    
    # STT Configuration
    stt_provider: str = Field(default="deepgram", description="STT provider (deepgram/google)")
    deepgram_api_key: Optional[str] = Field(default=None, description="Deepgram API key")
    stt_model: str = Field(default="nova-2", description="STT model name")
    stt_language: str = Field(default="en-US", description="STT language code")
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider (openai/huggingface/pollinations/mistral)")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="Hugging Face API key")
    mistral_api_key: Optional[str] = Field(default=None, description="Mistral AI API key")
    pollinations_api_key: Optional[str] = Field(default=None, description="Pollinations API key (optional)")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model")
    llm_fallback_provider: Optional[str] = Field(default=None, description="Fallback LLM provider")
    llm_fallback_model: Optional[str] = Field(default=None, description="Fallback LLM model")
    system_prompt: str = Field(
        default="You are a helpful voice assistant. Provide concise, natural responses suitable for speech output.",
        description="System prompt for LLM"
    )
    
    # TTS Configuration
    tts_provider: str = Field(default="openai", description="TTS provider (openai/elevenlabs/edge-tts/pollinations)")
    tts_voice: str = Field(default="alloy", description="TTS voice name")
    tts_model: str = Field(default="tts-1", description="TTS model (tts-1/tts-1-hd)")
    tts_fallback_provider: Optional[str] = Field(default=None, description="Fallback TTS provider")
    tts_fallback_voice: Optional[str] = Field(default=None, description="Fallback TTS voice")
    tts_fallback_model: Optional[str] = Field(default=None, description="Fallback TTS model")
    elevenlabs_api_key: Optional[str] = Field(default=None, description="ElevenLabs API key")
    
    # VAD Configuration
    vad_provider: str = Field(default="silero", description="VAD provider (silero/deepgram)")
    vad_threshold: float = Field(
        default=0.62,
        description="VAD speech probability threshold; higher = less sensitive to background/TV",
    )
    vad_silence_duration_ms: int = Field(default=700, description="Silence duration before endpoint (ms)")
    
    # Barge-in (interrupt while agent speaks)
    barge_in_min_confidence: float = Field(
        default=0.78,
        description="Min VAD confidence (0-1) to count as user speech during barge-in",
    )
    interrupt_debounce_ms: int = Field(
        default=4500,
        description="Min milliseconds between barge-in interrupts",
    )
    barge_in_on: str = Field(
        default="speech_continue",
        description='Barge-in trigger: "speech_start" (faster, more false triggers) or '
        '"speech_continue" (needs sustained speech; fewer false triggers)',
    )
    
    # Audio Configuration
    sample_rate: int = Field(default=16000, description="Audio sample rate (Hz)")
    chunk_duration_ms: int = Field(default=100, description="Audio chunk duration (ms)")
    
    # Microphone DSP (reduces rumble/hiss; not full AEC without playback reference)
    input_dsp: str = Field(
        default="none",
        description="none | highpass | noise_gate | highpass_noise_gate",
    )
    input_dsp_highpass_hz: float = Field(default=80.0, description="High-pass corner frequency (Hz)")
    input_dsp_noise_gate_rms: float = Field(
        default=800.0,
        description="RMS below this (approx) gets attenuated in noise_gate modes",
    )
    
    # Input gating (noisy environments)
    voice_input_mode: str = Field(
        default="continuous",
        description="continuous | push_to_talk | wake_word",
    )
    push_to_talk_key: str = Field(default="space", description="Key name for push_to_talk (e.g. space)")
    push_to_talk_stdin_arm_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Press Enter in the terminal to arm mic for N seconds (0=off); helps IDE terminals where Space may not reach pynput",
    )
    push_to_talk_startup_open_seconds: float = Field(
        default=12.0,
        ge=0.0,
        description="Seconds the mic gate is open right after start (0=off); avoids silent PTT when IDE keys/stdin fail",
    )
    push_to_talk_after_response_open_seconds: float = Field(
        default=25.0,
        ge=0.0,
        description="Re-open mic gate for this many seconds after each reply (and after barge-in); 0=off (Enter/Space only)",
    )
    wake_word_models: list[str] = Field(
        default_factory=lambda: ["alexa"],
        description="OpenWakeWord model names to load",
    )
    wake_word_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    wake_word_window_seconds: float = Field(
        default=30.0,
        description="Seconds to keep listening after wake word fires",
    )
    
    # Latency observability
    metrics_enabled: bool = Field(default=True, description="Log LLM/TTS latency metrics per turn")
    
    # Memory Configuration
    memory_max_turns: int = Field(default=10, description="Maximum conversation turns to store")
    memory_similarity_top_k: int = Field(default=5, description="Number of similar turns to retrieve")
    memory_recent_turns: int = Field(default=3, description="Number of recent turns to include")
    vector_db_path: str = Field(default="./data/conversation_memory", description="Path to FAISS index")
    
    # Performance Configuration
    target_latency_ms: int = Field(default=1500, description="Target end-to-end latency (ms)")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator("vad_threshold")
    @classmethod
    def validate_vad_threshold(cls, v: float) -> float:
        """Validate VAD threshold is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("vad_threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("barge_in_min_confidence")
    @classmethod
    def validate_barge_in_min_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("barge_in_min_confidence must be between 0.0 and 1.0")
        return v
    
    @field_validator("barge_in_on")
    @classmethod
    def validate_barge_in_on(cls, v: str) -> str:
        allowed = {"speech_start", "speech_continue"}
        if v not in allowed:
            raise ValueError(f'barge_in_on must be one of {allowed}')
        return v
    
    @field_validator("voice_input_mode")
    @classmethod
    def validate_voice_input_mode(cls, v: str) -> str:
        allowed = {"continuous", "push_to_talk", "wake_word"}
        if v not in allowed:
            raise ValueError(f"voice_input_mode must be one of {allowed}")
        return v
    
    @field_validator("input_dsp")
    @classmethod
    def validate_input_dsp(cls, v: str) -> str:
        allowed = {"none", "highpass", "noise_gate", "highpass_noise_gate"}
        if v not in allowed:
            raise ValueError(f"input_dsp must be one of {allowed}")
        return v
    
    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate sample rate is a common value."""
        valid_rates = [8000, 16000, 24000, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"sample_rate must be one of {valid_rates}")
        return v
    
    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        valid_providers = ["openai", "huggingface", "pollinations", "mistral"]
        if v not in valid_providers:
            raise ValueError(f"llm_provider must be one of {valid_providers}")
        return v
    
    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, v: str) -> str:
        """Validate LLM model name."""
        valid_openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        # For Hugging Face, we allow any model name (too many to validate)
        if v in valid_openai_models or "/" in v:  # HF models have format "org/model"
            return v
        logger.warning(f"LLM model '{v}' not in known OpenAI models: {valid_openai_models}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper


class ConfigManager:
    """Manages configuration loading and validation."""
    
    @staticmethod
    def load(config_path: Optional[str] = None) -> Config:
        """
        Load configuration from file and environment variables.
        
        Priority: environment variables > config file > defaults
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Config: Validated configuration object
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file specified but not found
        """
        # Load environment variables from .env file
        load_dotenv()
        
        config_dict: Dict[str, Any] = {}
        
        # Load from config file if provided
        if config_path:
            config_dict = ConfigManager._load_from_file(config_path)
        else:
            # Try default config paths
            default_paths = [
                "config/config.yaml",
                "config/config.json",
                "config.yaml",
                "config.json"
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    logger.info(f"Loading configuration from {path}")
                    config_dict = ConfigManager._load_from_file(path)
                    break
        
        # Override with environment variables
        env_overrides = ConfigManager._load_from_env()
        config_dict.update(env_overrides)
        
        # Validate and create Config object
        try:
            config = Config(**config_dict)
            ConfigManager.validate(config)
            logger.info("Configuration loaded and validated successfully")
            return config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    @staticmethod
    def _load_from_file(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary of configuration values
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config_dict or {}
    
    @staticmethod
    def _load_from_env() -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Dictionary of configuration values from environment
        """
        env_mapping = {
            "DEEPGRAM_API_KEY": "deepgram_api_key",
            "OPENAI_API_KEY": "openai_api_key",
            "HUGGINGFACE_API_KEY": "huggingface_api_key",
            "MISTRAL_API_KEY": "mistral_api_key",
            "POLLINATIONS_API_KEY": "pollinations_api_key",
            "LLM_PROVIDER": "llm_provider",
            "LLM_FALLBACK_PROVIDER": "llm_fallback_provider",
            "LLM_FALLBACK_MODEL": "llm_fallback_model",
            "TTS_FALLBACK_PROVIDER": "tts_fallback_provider",
            "TTS_FALLBACK_VOICE": "tts_fallback_voice",
            "TTS_FALLBACK_MODEL": "tts_fallback_model",
            "STT_PROVIDER": "stt_provider",
            "STT_MODEL": "stt_model",
            "STT_LANGUAGE": "stt_language",
            "LLM_MODEL": "llm_model",
            "SYSTEM_PROMPT": "system_prompt",
            "TTS_PROVIDER": "tts_provider",
            "TTS_VOICE": "tts_voice",
            "TTS_MODEL": "tts_model",
            "VAD_PROVIDER": "vad_provider",
            "VAD_THRESHOLD": "vad_threshold",
            "VAD_SILENCE_DURATION_MS": "vad_silence_duration_ms",
            "BARGE_IN_MIN_CONFIDENCE": "barge_in_min_confidence",
            "INTERRUPT_DEBOUNCE_MS": "interrupt_debounce_ms",
            "BARGE_IN_ON": "barge_in_on",
            "VOICE_INPUT_MODE": "voice_input_mode",
            "PUSH_TO_TALK_KEY": "push_to_talk_key",
            "PUSH_TO_TALK_STDIN_ARM_SECONDS": "push_to_talk_stdin_arm_seconds",
            "PUSH_TO_TALK_STARTUP_OPEN_SECONDS": "push_to_talk_startup_open_seconds",
            "PUSH_TO_TALK_AFTER_RESPONSE_OPEN_SECONDS": "push_to_talk_after_response_open_seconds",
            "INPUT_DSP": "input_dsp",
            "INPUT_DSP_HIGHPASS_HZ": "input_dsp_highpass_hz",
            "INPUT_DSP_NOISE_GATE_RMS": "input_dsp_noise_gate_rms",
            "WAKE_WORD_SENSITIVITY": "wake_word_sensitivity",
            "WAKE_WORD_WINDOW_SECONDS": "wake_word_window_seconds",
            "METRICS_ENABLED": "metrics_enabled",
            "SAMPLE_RATE": "sample_rate",
            "CHUNK_DURATION_MS": "chunk_duration_ms",
            "MEMORY_MAX_TURNS": "memory_max_turns",
            "MEMORY_SIMILARITY_TOP_K": "memory_similarity_top_k",
            "MEMORY_RECENT_TURNS": "memory_recent_turns",
            "VECTOR_DB_PATH": "vector_db_path",
            "TARGET_LATENCY_MS": "target_latency_ms",
            "LOG_LEVEL": "log_level"
        }
        
        config_dict = {}
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion for numeric values
                if config_key in [
                    "vad_threshold",
                    "barge_in_min_confidence",
                    "input_dsp_highpass_hz",
                    "input_dsp_noise_gate_rms",
                    "wake_word_sensitivity",
                    "wake_word_window_seconds",
                    "push_to_talk_stdin_arm_seconds",
                    "push_to_talk_startup_open_seconds",
                    "push_to_talk_after_response_open_seconds",
                ]:
                    config_dict[config_key] = float(value)
                elif config_key in [
                    "vad_silence_duration_ms", "sample_rate", "chunk_duration_ms",
                    "memory_max_turns", "memory_similarity_top_k", "memory_recent_turns",
                    "target_latency_ms", "interrupt_debounce_ms"
                ]:
                    config_dict[config_key] = int(value)
                elif config_key == "metrics_enabled":
                    config_dict[config_key] = value.lower() in ("1", "true", "yes")
                else:
                    config_dict[config_key] = value
        
        if config_dict:
            logger.debug(f"Loaded {len(config_dict)} values from environment variables")
        
        return config_dict
    
    @staticmethod
    def validate(config: Config) -> None:
        """
        Validate configuration for required fields and consistency.
        
        Args:
            config: Configuration object to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        errors = []
        
        # Validate required API keys based on providers
        if config.stt_provider == "deepgram" and not config.deepgram_api_key:
            errors.append("deepgram_api_key is required when stt_provider is 'deepgram'")
        
        if config.llm_provider == "openai" and not config.openai_api_key:
            errors.append("openai_api_key is required when llm_provider is 'openai'")
        
        if config.llm_provider == "huggingface" and not config.huggingface_api_key:
            errors.append("huggingface_api_key is required when llm_provider is 'huggingface'")
        
        if config.llm_provider == "mistral" and not config.mistral_api_key:
            errors.append("mistral_api_key is required when llm_provider is 'mistral'")
        
        # Pollinations doesn't require API key for basic usage
        
        # OpenAI key still needed for embeddings in memory manager (unless using Pollinations only)
        if not config.openai_api_key and config.llm_provider == "openai":
            errors.append("openai_api_key is required for LLM and embeddings")
        
        if config.tts_provider == "openai" and not config.openai_api_key:
            errors.append("openai_api_key is required when tts_provider is 'openai'")
        
        # Validate provider values
        valid_stt_providers = ["deepgram", "google"]
        if config.stt_provider not in valid_stt_providers:
            errors.append(f"stt_provider must be one of {valid_stt_providers}")
        
        valid_tts_providers = ["openai", "elevenlabs", "edge-tts", "pollinations"]
        if config.tts_provider not in valid_tts_providers:
            errors.append(f"tts_provider must be one of {valid_tts_providers}")
        
        if config.tts_provider == "elevenlabs" and not config.elevenlabs_api_key:
            errors.append("elevenlabs_api_key is required when tts_provider is 'elevenlabs'")
        
        valid_vad_providers = ["silero", "deepgram"]
        if config.vad_provider not in valid_vad_providers:
            errors.append(f"vad_provider must be one of {valid_vad_providers}")
        
        # Raise all errors together
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        
        logger.debug("Configuration validation passed")
    
    @staticmethod
    def save(config: Config, config_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
            config_path: Path to save configuration file
        """
        path = Path(config_path)
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = config.model_dump()
        
        # Save based on file extension
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        logger.info(f"Configuration saved to {config_path}")
