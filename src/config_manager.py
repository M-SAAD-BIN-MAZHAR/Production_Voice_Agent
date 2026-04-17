"""Configuration manager with validation."""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import json
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Configuration for the voice agent system."""
    
    # STT Configuration
    stt_provider: str = Field(default="deepgram", description="STT provider (deepgram/google)")
    deepgram_api_key: Optional[str] = Field(default=None, description="Deepgram API key")
    stt_model: str = Field(default="nova-2", description="STT model name")
    stt_language: str = Field(default="en-US", description="STT language code")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model (gpt-4o-mini/gpt-4o)")
    system_prompt: str = Field(
        default="You are a helpful voice assistant. Provide concise, natural responses suitable for speech output.",
        description="System prompt for LLM"
    )
    
    # TTS Configuration
    tts_provider: str = Field(default="openai", description="TTS provider (openai/elevenlabs/edge-tts)")
    tts_voice: str = Field(default="alloy", description="TTS voice name")
    tts_model: str = Field(default="tts-1", description="TTS model (tts-1/tts-1-hd)")
    elevenlabs_api_key: Optional[str] = Field(default=None, description="ElevenLabs API key")
    
    # VAD Configuration
    vad_provider: str = Field(default="silero", description="VAD provider (silero/deepgram)")
    vad_threshold: float = Field(default=0.5, description="VAD speech detection threshold (0.0-1.0)")
    vad_silence_duration_ms: int = Field(default=700, description="Silence duration before endpoint (ms)")
    
    # Audio Configuration
    sample_rate: int = Field(default=16000, description="Audio sample rate (Hz)")
    chunk_duration_ms: int = Field(default=100, description="Audio chunk duration (ms)")
    
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
    
    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate sample rate is a common value."""
        valid_rates = [8000, 16000, 24000, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"sample_rate must be one of {valid_rates}")
        return v
    
    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, v: str) -> str:
        """Validate LLM model name."""
        valid_models = ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        if v not in valid_models:
            logger.warning(f"LLM model '{v}' not in known models: {valid_models}")
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
                if config_key in ["vad_threshold"]:
                    config_dict[config_key] = float(value)
                elif config_key in [
                    "vad_silence_duration_ms", "sample_rate", "chunk_duration_ms",
                    "memory_max_turns", "memory_similarity_top_k", "memory_recent_turns",
                    "target_latency_ms"
                ]:
                    config_dict[config_key] = int(value)
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
        
        if not config.openai_api_key:
            errors.append("openai_api_key is required for LLM and embeddings")
        
        if config.tts_provider == "openai" and not config.openai_api_key:
            errors.append("openai_api_key is required when tts_provider is 'openai'")
        
        # Validate provider values
        valid_stt_providers = ["deepgram", "google"]
        if config.stt_provider not in valid_stt_providers:
            errors.append(f"stt_provider must be one of {valid_stt_providers}")
        
        valid_tts_providers = ["openai", "elevenlabs", "edge-tts"]
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
