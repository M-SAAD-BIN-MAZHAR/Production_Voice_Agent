# 🎤 Real-Time Voice Agent

A production-ready, multi-provider voice agent system with streaming capabilities, intelligent fallback mechanisms, and sub-1.5s latency. Built with Python asyncio for maximum performance.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

### 🚀 Performance
- **Sub-1.5s Latency** - Optimized streaming pipeline
- **Parallel Processing** - LLM + TTS run concurrently
- **Response Caching** - Instant playback for repeated queries (24h TTL)
- **Streaming Architecture** - Real-time token-by-token processing

### 🎯 Intelligence
- **Multi-Provider LLM Support**
  - Pollinations AI (free, unlimited) - Primary
  - Mistral AI (fast, high quality) - Fallback
  - Hugging Face (open-source models)
  - OpenAI (GPT-4o-mini, GPT-4o)
- **Conversation Memory** - Vector-based context retrieval with FAISS
- **Context-Aware Responses** - Remembers last 10 turns with similarity search

### 🎙️ Audio Processing
- **Advanced VAD** - Silero VAD with configurable sensitivity
- **Barge-in Support** - Interrupt agent mid-response
- **Noise Reduction** - High-pass filter + noise gate
- **Multiple Input Modes**
  - Continuous listening
  - Push-to-talk (hold Space key)
  - Wake word detection (optional)

### 🔊 Voice Quality
- **Multi-Provider TTS**
  - OpenAI TTS (HD quality, multiple voices)
  - Pollinations TTS (free fallback)
  - ElevenLabs (premium natural voices)
  - Edge-TTS (free alternative)
- **Voice Options** - Male/female, multiple accents
- **Adjustable Speed** - 0.25x to 4x playback speed
- **Volume Control** - Configurable amplification

### 🛡️ Reliability
- **Automatic Fallback** - Seamless provider switching on failure
- **Error Recovery** - Graceful degradation
- **Component Isolation** - Async task-based architecture
- **Comprehensive Logging** - Debug and performance metrics

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐
│ Microphone  │───▶│ Audio   │───▶│   VAD    │───▶│   STT   │───▶│   LLM   │───▶│   TTS    │
│   Input     │    │   DSP   │    │ (Silero) │    │(Deepgram)│   │(Multi)  │    │ (Multi)  │
└─────────────┘    └─────────┘    └──────────┘    └──────────┘    └─────────┘    └──────────┘
                                         │                               │              │
                                         │                               ▼              │
                                         │                          ┌─────────┐         │
                                         │                          │ Memory  │         │
                                         │                          │ Manager │         │
                                         │                          └─────────┘         │
                                         │                               │              │
                                         ▼                               ▼              ▼
┌─────────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
│   Speaker   │◀───│  Audio  │◀───│ Interrupt│◀───│ Response │◀───│  Voice  │◀───│ Terminal │
│   Output    │    │ Player  │    │ Manager  │    │  Cache   │    │  Agent  │    │    UI    │
└─────────────┘    └─────────┘    └──────────┘    └──────────┘    └─────────┘    └──────────┘
```

### Component Details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Audio Input** | sounddevice | Captures microphone audio at 16kHz |
| **Audio DSP** | scipy | High-pass filter + noise gate |
| **VAD** | Silero VAD | Detects speech start/end |
| **STT** | Deepgram | Real-time speech-to-text (WebSocket) |
| **LLM** | Multi-provider | Generates intelligent responses |
| **Memory** | FAISS + OpenAI Embeddings | Vector-based context retrieval |
| **TTS** | Multi-provider | Text-to-speech synthesis |
| **Audio Output** | sounddevice | Plays audio at 24kHz |
| **Orchestrator** | asyncio | Coordinates all components |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Microphone and speakers
- API keys (see Configuration section)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/M-SAAD-BIN-MAZHAR/Production_Voice_Agent.git
cd Production_Voice_Agent
```

2. **Create virtual environment**
```bash
python -m venv venv
```

3. **Activate virtual environment**
```bash
# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Configure API keys**

Copy the example config:
```bash
cp config/config.example.yaml config/config.yaml
```

Or use environment variables (copy `.env.example` to `.env`):
```bash
cp config/.env.example config/.env
```

Edit `config/config.yaml` or `config/.env` and add your API keys:
- `DEEPGRAM_API_KEY` - Get from [Deepgram Console](https://console.deepgram.com/)
- `OPENAI_API_KEY` - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- `MISTRAL_API_KEY` - Get from [Mistral AI](https://console.mistral.ai/)
- `HUGGINGFACE_API_KEY` - Get from [Hugging Face](https://huggingface.co/settings/tokens)

### Run

```bash
python main.py
```

Press `Ctrl+C` to stop.

## ⚙️ Configuration

### LLM Providers

The system supports multiple LLM providers with automatic fallback:

```yaml
# Primary provider (choose one):
llm_provider: "pollinations"  # Free, unlimited (recommended)
# llm_provider: "mistral"     # Fast, high quality
# llm_provider: "huggingface" # Open-source models
# llm_provider: "openai"      # Best quality

# Fallback provider:
llm_fallback_provider: "mistral"
llm_fallback_model: "mistral-small-latest"
```

**Available Models:**

| Provider | Models | Cost | Rate Limits |
|----------|--------|------|-------------|
| **Pollinations** | openai, mistral, llama | FREE | Unlimited |
| **Mistral** | mistral-small-latest, mistral-large-latest | Paid | Good limits |
| **Hugging Face** | meta-llama/Llama-3.2-3B-Instruct, etc. | FREE | ~1000/day |
| **OpenAI** | gpt-4o-mini, gpt-4o | Paid | 3/min (free tier) |

### TTS Providers

```yaml
# Primary TTS provider:
tts_provider: "openai"  # Best quality (recommended)
# tts_provider: "pollinations"  # Free alternative
# tts_provider: "elevenlabs"    # Most natural voices
# tts_provider: "edge-tts"      # Free, good quality

# Fallback TTS:
tts_fallback_provider: "pollinations"
```

**Available Voices:**

| Provider | Voices | Quality |
|----------|--------|---------|
| **OpenAI** | alloy, echo, fable, onyx, nova, shimmer | Excellent |
| **Pollinations** | Same as OpenAI | Good |
| **ElevenLabs** | bella, rachel, adam, arnold, etc. | Best |
| **Edge-TTS** | en-US-AriaNeural, etc. | Good |

### Voice Activity Detection

```yaml
# Speech detection sensitivity (0.0 to 1.0)
vad_threshold: 0.88  # Higher = less sensitive to background noise

# Silence duration before considering speech ended
vad_silence_duration_ms: 1300  # Milliseconds
```

### Barge-in (Interruption)

```yaml
# Enable interrupting the agent while speaking
barge_in_on: "speech_continue"  # Requires sustained speech
# barge_in_on: "speech_start"   # Faster but more false triggers

# Minimum confidence to trigger interrupt
barge_in_min_confidence: 0.82  # 0.0 to 1.0

# Minimum time between interrupts
interrupt_debounce_ms: 5000  # Milliseconds
```

### Input Modes

```yaml
# Choose input mode:
voice_input_mode: "continuous"     # Always listening (default)
# voice_input_mode: "push_to_talk" # Hold Space key to talk
# voice_input_mode: "wake_word"    # Say "Alexa" to activate
```

### Performance Tuning

```yaml
# For SPEED (lower latency):
chunk_duration_ms: 50
vad_silence_duration_ms: 800
llm_model: "mistral-small-latest"

# For QUALITY (better responses):
tts_model: "tts-1-hd"
llm_model: "mistral-large-latest"
memory_max_turns: 20
```

## 📁 Project Structure

```
Production_Voice_Agent/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── config/                # Configuration files
│   ├── config.yaml        # Main config (create from example)
│   ├── config.example.yaml
│   ├── .env               # Environment variables (create from example)
│   └── .env.example
│
├── src/                   # Source code
│   ├── voice_agent.py     # Main orchestrator
│   ├── audio_input.py     # Microphone capture
│   ├── audio_output.py    # Speaker playback
│   ├── audio_dsp.py       # Audio processing (filters, noise gate)
│   ├── vad.py             # Voice activity detection
│   ├── stt_service.py     # Speech-to-text (Deepgram)
│   ├── llm_service.py     # LLM providers (multi-provider)
│   ├── tts_service.py     # Text-to-speech (multi-provider)
│   ├── memory_manager.py  # Conversation memory (FAISS)
│   ├── response_cache.py  # Response caching
│   ├── interrupt_manager.py  # Barge-in handling
│   ├── terminal_ui.py     # Terminal interface
│   ├── config_manager.py  # Configuration management
│   ├── latency_metrics.py # Performance monitoring
│   ├── input_controls.py  # Push-to-talk, wake word
│   └── models.py          # Data models
│
├── tests/                 # Unit tests
│   ├── test_audio_dsp.py
│   ├── test_audio_input.py
│   ├── test_audio_output.py
│   ├── test_llm_service.py
│   ├── test_stt_service.py
│   ├── test_tts_service.py
│   └── test_vad.py
│
└── data/                  # Runtime data
    ├── conversation_memory/  # FAISS vector store
    └── response_cache/       # Cached responses
```

## 🔧 Advanced Features

### Response Caching

Frequently asked questions are cached for instant playback:

```python
# Automatic caching (24-hour TTL)
- Text responses cached by query
- Audio responses cached by text
- Instant playback on cache hit
```

### Conversation Memory

Context-aware responses using vector similarity:

```yaml
memory_max_turns: 10           # Store last 10 turns
memory_similarity_top_k: 5     # Retrieve 5 similar turns
memory_recent_turns: 3         # Always include 3 recent turns
```

### Latency Metrics

Track performance in real-time:

```yaml
metrics_enabled: true  # Log latency metrics

# Metrics tracked:
- Time to first LLM token
- Time to first TTS audio chunk
- Total turn duration
- Cache hit rate
```

### Push-to-Talk

Hold Space key while speaking:

```yaml
voice_input_mode: "push_to_talk"
push_to_talk_key: "space"
push_to_talk_after_response_open_seconds: 25  # Auto-open after response
```

### Wake Word Detection

Say "Alexa" to activate:

```yaml
voice_input_mode: "wake_word"
wake_word_models: ["alexa"]
wake_word_sensitivity: 0.5
wake_word_window_seconds: 30  # Stay active for 30s
```

## 🐛 Troubleshooting

### Common Issues

**1. No audio input/output**
```bash
# List available devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Set default device in config
# Or use environment variables
```

**2. Rate limit errors (OpenAI)**
```yaml
# Switch to free providers:
llm_provider: "pollinations"  # Unlimited
tts_provider: "pollinations"  # Unlimited
```

**3. High latency**
```yaml
# Optimize for speed:
chunk_duration_ms: 50
vad_silence_duration_ms: 800
llm_model: "mistral-small-latest"
```

**4. False speech detection**
```yaml
# Increase VAD threshold:
vad_threshold: 0.90  # Less sensitive
barge_in_min_confidence: 0.85
```

**5. Agent not responding**
- Check API keys are valid
- Verify internet connection
- Check logs for errors: `LOG_LEVEL=DEBUG python main.py`

## 📊 Performance

Typical latency breakdown (optimized setup):

| Stage | Time | Notes |
|-------|------|-------|
| Speech Detection | 50-200ms | VAD processing |
| Speech-to-Text | 200-500ms | Deepgram streaming |
| LLM Generation | 300-800ms | First token latency |
| Text-to-Speech | 200-500ms | First audio chunk |
| **Total** | **750-2000ms** | Target: <1500ms |

With caching: **<100ms** for repeated queries

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Deepgram](https://deepgram.com/) - Speech-to-text API
- [OpenAI](https://openai.com/) - GPT models and TTS
- [Mistral AI](https://mistral.ai/) - Fast, high-quality LLM
- [Pollinations AI](https://pollinations.ai/) - Free LLM and TTS
- [Hugging Face](https://huggingface.co/) - Open-source models
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/M-SAAD-BIN-MAZHAR/Production_Voice_Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/M-SAAD-BIN-MAZHAR/Production_Voice_Agent/discussions)

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Voice cloning
- [ ] Web interface
- [ ] Mobile app
- [ ] Docker deployment
- [ ] Cloud hosting support
- [ ] Custom wake words
- [ ] Emotion detection
- [ ] Multi-modal input (text + voice)

---
![alt text](<ChatGPT Image Apr 20, 2026, 12_48_16 AM.png>)