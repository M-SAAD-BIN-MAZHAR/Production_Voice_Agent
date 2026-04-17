# Production_Voice_Agent

Real-time voice agent system with streaming speech recognition, LLM processing, and text-to-speech synthesis.

## Features

- **Real-time Speech Recognition** - Deepgram streaming STT
- **Intelligent Responses** - OpenAI GPT-4o-mini
- **Natural Voice Output** - OpenAI TTS (Nova voice)
- **Voice Activity Detection** - Silero VAD
- **Response Caching** - 24-hour cache for instant responses
- **Parallel Processing** - LLM + TTS sentence-by-sentence synthesis
- **Interrupt Handling** - Stop agent mid-response
- **Conversation Memory** - Vector-based context retrieval

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `config/config.example.yaml` to `config/config.yaml`
6. Add your API keys to `config/config.yaml`:
   - `openai_api_key` - Get from https://platform.openai.com/api-keys
   - `deepgram_api_key` - Get from https://console.deepgram.com/

## Usage

```bash
python main.py
```

## Architecture

- **Audio Input** → **VAD** → **STT** → **LLM** → **TTS** → **Audio Output**
- Async/await architecture with component isolation
- Queue-based inter-component communication
- Graceful error handling and recovery

## Requirements

- Python 3.10+
- Microphone and speakers
- OpenAI API key
- Deepgram API key

## Configuration

Edit `config/config.yaml` to customize:
- VAD threshold and silence duration
- TTS voice and model
- LLM model and system prompt
- Memory and caching settings
