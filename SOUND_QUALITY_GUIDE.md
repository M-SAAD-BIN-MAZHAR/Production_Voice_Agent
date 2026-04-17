# Sound Quality Configuration Guide

This guide explains all parameters that affect voice output quality in the Real-Time Voice Agent.

---

## 📋 Table of Contents
1. [Quick Settings](#quick-settings-most-common-adjustments)
2. [Advanced Settings](#advanced-settings)
3. [Common Scenarios](#common-scenarios)
4. [File Reference](#file-reference)
5. [All Commented Files](#all-commented-files)

---

## Quick Settings (Most Common Adjustments)

### 1. Voice Selection
**Location:** `config/config.yaml` → `tts_voice`

**Options:**
- `"onyx"` - Deep, authoritative male voice (CURRENT)
- `"echo"` - Regular male voice
- `"fable"` - British male voice
- `"alloy"` - Neutral, balanced female voice
- `"nova"` - Warm female voice
- `"shimmer"` - Soft female voice

### 2. Speech Speed
**Location:** `src/tts_service.py` → Line ~148 → `speed` parameter

**Values:**
- `0.85` - 15% slower (CURRENT - better clarity)
- `1.0` - Normal speed
- `1.2` - 20% faster
- Range: 0.25 to 4.0

### 3. Volume (Loudness)
**Location:** `src/tts_service.py` → Line ~177 → Volume multiplication factor

**Values:**
- `1.5` - 50% louder (CURRENT)
- `1.0` - Original volume
- `2.0` - 100% louder (may cause distortion)
- `0.8` - 20% quieter

**Code:**
```python
audio_data = np.clip(audio_data * 1.5, -32768, 32767).astype(np.int16)
#                              ^^^
#                         Change this value
```

---

## Advanced Settings

### 4. TTS Model Quality
**Location:** `config/config.yaml` → `tts_model`

**Options:**
- `"tts-1"` - Faster, good quality (CURRENT)
- `"tts-1-hd"` - Slower, higher quality

**Trade-off:** HD model has better audio quality but takes longer to generate.

### 5. Audio Chunk Size (Smoothness)
**Location:** `src/tts_service.py` → Line ~158 → `chunk_size`

**Values:**
- `4096` (4KB) - Low latency, may be choppy
- `8192` (8KB) - Balanced
- `16384` (16KB) - Smooth playback (CURRENT)

**Trade-off:** Larger chunks = smoother but slightly higher latency.

### 6. Audio Output Buffer (Playback Smoothness)
**Location:** `src/audio_output.py` → Line ~82 → `blocksize`

**Values:**
- `512` - Low latency, may be choppy
- `1024` - Balanced
- `2048` - Smooth playback (CURRENT)
- `4096` - Very smooth, higher latency

**Trade-off:** Larger buffer = smoother but more delay.

### 7. VAD Threshold (Speech Detection Sensitivity)
**Location:** `config/config.yaml` → `vad_threshold`

**Values:**
- `0.5-0.7` - More sensitive, picks up background noise
- `0.8-0.9` - Balanced
- `0.93` - Less sensitive, cleaner detection (CURRENT)

**Effect:** Higher threshold = cleaner audio input, less background noise.

### 8. VAD Silence Duration
**Location:** `config/config.yaml` → `vad_silence_duration_ms`

**Values:**
- `500-1000ms` - Faster response, may cut off speech
- `1300ms` - Balanced (CURRENT)
- `1500-2000ms` - More patient, won't cut off pauses

**Effect:** Longer duration = won't cut off natural pauses in speech.

### 9. Audio Sample Rates
**Locations:**
- Input: `config/config.yaml` → `sample_rate` (16000 Hz)
- Output: `src/audio_output.py` → `sample_rate` (24000 Hz)

**Input Sample Rate:**
- `8000 Hz` - Telephone quality
- `16000 Hz` - Wideband, good for speech (CURRENT)
- `24000 Hz` - High quality
- `48000 Hz` - Studio quality (overkill)

**Output Sample Rate:**
- `24000 Hz` - OpenAI TTS default (FIXED, don't change)

---

## Common Scenarios

### Make Voice Clearer
1. Reduce speed to `0.75` in `src/tts_service.py`
2. Use `"tts-1-hd"` model in `config/config.yaml`
3. Increase VAD threshold to `0.95` in `config/config.yaml`

### Make Voice Louder
1. Increase volume multiplier to `2.0` in `src/tts_service.py`
2. Check system volume settings

### Reduce Choppy/Interrupted Audio
1. Increase chunk_size to `32768` in `src/tts_service.py`
2. Increase blocksize to `4096` in `src/audio_output.py`
3. Close other audio applications

### Faster Response Time
1. Reduce chunk_size to `8192` in `src/tts_service.py`
2. Reduce blocksize to `1024` in `src/audio_output.py`
3. Use `"tts-1"` model instead of `"tts-1-hd"`

---

## File Reference

| Parameter | File | Line | Current Value |
|-----------|------|------|---------------|
| Voice | `config/config.yaml` | ~30 | `"onyx"` |
| TTS Model | `config/config.yaml` | ~31 | `"tts-1"` |
| Speed | `src/tts_service.py` | ~148 | `0.85` |
| Volume | `src/tts_service.py` | ~177 | `1.5` |
| TTS Chunk Size | `src/tts_service.py` | ~158 | `16384` |
| Output Blocksize | `src/audio_output.py` | ~82 | `2048` |
| VAD Threshold | `config/config.yaml` | ~48 | `0.93` |
| VAD Silence | `config/config.yaml` | ~49 | `1300` |
| Input Sample Rate | `config/config.yaml` | ~63 | `16000` |
| Output Sample Rate | `src/audio_output.py` | ~75 | `24000` |

---

## All Commented Files

Every file in the codebase now has comprehensive comments marked with emojis:

### 🔊 Sound Quality Parameters
Files with sound quality settings:
- `src/tts_service.py` - Speed, volume, chunk size, voice selection
- `src/audio_output.py` - Sample rate, buffer size, channels
- `src/audio_input.py` - Input sample rate, audio conversion
- `src/vad.py` - Speech detection threshold, silence duration
- `config/config.yaml` - All configurable sound parameters

### ⚡ Performance Parameters
Files with performance settings:
- `src/voice_agent.py` - Rate limiting, caching
- `src/response_cache.py` - Cache expiration, storage
- `src/memory_manager.py` - Context retrieval, vector search
- `src/interrupt_manager.py` - Debounce timing

### 🤖 LLM Parameters
Files with AI model settings:
- `src/llm_service.py` - Model selection, temperature, max tokens
- `src/memory_manager.py` - Conversation context management

### 📚 Documentation
Files with comprehensive module documentation:
- `src/models.py` - Data structures
- `src/config_manager.py` - Configuration loading
- `src/terminal_ui.py` - User interface
- `src/stt_service.py` - Speech recognition
- `main.py` - Application entry point

### 🔍 Comment Markers
Look for these emoji markers in the code:
- 🔊 = Sound quality parameter
- ⚡ = Performance parameter
- 🤖 = LLM/AI parameter
- 💾 = Storage/caching parameter
- ⚠️ = Important warning or limitation

---

## Notes

- All changes require restarting the application
- Volume multiplier above 2.0 may cause audio distortion
- Speed below 0.5 or above 2.0 may sound unnatural
- Always test changes incrementally
- Keep backups of working configurations
- Comments are marked with emojis for easy identification
- Each parameter includes recommended ranges and trade-offs
