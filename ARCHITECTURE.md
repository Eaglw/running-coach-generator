# 🏛️ Architecture & Agent Handover Notes

> **Document Version:** 1.0.0  
> **Target Audience:** AI Coding Agents & Backend/Audio Engineers resuming work on this codebase.

---

## 1. Executive Summary & Purpose

`running-coach-generator` is a self-hosted, single-container Python (FastAPI) application designed to solve a specific problem: **generating personalized, AI-coached running workouts mixed with high-energy music, and synchronizing them automatically to an Apple Watch for phone-free offline running.**

### The Core Flow
1. **Input:** The user supplies a running prompt (*"10m warmup @ 5:30, 20m tempo @ 4:40, 5m cooldown"*) or a structured JSON payload via the Web GUI / REST API.
2. **LLM Coaching:** An OpenAI-compatible LLM parses the prompt and writes context-aware spoken coaching cues in Italian according to a selected persona (*tecnico_motivazionale*, *calmo_mindful*, *militare_severo*).
3. **Neural TTS:** A neural TTS engine (Piper / Kokoro) synthesizes the cues into `.wav` audio files (with automatic on-demand ONNX model downloading).
4. **Music Matching:** The music indexer reads ID3 `TBPM` tags or runs fallback `librosa` spectral tempo analysis to pick tracks matching each segment's target BPM range without duplicates.
5. **FFmpeg Engine:** A two-step disk-stream pipeline builds segment-aligned music beds with crossfades, places voice cues, applies dynamic sidechain ducking, and encodes to MP3.
6. **Publication:** The episode is added to `feed.xml` conforming to iTunes/Apple Podcasts specs with `HTTP Range` byte-serving for Apple Watch offline syncing.

---

## 2. Directory Structure & Component Roles

```text
running-coach-generator/
├── Dockerfile                  # python:3.11-slim + ffmpeg, espeak-ng, libsndfile1
├── docker-compose.yml          # Container service definition with ./data volume binding
├── requirements.txt            # FastAPI, Uvicorn, Pydantic v2, mutagen, librosa, piper-tts
├── .env.example                # Canonical environment variable defaults
├── README.md                   # User-facing English documentation
├── ARCHITECTURE.md             # This document (technical context for agents)
├── app/
│   ├── main.py                 # FastAPI app, background workers, REST routes, static mounts
│   ├── config.py               # Pydantic BaseSettings, path resolution & ensure_directories()
│   ├── models.py               # Pydantic v2 data contracts (WorkoutRequest, Segment, CoachCue)
│   ├── static/
│   │   └── index.html          # Single-Page Web Dashboard (Tailwind CSS dark mode)
│   └── services/
│       ├── audio_mixer.py      # Two-step FFmpeg engine (crossfades, ducking, MP3 mixdown)
│       ├── library_indexer.py  # ID3 extraction, librosa fallback BPM & segment track selection
│       ├── llm_coach.py        # Natural language parser & Italian cue script generator
│       ├── tts_engine.py       # Strategy pattern: Piper, Kokoro & system fallback
│       └── rss_generator.py    # Atomic RFC 2822 iTunes-compliant feed.xml generator
└── data/                       # Host-mounted storage
    ├── music/                  # Source MP3/FLAC music tracks
    ├── cache/                  # library_index.json & tts_models/ (auto-downloaded ONNX models)
    ├── podcasts/               # feed.xml & episodes/*.mp3
    └── temp/                   # Temporary session files (auto-purged)
```

---

## 3. Critical Technical Decisions & Bugfixes Applied

When maintaining or extending this codebase, **do not revert** these design decisions:

### A. Segment-to-Music Alignment (Strict BPM Sync & Zero Drift)
* **The Problem:** Passing a flat `List[Path]` to FFmpeg causes tracks from Segment 1 (e.g. Warmup at 120 BPM) to spill over into Segment 2 (e.g. Threshold at 165 BPM) if track durations do not align perfectly with segment boundaries.
* **The Architecture:**
  - `library_indexer.select_tracks_for_segments` returns `List[Tuple[int, List[Path]]]`, representing `[(segment_duration_sec, [tracks]), ...]`.
  - In `audio_mixer.generate_music_bed`, tracks within each segment are crossfaded (`acrossfade=d=3`), guaranteed with `apad,atrim=0:{seg_dur}` to match the exact segment duration down to the millisecond, and smoothed at the boundaries with a 1.5s fade-in and fade-out (`afade=t=in:st=0:d=1.5,afade=t=out:st={max(0, seg_dur - 1.5)}:d=1.5`).
  - Concatenation of segment sub-beds is done via `concat=n={num_segments}:v=0:a=1` so BPM transitions align strictly with segment boundaries and coach cues with zero audio drift.

### B. Dedicated Stream Allocation (Zero Stream Reuse Crash)
* **The Problem:** Reusing track input streams `[in_idx:a]` multiple times without `asplit` when a song is repeated causes FFmpeg to crash with `Invalid argument` (stream already consumed).
* **The Fix:** `generate_music_bed` never deduplicates input files; it passes every track occurrence as an independent `-i` argument, ensuring every segment has dedicated input stream indices.

### C. Zero-RAM Policy & Removal of `aloop`
* **The Problem:** FFmpeg's `aloop=loop=-1:size=2e+09` buffers gigabytes of uncompressed PCM in RAM, causing Docker OOM kills. Furthermore, `aloop=size=44100*10` only loops the first 10 seconds of a track.
* **The Architecture:** `aloop` is eliminated completely. If tracks in a segment are shorter than the segment's required duration, `library_indexer.py` and `audio_mixer.py` cycle and repeat the tracks in the list until `seg_dur` is covered, and `apad,atrim=0:{seg_dur}` guarantees the exact duration.

### D. Sidechain Ducking & Unity Gain Conservation
* When coach speech is playing, the background music bed volume drops by approximately -18dB smoothly and recovers cleanly when voice stops.
* FFmpeg Filtergraph implementation:
  1. Voice cues are positioned on a silent background track via `adelay={offset_ms}` and combined via `amix`.
  2. `sidechaincompress` smoothly attenuates the music bed when speech is detected:
     `sidechaincompress=threshold=0.015:ratio=6:attack=50:release=600`
  3. `amix` uses `normalize=0` with `duration=first` for unity gain preservation:
     `amix=inputs=2:duration=first:dropout_transition=0:normalize=0`

### E. Real Neural TTS with In-Memory ONNX Model Caching
* `PiperTTSEngine` and `KokoroTTSEngine` cache their loaded model instances (`_loaded_voice_cache` and `_kokoro_instance_cache`) in memory, avoiding reloading multi-megabyte ONNX graphs and weights for every cue.
* Automatically downloads `.onnx` and `.onnx.json` from Hugging Face on demand if not present.
* Silent fallback to `SystemFallbackTTSEngine` ensures synthesis never crashes even if external dependencies are missing.

### F. Lifecycle Resource Management & Cover Artwork
* Temporary session directories (`data/temp/job_{job_id}`) containing intermediate `cue_*.wav` files are strictly cleaned up via `finally: shutil.rmtree(...)` in `_process_workout_job`.
* On startup, `lifespan` ensures a compliant 1400x1400 podcast cover art (`data/podcasts/cover.jpg`) is generated via FFmpeg if not already present, preventing 404 errors during Apple Podcasts validation.

### G. Apple Watch Byte-Serving (`HTTP Range`)
* Apple Watch podcast sync requires byte-range requests (`Accept-Ranges: bytes`).
* FastAPI's `StaticFiles` natively supports `HTTP Range`. Thus, `/podcast` is mounted via `StaticFiles(directory=settings.PODCAST_DIR)`.
* An explicit `GET /podcast/feed.xml` route is declared before the mount to guarantee `Content-Type: application/rss+xml`.

### H. Multi-Provider LLM Dispatch & Settings Persistence
* `LLMCoach` (`app/services/llm_coach.py`) automatically dispatches requests according to `settings.LLM_PROVIDER`:
  - **Ollama & LM Studio:** Uses `openai.OpenAI` pointing to `http://localhost:11434/v1` or `http://localhost:1234/v1` with automatic fallback when local models reject strict `json_object` enforcement.
  - **Google Gemini & OpenAI:** Calls official OpenAI-compatible endpoints with structured JSON schemas.
  - **Anthropic Claude:** Executes direct HTTP calls to the Anthropic Messages API (`https://api.anthropic.com/v1/messages`) with headers `x-api-key` and `anthropic-version: 2023-06-01` without requiring third-party Anthropic SDK bloat.
  - **Connection Testing:** `POST /api/v1/settings/test-llm` sends a lightweight ping prompt to measure roundtrip latency and catch model configuration issues prior to running workouts.
* Settings edited in the Web GUI are saved to `/data/cache/settings.json` via `settings.save_dynamic_settings()`, surviving container rebuilds and server restarts.

### I. Dynamic Music Directory Configuration
* In addition to the default `./data/music` container directory, `Settings.MUSIC_DIR` is dynamically configurable at runtime via the Web GUI and `POST /api/v1/settings`.
* `LibraryIndexer.music_dir` is implemented as a dynamic property referencing `settings.MUSIC_DIR`. This allows scanning any host folder (such as a local Windows user music library `C:\Users\<user>\Music`) seamlessly without code modifications.

---

## 4. Background Job Pipeline Flow

When `POST /api/v1/workout/from-text` or `POST /api/v1/workout/from-json` is called:
```text
[Client] -> 202 Accepted {job_id, status: "queued"}
                 │
                 ▼ (FastAPI BackgroundTasks)
      1. GENERATING_SCRIPT (progress: 0.15)
         └── llm_coach.generate_spoken_script()
      2. SYNTHESIZING_TTS (progress: 0.35)
         └── tts_engine.synthesize() for each cue
         └── global_offset_ms = (segment_start_sec + cue.offset_sec) * 1000
      3. MIXING_AUDIO (progress: 0.60 -> 0.80)
         └── library_indexer.select_tracks_for_segments()
         └── audio_mixer.render_workout_audio() (Step A + Step B)
      4. RSS PUBLICATION (progress: 0.95)
         └── rss_generator.add_episode() -> updates feed.xml atomically
      5. COMPLETED (progress: 1.0)
         └── episode_url = f"{BASE_URL}/podcast/episodes/{filename}"
```

---

## 5. Standalone Verification & Testing

To test components in isolation without launching the full server:

```bash
# 1. Test the FFmpeg 2-step Ducking & Mixing Engine
python -m app.services.audio_mixer

# 2. Test End-to-End API and Job Execution
python -c "from fastapi.testclient import TestClient; from app.main import app; c = TestClient(app); print(c.get('/health').json())"

# 3. Start local development server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 6. Recommended Future Enhancements

If you are an agent tasked with extending this project, consider these natural additions:
1. **Web GUI Music Upload:** Add drag-and-drop file upload in the Web GUI (`/api/v1/library/upload`) directly to `/data/music`.
2. **GPX / Apple HealthKit Sync:** Allow importing a running route GPX file to synchronize coach cues with elevation changes or kilometer markers.
3. **Voice Persona Customizer:** Expose custom LLM system prompts in the Web GUI settings for personalized coaching styles.
