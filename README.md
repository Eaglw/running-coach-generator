# 🏃 Running Coach Generator

> **Self-hosted, single-container AI guided running workout generator** that blends your local music library with neural voice coaching cues, automatically publishing episodes via a static Podcast RSS feed for offline sync on **Apple Podcasts** and **Apple Watch** (run without your iPhone!).

---

## 🌟 Key Features

- 🖥️ **Integrated Web Dashboard GUI:** Modern, dark-mode sports dashboard to generate workouts via natural language or structured JSON, track background generation in real time, listen with an embedded HTML5 player, and inspect your music library.
- 🎵 **Segment-Aligned Music Indexer & BPM Detection:** Scans `/data/music` for audio files (`.mp3`, `.flac`, `.m4a`), extracts ID3 tags (`TBPM`, artist, title), and runs automated spectral tempo estimation via `librosa` on a 60-second slice if tags are missing. Results are cached in `library_index.json`.
- 🎛️ **Two-Step Zero-RAM FFmpeg Audio Mixing Engine:**
  - **Step A (Segment-Aligned Music Bed):** Groups tracks by segment, applies 3-second triangular crossfades (`acrossfade=d=3:c1=tri:c2=tri`) accounting for duration loss, trims exactly at segment boundaries (`atrim=0:{segment_duration}`), and applies broadcast EBU R128 loudness normalization (`loudnorm=I=-16:TP=-1.5:LRA=11`) with **zero giant RAM buffers** (`aloop` OOM eliminated).
  - **Step B (Precision Ducking & MP3 Mixdown):** Places voice cues with millisecond precision (`adelay`), applies dynamic **Sidechain Compression Ducking** (`sidechaincompress=threshold=0.015:ratio=6:attack=50:release=600`), and mixes with unity gain compensation (`amix` with `normalize=0`) encoded to 192k CBR 44.1kHz MP3.
- 🤖 **Multi-Provider LLM Coach:** OpenAI-compatible client (Google Gemini, OpenAI, local Ollama) using Structured Outputs to parse free-form dictation into strict `WorkoutRequest` schemas and generate natural Italian coaching scripts according to the coach persona (`tecnico_motivazionale`, `calmo_mindful`, `militare_severo`). Includes a rule-based heuristic Italian fallback.
- 🎙️ **Real Neural TTS Engine:** Strategy pattern supporting `piper-tts` and `kokoro-onnx`. Includes an **automatic on-demand downloader** that fetches Italian neural voice models from Hugging Face (`rhasspy/piper-voices`) into `/data/cache/tts_models/`.
- 📡 **Podcast RSS 2.0 & Apple Watch Byte-Serving:** Generates RFC 2822 compliant, iTunes-compatible `feed.xml` and serves audio files with mandatory `HTTP Range` byte-serving for seamless Apple Podcasts and Apple Watch background syncing.

---

## 🚀 Quickstart with Docker Compose

Running the entire service inside a single container is simple:

```bash
# 1. Clone the repository
git clone https://github.com/Eaglw/running-coach-generator.git
cd running-coach-generator

# 2. Configure environment variables
cp .env.example .env

# 3. Add your music tracks (.mp3, .flac) to the local folder:
# ./data/music/

# 4. Build and start the container
docker compose up -d --build
```

The Web Dashboard will be live at:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## ⌚ How to Sync with Apple Watch (Offline Running)

1. Open the **Apple Podcasts** app on your iPhone.
2. Tap the **Library** tab.
3. Tap the **three dots (...)** in the top right corner and select **"Follow a Show by URL..."**.
4. Paste your feed URL: `http://<YOUR-SERVER-IP>:8000/podcast/feed.xml`.
5. Open the **Podcasts app on your Apple Watch**: new generated workout tracks will automatically download to your watch over Wi-Fi/Bluetooth, allowing you to run untethered with Bluetooth headphones while leaving your iPhone at home!

---

## 📁 Directory Structure

```text
running-coach-generator/
├── docker-compose.yml          # Container configuration with host volume binding
├── Dockerfile                  # python:3.11-slim + ffmpeg + espeak-ng + libsndfile1
├── requirements.txt            # FastAPI, Uvicorn, Pydantic v2, mutagen, librosa, piper-tts
├── .env.example                # Example configuration and environment settings
├── README.md                   # Project documentation and setup guide
├── app/
│   ├── main.py                 # FastAPI application, background tasks, static mounts & routes
│   ├── config.py               # Pydantic Settings and path resolution
│   ├── models.py               # Pydantic v2 schemas (WorkoutRequest, Segment, CoachCue, etc.)
│   ├── static/
│   │   └── index.html          # Interactive Web Dashboard (Tailwind CSS, dark mode)
│   └── services/
│       ├── audio_mixer.py      # Two-step FFmpeg engine (crossfade, sidechain ducking, mixdown)
│       ├── library_indexer.py  # ID3 extraction, librosa fallback BPM & segment track selector
│       ├── llm_coach.py        # Natural language parser & Italian cue script generator
│       ├── tts_engine.py       # Strategy pattern for Piper, Kokoro & system fallback
│       └── rss_generator.py    # Atomic RFC 2822 iTunes-compliant feed.xml generator
└── data/
    ├── music/                  # Place your source MP3/FLAC music tracks here
    ├── cache/                  # library_index.json & tts_models/ (auto-downloaded ONNX models)
    ├── podcasts/               # feed.xml & generated workout episodes (.mp3)
    └── temp/                   # Temporary working audio files (cleaned up automatically)
```

---

## 📡 REST API Endpoints

Interactive Swagger API documentation is available at **`http://localhost:8000/docs`**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Dashboard GUI |
| `GET` | `/health` | Healthcheck endpoint (`{"status": "ok"}`) |
| `POST` | `/api/v1/workout/from-text` | Generate workout from natural language prompt or voice dictation |
| `POST` | `/api/v1/workout/from-json` | Generate workout from strict structured JSON specification |
| `GET` | `/api/v1/jobs/{job_id}` | Query status and progress of background generation job |
| `POST` | `/api/v1/library/scan` | Trigger re-scan and BPM indexing of `/data/music` |
| `GET` | `/api/v1/library/tracks` | Retrieve list of all indexed music tracks and BPM values |
| `GET` | `/api/v1/episodes` | List all published workout episodes |
| `GET` | `/podcast/feed.xml` | Serve Podcast 2.0 RSS feed (`application/rss+xml`) |
| `GET` | `/podcast/episodes/{file}` | Serve episode audio files with `HTTP Range` byte-serving |

---

## 📋 Workout JSON Contract Example

```json
{
  "workout_title": "Progressivo 45m - Focus Soglia",
  "coach_persona": "tecnico_motivazionale",
  "total_duration_sec": 2700,
  "segments": [
    {
      "name": "Riscaldamento Progressivo",
      "duration_sec": 600,
      "intensity": "low",
      "target_bpm_range": [115, 130],
      "target_pace": "5:30/km",
      "coach_cues": [
        {
          "offset_sec": 5,
          "intent": "Benvenuto, riepilogo struttura allenamento e avvio riscaldamento.",
          "spoken_text": null
        },
        {
          "offset_sec": 570,
          "intent": "Preavviso: tra 30 secondi si passa al blocco centrale a ritmo medio.",
          "spoken_text": null
        }
      ]
    }
  ]
}
```

*Note:* If `spoken_text` is `null`, `llm_coach.py` dynamically writes the spoken phrase based on `intent`, `coach_persona`, `segment.name`, and `target_pace`.

---

## 🧪 Local Testing & Verification

You can verify the audio mixing pipeline standalone (without needing API keys or local music files) using the built-in test suite:

```bash
# Test the FFmpeg 2-step sidechain ducking engine
python -m app.services.audio_mixer
```

To run the application locally without Docker:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
