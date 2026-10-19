# 🏃 Running Coach Generator

> **Self-hosted AI Running Workout Generator** • Mixa la tua libreria musicale con indicazioni vocali generate dall'intelligenza artificiale e pubblica automaticamente gli allenamenti su un feed RSS per **Apple Podcasts** e **Apple Watch** (senza iPhone durante la corsa).

---

## 🌟 Caratteristiche Principali

- 🖥️ **Web Dashboard Integrata:** Interfaccia grafica sportiva per creare allenamenti da prompt vocale/testuale, monitorare l'avanzamento in tempo reale, ascoltare gli episodi e gestire la libreria musicale.
- 🎵 **Music Indexer & BPM Detection:** Scansione automatica di `/data/music` con lettura dei tag ID3 (`TBPM`) e fallback intelligente tramite `librosa` (analisi spettrale su finestra di 60s) con cache JSON.
- 🎛️ **Engine di Mixaggio Audio FFmpeg a Due Fasi:**
  - **Step A:** Concatenazione delle tracce musicali con crossfade triangolare da 3 secondi (`acrossfade=d=3:c1=tri:c2=tri`) e normalizzazione loudness broadcast EBU R128 (`loudnorm=I=-16:TP=-1.5:LRA=11`).
  - **Step B:** Posizionamento dei cue vocali al millisecondo (`adelay`) e **Sidechain Ducking** dinamico (`sidechaincompress=threshold=0.015:ratio=6:attack=50:release=600`) a guadagno unitario (`normalize=0`), con codifica MP3 a 192k CBR 44.1kHz.
- 🤖 **LLM Coach Multi-Provider:** Supporto compatibile OpenAI (Google Gemini, OpenAI, Ollama locale) per il parsing da linguaggio naturale a struttura workout e la stesura dei testi vocali dei coach (`tecnico_motivazionale`, `calmo_mindful`, `militare_severo`). Include fallback euristico in italiano integrato.
- 🎙️ **TTS Neurale in Italiano:** Pattern Strategy per `piper-tts`, `kokoro-onnx` e fallback di sistema.
- 📡 **Podcast RSS Feed 2.0 & Byte-Serving:** Generazione atomica di `feed.xml` conforme agli standard iTunes/Apple Podcasts con date RFC 2822 e supporto obbligatorio `HTTP Range` per Apple Watch.

---

## 🚀 Avvio Rapido con Docker

Il modo più rapido per avviare il container è tramite Docker Compose:

```bash
# 1. Clona il repository
git clone https://github.com/Eaglw/running-coach-generator.git
cd running-coach-generator

# 2. Configura le variabili d'ambiente
cp .env.example .env

# 3. Posiziona i tuoi file musicali (.mp3, .flac) nella cartella locale
# ./data/music/

# 4. Avvia il container
docker compose up -d --build
```

La **Web GUI** sarà immediatamente disponibile all'indirizzo:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## ⌚ Come Sincronizzare con Apple Watch

1. Apri l'app **Apple Podcasts** su iPhone.
2. Vai nella scheda **Libreria**.
3. Tocca i tre puntini (**...**) in alto a destra e seleziona **"Segui uno spettacolo tramite URL..."**.
4. Incolla l'URL del tuo feed: `http://<IP-DEL-TUO-SERVER>:8000/podcast/feed.xml`.
5. Apri l'app **Podcast sull'Apple Watch**: i nuovi allenamenti generati verranno scaricati automaticamente nella memoria dell'orologio per essere ascoltati offline con cuffie Bluetooth mentre corri, lasciando l'iPhone a casa!

---

## 🛠️ Struttura delle Cartelle

```text
running-coach-generator/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── app/
│   ├── main.py                 # Server FastAPI, Web GUI, background workers & static files
│   ├── config.py               # Pydantic Settings & risoluzione cartelle
│   ├── models.py               # Schemi Pydantic v2 (WorkoutRequest, CoachCue, Segment)
│   ├── static/
│   │   └── index.html          # Dashboard Single Page Application (Tailwind CSS)
│   └── services/
│       ├── audio_mixer.py      # Engine FFmpeg (crossfade, sidechain ducking, MP3 mixdown)
│       ├── library_indexer.py  # Scansione ID3 & analisi BPM con fallback librosa
│       ├── llm_coach.py        # Parser prompt & generatore script coach in italiano
│       ├── tts_engine.py       # Strategy pattern per Piper, Kokoro e fallback vocale
│       └── rss_generator.py    # Generatore atomico feed.xml conforme Apple Podcasts
└── data/
    ├── music/                  # Cartella dove posizionare i tuoi brani musicali
    ├── cache/                  # Cache library_index.json (BPM e metadati)
    ├── podcasts/               # feed.xml ed episodi generati
    └── temp/                   # File temporanei di elaborazione audio
```

---

## 📡 API Endpoints

I documenti OpenAPI Swagger interattivi sono consultabili all'indirizzo **`http://localhost:8000/docs`**.

| Metodo | Endpoint | Descrizione |
|---|---|---|
| `GET` | `/` | Web Dashboard GUI |
| `POST` | `/api/v1/workout/from-text` | Genera workout partendo da descrizione testuale o vocale |
| `POST` | `/api/v1/workout/from-json` | Genera workout da payload JSON strutturato |
| `GET` | `/api/v1/jobs/{job_id}` | Stato di avanzamento del task in background |
| `POST` | `/api/v1/library/scan` | Forza la scansione e indicizzazione della libreria musicale |
| `GET` | `/api/v1/library/tracks` | Elenco delle tracce musicali con BPM rilevati |
| `GET` | `/api/v1/episodes` | Elenco degli episodi pubblicati nel podcast |
| `GET` | `/podcast/feed.xml` | Feed RSS conforme Apple Podcasts |
| `GET` | `/podcast/episodes/{file}` | Streaming statico MP3 con supporto Byte-Serving `HTTP Range` |
| `GET` | `/health` | Healthcheck del container |

---

## 🧪 Test di Verifica Locale

Puoi verificare autonomamente la correttezza del mixaggio audio FFmpeg senza configurazioni esterne con il test CLI integrato:

```bash
python -m app.services.audio_mixer
```
