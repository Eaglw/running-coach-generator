from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import (
    EpisodeMetadata,
    JobStatus,
    JobStatusResponse,
    PromptWorkoutRequest,
    WorkoutRequest,
)
from app.services.audio_mixer import AudioMixer
from app.services.library_indexer import TrackMetadata, library_indexer
from app.services.llm_coach import llm_coach
from app.services.rss_generator import RSSGenerator
from app.services.tts_engine import get_tts_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("running_coach")

# In-memory store for background job statuses
jobs_db: Dict[str, JobStatusResponse] = {}

# Services
audio_mixer = AudioMixer()
rss_generator = RSSGenerator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown hooks."""
    settings.ensure_directories()
    # Ensure initial feed.xml exists
    if not settings.RSS_FEED_PATH.exists():
        rss_generator.generate_feed([])

    # Ensure default podcast cover artwork exists (1400x1400 requirement for Apple Podcasts)
    cover_path = settings.PODCAST_DIR / "cover.jpg"
    if not cover_path.exists():
        try:
            logger.info("Generating default 1400x1400 podcast cover artwork...")
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "color=c=#0f172a:s=1400x1400:d=1",
                "-frames:v", "1",
                "-q:v", "2",
                str(cover_path)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Generated %s successfully.", cover_path)
        except Exception as e:
            logger.warning("Failed to generate default cover image: %s", e)

    # Trigger initial non-forced scan if library cache is empty
    if not library_indexer.get_tracks():
        logger.info("Initializing music library index on startup...")
        try:
            library_indexer.scan_library(force_rescan=False)
        except Exception as e:
            logger.warning("Initial music library scan failed: %s", e)

    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="Running Coach Generator",
    description="Self-hosted AI running workout generator publishing to Apple Podcasts RSS",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local dashboards or external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _process_workout_job(job_id: str, workout: WorkoutRequest) -> None:
    """
    Asynchronous pipeline orchestrating:
    1. LLM Cue Script generation (generating_script)
    2. TTS voice cue synthesis (synthesizing_tts)
    3. Music selection & BPM matching (mixing_audio)
    4. FFmpeg Step A (Segment-aligned Music bed) & Step B (Ducking & MP3 Mix)
    5. Apple Podcasts RSS feed publication (rss_generator.add_episode -> completed)
    """
    job = jobs_db.get(job_id)
    if not job:
        return

    session_temp_dir = settings.TEMP_DIR / f"job_{job_id}"
    session_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Phase 1: Coach Script generation (LLM)
        job.status = JobStatus.GENERATING_SCRIPT
        job.progress = 0.15
        job.updated_at = datetime.now(timezone.utc)
        logger.info("[Job %s] Generating coach cue scripts with persona '%s'...", job_id, workout.coach_persona)
        workout_with_scripts = llm_coach.generate_spoken_script(workout)

        # Phase 2: TTS Synthesis
        job.status = JobStatus.SYNTHESIZING_TTS
        job.progress = 0.35
        job.updated_at = datetime.now(timezone.utc)
        logger.info("[Job %s] Synthesizing TTS audio cues...", job_id)

        tts_engine = get_tts_engine()
        voice_cues = []  # List[Tuple[Path, int]] -> (wav_path, global_offset_ms)

        current_segment_start_sec = 0
        cue_idx = 0

        for segment in workout_with_scripts.segments:
            for cue in segment.coach_cues:
                cue_idx += 1
                global_offset_sec = current_segment_start_sec + cue.offset_sec
                global_offset_ms = int(global_offset_sec * 1000)

                cue_wav_path = session_temp_dir / f"cue_{cue_idx}.wav"
                spoken_text = cue.spoken_text or cue.intent
                tts_engine.synthesize(spoken_text, cue_wav_path)
                voice_cues.append((cue_wav_path, global_offset_ms))

            current_segment_start_sec += segment.duration_sec

        # Phase 3: Track Selection per Segment
        job.status = JobStatus.MIXING_AUDIO
        job.progress = 0.60
        job.updated_at = datetime.now(timezone.utc)
        logger.info("[Job %s] Selecting music tracks for segments...", job_id)

        try:
            segment_tracks = library_indexer.select_tracks_for_segments(workout_with_scripts.segments)
        except Exception as e:
            logger.warning("[Job %s] Music selection fallback: %s. Generating ambient bed.", job_id, e)
            fallback_music = session_temp_dir / "fallback_bed.mp3"
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "sine=frequency=130:duration=60",
                "-af", "volume=0.2",
                "-c:a", "libmp3lame", "-b:a", "192k", str(fallback_music)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            segment_tracks = [(workout_with_scripts.total_duration_sec, [fallback_music])]

        # Phase 4: FFmpeg Mixing (Step A & Step B)
        job.progress = 0.80
        job.updated_at = datetime.now(timezone.utc)
        output_filename = f"workout_{job_id}.mp3"
        logger.info("[Job %s] Rendering final audio track with FFmpeg ducking...", job_id)

        final_mp3_path = audio_mixer.render_workout_audio(
            segment_tracks=segment_tracks,
            voice_cues=voice_cues,
            total_duration_sec=workout_with_scripts.total_duration_sec,
            output_filename=output_filename
        )

        # Phase 5: RSS Feed Publication
        job.progress = 0.95
        job.updated_at = datetime.now(timezone.utc)
        file_size = final_mp3_path.stat().st_size
        duration_sec = int(audio_mixer.get_audio_duration_sec(final_mp3_path))

        desc = (
            f"Allenamento guidato: {workout.workout_title}. "
            f"Durata: {duration_sec // 60} minuti. "
            f"Persona coach: {workout.coach_persona}."
        )

        episode = rss_generator.add_episode(
            guid=f"episode-{job_id}",
            title=workout.workout_title,
            description=desc,
            filename=output_filename,
            file_size_bytes=file_size,
            duration_sec=duration_sec
        )

        # Completed
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.episode_filename = output_filename
        job.episode_url = episode.enclosure_url
        job.updated_at = datetime.now(timezone.utc)
        logger.info("[Job %s] Successfully completed! Published at %s", job_id, episode.enclosure_url)

    except Exception as e:
        logger.exception("[Job %s] Failed: %s", job_id, e)
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.updated_at = datetime.now(timezone.utc)
    finally:
        shutil.rmtree(session_temp_dir, ignore_errors=True)
        logger.debug("[Job %s] Cleaned up temporary directory %s", job_id, session_temp_dir)


# =====================================================================
# API Endpoints
# =====================================================================

@app.get(
    "/health",
    summary="Health check endpoint"
)
async def health():
    """Healthcheck endpoint returning {'status': 'ok'}."""
    return {"status": "ok"}


@app.post(
    "/api/v1/workout/from-json",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobStatusResponse,
    summary="Create workout from structured JSON specification"
)
async def create_workout_from_json(
    workout: WorkoutRequest,
    background_tasks: BackgroundTasks
):
    """
    Accepts a strictly validated WorkoutRequest JSON payload,
    triggers background processing, and returns 202 Accepted with a job_id.
    """
    job_id = uuid.uuid4().hex[:12]
    job_resp = JobStatusResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0.0,
        workout_title=workout.workout_title
    )
    jobs_db[job_id] = job_resp
    background_tasks.add_task(_process_workout_job, job_id, workout)
    return job_resp


@app.post(
    "/api/v1/workout/from-text",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobStatusResponse,
    summary="Create workout from natural language prompt or voice dictation"
)
async def create_workout_from_text(
    payload: PromptWorkoutRequest,
    background_tasks: BackgroundTasks
):
    """
    Accepts a natural language prompt, converts it into a WorkoutRequest using LLM Structured Outputs,
    and enqueues the generation job.
    """
    parsed_workout = llm_coach.parse_prompt_to_workout(
        prompt=payload.prompt,
        coach_persona=payload.coach_persona
    )
    return await create_workout_from_json(parsed_workout, background_tasks)


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Query workout generation job status"
)
async def get_job_status(job_id: str):
    """
    Returns the current status (queued, generating_script, synthesizing_tts, mixing_audio, completed, failed)
    and progress of a generation job.
    """
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job


@app.post(
    "/api/v1/library/scan",
    summary="Trigger re-scan and BPM indexing of music library"
)
async def scan_library(force: bool = False):
    """
    Scans /data/music, extracts ID3 tags, and runs librosa fallback BPM detection.
    Returns the count and list of indexed tracks.
    """
    tracks = library_indexer.scan_library(force_rescan=force)
    return {"count": len(tracks), "tracks": tracks}


@app.get(
    "/api/v1/library/tracks",
    response_model=List[TrackMetadata],
    summary="Get all indexed music tracks"
)
async def get_library_tracks():
    """Return all tracks currently indexed in the music library."""
    return library_indexer.get_tracks()


@app.get(
    "/api/v1/episodes",
    response_model=List[EpisodeMetadata],
    summary="List all published podcast episodes"
)
async def get_episodes():
    """Return list of all published episodes in feed.xml."""
    return rss_generator._load_episodes_metadata()


@app.get(
    "/api/v1/settings",
    summary="Get active configuration parameters"
)
async def get_active_settings():
    """Return runtime settings for Web GUI."""
    return {
        "BASE_URL": settings.BASE_URL,
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "LLM_MODEL": settings.LLM_MODEL,
        "LLM_BASE_URL": settings.LLM_BASE_URL,
        "LLM_API_KEY": settings.LLM_API_KEY,
        "TTS_ENGINE": settings.TTS_ENGINE,
        "TTS_VOICE": settings.TTS_VOICE,
        "MUSIC_DIR": str(settings.MUSIC_DIR),
        "PODCAST_DIR": str(settings.PODCAST_DIR),
        "PODCAST_TITLE": settings.PODCAST_TITLE
    }


@app.post(
    "/api/v1/settings",
    summary="Update and persist configuration parameters"
)
async def update_settings(updates: Dict[str, Any]):
    """Update settings and persist to data/cache/settings.json."""
    allowed = {
        "BASE_URL", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL",
        "LLM_API_KEY", "TTS_ENGINE", "TTS_VOICE", "PODCAST_TITLE",
        "MUSIC_DIR"
    }
    filtered = {k: v for k, v in updates.items() if k in allowed}
    settings.save_dynamic_settings(filtered)
    if "BASE_URL" in filtered or "PODCAST_TITLE" in filtered:
        try:
            rss_generator.refresh_episodes_and_regenerate()
            logger.info("Regenerated podcast RSS feed with updated BASE_URL/title: %s", settings.BASE_URL)
        except Exception as e:
            logger.warning("Failed to refresh RSS feed on settings update: %s", e)
    return {"message": "Settings updated and saved successfully.", "settings": await get_active_settings()}


@app.post(
    "/api/v1/settings/test-llm",
    summary="Test current LLM connection and configuration"
)
async def test_llm_connection():
    """Execute a test prompt with current LLM provider settings."""
    return llm_coach.test_connection()


@app.get(
    "/podcast/feed.xml",
    summary="Serve the Podcast 2.0 RSS Feed"
)
async def get_podcast_feed(request: Request):
    """
    Serves the podcast feed.xml with Content-Type application/rss+xml.
    Dynamically adapts localhost URLs to the incoming Host header or settings.BASE_URL
    so Apple Podcasts clients on mobile devices can download episodes seamlessly.
    """
    if not settings.RSS_FEED_PATH.exists():
        rss_generator.generate_feed([])
    raw_xml = settings.RSS_FEED_PATH.read_text(encoding="utf-8")

    # If accessed from an external IP/hostname, replace localhost with the actual host
    host_header = request.headers.get("host")
    if host_header:
        scheme = request.url.scheme or "http"
        request_base = f"{scheme}://{host_header}"
        if "localhost" not in host_header and "127.0.0.1" not in host_header:
            raw_xml = raw_xml.replace("http://localhost:8000", request_base).replace("http://127.0.0.1:8000", request_base)
    elif settings.BASE_URL:
        raw_xml = raw_xml.replace("http://localhost:8000", settings.BASE_URL).replace("http://127.0.0.1:8000", settings.BASE_URL)

    return Response(
        content=raw_xml,
        media_type="application/rss+xml"
    )


# =====================================================================
# Static File Mounts (Apple Watch Byte-Serving & Web GUI)
# =====================================================================

# Mount /podcast using FastAPI StaticFiles with mandatory HTTP Range support
# This automatically serves /podcast/episodes/*.mp3 with byte-serving for Apple Watch
app.mount(
    "/podcast",
    StaticFiles(directory=str(settings.PODCAST_DIR)),
    name="podcast_static"
)

# Serve Web GUI static assets
static_assets_dir = Path(__file__).parent / "static"
static_assets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_assets_dir)), name="static_gui")


@app.get("/", summary="Web Dashboard GUI")
async def serve_gui():
    """Serves the interactive Web GUI single-page application."""
    index_file = static_assets_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return {"message": "Running Coach Generator API is active. Web GUI not found."}
