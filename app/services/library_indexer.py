import json
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

try:
    import mutagen
    from mutagen.id3 import ID3
except ImportError:
    mutagen = None
    ID3 = None
import numpy as np

from app.config import settings
from app.models import Segment

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".wav", ".aac", ".ogg"}


class TrackMetadata(BaseModel):
    """Metadata representation of an indexed music track."""
    path: str
    filename: str
    title: str
    artist: str
    duration_sec: float
    bpm: int
    bpm_source: str  # "id3" or "librosa_analysis"
    energy: str = "medium"  # "low", "medium", "high", "max"
    intro_sec: float = 0.0  # seconds before main beat / drop kicks in
    mtime: float


class LibraryIndexer:
    """
    Scans the local music directory, reads ID3 tags (TBPM),
    computes fallback BPM with librosa on a 60s slice,
    and caches results in library_index.json.
    """

    def __init__(
        self,
        music_dir: Optional[Path] = None,
        cache_path: Optional[Path] = None
    ) -> None:
        self._custom_music_dir = music_dir
        self.cache_path = cache_path or settings.LIBRARY_CACHE_PATH
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._tracks: Dict[str, TrackMetadata] = {}
        self.load_cache()

    @property
    def music_dir(self) -> Path:
        return Path(self._custom_music_dir or settings.MUSIC_DIR)

    def load_cache(self) -> None:
        """Load indexed tracks from JSON cache if available, filtering out _Serato_ and short files (<60s)."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._tracks = {
                        k: TrackMetadata(**v) for k, v in data.items()
                        if "_serato_" not in k.lower() and v.get("duration_sec", 0.0) >= 60.0
                    }
                logger.info("Loaded %d tracks from library index cache.", len(self._tracks))
            except Exception as e:
                logger.warning("Failed to load library index cache: %s. Starting fresh.", e)
                self._tracks = {}

    def save_cache(self) -> None:
        """Persist indexed tracks to JSON cache atomically."""
        tmp_path = self.cache_path.with_suffix(".json.tmp")
        data = {k: v.model_dump() for k, v in self._tracks.items()}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.cache_path)
        logger.info("Saved %d tracks to library index cache at %s", len(self._tracks), self.cache_path)

    def _extract_id3_metadata(self, file_path: Path) -> Tuple[str, str, float, Optional[int]]:
        """
        Extract title, artist, duration, and BPM from ID3 tags.
        """
        title = file_path.stem
        artist = "Unknown Artist"
        duration_sec = 0.0
        bpm: Optional[int] = None

        try:
            audio = mutagen.File(str(file_path))
            if audio is not None:
                if hasattr(audio, "info") and hasattr(audio.info, "length"):
                    duration_sec = float(audio.info.length)

                # Try standard ID3 tags (for MP3)
                if isinstance(audio.tags, ID3):
                    # Title
                    if "TIT2" in audio.tags:
                        title = str(audio.tags["TIT2"].text[0])
                    # Artist
                    if "TPE1" in audio.tags:
                        artist = str(audio.tags["TPE1"].text[0])
                    # BPM
                    if "TBPM" in audio.tags:
                        raw_bpm = str(audio.tags["TBPM"].text[0]).strip()
                        try:
                            val = int(round(float(raw_bpm)))
                            if val > 0:
                                bpm = val
                        except ValueError:
                            pass
                elif audio.tags:
                    # Generic tags for FLAC/M4A/Vorbis
                    if "title" in audio.tags:
                        title = str(audio.tags["title"][0])
                    if "artist" in audio.tags:
                        artist = str(audio.tags["artist"][0])
                    for bpm_tag in ["bpm", "BPM", "tempo", "TEMPO", "tbpm"]:
                        if bpm_tag in audio.tags:
                            try:
                                val = int(round(float(str(audio.tags[bpm_tag][0]))))
                                if val > 0:
                                    bpm = val
                                    break
                            except ValueError:
                                pass
        except Exception as e:
            logger.debug("Mutagen could not read tags for %s: %s", file_path.name, e)

        return title, artist, duration_sec, bpm

    def _analyze_audio_properties(
        self,
        file_path: Path,
        duration_sec: float,
        id3_bpm: Optional[int] = None
    ) -> Tuple[int, str, str, float]:
        """
        Analyzes audio file using FFmpeg and librosa:
        1. Estimates BPM (using id3_bpm if present, else librosa beat_track on a 60s slice).
        2. Detects intro_sec (where the main beat / drop kicks in via RMS envelope).
        3. Classifies energy level ('low', 'medium', 'high', 'max') combining RMS density and tempo.
        Returns: (bpm, bpm_source, energy, intro_sec)
        """
        target_slice_duration = min(60.0, max(10.0, duration_sec))
        if duration_sec > target_slice_duration:
            offset = max(0.0, (duration_sec - target_slice_duration) / 2.0)
        else:
            offset = 0.0

        bpm = id3_bpm if (id3_bpm is not None and id3_bpm > 0) else 120
        bpm_source = "id3" if (id3_bpm is not None and id3_bpm > 0) else "fallback_default"
        energy = "medium"
        intro_sec = 0.0

        try:
            import librosa

            # 1. Decode head slice (first 40s) for intro & drop analysis
            head_duration = min(40.0, duration_sec)
            cmd_head = [
                "ffmpeg", "-v", "error",
                "-ss", "0",
                "-t", str(head_duration),
                "-i", str(file_path),
                "-f", "f32le",
                "-ac", "1",
                "-ar", "22050",
                "-"
            ]
            res_head = subprocess.run(cmd_head, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res_head.stdout and len(res_head.stdout) >= 22050 * 2:
                y_head = np.frombuffer(res_head.stdout, dtype=np.float32)
                # Compute RMS envelope to find where the main beat drops
                hop_len = 512
                rms_head = librosa.feature.rms(y=y_head, frame_length=2048, hop_length=hop_len)[0]
                if rms_head.size > 0:
                    loud_p80 = float(np.percentile(rms_head, 80))
                    if loud_p80 > 0.01:
                        threshold = 0.35 * loud_p80
                        times = librosa.frames_to_time(np.arange(rms_head.size), sr=22050, hop_length=hop_len)
                        sustain_frames = max(1, int(0.5 * 22050 / hop_len))
                        for i in range(rms_head.size):
                            if rms_head[i] >= threshold:
                                window = rms_head[i : i + sustain_frames]
                                if window.size and float(np.mean(window)) >= threshold:
                                    intro_sec = round(max(0.0, float(times[i])), 1)
                                    break

            # 2. Decode middle slice for BPM (if not in ID3) and overall energy density
            cmd_mid = [
                "ffmpeg", "-v", "error",
                "-ss", str(offset),
                "-t", str(target_slice_duration),
                "-i", str(file_path),
                "-f", "f32le",
                "-ac", "1",
                "-ar", "22050",
                "-"
            ]
            res_mid = subprocess.run(cmd_mid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res_mid.stdout and len(res_mid.stdout) >= 22050 * 4:
                y_mid = np.frombuffer(res_mid.stdout, dtype=np.float32)

                # Compute BPM with librosa if not present in ID3
                if id3_bpm is None or id3_bpm <= 0:
                    tempo, _ = librosa.beat.beat_track(y=y_mid, sr=22050, start_bpm=128.0)
                    if hasattr(tempo, "__iter__"):
                        computed = float(tempo[0])
                    else:
                        computed = float(tempo)
                    computed_bpm = int(round(computed))
                    if computed_bpm > 0:
                        bpm = max(60, min(computed_bpm, 220))
                        bpm_source = "librosa_analysis"

                # Compute RMS energy density
                rms_mid = float(np.sqrt(np.mean(y_mid ** 2)))
                # Energy classification combining tempo & sonic density
                if bpm >= 155 or (bpm >= 140 and rms_mid >= 0.12):
                    energy = "max"
                elif bpm >= 130 or rms_mid >= 0.08:
                    energy = "high"
                elif bpm >= 115 or rms_mid >= 0.04:
                    energy = "medium"
                else:
                    energy = "low"

        except Exception as e:
            logger.debug("Audio analysis fallback on %s: %s", file_path.name, e)
            if id3_bpm and id3_bpm > 0:
                bpm = id3_bpm
                bpm_source = "id3"
            else:
                bpm = 120
                bpm_source = "fallback_default"
            if bpm >= 155:
                energy = "max"
            elif bpm >= 135:
                energy = "high"
            elif bpm >= 118:
                energy = "medium"
            else:
                energy = "low"

        return bpm, bpm_source, energy, intro_sec

    def scan_library(self, force_rescan: bool = False) -> List[TrackMetadata]:
        """
        Scan music_dir for audio files, extract tags and calculate missing BPMs.
        Skips _Serato_ folders and audio files shorter than 60 seconds.
        """
        discovered_paths: Set[str] = set()
        updated = False

        for root, dirs, files in os.walk(self.music_dir):
            # Exclude _Serato_ folders completely from tree traversal
            dirs[:] = [d for d in dirs if "_serato_" not in d.lower()]
            if "_serato_" in root.lower():
                continue

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in AUDIO_EXTENSIONS:
                    file_path = Path(root) / file
                    path_str = str(file_path.resolve())
                    discovered_paths.add(path_str)
                    mtime = file_path.stat().st_mtime

                    # Check if already indexed and up to date
                    if not force_rescan and path_str in self._tracks:
                        cached = self._tracks[path_str]
                        if abs(cached.mtime - mtime) < 1.0 and cached.bpm_source != "fallback_default":
                            continue

                    title, artist, duration, bpm = self._extract_id3_metadata(file_path)

                    # If duration was not found via ID3, compute duration
                    if duration <= 0:
                        try:
                            import soundfile as sf
                            info = sf.info(str(file_path))
                            duration = float(info.duration)
                        except Exception:
                            try:
                                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
                                r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
                                duration = float(r.stdout.strip())
                            except Exception:
                                duration = 180.0

                    # Filter out short DJ samples / sound effects (< 60s)
                    if duration < 60.0:
                        logger.debug("Skipping short audio sample (< 60s): %s (%.1fs)", file_path.name, duration)
                        if path_str in self._tracks:
                            del self._tracks[path_str]
                            updated = True
                        continue

                    logger.info("Indexing track: %s (%.1fs)", file_path.name, duration)

                    bpm, bpm_source, energy, intro_sec = self._analyze_audio_properties(file_path, duration, id3_bpm=bpm)

                    track = TrackMetadata(
                        path=path_str,
                        filename=file_path.name,
                        title=title,
                        artist=artist,
                        duration_sec=duration,
                        bpm=bpm,
                        bpm_source=bpm_source,
                        energy=energy,
                        intro_sec=intro_sec,
                        mtime=mtime
                    )
                    self._tracks[path_str] = track
                    updated = True

        # Prune deleted tracks, _Serato_ folders, or short tracks from cache
        current_keys = list(self._tracks.keys())
        for k in current_keys:
            if k not in discovered_paths or "_serato_" in k.lower() or self._tracks[k].duration_sec < 60.0:
                del self._tracks[k]
                updated = True

        if updated:
            self.save_cache()

        return list(self._tracks.values())

    def get_tracks(self) -> List[TrackMetadata]:
        """Return all currently indexed tracks."""
        return list(self._tracks.values())

    def select_tracks_for_segments(
        self,
        segments: List[Segment]
    ) -> List[Tuple[int, List[Path]]]:
        """
        Track selection algorithm:
        Picks tracks matching each segment's target_bpm_range (with ±5% tolerance or
        half/double-time fallback) to cover segment.duration_sec,
        strictly avoiding duplicate tracks and identical song titles in the same workout.
        Returns a list of tuples: [(segment_duration_sec, [track_path_1, track_path_2, ...]), ...]
        """
        all_tracks = self.get_tracks()
        if not all_tracks:
            # If library is empty, rescan once
            all_tracks = self.scan_library()
        if not all_tracks:
            raise ValueError(f"No music tracks found in {self.music_dir}. Please place MP3/FLAC files in the music folder.")

        result: List[Tuple[int, List[Path]]] = []
        used_paths: Set[str] = set()
        used_song_keys: Set[Tuple[str, str]] = set()

        def is_available(t: TrackMetadata) -> bool:
            if t.path in used_paths:
                return False
            key = (t.title.strip().lower(), t.artist.strip().lower())
            if key in used_song_keys:
                return False
            return True

        energy_rank = {"low": 1, "medium": 2, "high": 3, "max": 4}

        for segment in segments:
            min_bpm, max_bpm = segment.target_bpm_range
            target_duration = segment.duration_sec
            seg_intensity = segment.intensity.lower()
            target_energy_level = energy_rank.get(seg_intensity, 2)
            target_mid = (min_bpm + max_bpm) / 2

            segment_accumulated_sec = 0.0
            segment_tracks: List[Path] = []

            def score_track(t: TrackMetadata) -> float:
                # 1. Energy compatibility score (0.0 to 1.0)
                t_energy_level = energy_rank.get(t.energy.lower(), 2)
                diff = abs(target_energy_level - t_energy_level)
                energy_score = max(0.1, 1.0 - diff * 0.3)

                # 2. BPM proximity score (0.0 to 1.0)
                if min_bpm <= t.bpm <= max_bpm:
                    bpm_score = 1.0
                elif min_bpm <= t.bpm * 2 <= max_bpm or min_bpm <= t.bpm / 2 <= max_bpm:
                    bpm_score = 0.9  # half/double time harmonic compatibility
                else:
                    tol_min = int(min_bpm * 0.92)
                    tol_max = int(max_bpm * 1.08)
                    if tol_min <= t.bpm <= tol_max:
                        bpm_score = 0.75
                    else:
                        bpm_score = max(0.1, 1.0 - abs(t.bpm - target_mid) / 50.0)

                # 3. Drop advantage for high intensity
                drop_score = 0.0
                if seg_intensity in ("high", "max") and t.intro_sec > 0:
                    drop_score = 0.1

                return energy_score * 0.5 + bpm_score * 0.4 + drop_score

            # 1. Available tracks matching target criteria
            available = [t for t in all_tracks if is_available(t)]
            if not available:
                available = list(all_tracks)

            candidates = sorted(
                available,
                key=lambda t: (score_track(t), t.duration_sec),
                reverse=True
            )

            for track in candidates:
                segment_tracks.append(Path(track.path))
                used_paths.add(track.path)
                used_song_keys.add((track.title.strip().lower(), track.artist.strip().lower()))
                # First track gives full duration; subsequent tracks lose 4.0s in acrossfade
                effective_duration = track.duration_sec if len(segment_tracks) == 1 else max(0.0, track.duration_sec - 4.0)
                segment_accumulated_sec += effective_duration
                if segment_accumulated_sec >= target_duration:
                    break

            # If segment still needs audio coverage, repeat tracks from pool until target_duration is covered
            if segment_accumulated_sec < target_duration and all_tracks:
                pool = sorted(all_tracks, key=lambda t: score_track(t), reverse=True)
                pool_idx = 0
                while segment_accumulated_sec < target_duration and pool:
                    track = pool[pool_idx % len(pool)]
                    segment_tracks.append(Path(track.path))
                    effective_duration = track.duration_sec if len(segment_tracks) == 1 else max(0.0, track.duration_sec - 4.0)
                    segment_accumulated_sec += effective_duration
                    pool_idx += 1

            result.append((target_duration, segment_tracks))

        return result


# Singleton indexer instance
library_indexer = LibraryIndexer()
