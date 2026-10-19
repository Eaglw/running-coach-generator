import json
import logging
import math
import os
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
        self.music_dir = music_dir or settings.MUSIC_DIR
        self.cache_path = cache_path or settings.LIBRARY_CACHE_PATH
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._tracks: Dict[str, TrackMetadata] = {}
        self.load_cache()

    def load_cache(self) -> None:
        """Load indexed tracks from JSON cache if available."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._tracks = {
                        k: TrackMetadata(**v) for k, v in data.items()
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

    def _compute_fallback_bpm(self, file_path: Path, duration_sec: float) -> int:
        """
        Calculate BPM using librosa analyzing a 60-second middle slice of the audio file.
        """
        try:
            import librosa
            # Analyze a 60-second middle slice for optimal speed and accuracy
            target_slice_duration = 60.0
            if duration_sec > target_slice_duration:
                offset = max(0.0, (duration_sec - target_slice_duration) / 2.0)
            else:
                offset = 0.0
                target_slice_duration = duration_sec if duration_sec > 0 else None

            y, sr = librosa.load(
                str(file_path),
                offset=offset,
                duration=target_slice_duration,
                sr=22050,
                mono=True
            )
            if len(y) == 0:
                return 120  # Default fallback tempo

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # Handle float or scalar numpy array
            if hasattr(tempo, "__iter__"):
                computed = float(tempo[0])
            else:
                computed = float(tempo)

            bpm = int(round(computed))
            return max(60, min(bpm, 220))
        except Exception as e:
            logger.warning("librosa BPM analysis failed for %s: %s. Using default 120 BPM.", file_path.name, e)
            return 120

    def scan_library(self, force_rescan: bool = False) -> List[TrackMetadata]:
        """
        Scan music_dir for audio files, extract tags and calculate missing BPMs.
        """
        discovered_paths: Set[str] = set()
        updated = False

        for root, _, files in os.walk(self.music_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in AUDIO_EXTENSIONS:
                    file_path = Path(root) / file
                    path_str = str(file_path.resolve())
                    discovered_paths.add(path_str)
                    mtime = file_path.stat().st_mtime

                    # Check if already indexed and up to date
                    if not force_rescan and path_str in self._tracks:
                        if abs(self._tracks[path_str].mtime - mtime) < 1.0:
                            continue

                    logger.info("Indexing track: %s", file_path.name)
                    title, artist, duration, bpm = self._extract_id3_metadata(file_path)

                    if bpm is not None and bpm > 0:
                        bpm_source = "id3"
                    else:
                        bpm = self._compute_fallback_bpm(file_path, duration)
                        bpm_source = "librosa_analysis"

                    # If duration was not found via ID3, compute duration
                    if duration <= 0:
                        try:
                            import soundfile as sf
                            info = sf.info(str(file_path))
                            duration = float(info.duration)
                        except Exception:
                            duration = 180.0  # Safe fallback estimate

                    track = TrackMetadata(
                        path=path_str,
                        filename=file_path.name,
                        title=title,
                        artist=artist,
                        duration_sec=duration,
                        bpm=bpm,
                        bpm_source=bpm_source,
                        mtime=mtime
                    )
                    self._tracks[path_str] = track
                    updated = True

        # Prune deleted tracks from cache
        current_keys = list(self._tracks.keys())
        for k in current_keys:
            if k not in discovered_paths:
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
        strictly avoiding duplicate tracks in the same workout.
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

        for segment in segments:
            min_bpm, max_bpm = segment.target_bpm_range
            target_duration = segment.duration_sec
            segment_accumulated_sec = 0.0
            segment_tracks: List[Path] = []

            # 1. Strict match: tracks in [min_bpm, max_bpm]
            candidates = [
                t for t in all_tracks
                if t.path not in used_paths and min_bpm <= t.bpm <= max_bpm
            ]

            # 2. Tolerance match (±5%)
            if not candidates:
                tol_min = int(min_bpm * 0.95)
                tol_max = int(max_bpm * 1.05)
                candidates = [
                    t for t in all_tracks
                    if t.path not in used_paths and tol_min <= t.bpm <= tol_max
                ]

            # 3. Half-time or double-time match (e.g. 160 BPM track for 80 BPM target)
            if not candidates:
                candidates = [
                    t for t in all_tracks
                    if t.path not in used_paths and (
                        min_bpm <= t.bpm * 2 <= max_bpm or
                        min_bpm <= t.bpm / 2 <= max_bpm
                    )
                ]

            # 4. Fallback: closest available unused track
            if not candidates:
                unused = [t for t in all_tracks if t.path not in used_paths]
                target_mid = (min_bpm + max_bpm) / 2
                candidates = sorted(
                    unused if unused else all_tracks,
                    key=lambda t: abs(t.bpm - target_mid)
                )

            # Sort candidate tracks by duration descending to minimize splices
            candidates.sort(key=lambda t: t.duration_sec, reverse=True)

            for track in candidates:
                segment_tracks.append(Path(track.path))
                used_paths.add(track.path)
                # First track gives full duration; subsequent tracks lose 3.0s in acrossfade
                effective_duration = track.duration_sec if len(segment_tracks) == 1 else max(0.0, track.duration_sec - 3.0)
                segment_accumulated_sec += effective_duration
                if segment_accumulated_sec >= target_duration:
                    break

            # If segment still needs audio coverage, allow reuse with lowest BPM distance
            if segment_accumulated_sec < target_duration:
                target_mid = (min_bpm + max_bpm) / 2
                pool = sorted(all_tracks, key=lambda t: abs(t.bpm - target_mid))
                for track in pool:
                    segment_tracks.append(Path(track.path))
                    effective_duration = track.duration_sec if len(segment_tracks) == 1 else max(0.0, track.duration_sec - 3.0)
                    segment_accumulated_sec += effective_duration
                    if segment_accumulated_sec >= target_duration:
                        break

            result.append((target_duration, segment_tracks))

        return result


# Singleton indexer instance
library_indexer = LibraryIndexer()
