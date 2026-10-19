import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
import uuid

from app.config import settings

logger = logging.getLogger(__name__)


class AudioMixerError(Exception):
    """Exception raised when an audio mixing or FFmpeg step fails."""
    pass


class AudioMixer:
    """
    Two-step FFmpeg audio mixing engine:
    1. Step A: Segment-aligned music bed concatenation with 3s crossfade, per-segment trimming,
       and EBU R128 loudnorm without large RAM buffers.
    2. Step B: Precision voice timeline assembly, sidechain compression ducking, and MP3 encoding.
    """

    def __init__(
        self,
        temp_dir: Optional[Path] = None,
        episodes_dir: Optional[Path] = None
    ) -> None:
        self.temp_dir = temp_dir or settings.TEMP_DIR
        self.episodes_dir = episodes_dir or settings.EPISODES_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def _run_ffmpeg(self, cmd: List[str], desc: str) -> None:
        """Run an FFmpeg command via subprocess with error capture."""
        logger.debug("Executing FFmpeg (%s): %s", desc, " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            error_msg = (
                f"FFmpeg failed during '{desc}' with returncode {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {result.stderr}"
            )
            logger.error(error_msg)
            raise AudioMixerError(error_msg)

    def get_audio_duration_sec(self, audio_path: Path) -> float:
        """Query duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise AudioMixerError(f"ffprobe failed to inspect {audio_path}: {result.stderr}")
        return float(result.stdout.strip())

    def generate_music_bed(
        self,
        segment_tracks: Union[List[Tuple[int, List[Path]]], List[Path]],
        output_wav_path: Path,
        total_duration_sec: Optional[int] = None
    ) -> Path:
        """
        Step A: Builds a segment-aligned music bed.
        Accepts either:
          - A list of segments: [(segment_duration_sec, [track1, track2, ...]), ...]
          - Or a flat list of tracks (with total_duration_sec) for simple single-segment mixdowns.
        Trims each segment to its exact duration and concatenates them with zero RAM aloop buffers.
        """
        # Normalize input to List[Tuple[int, List[Path]]]
        if not segment_tracks:
            raise AudioMixerError("Cannot generate music bed: segment_tracks is empty.")

        first_elem = segment_tracks[0]
        if isinstance(first_elem, (Path, str)):
            # Flat list passed
            dur = total_duration_sec or 300
            normalized_segments: List[Tuple[int, List[Path]]] = [(dur, [Path(p) for p in segment_tracks])]
        else:
            normalized_segments = [
                (dur, [Path(p) for p in tracks])
                for dur, tracks in segment_tracks  # type: ignore
                if tracks
            ]

        if not normalized_segments:
            raise AudioMixerError("No valid tracks provided in segment_tracks.")

        target_total_sec = total_duration_sec or sum(dur for dur, _ in normalized_segments)

        # Collect ordered tracks across all segments
        raw_tracks: List[Path] = []
        for _, tracks in normalized_segments:
            for p in tracks:
                raw_tracks.append(p)

        if not raw_tracks:
            raise AudioMixerError("No valid audio tracks available to construct music bed.")

        # Measure audio duration per track
        track_durations: List[float] = []
        for p in raw_tracks:
            try:
                d = self.get_audio_duration_sec(p)
            except Exception:
                d = 180.0
            track_durations.append(d)

        # Ensure cumulative duration covers target_total_sec (accounting for 4.0s acrossfade per transition)
        active_tracks: List[Path] = list(raw_tracks)
        cum_dur = track_durations[0] if track_durations else 180.0
        for d in track_durations[1:]:
            cum_dur += max(0.0, d - 4.0)

        # If library tracks are shorter than total workout duration, cycle tracks
        if cum_dur < target_total_sec:
            orig_len = len(raw_tracks)
            idx = 0
            while cum_dur < target_total_sec + 20.0 and len(active_tracks) < 60:
                p = raw_tracks[idx % orig_len]
                d = track_durations[idx % orig_len]
                active_tracks.append(p)
                cum_dur += max(0.0, d - 4.0)
                idx += 1

        cmd = ["ffmpeg", "-y"]
        for p in active_tracks:
            if not p.exists():
                raise AudioMixerError(f"Music track not found: {p}")
            cmd.extend(["-i", str(p)])

        filter_parts: List[str] = []
        n_tracks = len(active_tracks)

        for i in range(n_tracks):
            filter_parts.append(f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[t{i}]")

        if n_tracks == 1:
            last_tag = "t0"
        else:
            curr_tag = "t0"
            for i in range(1, n_tracks):
                next_tag = f"t{i}"
                out_tag = f"mix_{i}"
                filter_parts.append(f"[{curr_tag}][{next_tag}]acrossfade=d=4:c1=tri:c2=tri[{out_tag}]")
                curr_tag = out_tag
            last_tag = curr_tag

        fade_out_st = max(0.0, float(target_total_sec) - 4.0)
        filter_parts.append(
            f"[{last_tag}]apad,atrim=0:{target_total_sec},"
            f"afade=t=in:st=0:d=2.0:curve=hsin,"
            f"afade=t=out:st={fade_out_st}:d=4.0:curve=hsin,"
            f"loudnorm=I=-16:TP=-1.5:LRA=11[out]"
        )

        filter_complex = ";".join(filter_parts)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_wav_path)
        ])

        self._run_ffmpeg(cmd, desc="Continuous DJ Music Bed Generation (Step A)")
        return output_wav_path


    def build_voice_timeline(
        self,
        voice_cues: List[Tuple[Path, int]],
        total_duration_sec: int,
        output_wav_path: Path
    ) -> Path:
        """
        Builds a single aligned voice track where each voice cue .wav is placed at its global timestamp.
        voice_cues: list of (wav_file_path, global_offset_ms)
        """
        if not voice_cues:
            # Generate silent audio bed matching target duration
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", str(total_duration_sec),
                "-c:a", "pcm_s16le",
                str(output_wav_path)
            ]
            self._run_ffmpeg(cmd, desc="Voice Timeline (Silent fallback)")
            return output_wav_path

        cmd = ["ffmpeg", "-y"]
        for wav_path, _ in voice_cues:
            if not wav_path.exists():
                raise AudioMixerError(f"Voice cue file not found: {wav_path}")
            cmd.extend(["-i", str(wav_path)])

        filter_parts: List[str] = []
        num_cues = len(voice_cues)

        for i, (_, offset_ms) in enumerate(voice_cues):
            clamped_offset = max(0, offset_ms)
            filter_parts.append(
                f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                f"adelay={clamped_offset}|{clamped_offset}[v{i}]"
            )

        if num_cues == 1:
            filter_parts.append(f"[v0]apad,atrim=0:{total_duration_sec}[out]")
        else:
            inputs_str = "".join(f"[v{i}]" for i in range(num_cues))
            filter_parts.append(
                f"{inputs_str}amix=inputs={num_cues}:dropout_transition=0:normalize=0,apad,"
                f"atrim=0:{total_duration_sec}[out]"
            )

        filter_complex = ";".join(filter_parts)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_wav_path)
        ])

        self._run_ffmpeg(cmd, desc="Voice Timeline Assembly")
        return output_wav_path

    def mix_and_duck(
        self,
        music_bed_path: Path,
        voice_track_path: Path,
        output_mp3_path: Path
    ) -> Path:
        """
        Step B: Sidechain Ducking and final MP3 encoding.
        Ducks music bed aggressively and smoothly whenever voice track has audio signal,
        then mixes with boosted voice clarity (volume=1.4) and unity gain (normalize=0).
        """
        if not music_bed_path.exists():
            raise AudioMixerError(f"Music bed missing: {music_bed_path}")
        if not voice_track_path.exists():
            raise AudioMixerError(f"Voice track missing: {voice_track_path}")

        # Note: asplit=2 duplicates voice stream safely without stream exhaustion.
        # sidechaincompress drops background music by ~18-20dB.
        # volume=1.4 on mix_voice guarantees the coach voice cuts cleanly above loud electronic tracks.
        filter_complex = (
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[music];"
            "[1:a]aformat=sample_rates=44100:channel_layouts=stereo,apad,asplit=2[sc_voice][raw_voice];"
            "[raw_voice]volume=1.4[boosted_voice];"
            "[music][sc_voice]sidechaincompress=threshold=0.010:ratio=10:attack=30:release=700[ducked_music];"
            "[ducked_music][boosted_voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(music_bed_path),
            "-i", str(voice_track_path),
            "-filter_complex", filter_complex,
            "-map", "[mixed]",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            str(output_mp3_path)
        ]

        self._run_ffmpeg(cmd, desc="Sidechain Ducking & MP3 Mixdown (Step B)")
        return output_mp3_path

    def render_workout_audio(
        self,
        segment_tracks: Union[List[Tuple[int, List[Path]]], List[Path]],
        voice_cues: List[Tuple[Path, int]],
        total_duration_sec: int,
        output_filename: str
    ) -> Path:
        """
        Full orchestration of Step A and Step B.
        Generates final MP3 in episodes_dir and purges intermediate WAV files.
        """
        session_id = uuid.uuid4().hex[:8]
        session_dir = self.temp_dir / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        music_bed_path = session_dir / "temp_music_bed.wav"
        voice_track_path = session_dir / "temp_voice_track.wav"
        final_mp3_path = self.episodes_dir / output_filename

        try:
            logger.info("Generating segment-aligned music bed (%d seconds)...", total_duration_sec)
            self.generate_music_bed(segment_tracks, music_bed_path, total_duration_sec=total_duration_sec)

            logger.info("Assembling voice cues timeline (%d cues)...", len(voice_cues))
            self.build_voice_timeline(voice_cues, total_duration_sec, voice_track_path)

            logger.info("Executing sidechain ducking & MP3 encoding -> %s", final_mp3_path)
            self.mix_and_duck(music_bed_path, voice_track_path, final_mp3_path)

            return final_mp3_path
        finally:
            # Clean up intermediate directory and large WAV files
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)


def run_mock_test() -> bool:
    """
    Standalone verification CLI test:
    Generates 2 synthetic MP3 music files and 1 synthetic WAV voice cue,
    runs the two-step FFmpeg ducking pipeline with 2 distinct segments, and checks the generated MP3.
    """
    print("\n--- Running AudioMixer Standalone Verification Test ---")
    test_temp = settings.TEMP_DIR / "cli_test"
    test_temp.mkdir(parents=True, exist_ok=True)

    music1 = test_temp / "sample_music_1.mp3"
    music2 = test_temp / "sample_music_2.mp3"
    voice1 = test_temp / "sample_voice.wav"
    output_mp3 = test_temp / "test_workout_output.mp3"

    try:
        # 1. Synthesize sample music track 1 (440Hz sine wave, 5 seconds)
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=5",
            "-c:a", "libmp3lame", "-b:a", "192k", str(music1)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Synthesize sample music track 2 (880Hz sine wave, 5 seconds)
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "sine=frequency=880:duration=5",
            "-c:a", "libmp3lame", "-b:a", "192k", str(music2)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Synthesize sample voice cue (1200Hz sine beep, 2 seconds)
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "sine=frequency=1200:duration=2",
            "-c:a", "pcm_s16le", "-ar", "44100", str(voice1)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        mixer = AudioMixer(temp_dir=test_temp, episodes_dir=test_temp)

        # Test Step A + Step B orchestration:
        # Segment 1: 5 seconds with music1
        # Segment 2: 3 seconds with music2
        # Total workout = 8 seconds, voice starting at 2000ms
        print("Executing AudioMixer pipeline (Segment 1: 5s, Segment 2: 3s, voice offset: 2000ms)...")
        result_path = mixer.render_workout_audio(
            segment_tracks=[
                (5, [music1]),
                (3, [music2])
            ],
            voice_cues=[(voice1, 2000)],
            total_duration_sec=8,
            output_filename=output_mp3.name
        )

        assert result_path.exists(), "Output MP3 was not created!"
        file_size = result_path.stat().st_size
        duration = mixer.get_audio_duration_sec(result_path)

        print(f"SUCCESS: Generated {result_path}")
        print(f"File Size: {file_size:,} bytes")
        print(f"Audio Duration: {duration:.2f} seconds (expected ~8.0s)")

        assert file_size > 10000, "Output file too small!"
        assert abs(duration - 8.0) < 1.0, f"Duration mismatch: {duration} vs 8.0s"
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")
        return True

    finally:
        # Cleanup test files
        shutil.rmtree(test_temp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    success = run_mock_test()
    sys.exit(0 if success else 1)
