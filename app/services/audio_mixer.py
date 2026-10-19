import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import uuid

from app.config import settings

logger = logging.getLogger(__name__)


class AudioMixerError(Exception):
    """Exception raised when an audio mixing or FFmpeg step fails."""
    pass


class AudioMixer:
    """
    Two-step FFmpeg audio mixing engine:
    1. Step A: Music bed concatenation with 3s crossfade, trimming, and EBU R128 loudnorm.
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
        track_paths: List[Path],
        target_duration_sec: int,
        output_wav_path: Path
    ) -> Path:
        """
        Step A: Concatenate music tracks with 3-second crossfades and normalize loudness.
        """
        if not track_paths:
            raise AudioMixerError("Cannot generate music bed: track_paths is empty.")

        for p in track_paths:
            if not p.exists():
                raise AudioMixerError(f"Music track not found: {p}")

        num_tracks = len(track_paths)
        cmd = ["ffmpeg", "-y"]

        for p in track_paths:
            cmd.extend(["-i", str(p)])

        filter_parts: List[str] = []

        if num_tracks == 1:
            # Single track: format and ensure duration
            filter_parts.append(
                f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo,aloop=loop=-1:size=2e+09,"
                f"atrim=0:{target_duration_sec},loudnorm=I=-16:TP=-1.5:LRA=11[out]"
            )
        else:
            # Format all inputs to common sample rate and stereo layout
            for i in range(num_tracks):
                filter_parts.append(f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{i}]")

            # Chain crossfades: acrossfade=d=3:c1=tri:c2=tri
            current_tag = "a0"
            for i in range(1, num_tracks):
                next_tag = f"a{i}"
                out_tag = f"m{i}"
                filter_parts.append(
                    f"[{current_tag}][{next_tag}]acrossfade=d=3:c1=tri:c2=tri[{out_tag}]"
                )
                current_tag = out_tag

            # Loop if necessary to cover target duration, trim, and apply EBU R128 loudnorm
            filter_parts.append(
                f"[{current_tag}]aloop=loop=-1:size=2e+09,atrim=0:{target_duration_sec},"
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

        self._run_ffmpeg(cmd, desc="Music Bed Generation (Step A)")
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
                "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
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
        Ducks music bed whenever voice track has audio signal, then mixes with unity gain (normalize=0).
        """
        if not music_bed_path.exists():
            raise AudioMixerError(f"Music bed missing: {music_bed_path}")
        if not voice_track_path.exists():
            raise AudioMixerError(f"Voice track missing: {voice_track_path}")

        # Note: apad on voice track ensures sidechaincompress doesn't truncate the music bed early.
        filter_complex = (
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[music];"
            "[1:a]aformat=sample_rates=44100:channel_layouts=stereo,apad[voice];"
            "[music][voice]sidechaincompress=threshold=0.015:ratio=6:attack=50:release=600[ducked_music];"
            "[ducked_music][voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed]"
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
        track_paths: List[Path],
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
            logger.info("Generating music bed (%d seconds)...", total_duration_sec)
            self.generate_music_bed(track_paths, total_duration_sec, music_bed_path)

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
    runs the two-step FFmpeg ducking pipeline, and checks the generated MP3.
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

        # Test Step A + Step B orchestration for an 8-second total workout, voice starting at 2000ms
        print("Executing AudioMixer pipeline (target duration: 8 seconds, voice offset: 2000ms)...")
        result_path = mixer.render_workout_audio(
            track_paths=[music1, music2],
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
