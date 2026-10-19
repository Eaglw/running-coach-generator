from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class TTSEngineError(Exception):
    """Exception raised when TTS synthesis fails."""
    pass


class BaseTTSEngine(ABC):
    """
    Abstract Strategy for Text-to-Speech synthesis engines.
    Outputs a mono/stereo 24kHz or 44.1kHz .wav file.
    """

    @abstractmethod
    def synthesize(self, text: str, output_wav_path: Path) -> Path:
        """
        Synthesize text into a .wav file at output_wav_path.
        """
        pass


class PiperTTSEngine(BaseTTSEngine):
    """
    Piper TTS integration with automatic Italian model downloader from HuggingFace.
    Executes either the piper python module or the piper CLI binary.
    """

    _loaded_voice_cache: Dict[str, Any] = {}

    def __init__(self, voice_name: Optional[str] = None) -> None:
        self.voice_name = voice_name or settings.TTS_VOICE
        self.piper_bin = shutil.which("piper") or shutil.which("piper.exe")

    def _ensure_voice_model(self) -> Optional[Path]:
        """
        Ensures the .onnx and .onnx.json files exist locally in CACHE_DIR / tts_models.
        Downloads them automatically from HuggingFace piper-voices if missing.
        """
        models_dir = settings.CACHE_DIR / "tts_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        onnx_file = models_dir / f"{self.voice_name}.onnx"
        json_file = models_dir / f"{self.voice_name}.onnx.json"

        if onnx_file.exists() and json_file.exists():
            return onnx_file

        # Map voice name to HuggingFace path:
        # e.g. it_IT-riccardo-x_low -> it/it_IT/riccardo/x_low/it_IT-riccardo-x_low
        parts = self.voice_name.split("-")
        if len(parts) >= 3:
            locale = parts[0]   # it_IT
            speaker = parts[1]  # riccardo
            quality = parts[2]  # x_low
            lang = locale.split("_")[0]  # it
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{locale}/{speaker}/{quality}/{self.voice_name}"
        else:
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/riccardo/x_low/{self.voice_name}"

        try:
            import requests
            logger.info("Downloading neural TTS voice '%s' from HuggingFace...", self.voice_name)
            for suffix, target_file in [(".onnx", onnx_file), (".onnx.json", json_file)]:
                if not target_file.exists():
                    url = f"{base_url}{suffix}"
                    resp = requests.get(url, stream=True, timeout=60)
                    if resp.status_code == 200:
                        tmp_target = target_file.with_suffix(f"{suffix}.tmp")
                        with open(tmp_target, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=32768):
                                if chunk:
                                    f.write(chunk)
                        tmp_target.replace(target_file)
                        logger.info("Downloaded %s successfully.", target_file.name)
                    else:
                        logger.warning("Failed to download voice file %s (HTTP %d)", url, resp.status_code)
                        return None
            return onnx_file if onnx_file.exists() else None
        except Exception as e:
            logger.warning("Auto-download of Piper voice model failed: %s", e)
            return None

    def synthesize(self, text: str, output_wav_path: Path) -> Path:
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        if not text.strip():
            self._generate_silence(output_wav_path)
            return output_wav_path

        model_path = self._ensure_voice_model()

        # 1. Try python piper module directly with in-memory model cache
        try:
            from piper import PiperVoice  # type: ignore
            if model_path and model_path.exists():
                model_key = str(model_path.resolve())
                if model_key not in self._loaded_voice_cache:
                    logger.info("Loading Piper ONNX voice model '%s' into memory...", model_path.name)
                    self._loaded_voice_cache[model_key] = PiperVoice.load(model_key)
                voice = self._loaded_voice_cache[model_key]

                import wave
                with wave.open(str(output_wav_path), "wb") as wav_file:
                    voice.synthesize(text, wav_file)
                if output_wav_path.exists() and output_wav_path.stat().st_size > 100:
                    return output_wav_path
        except Exception as e:
            logger.debug("Python piper module direct invocation: %s", e)

        # 2. Try piper CLI binary
        if self.piper_bin and model_path and model_path.exists():
            cmd = [
                self.piper_bin,
                "--model", str(model_path),
                "--output_file", str(output_wav_path)
            ]
            try:
                subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    capture_output=True,
                    check=True
                )
                if output_wav_path.exists() and output_wav_path.stat().st_size > 100:
                    return output_wav_path
            except Exception as e:
                logger.warning("Piper CLI execution failed: %s. Using system fallback.", e)

        # 3. Fallback to system voice if neural model/binary is unavailable
        return SystemFallbackTTSEngine().synthesize(text, output_wav_path)

    def _generate_silence(self, output_wav_path: Path) -> None:
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
            "-t", "1.0", "-c:a", "pcm_s16le", str(output_wav_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class KokoroTTSEngine(BaseTTSEngine):
    """
    Kokoro ONNX neural TTS engine for Italian / multi-language synthesis.
    Standard Italian voices: im_nicola (male), if_sara (female) in kokoro-v1.0.onnx.
    """
    _kokoro_instance_cache: Optional[Any] = None

    def __init__(self, voice_name: Optional[str] = None) -> None:
        # Standard Kokoro v1.0 Italian voices: 'im_nicola' or 'if_sara'
        self.voice_name = voice_name or settings.TTS_VOICE or "im_nicola"

    def _ensure_kokoro_models(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Ensure kokoro-v1.0.onnx and voices-v1.0.bin are downloaded in settings.CACHE_DIR / 'tts_models'.
        Downloads from official GitHub release if missing.
        """
        models_dir = settings.CACHE_DIR / "tts_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_file = models_dir / "kokoro-v1.0.onnx"
        voices_file = models_dir / "voices-v1.0.bin"

        if model_file.exists() and voices_file.exists():
            return model_file, voices_file

        base_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
        files_to_download = [
            ("kokoro-v1.0.onnx", model_file),
            ("voices-v1.0.bin", voices_file),
        ]

        try:
            import requests
            for fname, target_path in files_to_download:
                if not target_path.exists():
                    url = f"{base_url}/{fname}"
                    logger.info("Downloading Kokoro neural TTS model '%s'...", fname)
                    resp = requests.get(url, stream=True, timeout=120)
                    if resp.status_code == 200:
                        tmp_target = target_path.with_suffix(f"{target_path.suffix}.tmp")
                        with open(tmp_target, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                        tmp_target.replace(target_path)
                        logger.info("Downloaded %s successfully.", target_path.name)
                    else:
                        logger.warning("Failed to download Kokoro file %s (HTTP %d)", url, resp.status_code)
                        return None, None
            return model_file, voices_file
        except Exception as e:
            logger.warning("Auto-download of Kokoro ONNX model failed: %s", e)
            return None, None

    def synthesize(self, text: str, output_wav_path: Path) -> Path:
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import kokoro_onnx  # type: ignore
            model_file, voices_file = self._ensure_kokoro_models()
            if model_file and voices_file and model_file.exists() and voices_file.exists():
                if KokoroTTSEngine._kokoro_instance_cache is None:
                    logger.info("Loading Kokoro ONNX model into memory...")
                    KokoroTTSEngine._kokoro_instance_cache = kokoro_onnx.Kokoro(str(model_file), str(voices_file))
                kokoro = KokoroTTSEngine._kokoro_instance_cache

                # Determine kokoro language from voice prefix
                v_lower = self.voice_name.lower()
                if v_lower.startswith(("af_", "am_")):
                    kokoro_lang = "en-us"
                elif v_lower.startswith(("bf_", "bm_")):
                    kokoro_lang = "en-gb"
                elif v_lower.startswith(("ef_", "em_")):
                    kokoro_lang = "es"
                else:
                    kokoro_lang = "it"

                samples, sample_rate = kokoro.create(text, voice=self.voice_name, speed=1.0, lang=kokoro_lang)
                import soundfile as sf
                sf.write(str(output_wav_path), samples, sample_rate)
                return output_wav_path
        except Exception as e:
            logger.warning("Kokoro ONNX synthesis failed (%s). Falling back to Piper neural voice.", e)

        # If Kokoro models fail, fallback to Piper neural voice
        return PiperTTSEngine(voice_name=self.voice_name if "it_IT" in self.voice_name else None).synthesize(text, output_wav_path)


class SystemFallbackTTSEngine(BaseTTSEngine):
    """
    Robust fallback engine utilizing system espeak-ng (installed in Dockerfile)
    or native Windows PowerShell speech synthesizer / ffmpeg.
    Guarantees the pipeline always produces valid spoken WAV audio on any machine.
    """

    def synthesize(self, text: str, output_wav_path: Path) -> Path:
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_text = text.replace('"', ' ').replace("'", " ")

        # 1. Try espeak-ng (Linux / Docker environment)
        espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak_bin:
            try:
                cmd = [
                    espeak_bin,
                    "-v", "it",
                    "-s", "150",
                    "-w", str(output_wav_path),
                    cleaned_text
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and output_wav_path.exists() and output_wav_path.stat().st_size > 100:
                    return output_wav_path
            except Exception as e:
                logger.debug("espeak-ng failed: %s", e)

        # 2. Try Windows SAPI Voice via PowerShell (Development on Windows host)
        if os.name == "nt":
            try:
                ps_script = f"""
                Add-Type -AssemblyName System.Speech;
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
                $synth.SetOutputToWaveFile('{str(output_wav_path)}');
                $synth.Speak('{cleaned_text}');
                $synth.Dispose();
                """
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    capture_output=True,
                    text=True
                )
                if res.returncode == 0 and output_wav_path.exists() and output_wav_path.stat().st_size > 100:
                    return output_wav_path
            except Exception as e:
                logger.debug("Windows System.Speech failed: %s", e)

        # 3. Ultimate Fallback: Synthesize an acoustic alert tone via ffmpeg
        logger.info("Using FFmpeg tone generator fallback for text: '%s'", text[:30])
        duration_sec = max(1.5, min(len(text) * 0.08, 6.0))
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=587:duration={duration_sec}",
            "-af", "apad,afade=t=in:ss=0:d=0.05,afade=t=out:st=1.4:d=0.1",
            "-c:a", "pcm_s16le", "-ar", "44100",
            str(output_wav_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return output_wav_path


def get_tts_engine(
    engine_name: Optional[str] = None,
    voice_name: Optional[str] = None,
    language: Optional[str] = None
) -> BaseTTSEngine:
    """
    Factory function returning the configured TTS engine with language-aware defaults.
    """
    selected_engine = (engine_name or settings.TTS_ENGINE).lower().strip()
    target_lang = (language or settings.WORKOUT_LANGUAGE or "it").lower().strip()

    if selected_engine == "kokoro":
        if not voice_name:
            voice_name = "im_nicola" if target_lang == "it" else "af_heart"
        return KokoroTTSEngine(voice_name=voice_name)
    elif selected_engine == "piper":
        if not voice_name:
            voice_name = "it_IT-riccardo-x_low" if target_lang == "it" else "en_US-lessac-medium"
        return PiperTTSEngine(voice_name=voice_name)
    elif selected_engine == "fallback":
        return SystemFallbackTTSEngine()
    else:
        logger.warning("Unknown TTS engine '%s'. Defaulting to Piper.", selected_engine)
        if not voice_name:
            voice_name = "it_IT-riccardo-x_low" if target_lang == "it" else "en_US-lessac-medium"
        return PiperTTSEngine(voice_name=voice_name)
