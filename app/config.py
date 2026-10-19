from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and/or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base storage directory
    DATA_DIR: Path = Field(default=Path("./data"), description="Root data directory")

    # Subdirectories
    MUSIC_DIR: Path = Field(default=Path("./data/music"), description="Directory where source music tracks reside")
    CACHE_DIR: Path = Field(default=Path("./data/cache"), description="Directory for library index and analysis cache")
    PODCAST_DIR: Path = Field(default=Path("./data/podcasts"), description="Directory where podcast feeds and episodes are stored")
    TEMP_DIR: Path = Field(default=Path("./data/temp"), description="Directory for temporary audio processing files")

    # Web & RSS configuration
    BASE_URL: str = Field(
        default="http://localhost:8000",
        description="Public base URL accessible by Apple Podcasts clients (e.g. https://coach.local)"
    )
    PODCAST_TITLE: str = Field(default="Running Coach Daily", description="Podcast feed channel title")
    PODCAST_DESCRIPTION: str = Field(
        default="Personalized AI-guided running workouts synced to your music and Apple Podcasts",
        description="Podcast channel description"
    )
    PODCAST_AUTHOR: str = Field(default="AI Running Coach", description="Podcast author name")
    PODCAST_LANGUAGE: str = Field(default="it-IT", description="Podcast language (RFC 5646)")
    PODCAST_IMAGE_URL: Optional[str] = Field(
        default="http://localhost:8000/podcast/cover.jpg",
        description="Public URL for podcast artwork"
    )

    # LLM Settings (OpenAI-compatible)
    LLM_API_KEY: str = Field(default="", description="API key for LLM provider")
    LLM_BASE_URL: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description="Base URL for OpenAI-compatible client"
    )
    LLM_MODEL: str = Field(default="gemini-2.5-flash", description="Model name to use for cue script generation")

    # TTS Settings
    TTS_ENGINE: str = Field(default="piper", description="TTS engine to use: 'piper' or 'kokoro'")
    TTS_VOICE: str = Field(default="it_IT-riccardo-x_low", description="Voice identifier for TTS")

    @property
    def EPISODES_DIR(self) -> Path:
        return self.PODCAST_DIR / "episodes"

    @property
    def RSS_FEED_PATH(self) -> Path:
        return self.PODCAST_DIR / "feed.xml"

    @property
    def LIBRARY_CACHE_PATH(self) -> Path:
        return self.CACHE_DIR / "library_index.json"

    def ensure_directories(self) -> None:
        """Ensure all required runtime directories exist."""
        for path in [
            self.DATA_DIR,
            self.MUSIC_DIR,
            self.CACHE_DIR,
            self.PODCAST_DIR,
            self.EPISODES_DIR,
            self.TEMP_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Singleton configuration instance
settings = Settings()
settings.ensure_directories()
