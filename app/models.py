from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class IntensityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class CoachCue(BaseModel):
    """
    A single coaching voice intervention within a workout segment.
    """
    offset_sec: int = Field(
        ...,
        ge=0,
        description="Seconds relative to segment start when this cue should play"
    )
    intent: str = Field(
        ...,
        min_length=1,
        description="Functional goal of the cue (e.g., pace instruction, motivation, interval preview)"
    )
    spoken_text: Optional[str] = Field(
        default=None,
        description="Exact spoken text for TTS. If null, generated dynamically by LLM Coach."
    )


class Segment(BaseModel):
    """
    A structured phase of the running workout (e.g., Warmup, Interval, Cooldown).
    """
    name: str = Field(..., min_length=1, description="Descriptive name of the segment")
    duration_sec: int = Field(..., gt=0, description="Duration of this segment in seconds")
    intensity: str = Field(default="medium", description="Intensity label (e.g., low, medium, high, max)")
    target_bpm_range: Tuple[int, int] = Field(
        ...,
        description="Target music tempo range [min_bpm, max_bpm]"
    )
    target_pace: Optional[str] = Field(
        default=None,
        description="Pace target for running (e.g., '5:30/km' or 'Zona 2')"
    )
    coach_cues: List[CoachCue] = Field(
        default_factory=list,
        description="Chronological voice interventions during this segment"
    )

    @field_validator("target_bpm_range")
    @classmethod
    def validate_bpm_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        min_bpm, max_bpm = v
        if min_bpm <= 0 or max_bpm <= 0:
            raise ValueError("BPM values must be positive integers")
        if min_bpm > max_bpm:
            raise ValueError(f"min_bpm ({min_bpm}) cannot be greater than max_bpm ({max_bpm})")
        return v

    @field_validator("coach_cues")
    @classmethod
    def validate_cue_offsets(cls, cues: List[CoachCue]) -> List[CoachCue]:
        # Ensure cues are ordered by offset
        return sorted(cues, key=lambda c: c.offset_sec)


class WorkoutRequest(BaseModel):
    """
    Complete specification for generating a coach-guided workout audio track.
    """
    workout_title: str = Field(..., min_length=1, description="Title of the workout episode")
    coach_persona: str = Field(
        default="tecnico_motivazionale",
        description="Tone and style of the voice coach (e.g., 'tecnico_motivazionale', 'calmo_mindful', 'militare_severo')"
    )
    total_duration_sec: int = Field(..., gt=0, description="Expected total duration in seconds")
    language: str = Field(default="it", description="Language for workout titles, cues and voice: 'it' or 'en'")
    segments: List[Segment] = Field(..., min_length=1, description="Ordered list of workout segments")

    @model_validator(mode="after")
    def validate_total_duration(self) -> "WorkoutRequest":
        calculated_total = sum(s.duration_sec for s in self.segments)
        if abs(calculated_total - self.total_duration_sec) > 5:
            # Reconcile if slightly off or raise if significantly inconsistent
            self.total_duration_sec = calculated_total
        return self


class PromptWorkoutRequest(BaseModel):
    """
    Payload for natural language workout prompt or transcribed voice dictation.
    """
    prompt: str = Field(..., min_length=5, description="Natural language description of the workout")
    coach_persona: str = Field(default="tecnico_motivazionale", description="Tone and persona of coach")
    language: str = Field(default="it", description="Coaching voice language: 'it' or 'en'")


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING_SCRIPT = "generating_script"
    SYNTHESIZING_TTS = "synthesizing_tts"
    MIXING_AUDIO = "mixing_audio"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatusResponse(BaseModel):
    """
    Status of an asynchronous workout generation job.
    """
    job_id: str
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    workout_title: Optional[str] = None
    episode_filename: Optional[str] = None
    episode_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EpisodeMetadata(BaseModel):
    """
    Metadata representation for an episode in the podcast feed.
    """
    guid: str
    title: str
    description: str
    filename: str
    file_size_bytes: int
    duration_sec: int
    pub_date: datetime
    enclosure_url: str
