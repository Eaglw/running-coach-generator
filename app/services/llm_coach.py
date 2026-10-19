import json
import logging
import re
import time
from typing import Dict, Optional, Tuple
from openai import OpenAI
import requests

from app.config import settings
from app.models import CoachCue, Segment, WorkoutRequest

logger = logging.getLogger(__name__)

WORKOUT_SYSTEM_PROMPT = """You are an expert running coach and athletics trainer.
Your job is to convert a natural language workout description into a strict, validated JSON specification for an audio-guided workout.

Expected JSON structure:
{
  "workout_title": "Descriptive Workout Title",
  "coach_persona": "tecnico_motivazionale",
  "total_duration_sec": 2700,
  "segments": [
    {
      "name": "Progressive Warmup",
      "duration_sec": 600,
      "intensity": "low",
      "target_bpm_range": [115, 130],
      "target_pace": "5:30/km",
      "coach_cues": [
        {
          "offset_sec": 5,
          "intent": "Welcome runner, explain workout structure and start easy jog.",
          "spoken_text": null
        },
        {
          "offset_sec": 570,
          "intent": "Warmup ending soon: 30 seconds until the main tempo block.",
          "spoken_text": null
        }
      ]
    }
  ]
}

Key rules:
1. `coach_cues` must contain at least 1 cue at the start of each segment (offset_sec ~5s) and 1 preview cue before ending if segment > 3 minutes.
2. `target_bpm_range` must align with intensity: low -> [115, 130], medium -> [130, 150], high/threshold -> [150, 175], max/sprint -> [170, 190].
3. `total_duration_sec` must strictly equal the sum of all segment `duration_sec`.
4. Output ONLY the valid JSON object, without markdown formatting or surrounding conversational text.
"""

SCRIPT_SYSTEM_PROMPT = """You are the voice of the running coach in the runner's headphones during a guided workout.
Generate the exact spoken text for this coaching cue.

Style guidelines:
- Persona: {coach_persona}
  * 'tecnico_motivazionale': professional, encouraging, focused on cadence, breathing, and tall posture.
  * 'calmo_mindful': relaxed, mindful, focused on bodily sensations and deep fluid breathing.
  * 'militare_severo': energetic, disciplined, no excuses, high cadence, push through the fatigue.
- Length: concise (1 to 3 short sentences, under 35 words) so background music stays prominent.
- Output ONLY the spoken text, without quotation marks or explanations.
"""


class LLMCoach:
    """
    Multi-provider LLM integration supporting:
    - Ollama (Local)
    - LM Studio (Local)
    - Google Gemini (OpenAI-compatible)
    - OpenAI / OpenAI-compatible
    - Anthropic Claude (Messages API)
    """

    def _get_provider_info(self) -> Tuple[str, str, str, str]:
        """Fetch current settings dynamically."""
        provider = settings.LLM_PROVIDER.lower().strip()
        base_url = settings.LLM_BASE_URL.strip()
        model = settings.LLM_MODEL.strip()
        api_key = settings.LLM_API_KEY.strip()

        # Handle local providers with empty keys
        if provider in ["ollama", "lmstudio"] and not api_key:
            api_key = provider

        return provider, base_url, model, api_key

    def test_connection(self) -> Dict[str, any]:
        """Test the current LLM connection and return status, response time, and output."""
        provider, base_url, model, api_key = self._get_provider_info()
        start_time = time.time()

        try:
            prompt = "Rispondi esclusivamente con la frase 'Running Coach Pronto' se ricevi questo messaggio."
            text = self._call_llm(
                system_prompt="Sei un assistente AI di test.",
                user_prompt=prompt,
                force_json=False
            )
            elapsed = round((time.time() - start_time) * 1000)
            return {
                "status": "success",
                "provider": provider,
                "model": model,
                "response": text,
                "elapsed_ms": elapsed
            }
        except Exception as e:
            elapsed = round((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "provider": provider,
                "model": model,
                "error": str(e),
                "elapsed_ms": elapsed
            }

    def _call_claude(
        self,
        base_url: str,
        model: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """Direct call to Anthropic Claude Messages API."""
        url = base_url.rstrip("/")
        if not url.endswith("/messages"):
            url = f"{url}/messages" if "/v1" in url else f"{url}/v1/messages"

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        body = {
            "model": model,
            "max_tokens": 1500,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        resp = requests.post(url, headers=headers, json=body, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Claude API Error (HTTP {resp.status_code}): {resp.text}")
        data = resp.json()
        return data["content"][0]["text"].strip()

    def _call_openai_compatible(
        self,
        base_url: str,
        model: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        force_json: bool = False
    ) -> str:
        """Call an OpenAI-compatible endpoint (Gemini, OpenAI, Ollama, LM Studio)."""
        client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2 if force_json else 0.7,
            "max_tokens": 1200 if force_json else 150
        }
        if force_json:
            try:
                # Some local models fail if response_format is passed
                kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or "{}"
            except Exception:
                # Retry without explicit response_format for Ollama / LM Studio models
                kwargs.pop("response_format", None)
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or "{}"
        else:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        force_json: bool = False
    ) -> str:
        """Unified dispatch to the configured LLM provider."""
        provider, base_url, model, api_key = self._get_provider_info()

        if provider == "claude" or "anthropic.com" in base_url:
            if not api_key:
                raise ValueError("Claude API Key mancante nelle impostazioni!")
            return self._call_claude(base_url, model, api_key, system_prompt, user_prompt)
        else:
            return self._call_openai_compatible(base_url, model, api_key, system_prompt, user_prompt, force_json=force_json)

    def parse_prompt_to_workout(
        self,
        prompt: str,
        coach_persona: str = "tecnico_motivazionale"
    ) -> WorkoutRequest:
        """
        Converts natural language dictation into a structured WorkoutRequest.
        Uses LLM if configured, or smart heuristic parser as robust fallback.
        """
        _, _, _, api_key = self._get_provider_info()
        provider = settings.LLM_PROVIDER.lower().strip()

        # Try LLM call if key is present or using local provider (Ollama / LM Studio)
        if api_key or provider in ["ollama", "lmstudio"]:
            try:
                raw_response = self._call_llm(
                    system_prompt=WORKOUT_SYSTEM_PROMPT,
                    user_prompt=f"Persona allenatore: {coach_persona}\nDescrizione: {prompt}",
                    force_json=True
                )
                # Extract JSON even if wrapped in markdown code blocks
                clean_json = raw_response.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()

                # Find first { and last }
                start_idx = clean_json.find("{")
                end_idx = clean_json.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    clean_json = clean_json[start_idx:end_idx+1]

                data = json.loads(clean_json)
                return WorkoutRequest(**data)
            except Exception as e:
                logger.warning("LLM parser failed: %s. Using heuristic fallback parser.", e)

        # Fallback Heuristic Parser
        return self._heuristic_parse(prompt, coach_persona)

    def _heuristic_parse(self, prompt: str, persona: str) -> WorkoutRequest:
        """
        Heuristic rule-based parser in Italian when no LLM is configured.
        """
        prompt_lower = prompt.lower()
        segments: List[Segment] = []

        minutes_matches = re.findall(r"(\d+)\s*(?:min|minuti|m\b)", prompt_lower)
        durations = [int(m) * 60 for m in minutes_matches] if minutes_matches else [600, 1200, 600]

        if len(durations) == 1:
            total = durations[0]
            w_time = max(300, int(total * 0.2))
            c_time = max(300, int(total * 0.2))
            m_time = total - w_time - c_time
            durations = [w_time, m_time, c_time]

        # Build Warmup segment
        segments.append(Segment(
            name="Progressive Warmup",
            duration_sec=durations[0],
            intensity="low",
            target_bpm_range=(118, 128),
            target_pace="5:45/km",
            coach_cues=[
                CoachCue(offset_sec=5, intent="Welcome runner, start easy jog with tall posture"),
                CoachCue(offset_sec=max(10, durations[0] - 30), intent="Notice: 30 seconds until the main workout block")
            ]
        ))

        # Build Central segment
        segments.append(Segment(
            name="Main Workout Block",
            duration_sec=durations[1],
            intensity="medium",
            target_bpm_range=(135, 150),
            target_pace="4:50/km",
            coach_cues=[
                CoachCue(offset_sec=5, intent="Start main block: lock in your target cadence and steady breathing"),
                CoachCue(offset_sec=max(20, int(durations[1] / 2)), intent="Halfway through block: check shoulders and arm swing"),
                CoachCue(offset_sec=max(10, durations[1] - 30), intent="Final 30 seconds: hold your form strong before recovery")
            ]
        ))

        # Build Cooldown segment
        cooldown_dur = durations[2] if len(durations) > 2 else 300
        segments.append(Segment(
            name="Recovery Cooldown",
            duration_sec=cooldown_dur,
            intensity="low",
            target_bpm_range=(110, 125),
            target_pace="6:00/km",
            coach_cues=[
                CoachCue(offset_sec=5, intent="Great work today, bring your heart rate down with an easy jog"),
                CoachCue(offset_sec=max(10, cooldown_dur - 20), intent="Workout complete: summary and recovery cooldown")
            ]
        ))

        total_sec = sum(s.duration_sec for s in segments)
        title = "Custom Guided Running Workout"
        if "soglia" in prompt_lower or "threshold" in prompt_lower or "tempo" in prompt_lower:
            title = "Threshold & Tempo Focus Run"
        elif "progressivo" in prompt_lower or "progressive" in prompt_lower:
            title = "Progressive Tempo Run"
        elif "ripetute" in prompt_lower or "interval" in prompt_lower or "fartlek" in prompt_lower:
            title = "Interval & Fartlek Workout"

        return WorkoutRequest(
            workout_title=title,
            coach_persona=persona,
            total_duration_sec=total_sec,
            segments=segments
        )

    def generate_spoken_script(self, workout: WorkoutRequest) -> WorkoutRequest:
        """
        Populates spoken_text for any coach cue where spoken_text is null.
        """
        for segment in workout.segments:
            for cue in segment.coach_cues:
                if cue.spoken_text and cue.spoken_text.strip():
                    continue

                cue.spoken_text = self._generate_cue_text(
                    intent=cue.intent,
                    persona=workout.coach_persona,
                    segment_name=segment.name,
                    target_pace=segment.target_pace,
                    intensity=segment.intensity
                )

        return workout

    def _generate_cue_text(
        self,
        intent: str,
        persona: str,
        segment_name: str,
        target_pace: Optional[str],
        intensity: str
    ) -> str:
        """
        Generate English/Italian coaching phrase for a single cue.
        """
        _, _, _, api_key = self._get_provider_info()
        provider = settings.LLM_PROVIDER.lower().strip()

        if api_key or provider in ["ollama", "lmstudio"]:
            try:
                system_prompt = SCRIPT_SYSTEM_PROMPT.format(coach_persona=persona)
                user_msg = (
                    f"Segment: {segment_name} (Intensity: {intensity}, Target pace: {target_pace or 'N/A'})\n"
                    f"Cue goal: {intent}\n"
                    f"Generate the exact spoken sentence you will say to the runner."
                )
                text = self._call_llm(system_prompt, user_msg, force_json=False)
                clean = text.strip().strip('"').strip("'")
                if clean:
                    return clean
            except Exception as e:
                logger.debug("LLM cue script generation error: %s. Using heuristic template.", e)

        # Fallback Script Generator
        intent_lower = intent.lower()
        pace_str = f" at {target_pace}" if target_pace else ""

        if any(w in intent_lower for w in ["benvenuto", "welcome", "avvio", "start", "inizio"]):
            if persona == "militare_severo":
                return f"Let's move! Starting with {segment_name}{pace_str}. Stay disciplined, eyes ahead!"
            elif persona == "calmo_mindful":
                return f"Welcome runner. Beginning {segment_name}{pace_str}. Find your rhythm and breathe easy."
            else:
                return f"Welcome runner! Let's kick off with {segment_name}{pace_str}. Relax your shoulders and enjoy the run!"

        elif any(w in intent_lower for w in ["metà", "halfway", "postura", "posture"]):
            if persona == "militare_severo":
                return "Halfway mark! Maintain your cadence, drive through the ankles, no slowing down!"
            else:
                return "You are halfway through this block. Check your posture and breathing: rhythm is spot on, keep going!"

        elif any(w in intent_lower for w in ["preavviso", "30 second", "30 secondi", "tra poco", "soon"]):
            return "Attention runner, final 30 seconds in this block. Prepare for the upcoming pace change."

        elif any(w in intent_lower for w in ["fine", "conclusione", "finish", "complete", "ottimo", "great"]):
            if persona == "militare_severo":
                return "Halt! Workout finished with dedication. Mission accomplished, now recover."
            else:
                return "Workout complete! Fantastic effort and energy today. Walk it out and enjoy your recovery."

        return f"{intent}. Hold your rhythm{pace_str} and stay focused on your movement."


# Singleton LLM coach instance
llm_coach = LLMCoach()
