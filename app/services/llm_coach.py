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

def get_workout_system_prompt(language: str = "it") -> str:
    lang = language.lower().strip()
    if lang == "en":
        return """You are an expert running coach and athletics trainer.
Your job is to convert a natural language workout description into a strict, validated JSON specification for an audio-guided workout.

CRITICAL REQUIREMENT:
All user-facing titles, segment names, cue intents, and coaching speech MUST BE IN NATURAL MOTIVATING ENGLISH.

Expected JSON structure:
{
  "workout_title": "Progressive Tempo Run with Strides",
  "coach_persona": "tecnico_motivazionale",
  "total_duration_sec": 2700,
  "language": "en",
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
          "intent": "Welcome runner, explain workout goals and start easy jog.",
          "spoken_text": null
        },
        {
          "offset_sec": 180,
          "intent": "Posture check: relax shoulders, upright posture and relaxed breathing.",
          "spoken_text": null
        },
        {
          "offset_sec": 570,
          "intent": "Warmup ending notice: 30 seconds until tempo block.",
          "spoken_text": null
        }
      ]
    }
  ]
}

Key rules:
1. Language: Strictly ENGLISH for `workout_title`, `segment.name`, `intent`, and `spoken_text`.
2. `coach_cues` placement:
   - Segment start (offset_sec ~5s): Announce segment goal, target pace and mindset.
   - Periodic cues: In any segment longer than 3 minutes, schedule a coaching cue EVERY 120 TO 180 SECONDS (every 2-3 minutes maximum). Alternate topics:
     * Posture & Form (relax shoulders, gaze forward at horizon, compact arm swing).
     * Cadence & Footstrike (quick light steps, 170-180 SPM, landing under hips).
     * Breathing (deep diaphragmatic rhythm, exhaling fully to clear fatigue).
     * Effort Check (confirming perceived exertion matches the current intensity zone).
   - Segment warning (offset_sec ~30s before segment duration): Alert runner of upcoming pace change.
3. `target_bpm_range` must align with intensity: low -> [115, 130], medium -> [130, 150], high/threshold -> [150, 175], max/sprint -> [170, 190].
4. `total_duration_sec` must strictly equal the sum of all segment `duration_sec`.
5. Output ONLY the valid JSON object, without markdown formatting or surrounding conversational text.
"""
    # Default: Italian
    return """You are an expert running coach and athletics trainer.
Your job is to convert a natural language workout description into a strict, validated JSON specification for an audio-guided workout.

CRITICAL REQUIREMENT:
All user-facing titles, segment names, cue intents, and coaching speech MUST BE IN NATURAL MOTIVATIONAL ITALIAN, because the neural voice synthesizers (Piper and Kokoro) are strictly Italian models. Do not use English for workout titles, segment names, or cue intents.

Expected JSON structure:
{
  "workout_title": "Corsa Progressiva con Variazioni di Ritmo",
  "coach_persona": "tecnico_motivazionale",
  "total_duration_sec": 2700,
  "language": "it",
  "segments": [
    {
      "name": "Riscaldamento Progressivo",
      "duration_sec": 600,
      "intensity": "low",
      "target_bpm_range": [115, 130],
      "target_pace": "5:30/km",
      "coach_cues": [
        {
          "offset_sec": 5,
          "intent": "Benvenuto runner, spieghiamo la sessione e avviamo una corsa leggera di riscaldamento.",
          "spoken_text": null
        },
        {
          "offset_sec": 180,
          "intent": "Check postura: spalle rilassate, busto eretto e respirazione sciolta.",
          "spoken_text": null
        },
        {
          "offset_sec": 570,
          "intent": "Preavviso fine riscaldamento: 30 secondi al cambio ritmo verso il blocco centrale.",
          "spoken_text": null
        }
      ]
    }
  ]
}

Key rules:
1. Language: Strictly ITALIAN for `workout_title`, `segment.name`, `intent`, and `spoken_text`.
2. `coach_cues` placement:
   - Segment start (offset_sec ~5s): Announce segment goal, target pace and mindset in Italian.
   - Periodic cues: In any segment longer than 3 minutes, schedule a coaching cue EVERY 120 TO 180 SECONDS (every 2-3 minutes maximum). Alternate topics:
     * Postura e Assetto (rilassare le spalle, sguardo all'orizzonte, oscillazione gomiti compatta).
     * Cadenza e Appoggio (passi rapidi e leggeri, 170-180 SPM, appoggio sotto il bacino).
     * Respirazione (ritmo costante e profondo, espirazione completa).
     * Gestione dello Sforzo (percezione della fatica adeguata alla zona d'intensità).
   - Segment warning (offset_sec ~30s before segment duration): Alert runner of upcoming pace change in Italian.
3. `target_bpm_range` must align with intensity: low -> [115, 130], medium -> [130, 150], high/threshold -> [150, 175], max/sprint -> [170, 190].
4. `total_duration_sec` must strictly equal the sum of all segment `duration_sec`.
5. Output ONLY the valid JSON object, without markdown formatting or surrounding conversational text.
"""

def get_script_system_prompt(language: str = "it", coach_persona: str = "tecnico_motivazionale") -> str:
    lang = language.lower().strip()
    if lang == "en":
        return f"""You are the voice of the running coach in the runner's headphones during a guided workout.
Generate the exact spoken text for this coaching cue IN NATURAL ENGLISH.

Style guidelines:
- Language: ENGLISH ONLY.
- Persona: {coach_persona}
  * 'tecnico_motivazionale': professional, encouraging, focused on cadence, breathing, and tall posture.
  * 'calmo_mindful': relaxed, mindful, focused on bodily sensations and deep fluid breathing.
  * 'militare_severo': energetic, disciplined, no excuses, high cadence, push through the fatigue.
- Length: concise (1 to 2 short sentences, 15 to 30 words) so background music stays prominent.
- Output ONLY the spoken English text, without quotation marks or explanations.
"""
    # Default: Italian
    return f"""You are the voice of the running coach in the runner's headphones during a guided workout.
Generate the exact spoken text for this coaching cue IN NATURAL ITALIAN.

CRITICAL:
The voice synthesizers are Italian neural models (Piper / Kokoro). Output ONLY natural, motivating spoken Italian text. Never output English text.

Style guidelines:
- Language: ITALIAN ONLY.
- Persona: {coach_persona}
  * 'tecnico_motivazionale': professional, encouraging, focused on cadence, breathing, and tall posture.
  * 'calmo_mindful': relaxed, mindful, focused on bodily sensations and deep fluid breathing.
  * 'militare_severo': energetic, disciplined, no excuses, high cadence, push through the fatigue.
- Length: concise (1 to 2 short sentences, 15 to 30 words) so background music stays prominent.
- Output ONLY the spoken Italian text, without quotation marks or explanations.
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
        coach_persona: str = "tecnico_motivazionale",
        language: Optional[str] = None
    ) -> WorkoutRequest:
        """
        Converts natural language dictation into a structured WorkoutRequest.
        Uses LLM if configured, or smart heuristic parser as robust fallback.
        """
        target_lang = (language or settings.WORKOUT_LANGUAGE or "it").lower().strip()
        _, _, _, api_key = self._get_provider_info()
        provider = settings.LLM_PROVIDER.lower().strip()

        # Try LLM call if key is present or using local provider (Ollama / LM Studio)
        if api_key or provider in ["ollama", "lmstudio"]:
            try:
                system_prompt = get_workout_system_prompt(target_lang)
                user_msg = f"Coach Persona: {coach_persona}\nTarget Language: {target_lang}\nWorkout Prompt: {prompt}"
                raw_response = self._call_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_msg,
                    force_json=True
                )
                clean_json = raw_response.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()

                start_idx = clean_json.find("{")
                end_idx = clean_json.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    clean_json = clean_json[start_idx:end_idx+1]

                data = json.loads(clean_json)
                data["language"] = target_lang
                parsed = WorkoutRequest(**data)
                return self._ensure_regular_cue_cadence(parsed)
            except Exception as e:
                logger.warning("LLM parser failed: %s. Using heuristic fallback parser.", e)

        # Fallback Heuristic Parser
        heuristic_workout = self._heuristic_parse(prompt, coach_persona, target_lang)
        return self._ensure_regular_cue_cadence(heuristic_workout)

    def _ensure_regular_cue_cadence(self, workout: WorkoutRequest) -> WorkoutRequest:
        """
        Guarantees that no gap between coach interventions exceeds 150-180 seconds (2.5-3 minutes).
        Automatically injects contextual cues (posture, breathing, cadence, effort check)
        tailored to each segment's intensity in the target language.
        """
        lang = (workout.language or settings.WORKOUT_LANGUAGE or "it").lower().strip()

        if lang == "en":
            periodic_cues_by_intensity = {
                "low": [
                    ("respiro", "Breathing check: deep nasal inhale and smooth exhale through the mouth."),
                    ("postura", "Posture check: upright spine, relaxed shoulders and eyes forward on the horizon."),
                    ("cadenza", "Cadence check: light, agile footstrike landing under your hips."),
                    ("ritmo", "Maintain this easy pace, ideal to build aerobic endurance.")
                ],
                "medium": [
                    ("postura", "Posture check: relax your shoulders, compact elbow swing and slight forward lean."),
                    ("respiro", "Breathing check: settle into a rhythmic diaphragmatic breathing pattern."),
                    ("cadenza", "Cadence check: aim for 175-180 strides per minute with reactive footstrikes."),
                    ("sforzo", "Effort check: you are right in Zone 2, conversational pace.")
                ],
                "high": [
                    ("cadenza", "Cadence and drive: keep turnover rapid and crisp, hold your rhythm!"),
                    ("respiro", "Deep breathing: exhale forcefully to clear fatigue and oxygenate your legs."),
                    ("postura", "Strong posture: tall chest to open your lungs, driving elbows."),
                    ("motivazione", "Lock into race pace! Full focus, stay strong and ride the music.")
                ],
                "max": [
                    ("massimale", "Everything you've got! Drive the arms, eyes up and explode off the ground!"),
                    ("respiro", "Breathe with power, keep high cadence and do not yield an inch!"),
                    ("determinazione", "Final seconds of the sprint, empty the tank and finish strong!")
                ]
            }
            start_fmt = "Starting {name}: lock into your target pace of {pace}"
            warn_fmt = "Attention: 30 seconds until the end of {name}, prepare for pace change"
        else:
            periodic_cues_by_intensity = {
                "low": [
                    ("respiro", "Controllo respiro: inspira profondamente con il naso ed espira con la bocca, ritmo fluido e rilassato."),
                    ("postura", "Check postura: busto eretto, spalle morbide e sguardo dritto avanti a te."),
                    ("cadenza", "Frequenza passi: concentrati su appoggi leggeri e agili, senza sforzare."),
                    ("ritmo", "Mantieni questo passo blando, ideale per ossigenare i muscoli e costruire la base aerobica.")
                ],
                "medium": [
                    ("postura", "Check postura: rilassa le spalle, tieni i gomiti vicini al corpo e il busto leggermente proiettato in avanti."),
                    ("respiro", "Controllo respiro: trova una respirazione diaframmatica ritmica e costante con i tuoi passi."),
                    ("cadenza", "Cadenza: punta a centoottanta passi al minuto, appoggio reattivo sotto il bacino senza allungare la falcata."),
                    ("sforzo", "Verifica sforzo: sei in piena Zona Due, devi riuscire a parlare senza andare in debito d'ossigeno.")
                ],
                "high": [
                    ("cadenza", "Spinta e frequenza: mantieni i passi rapidi e scattanti, non cedere di un solo secondo!"),
                    ("respiro", "Respiro profondo: espira con decisione per ripulire l'acido lattico e ossigenare le gambe."),
                    ("postura", "Postura forte: spalle aperte per aprire la cassa toracica, braccia decise che danno il ritmo."),
                    ("motivazione", "Sei nel cuore del ritmo gara! Concentrazione totale, tieni duro e cavalca la musica.")
                ],
                "max": [
                    ("massimale", "Tutto quello che hai! Spingi con le braccia, sguardo alto e scarica tutta l'energia a terra!"),
                    ("respiro", "Espira con forza, tieni alta la frequenza e non mollare nemmeno un metro!"),
                    ("determinazione", "Mancano pochi secondi, dai fondo alle riserve e chiudi questo sprint al massimo!")
                ]
            }
            start_fmt = "Inizio {name}: imposta il ritmo a {pace}"
            warn_fmt = "Attenzione: mancano trenta secondi alla fine di {name}, preparati al cambio di ritmo"

        for segment in workout.segments:
            seg_dur = segment.duration_sec
            if seg_dur < 120:
                continue

            existing_cues = sorted(segment.coach_cues, key=lambda c: c.offset_sec)
            offsets = [c.offset_sec for c in existing_cues]

            # Ensure start cue exists
            if not offsets or offsets[0] > 30:
                existing_cues.insert(0, CoachCue(
                    offset_sec=5,
                    intent=start_fmt.format(name=segment.name, pace=segment.target_pace or ("target pace" if lang == "en" else "ritmo target"))
                ))

            # Ensure end-warning cue exists (30s before end)
            if seg_dur > 180 and (not offsets or (seg_dur - offsets[-1]) > 55):
                existing_cues.append(CoachCue(
                    offset_sec=max(10, seg_dur - 30),
                    intent=warn_fmt.format(name=segment.name)
                ))

            existing_cues.sort(key=lambda c: c.offset_sec)

            # Detect any gap > 180 seconds and insert periodic cues every ~140s
            filled_cues: List[CoachCue] = []
            pool = periodic_cues_by_intensity.get(segment.intensity.lower(), periodic_cues_by_intensity["medium"])
            cue_idx = 0

            for i in range(len(existing_cues)):
                curr_cue = existing_cues[i]
                if filled_cues:
                    prev_offset = filled_cues[-1].offset_sec
                    gap = curr_cue.offset_sec - prev_offset
                    while gap > 180:
                        interp_offset = prev_offset + 140
                        if interp_offset >= curr_cue.offset_sec - 35:
                            break
                        _, cue_intent = pool[cue_idx % len(pool)]
                        cue_idx += 1
                        filled_cues.append(CoachCue(
                            offset_sec=interp_offset,
                            intent=cue_intent
                        ))
                        prev_offset = interp_offset
                        gap = curr_cue.offset_sec - prev_offset

                filled_cues.append(curr_cue)

            segment.coach_cues = filled_cues

        return workout

    def _heuristic_parse(self, prompt: str, persona: str, language: str = "it") -> WorkoutRequest:
        """
        Heuristic rule-based parser in Italian or English when no LLM is configured.
        """
        prompt_lower = prompt.lower()
        segments: List[Segment] = []
        lang = language.lower().strip()

        minutes_matches = re.findall(r"(\d+)\s*(?:min|minuti|m\b)", prompt_lower)
        durations = [int(m) * 60 for m in minutes_matches] if minutes_matches else [600, 1200, 600]

        if len(durations) == 1:
            total = durations[0]
            w_time = max(300, int(total * 0.2))
            c_time = max(300, int(total * 0.2))
            m_time = total - w_time - c_time
            durations = [w_time, m_time, c_time]

        warmup_dur = durations[0]
        central_dur = durations[1]
        cooldown_dur = durations[2] if len(durations) > 2 else 300

        if lang == "en":
            # Warmup English
            warmup_cues = [
                CoachCue(offset_sec=5, intent="Welcome runner, start easy jog with tall posture and relaxed breathing"),
                CoachCue(offset_sec=max(10, warmup_dur - 30), intent="Notice: 30 seconds until the main workout block")
            ]
            if warmup_dur >= 360:
                warmup_cues.insert(1, CoachCue(offset_sec=180, intent="Posture check: relax shoulders and breathe smoothly"))

            segments.append(Segment(
                name="Progressive Warmup",
                duration_sec=warmup_dur,
                intensity="low",
                target_bpm_range=(118, 128),
                target_pace="5:45/km",
                coach_cues=warmup_cues
            ))

            # Central English
            central_cues = [
                CoachCue(offset_sec=5, intent="Start main block: lock into target cadence and steady breathing"),
                CoachCue(offset_sec=max(10, central_dur - 30), intent="Notice: final 30 seconds of effort before recovery")
            ]
            segments.append(Segment(
                name="Main Workout Block",
                duration_sec=central_dur,
                intensity="medium",
                target_bpm_range=(135, 150),
                target_pace="4:50/km",
                coach_cues=central_cues
            ))

            # Cooldown English
            segments.append(Segment(
                name="Cool Down & Recovery",
                duration_sec=cooldown_dur,
                intensity="low",
                target_bpm_range=(110, 125),
                target_pace="6:00/km",
                coach_cues=[
                    CoachCue(offset_sec=5, intent="Great work! Ease down the pace into an easy recovery jog"),
                    CoachCue(offset_sec=max(10, cooldown_dur - 20), intent="Workout complete: walk and recover")
                ]
            ))

            title = "Guided Running Workout"
            if "threshold" in prompt_lower or "tempo" in prompt_lower or "soglia" in prompt_lower:
                title = "Threshold & Tempo Workout"
            elif "progressive" in prompt_lower or "progressivo" in prompt_lower:
                title = "Progressive Tempo Run"
            elif "interval" in prompt_lower or "fartlek" in prompt_lower or "ripetute" in prompt_lower:
                title = "Interval & Fartlek Session"

        else:
            # Warmup Italian
            warmup_cues = [
                CoachCue(offset_sec=5, intent="Benvenuto runner, iniziamo con una corsa facile e postura rilassata"),
                CoachCue(offset_sec=max(10, warmup_dur - 30), intent="Preavviso: mancano 30 secondi all'inizio del blocco centrale")
            ]
            if warmup_dur >= 360:
                warmup_cues.insert(1, CoachCue(offset_sec=180, intent="Check postura e spalle rilassate per riscaldare il corpo"))

            segments.append(Segment(
                name="Riscaldamento Progressivo",
                duration_sec=warmup_dur,
                intensity="low",
                target_bpm_range=(118, 128),
                target_pace="5:45/km",
                coach_cues=warmup_cues
            ))

            # Central Italian
            central_cues = [
                CoachCue(offset_sec=5, intent="Inizio blocco principale: trova il tuo ritmo e stabilizza la respirazione"),
                CoachCue(offset_sec=max(10, central_dur - 30), intent="Preavviso: ultimi 30 secondi di sforzo prima del defaticamento")
            ]
            segments.append(Segment(
                name="Blocco Centrale",
                duration_sec=central_dur,
                intensity="medium",
                target_bpm_range=(135, 150),
                target_pace="4:50/km",
                coach_cues=central_cues
            ))

            # Cooldown Italian
            segments.append(Segment(
                name="Defaticamento Finale",
                duration_sec=cooldown_dur,
                intensity="low",
                target_bpm_range=(110, 125),
                target_pace="6:00/km",
                coach_cues=[
                    CoachCue(offset_sec=5, intent="Ottimo lavoro! Rallenta il ritmo e inizia il defaticamento a corsa sciolta"),
                    CoachCue(offset_sec=max(10, cooldown_dur - 20), intent="Allenamento concluso con successo: complimenti e recupero camminando")
                ]
            ))

            title = "Allenamento Guidato di Corsa"
            if "soglia" in prompt_lower or "threshold" in prompt_lower or "tempo" in prompt_lower:
                title = "Corsa a Ritmo Medio e Soglia"
            elif "progressivo" in prompt_lower or "progressive" in prompt_lower:
                title = "Corsa a Ritmo Progressivo"
            elif "ripetute" in prompt_lower or "interval" in prompt_lower or "fartlek" in prompt_lower:
                title = "Ripetute e Variazioni di Ritmo"

        total_sec = sum(s.duration_sec for s in segments)
        return WorkoutRequest(
            workout_title=title,
            coach_persona=persona,
            total_duration_sec=total_sec,
            language=lang,
            segments=segments
        )

    def generate_spoken_script(self, workout: WorkoutRequest, language: Optional[str] = None) -> WorkoutRequest:
        """
        Populates spoken_text for any coach cue where spoken_text is null.
        """
        target_lang = (language or workout.language or settings.WORKOUT_LANGUAGE or "it").lower().strip()
        for segment in workout.segments:
            for cue in segment.coach_cues:
                if cue.spoken_text and cue.spoken_text.strip():
                    continue

                cue.spoken_text = self._generate_cue_text(
                    intent=cue.intent,
                    persona=workout.coach_persona,
                    segment_name=segment.name,
                    target_pace=segment.target_pace,
                    intensity=segment.intensity,
                    language=target_lang
                )

        return workout

    def _generate_cue_text(
        self,
        intent: str,
        persona: str,
        segment_name: str,
        target_pace: Optional[str],
        intensity: str,
        language: str = "it"
    ) -> str:
        """
        Generate English or Italian coaching phrase for a single cue.
        """
        target_lang = language.lower().strip()
        _, _, _, api_key = self._get_provider_info()
        provider = settings.LLM_PROVIDER.lower().strip()

        if api_key or provider in ["ollama", "lmstudio"]:
            try:
                system_prompt = get_script_system_prompt(target_lang, persona)
                user_msg = (
                    f"Segment: {segment_name} (Intensity: {intensity}, Target pace: {target_pace or 'N/A'})\n"
                    f"Cue goal: {intent}\n"
                    f"Generate the exact spoken sentence you will say to the runner in {target_lang}."
                )
                text = self._call_llm(system_prompt, user_msg, force_json=False)
                clean = text.strip().strip('"').strip("'")
                if clean:
                    return clean
            except Exception as e:
                logger.debug("LLM cue script generation error: %s. Using heuristic template.", e)

        intent_lower = intent.lower()

        # English Fallback
        if target_lang == "en":
            pace_str = f" at {target_pace}" if target_pace else ""
            if any(w in intent_lower for w in ["welcome", "start", "benvenuto", "inizio"]):
                if persona == "militare_severo":
                    return f"Let's move! Starting with {segment_name}{pace_str}. Eyes forward, full focus and hold your cadence!"
                elif persona == "calmo_mindful":
                    return f"Welcome runner. Starting into {segment_name}{pace_str}. Find your natural stride and breathe with ease."
                else:
                    return f"Welcome runner! Let's get moving with {segment_name}{pace_str}. Relax your shoulders, springy steps and enjoy the run!"

            elif any(w in intent_lower for w in ["posture", "shoulder", "form", "postura"]):
                if persona == "militare_severo":
                    return "Running posture! Shoulders down, solid core, ninety-degree elbows and eyes on the horizon!"
                elif persona == "calmo_mindful":
                    return "Check in with your body: unclench your jaw, soften your neck and keep a tall, relaxed spine."
                else:
                    return "Posture check: upright spine, loose shoulders and compact elbows. A relaxed runner is an efficient runner!"

            elif any(w in intent_lower for w in ["breath", "respiro", "respirazione"]):
                return "Breathing check: find a steady two-two rhythm with your footsteps. Exhale fully to oxygenate your muscles."

            elif any(w in intent_lower for w in ["cadence", "step", "frequency", "cadenza"]):
                if persona == "militare_severo":
                    return "Shorten your stride and boost turnover! Quick light taps on the ground, zero wasted bounce!"
                else:
                    return "Focus on cadence: light, quick footstrikes right beneath your hips. Run smooth like silk."

            elif any(w in intent_lower for w in ["halfway", "effort", "pace", "metà", "sforzo"]):
                return "Halfway mark! The pace is strong and consistent: manage your energy smartly."

            elif any(w in intent_lower for w in ["warning", "30 second", "soon", "preavviso"]):
                return "Heads up runner, final thirty seconds in this phase. Prepare mentally for the pace shift!"

            elif any(w in intent_lower for w in ["finish", "complete", "great", "fine", "conclusione"]):
                if persona == "militare_severo":
                    return "Halt! Workout completed with discipline. Mission accomplished, now walk and recover."
                else:
                    return "Workout complete! Fantastic effort today, you gave it your all. Ease down to a walk and enjoy your recovery."

            return f"{intent}. Hold your rhythm{pace_str} and stay smooth."

        # Italian Fallback (Default)
        pace_str = f" a {target_pace}" if target_pace else ""
        if any(w in intent_lower for w in ["benvenuto", "welcome", "avvio", "start", "inizio"]):
            if persona == "militare_severo":
                return f"Si parte! Iniziamo con {segment_name}{pace_str}. Concentrazione massima, sguardo avanti e mantieni la cadenza!"
            elif persona == "calmo_mindful":
                return f"Benvenuto runner. Iniziamo con {segment_name}{pace_str}. Trova il tuo ritmo e respira in modo fluido e rilassato."
            else:
                return f"Ciao runner! Si parte con {segment_name}{pace_str}. Rilassa le spalle, appoggio reattivo sotto il bacino e goditi la corsa!"

        elif any(w in intent_lower for w in ["postura", "posture", "spalle"]):
            if persona == "militare_severo":
                return "Assetto da corsa! Spalle basse, busto saldo, gomiti a novanta gradi e sguardo dritto all'orizzonte!"
            elif persona == "calmo_mindful":
                return "Senti il tuo corpo: decontrai il collo, rilassa i muscoli del viso e mantieni la schiena piacevolmente allungata."
            else:
                return "Check postura: busto eretto, spalle morbide e gomiti compatti. Una corsa rilassata è una corsa efficiente!"

        elif any(w in intent_lower for w in ["respiro", "respirazione", "breath"]):
            if persona == "calmo_mindful":
                return "Focalizzati sul respiro: inspira aria fresca e riempi la pancia, espira a fondo liberando ogni tensione."
            else:
                return "Ascolta il respiro: trova un ritmo regolare due-due con i tuoi passi. Espira a fondo per ossigenare le gambe."

        elif any(w in intent_lower for w in ["cadenza", "frequenza", "passi"]):
            if persona == "militare_severo":
                return "Accorcia la falcata e alza la frequenza! Passi rapidi ed elastici sul terreno, nessun rimbalzo inutile!"
            else:
                return "Concentrati sulla cadenza: passi brevi, leggeri e frequenti. Immagina di correre sul velluto."

        elif any(w in intent_lower for w in ["metà", "halfway", "sforzo", "ritmo", "zona"]):
            if persona == "militare_severo":
                return "Metà blocco completato! La fatica sale ma la determinazione non cala, spingi con grinta!"
            else:
                return "Sei a metà di questa fase. Il ritmo è ottimo e costante: gestisci le energie con intelligenza."

        elif any(w in intent_lower for w in ["preavviso", "30 second", "30 secondi", "tra poco", "soon"]):
            return f"Attenzione runner, ultimi trenta secondi in questa fase. Preparati mentalmente al cambio di ritmo!"

        elif any(w in intent_lower for w in ["fine", "conclusione", "finish", "complete", "ottimo", "great"]):
            if persona == "militare_severo":
                return "Alt! Allenamento portato a termine con successo. Ottima disciplina, ora cammina e recupera."
            else:
                return "Allenamento concluso! Grandissimo impegno oggi, hai dato il massimo. Rallenta il passo e goditi il recupero."

        return f"{intent}. Mantieni il ritmo{pace_str} e concentrati sul tuo movimento."


# Singleton LLM coach instance
llm_coach = LLMCoach()
