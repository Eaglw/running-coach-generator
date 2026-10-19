import json
import logging
import re
from typing import Optional
from openai import OpenAI

from app.config import settings
from app.models import CoachCue, Segment, WorkoutRequest

logger = logging.getLogger(__name__)

WORKOUT_SYSTEM_PROMPT = """Sei un esperto allenatore di atletica e corsa su strada.
Il tuo compito è convertire la descrizione naturale di un allenamento in un oggetto JSON rigoroso e valido per un audio-workout.

Formato JSON atteso:
{
  "workout_title": "Titolo descrittivo dell'allenamento",
  "coach_persona": "tecnico_motivazionale",
  "total_duration_sec": 2700,
  "segments": [
    {
      "name": "Riscaldamento",
      "duration_sec": 600,
      "intensity": "low",
      "target_bpm_range": [115, 130],
      "target_pace": "5:30/km",
      "coach_cues": [
        {
          "offset_sec": 5,
          "intent": "Benvenuto, spiegazione allenamento e avvio corsa blanda.",
          "spoken_text": null
        },
        {
          "offset_sec": 570,
          "intent": "Avviso fine riscaldamento: tra 30 secondi inizia la fase centrale.",
          "spoken_text": null
        }
      ]
    }
  ]
}

Regole importanti:
1. `coach_cues` deve contenere almeno 1 cue iniziale all'inizio di ciascun segmento (offset_sec ~5s) e 1 cue prima del termine se il segmento dura più di 3 minuti.
2. `target_bpm_range` deve essere coerente: low intensity -> [115, 130], medium -> [130, 150], high/threshold -> [150, 175], max/sprint -> [170, 190].
3. `total_duration_sec` deve essere la somma esatta dei `duration_sec` di tutti i segmenti.
4. Restituisci SOLO ed esclusivamente il codice JSON valido, senza testo o markdown aggiuntivo.
"""

SCRIPT_SYSTEM_PROMPT = """Sei la voce del running coach in cuffia durante un allenamento guidato.
Genera il testo parlato esatto in lingua italiana per questo cue audio.

Linee guida di stile:
- Persona: {coach_persona}
  * 'tecnico_motivazionale': professionale, incoraggiante, focalizzato su ritmo, respirazione e postura.
  * 'calmo_mindful': rilassato, consapevole, attenzione alle sensazioni del corpo e al respiro.
  * 'militare_severo': energico, disciplinato, senza scuse, ritmo sostenuto e spinta massimale.
- Lunghezza: breve e concisa (da 1 a 3 frasi, massimo 35 parole) così da lasciare spazio alla musica.
- Scrivi solo il testo parlato, senza virgolette o introduzioni.
"""


class LLMCoach:
    """
    LLM integration for:
    1. Parsing natural language workout prompts to structured WorkoutRequest.
    2. Generating natural Italian spoken script for each coach cue.
    """

    def __init__(self) -> None:
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL
        self.model = settings.LLM_MODEL
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> Optional[OpenAI]:
        if self._client is None and self.api_key:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    def parse_prompt_to_workout(
        self,
        prompt: str,
        coach_persona: str = "tecnico_motivazionale"
    ) -> WorkoutRequest:
        """
        Converts natural language dictation into a structured WorkoutRequest.
        Uses LLM if API key is provided, or smart heuristic parser as robust fallback.
        """
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": WORKOUT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Persona allenatore: {coach_persona}\nDescrizione: {prompt}"}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                raw_json = response.choices[0].message.content or "{}"
                data = json.loads(raw_json)
                return WorkoutRequest(**data)
            except Exception as e:
                logger.warning("LLM parser failed: %s. Using heuristic fallback parser.", e)

        # Fallback Heuristic Parser
        return self._heuristic_parse(prompt, coach_persona)

    def _heuristic_parse(self, prompt: str, persona: str) -> WorkoutRequest:
        """
        Heuristic rule-based parser in Italian when no LLM API key is present.
        """
        prompt_lower = prompt.lower()
        segments: List[Segment] = []

        # Find numbers of minutes mentioned in prompt
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
            name="Riscaldamento Progressivo",
            duration_sec=durations[0],
            intensity="low",
            target_bpm_range=(118, 128),
            target_pace="5:45/km",
            coach_cues=[
                CoachCue(offset_sec=5, intent="Benvenuto, avvio riscaldamento e postura"),
                CoachCue(offset_sec=max(10, durations[0] - 30), intent="Preavviso passaggio al ritmo centrale")
            ]
        ))

        # Build Central segment
        segments.append(Segment(
            name="Fase Centrale di Lavoro",
            duration_sec=durations[1],
            intensity="medium",
            target_bpm_range=(135, 150),
            target_pace="4:50/km",
            coach_cues=[
                CoachCue(offset_sec=5, intent="Inizio blocco centrale: trova il ritmo e respira costante"),
                CoachCue(offset_sec=max(20, int(durations[1] / 2)), intent="Metà segmento: verifica postura e braccia"),
                CoachCue(offset_sec=max(10, durations[1] - 30), intent="Ultimi 30 secondi: tieni forte prima del defaticamento")
            ]
        ))

        # Build Cooldown segment
        cooldown_dur = durations[2] if len(durations) > 2 else 300
        segments.append(Segment(
            name="Defaticamento e Rientro",
            duration_sec=cooldown_dur,
            intensity="low",
            target_bpm_range=(110, 125),
            target_pace="6:00/km",
            coach_cues=[
                CoachCue(offset_sec=5, intent="Ottimo lavoro, rallenta progressivamente il passo"),
                CoachCue(offset_sec=max(10, cooldown_dur - 20), intent="Conclusione allenamento e riepilogo finale")
            ]
        ))

        total_sec = sum(s.duration_sec for s in segments)
        title = "Allenamento Guidato Personalizzato"
        if "soglia" in prompt_lower:
            title = "Focus Soglia e Ritmo Medio"
        elif "progressivo" in prompt_lower:
            title = "Corsa Progressiva Guidata"
        elif "ripetute" in prompt_lower or "intervalli" in prompt_lower:
            title = "Intervalli & Fartlek"

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
                    continue  # Already provided by user

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
        Generate Italian coaching phrase for a single cue.
        """
        if self.client:
            try:
                system_prompt = SCRIPT_SYSTEM_PROMPT.format(coach_persona=persona)
                user_msg = (
                    f"Segmento: {segment_name} (Intensità: {intensity}, Passo target: {target_pace or 'N/A'})\n"
                    f"Obiettivo cue: {intent}\n"
                    f"Genera la frase esatta che dirai al runner."
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.7,
                    max_tokens=80
                )
                text = (response.choices[0].message.content or "").strip().strip('"')
                if text:
                    return text
            except Exception as e:
                logger.debug("LLM cue script generation error: %s. Using heuristic template.", e)

        # Fallback Italian Script Generator
        intent_lower = intent.lower()
        pace_str = f" a {target_pace}" if target_pace else ""

        if "benvenuto" in intent_lower or "avvio" in intent_lower or "inizio" in intent_lower:
            if persona == "militare_severo":
                return f"Si comincia! Partiamo con {segment_name}{pace_str}. Concentrati e non distrarti!"
            elif persona == "calmo_mindful":
                return f"Ciao runner. Iniziamo con {segment_name}{pace_str}. Ascolta il tuo respiro e trova fluidità."
            else:
                return f"Ciao runner! Iniziamo con {segment_name}{pace_str}. Spalle rilassate, postura alta e buona corsa!"

        elif "metà" in intent_lower or "postura" in intent_lower:
            if persona == "militare_severo":
                return "Siamo a metà! Mantieni la cadenza, niente cali di ritmo, spingi con le caviglie!"
            else:
                return "Siamo esattamente a metà blocco. Controlla le spalle e la respirazione: ritmo perfetto, continua così!"

        elif "preavviso" in intent_lower or "30 secondi" in intent_lower or "tra poco" in intent_lower:
            return "Attenzione runner, ultimi 30 secondi in questo blocco. Preparati al prossimo cambio di ritmo."

        elif "fine" in intent_lower or "conclusione" in intent_lower or "ottimo" in intent_lower:
            if persona == "militare_severo":
                return "Stop! Allenamento completato con determinazione. Ottimo lavoro, ora recupera."
            else:
                return "Allenamento concluso con successo! Ottima gestione delle energie. Fai qualche minuto di stretching."

        # Generic default
        return f"{intent}. Mantieni il ritmo{pace_str} e concentrati sul movimento."


# Singleton LLM coach instance
llm_coach = LLMCoach()
