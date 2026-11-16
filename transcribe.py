"""Local speech-to-text for PawPal+ using faster-whisper.

Runs entirely on this machine: no API key, no cost, and the audio never leaves
the computer. It never calls Claude. The transcript is shown in an editable
box so the user can fix mistakes before parsing.
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import List, Optional

# base.en: about 145 MB, a good speed/accuracy balance for short English
# commands. Try "small.en" (more accurate, slower) or "tiny.en" (faster, less
# accurate). The model downloads once on first use, then works offline.
MODEL_SIZE = "base.en"


class TranscribeError(Exception):
    """Speech couldn't be turned into text; the message is safe to show the user."""


@lru_cache(maxsize=1)
def _load_model(size: str = MODEL_SIZE):
    """Load the Whisper model once and reuse it for every later recording."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscribeError(
            "Speech recognition isn't installed. Run: pip install faster-whisper"
        )
    try:
        return WhisperModel(size, device="cpu", compute_type="int8")
    except Exception as error:  # e.g. first-run download failed because there is no internet
        raise TranscribeError(
            f"Couldn't load the speech model '{size}'. The first run needs internet "
            f"to download it. ({error})"
        )


def transcribe(audio_bytes: bytes, pet_names: Optional[List[str]] = None) -> str:
    """Turn a recording (WAV bytes from st.audio_input) into text.

    `pet_names` is given to Whisper as a hint so names like "Mochi" are
    spelled correctly; it helps but doesn't guarantee it. Raises
    TranscribeError if the recording is empty, nothing was heard, or speech
    recognition isn't available.
    """
    if not audio_bytes:
        raise TranscribeError("The recording is empty. Please record again.")

    model = _load_model()

    prompt = "Pet care tasks."
    if pet_names:
        prompt += f" Pets: {', '.join(pet_names)}."

    try:
        segments, _info = model.transcribe(
            io.BytesIO(audio_bytes),
            language="en",
            initial_prompt=prompt,
            vad_filter=True,  # skip silence, which Whisper can otherwise "hear" words in
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        raise TranscribeError(f"Couldn't read that recording. Please try again. ({error})")

    if not text:
        raise TranscribeError("I didn't hear anything. Please record again, a bit closer to the mic.")
    return text
