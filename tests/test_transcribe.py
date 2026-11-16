"""Tests for transcribe.py. They use a fake Whisper model, so no model download,
microphone or network access is needed."""
from types import SimpleNamespace

import pytest

import transcribe as tr


class FakeModel:
    """Mimics WhisperModel.transcribe: returns (segments, info)."""

    def __init__(self, texts=(), error=None):
        self.texts, self.error, self.kwargs = texts, error, None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return iter(SimpleNamespace(text=t) for t in self.texts), SimpleNamespace()


def use(monkeypatch, model):
    monkeypatch.setattr(tr, "_load_model", lambda *args, **kwargs: model)


def test_empty_recording_is_rejected(monkeypatch):
    use(monkeypatch, FakeModel(["never used"]))

    with pytest.raises(tr.TranscribeError, match="empty"):
        tr.transcribe(b"")


def test_segments_are_joined_and_trimmed(monkeypatch):
    use(monkeypatch, FakeModel([" walk Mochi ", "at 8 am "]))

    assert tr.transcribe(b"audio") == "walk Mochi at 8 am"


def test_pet_names_are_sent_as_a_hint(monkeypatch):
    model = FakeModel(["hi"])
    use(monkeypatch, model)

    tr.transcribe(b"audio", ["Mochi", "Max"])

    assert "Mochi, Max" in model.kwargs["initial_prompt"]
    assert model.kwargs["language"] == "en"
    assert model.kwargs["vad_filter"] is True


def test_no_pet_names_still_sends_a_prompt(monkeypatch):
    model = FakeModel(["hi"])
    use(monkeypatch, model)

    tr.transcribe(b"audio")

    assert "Pets:" not in model.kwargs["initial_prompt"]


def test_silence_gives_a_friendly_error(monkeypatch):
    use(monkeypatch, FakeModel([]))

    with pytest.raises(tr.TranscribeError, match="didn't hear anything"):
        tr.transcribe(b"audio")


def test_whitespace_only_result_counts_as_silence(monkeypatch):
    use(monkeypatch, FakeModel(["  ", " "]))

    with pytest.raises(tr.TranscribeError, match="didn't hear anything"):
        tr.transcribe(b"audio")


def test_unreadable_recording_gives_a_friendly_error(monkeypatch):
    use(monkeypatch, FakeModel(error=RuntimeError("bad audio")))

    with pytest.raises(tr.TranscribeError, match="Couldn't read that recording"):
        tr.transcribe(b"audio")


def test_model_load_failure_gives_a_friendly_error(monkeypatch):
    def broken_model(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setattr("faster_whisper.WhisperModel", broken_model)
    tr._load_model.cache_clear()
    try:
        with pytest.raises(tr.TranscribeError, match="Couldn't load the speech model"):
            tr._load_model("base.en")
    finally:
        tr._load_model.cache_clear()
