import pytest
from voice_ai.slm import TextCleaner, IntentExtractor, Formatter, SLMResult, BaseSLM

def test_cleaner_fillers():
    raw = "uh I think we should um go out like basically"
    cleaned = TextCleaner.remove_fillers(raw)
    assert "uh" not in cleaned.lower()
    assert "um" not in cleaned.lower()

def test_cleaner_repetitions():
    raw = "schedule schedule a meeting"
    cleaned = TextCleaner.remove_repetitions(raw)
    assert cleaned.lower() == "schedule a meeting"

    phrase_raw = "create a create a python function"
    phrase_cleaned = TextCleaner.remove_repetitions(phrase_raw)
    assert phrase_cleaned.lower() == "create a python function"

def test_cleaner_capitalization_and_punctuation():
    raw = "the weather in chennai is hot"
    cleaned = TextCleaner.fix_capitalization_and_punctuation(raw)
    assert cleaned.startswith("The")
    assert cleaned.endswith(".")
    assert "Chennai" in cleaned

def test_intent_reminder():
    raw = "remind me to submit the machine learning assignment tomorrow at 6"
    res = IntentExtractor.extract_fallback(raw)
    assert res["intent"] == "reminder"
    assert res["action"] == "submit"
    assert "assignment" in res["object"]
    assert "tomorrow at 6" in res["time"]

def test_intent_schedule():
    raw = "schedule a meeting with Rahul tomorrow at ten in the morning"
    res = IntentExtractor.extract_fallback(raw)
    assert res["intent"] == "schedule"
    assert res["action"] == "meeting"
    assert "rahul" in res.get("object", "").lower()


def test_intent_summarize():
    raw = "please summarize this document and give me the three points"
    res = IntentExtractor.extract_fallback(raw)
    assert res["intent"] == "summarize"
    assert "this document" in res["object"]

def test_intent_none():
    raw = "the weather in Chennai is quite hot today"
    res = IntentExtractor.extract_fallback(raw)
    assert res["intent"] == "none"

def test_formatter_bullet_list():
    raw = "the three things we need to finish are testing deployment and documentation make them bullet points"
    fmt_text, detected_fmt, fmt_data = Formatter.format_text(raw)
    assert detected_fmt == "bullet_list"
    assert "•" in fmt_text
    assert len(fmt_data["items"]) >= 3

def test_formatter_numbered_list():
    raw = "here are the tasks first prepare the data second train the model numbered list"
    fmt_text, detected_fmt, fmt_data = Formatter.format_text(raw)
    assert detected_fmt == "numbered_list"
    assert "1." in fmt_text
    assert "2." in fmt_text

def test_slm_result_dataclass():
    res = SLMResult(
        clean_text="Cleaned text.",
        intent="reminder",
        intent_data={"action": "test"},
        format="none",
        final_output="Cleaned text.",
        raw_transcript="raw text",
        latency_ms=12.5,
        model_name="TestSLM",
    )
    d = res.to_dict()
    assert d["clean_text"] == "Cleaned text."
    assert d["intent"] == "reminder"
    assert d["latency_ms"] == 12.5
