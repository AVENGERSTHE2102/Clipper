from types import SimpleNamespace
from unittest.mock import Mock

from src.media import transcription


def test_openai_transcription_uses_api_and_caches_segments(monkeypatch, tmp_path):
    media = tmp_path / "audio.mp3"
    media.write_bytes(b"audio")
    response = SimpleNamespace(
        text="hello world",
        language="en",
        segments=[SimpleNamespace(
            start=0.0,
            end=1.2,
            text=" hello world",
            words=[SimpleNamespace(word=" hello", start=0.0, end=0.5)],
        )],
    )
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.audio.transcriptions.create.return_value = response
    openai = Mock(return_value=client)
    monkeypatch.setattr(transcription, "_prepare_audio_for_transcription", lambda _: media)
    monkeypatch.setattr("openai.OpenAI", openai)

    result = transcription._get_transcript_with_openai(
        tmp_path / "video.mp4",
        SimpleNamespace(openai_api_key="test", transcription_language=""),
    )

    assert "hello world" in result
    assert client.audio.transcriptions.create.call_args.kwargs["model"] == "whisper-1"
    cache = transcription._whisper_result_to_transcript_data(
        {
            "text": response.text,
            "language": response.language,
            "segments": [{
                "start": response.segments[0].start,
                "end": response.segments[0].end,
                "text": response.segments[0].text,
                "words": [{"word": " hello", "start": 0.0, "end": 0.5}],
            }],
        }
    )
    assert cache["words"][0]["start"] == 0
    assert cache["utterances"][0]["end"] == 1200
