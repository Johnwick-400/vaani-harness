"""
Real-Time AWS Amazon Transcribe Streaming (`en-IN` & `te-IN`) Wrapper
Used identically as the Base STT on BOTH sides of the A/B comparison
(Left: Raw Amazon Transcribe without Harness | Right: Amazon Transcribe + VAANI-HARNESS).
"""

import asyncio
import os
import time
import wave
from pathlib import Path
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v.strip('"')


class _TranscriptCollector(TranscriptResultStreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.final_parts = []

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for res in transcript_event.transcript.results:
            if res.alternatives and not res.is_partial:
                self.final_parts.append(res.alternatives[0].transcript)


async def transcribe_wav_aws_streaming(wav_path: str, language_code: str = "en-IN") -> dict:
    t0 = time.time()
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    client = TranscribeStreamingClient(region=region)

    stream = await client.start_stream_transcription(
        language_code=language_code,
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )
    handler = _TranscriptCollector(stream.output_stream)

    async def _send_pcm():
        with wave.open(str(wav_path), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        chunk_bytes = 8192
        for i in range(0, len(pcm), chunk_bytes):
            await stream.input_stream.send_audio_event(audio_chunk=pcm[i : i + chunk_bytes])
            await asyncio.sleep(0.005)
        await stream.input_stream.end_stream()

    await asyncio.gather(_send_pcm(), handler.handle_events())
    return {
        "transcript": " ".join(handler.final_parts).strip(),
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "engine": f"Amazon Transcribe Streaming ({language_code})",
    }
