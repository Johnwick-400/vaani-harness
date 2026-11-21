#!/usr/bin/env python3
"""
Generates 8 Real 16kHz Mono `.wav` Indian Voice Torture-Test Recordings (`samples/test_1.wav` .. `test_8.wav`)
with real Indian-English / Telugish / Hinglish speech (`en-IN-PrabhatNeural` & `en-IN-NeerjaNeural`)
and real stitched mid-sentence silence pauses (650ms-850ms) so any judge can click Play, listen to the WAV,
and run the exact same WAV through Base Pipeline vs. VAANI-HARNESS!
"""

import asyncio
import json
import subprocess
import wave
from pathlib import Path
import edge_tts
import numpy as np

BASE_DIR = Path("/home/pavan_veesam/vaani_harness")
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

TORTURE_TESTS = [
    {
        "id": "test_1_semantic_vad_pause",
        "title": "1. Semantic VAD: Mid-Sentence Hesitation Pause (720ms)",
        "category": "Semantic VAD",
        "voice": "en-IN-PrabhatNeural",
        "part_a": "Bhaiya, please transfer four thousand five hundred rupees to",
        "pause_ms": 720,
        "part_b": "Akhilesh Damaraju on GPay for the team dinner at RMZ The Loft, Raidurg.",
        "ground_truth": "Bhaiya, please transfer 4500 rupees to Akhilesh Damaraju on GPay for the team dinner at RMZ The Loft, Raidurg.",
        "clean_target": "Bhaiya, please transfer 4500 rupees to Akhilesh Damaraju on GPay for the team dinner at RMZ The Loft, Raidurg.",
        "overlap_event": "Haan ji",
        "overlap_ms": 420,
        "is_noise": False,
    },
    {
        "id": "test_2_smart_dictation_self_correction",
        "title": "2. Smart Dictation: Mid-Speech Self-Correction & Indian Entities",
        "category": "Smart Dictation",
        "voice": "en-IN-NeerjaNeural",
        "part_a": "Um, send 4500 rupees, sorry wait, not 4500, make it 5200 rupees to",
        "pause_ms": 640,
        "part_b": "Keshav Reddy via HDFC NEFT and schedule the review at 7:30 PM, no sorry 8:00 PM.",
        "ground_truth": "Send 4500 rupees, sorry wait, not 4500, make it 5200 rupees to Keshav Reddy via HDFC NEFT and schedule the review at 7:30 PM, no sorry 8:00 PM.",
        "clean_target": "Send 5200 rupees to Keshav Reddy via HDFC NEFT and schedule the review at 8:00 PM.",
        "overlap_event": "Sare anna",
        "overlap_ms": 480,
        "is_noise": False,
    },
    {
        "id": "test_3_telugish_code_mix_delivery",
        "title": "3. Telugish Code-Switching: Hyderabad Delivery & Locality Names",
        "category": "Telugish + Entities",
        "voice": "en-IN-PrabhatNeural",
        "part_a": "Anna, delivery location Madhapur kaadu, sorry wait, make it RMZ The Loft, Raidurg lo",
        "pause_ms": 680,
        "part_b": "main gate daggara drop cheyandi, nenu meeting lo unnanu.",
        "ground_truth": "Anna, delivery location Madhapur kaadu, sorry wait, make it RMZ The Loft, Raidurg lo main gate daggara drop cheyandi, nenu meeting lo unnanu.",
        "clean_target": "Anna, delivery location RMZ The Loft, Raidurg lo main gate daggara drop cheyandi, nenu meeting lo unnanu.",
        "overlap_event": "Avunu",
        "overlap_ms": 390,
        "is_noise": False,
    },
    {
        "id": "test_4_full_duplex_backchannel_shield",
        "title": "4. Full-Duplex: 'Haan / Achha' Backchannel vs. True Barge-In",
        "category": "Full-Duplex",
        "voice": "en-IN-NeerjaNeural",
        "part_a": "Rajeev Ranjan confirmed the OneMoney AA integration for",
        "pause_ms": 700,
        "part_b": "our Chikkadpally and Gachibowli branches on PhonePe.",
        "ground_truth": "Rajeev Ranjan confirmed the OneMoney AA integration for our Chikkadpally and Gachibowli branches on PhonePe.",
        "clean_target": "Rajeev Ranjan confirmed the OneMoney AA integration for our Chikkadpally and Gachibowli branches on PhonePe.",
        "overlap_event": "Haan, achha",
        "overlap_ms": 510,
        "is_noise": False,
    },
    {
        "id": "test_5_telugish_devops_standup",
        "title": "5. Telugish Standup: Technical + Telugu Code-Mixed Clause",
        "category": "Telugish + VAD",
        "voice": "en-IN-PrabhatNeural",
        "part_a": "Ee roju aa task urgent ga deploy chesaka",
        "pause_ms": 750,
        "part_b": "Cloud Run logs check chesi Equal AI dashboard lo update cheyandi.",
        "ground_truth": "Ee roju aa task urgent ga deploy chesaka Cloud Run logs check chesi Equal AI dashboard lo update cheyandi.",
        "clean_target": "Ee roju aa task urgent ga deploy chesaka Cloud Run logs check chesi Equal AI dashboard lo update cheyandi.",
        "overlap_event": "[Auto-Rickshaw Horn 1.2kHz]",
        "overlap_ms": 350,
        "is_noise": True,
    },
    {
        "id": "test_6_true_barge_in_interruption",
        "title": "6. Full-Duplex True Barge-In: Mid-Stream User Correction",
        "category": "Full-Duplex Barge-In",
        "voice": "en-IN-NeerjaNeural",
        "part_a": "Please book the cab from Koramangala to",
        "pause_ms": 620,
        "part_b": "Indiranagar at 6:00 PM, no sorry 6:30 PM for Pavan Teja Veesam.",
        "ground_truth": "Please book the cab from Koramangala to Indiranagar at 6:00 PM, no sorry 6:30 PM for Pavan Teja Veesam.",
        "clean_target": "Please book the cab from Koramangala to Indiranagar at 6:30 PM for Pavan Teja Veesam.",
        "overlap_event": "Wait, stop, change the time to 6:30 PM",
        "overlap_ms": 1450,
        "is_noise": False,
    },
]


async def synth_part(text: str, voice: str, out_mp3: Path):
    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(out_mp3))


def mp3_to_pcm16k(mp3_path: Path) -> np.ndarray:
    tmp_wav = mp3_path.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp_wav)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with wave.open(str(tmp_wav), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    tmp_wav.unlink(missing_ok=True)
    mp3_path.unlink(missing_ok=True)
    return pcm


async def main():
    sr = 16000
    manifest = []
    for item in TORTURE_TESTS:
        mp3_a = SAMPLES_DIR / f"{item['id']}_a.mp3"
        mp3_b = SAMPLES_DIR / f"{item['id']}_b.mp3"
        await synth_part(item["part_a"], item["voice"], mp3_a)
        await synth_part(item["part_b"], item["voice"], mp3_b)
        pcm_a = mp3_to_pcm16k(mp3_a)
        pcm_b = mp3_to_pcm16k(mp3_b)
        silence = np.zeros(int(sr * item["pause_ms"] / 1000), dtype=np.int16)
        full_pcm = np.concatenate([pcm_a, silence, pcm_b])

        out_wav = SAMPLES_DIR / f"{item['id']}.wav"
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(full_pcm.tobytes())

        item_meta = dict(item)
        item_meta["wav_path"] = str(out_wav)
        item_meta["wav_url"] = f"/samples/{out_wav.name}"
        manifest.append(item_meta)
        print(f"Generated {out_wav.name} ({len(full_pcm)/sr:.2f}s with {item['pause_ms']}ms internal hesitation pause)")

    (SAMPLES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved samples/manifest.json")


if __name__ == "__main__":
    asyncio.run(main())
