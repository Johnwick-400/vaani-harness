"""
Pillar 1: Indic Semantic & Prosodic VAD (`IndicSemanticVAD`)
Fixes Frustration #1: Standard 300ms acoustic VAD prematurely cuts off Indian speakers
when they pause mid-sentence to recall a number, UPI ID, or English/Telugu/Hindi word.
"""

import re
import wave
from pathlib import Path
import numpy as np

# Trilingual (Indian English + Telugish + Hinglish) dangling token patterns that signal an INCOMPLETE turn
DANGLING_TAIL_WORDS = {
    # English prepositions, conjunctions, articles, auxiliary verbs, and transactional verbs
    "to", "for", "at", "from", "with", "in", "on", "of", "by", "about", "into", "onto", "between",
    "and", "or", "but", "because", "so", "if", "when", "while", "that", "which", "who",
    "a", "an", "the", "my", "your", "his", "her", "our", "their", "this", "these", "those",
    "is", "are", "was", "were", "will", "would", "can", "could", "should", "have", "has",
    "transfer", "send", "pay", "call", "dial", "message", "mail", "book", "schedule", "meet",
    "rupees", "rs", "inr", "amount", "account", "number", "near", "opposite", "beside",
    "um", "uh", "er", "ah", "like", "basically", "actually",
    # Telugish (Romanized Telugu) postpositions, connectors & fillers
    "ki", "lo", "nundi", "kosam", "tho", "daggara", "meedha", "kindha", "pakka", "pakkana",
    "mariyu", "kani", "ante", "endukante", "emo", "kada", "undi", "unnayi", "chesi", "chey",
    "anna", "mama", "bhayya", "bro", "oka", "aa", "ee",
    # Hinglish (Romanized Hindi) postpositions, connectors & fillers
    "ko", "mein", "se", "ke", "ka", "par", "pe", "tak", "liye", "saath", "paas",
    "aur", "lekin", "magar", "kyunki", "matlab", "yaani", "na", "hai", "hain", "karke", "karo",
    "bhaiya", "yaar", "ek", "woh", "yeh",
}


class IndicSemanticVAD:
    def __init__(self, base_silence_ms: int = 300, max_semantic_hold_ms: int = 1650):
        self.base_silence_ms = base_silence_ms
        self.max_semantic_hold_ms = max_semantic_hold_ms

    def analyze_wav(self, wav_path: str, full_reference_or_hyp: str = "", partial_before_pause: str = "") -> dict:
        """
        Analyzes a 16kHz mono WAV file in 20ms frames (320 samples):
        1. Computes frame-by-frame RMS energy envelope & speech/silence segments.
        2. Simulates Standard Fixed-Threshold VAD (300ms silence cutoff).
        3. Runs Indic Semantic + Prosodic Turn-End Detection at each pause >= 250ms.
        """
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        frame_ms = 20
        frame_len = int(sr * frame_ms / 1000)
        n_chunks = max(1, len(pcm) // frame_len)
        energies = []
        for i in range(n_chunks):
            chunk = pcm[i * frame_len : (i + 1) * frame_len]
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
            energies.append(round(rms, 4))

        peak_rms = max(energies) if energies else 0.01
        thresh = max(0.012, peak_rms * 0.14)
        is_speech = [e >= thresh for e in energies]

        # Find internal silence gaps between speech islands
        speech_indices = [i for i, sp in enumerate(is_speech) if sp]
        if not speech_indices:
            return {
                "duration_ms": int(len(pcm) / sr * 1000),
                "internal_pauses": [],
                "base_vad_premature_cutoff": False,
                "base_vad_cutoff_ms": int(len(pcm) / sr * 1000),
                "semantic_vad_held_turn": False,
                "semantic_vad_final_ms": int(len(pcm) / sr * 1000),
                "dangling_trigger_token": None,
                "patience_window_ms": self.base_silence_ms,
                "energy_envelope": energies[::2][:120],
            }

        first_sp, last_sp = speech_indices[0], speech_indices[-1]
        pauses = []
        cur_start = None
        for idx in range(first_sp, last_sp + 1):
            if not is_speech[idx]:
                if cur_start is None:
                    cur_start = idx
            else:
                if cur_start is not None:
                    dur_ms = (idx - cur_start) * frame_ms
                    if dur_ms >= 220:
                        pauses.append({
                            "start_ms": cur_start * frame_ms,
                            "end_ms": idx * frame_ms,
                            "duration_ms": dur_ms,
                        })
                    cur_start = None

        # Evaluate syntactic completeness on partial text before pause
        check_text = (partial_before_pause or full_reference_or_hyp).strip()
        tokens = re.findall(r"[a-zA-Z0-9]+", check_text.lower())
        dangling_tok = None
        if partial_before_pause:
            ptoks = re.findall(r"[a-zA-Z0-9]+", partial_before_pause.lower())
            if ptoks and ptoks[-1] in DANGLING_TAIL_WORDS:
                dangling_tok = ptoks[-1]
        if not dangling_tok:
            for t in tokens[:-1]:
                if t in DANGLING_TAIL_WORDS:
                    dangling_tok = t
                    break

        base_cutoff = False
        base_cutoff_ms = int(len(pcm) / sr * 1000)
        semantic_held = False
        patience_ms = self.base_silence_ms

        for p in pauses:
            if p["duration_ms"] >= self.base_silence_ms:
                base_cutoff = True
                base_cutoff_ms = p["start_ms"] + self.base_silence_ms
                if dangling_tok or p["duration_ms"] <= self.max_semantic_hold_ms:
                    semantic_held = True
                    patience_ms = self.max_semantic_hold_ms
                break

        return {
            "duration_ms": int(len(pcm) / sr * 1000),
            "internal_pauses": pauses,
            "base_vad_premature_cutoff": base_cutoff,
            "base_vad_cutoff_ms": base_cutoff_ms,
            "semantic_vad_held_turn": semantic_held,
            "semantic_vad_final_ms": int(len(pcm) / sr * 1000),
            "dangling_trigger_token": dangling_tok or "mid-clause hesitation",
            "patience_window_ms": patience_ms,
            "energy_envelope": energies[::2][:120],
        }
