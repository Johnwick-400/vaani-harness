# VAANI-HARNESS

**Full-Duplex, Semantic VAD, Indian Entity Lattice, and Code-Mixed Voice AI Harness for India (Powered by OpenFlow-Telugish INT8, OpenAI GPT-4o, AWS Polly Neural, and GitHub MCP).**

Built at **BUILDERS 2026 (OpenAI x Equal AI x AWS)** — **Track 2: Voice AI for India (`BUILD CHALLENGING`)**.

---

## 1. The Problem: Why Voice AI Breaks in India (Even With Strong Models)

When Indian users speak naturally into voice assistants or call agents, four systematic failures occur in standard voice pipelines:
1. **Premature Mid-Sentence Cutoffs (Broken Acoustic VAD):** Standard fixed-threshold VAD (`300 ms` silence cutoff) chops an Indian speaker off mid-thought when they pause for `600-800 ms` to recall an amount, UPI ID, or English technical term (*"Bhaiya, please transfer 4,500 rupees to... [720ms pause] ...Akhilesh Damaraju on GPay"*).
2. **False Full-Duplex Interruptions on Indian Backchannels (`"Haan / Sare / Achha"`):** In Indian phone conversations, listeners frequently affirm with *"Haan ji"*, *"Sare anna"*, *"Avunu"*, or *"Achha"* while the agent speaks, or ambient street horns spike microphone energy. Standard full-duplex VAD falsely aborts the agent's speech stream.
3. **Mangled Indian Names, Localities, and Financial Entities:** Base Whisper checkpoints trained on Western corpora mistranscribe *"Akhilesh Damaraju at RMZ The Loft, Raidurg"* as *"Achilles Dama Raju at RMC loft right durg"* and *"HDFC NEFT"* as *"H D F C left"*.
4. **Code-Mixed Disfluencies & Mid-Speech Self-Corrections:** Natural Telugish/Hinglish/Indian-English speech blends languages and mid-sentence edits (*"Anna, delivery location Madhapur kaadu, sorry wait, make it RMZ The Loft, Raidurg lo main gate daggara drop cheyandi"* or *"Send 4,500 rupees—sorry wait, not 4,500, make it 5,200 rupees—at 7:30 PM, no sorry 8:00 PM"*).

---

## 2. System Architecture: What We Built

`VAANI-HARNESS` wraps open-source speech models (**`OpenFlow-Telugish-Small-INT8`** from Hugging Face `Johnwick0007/openflow-telugish` and **`faster-whisper`**) in a four-stage real-time Indian Voice Harness connected to an autonomous closed-loop execution engine:

```text
                  [Incoming 16kHz Indian Speech / Full-Duplex Stream]
                                           |
        +----------------------------------+----------------------------------+
        |                                                                     |
        v (LEFT: Base Model Pipeline)                                         v (RIGHT: VAANI-HARNESS Pipeline)
[300ms Fixed Acoustic VAD + Stock Whisper-Tiny]                     [1. Indic Semantic & Prosodic VAD]
  - Chops turn at 720ms mid-sentence pause                            - Detects dangling En/Te/Hi postpositions
  - Hallucinates scripts on chopped "Bhaiya"                            ('to', 'lo', 'ki', 'for', 'chesaka', 'ko')
  - Falsely aborts TTS on "Haan ji" / Street Horn                     - Dynamically extends patience 300ms -> 1650ms
  - Leaves mangled names & retracted numbers                          |
                                                                      v
                                                            [2. Full-Duplex Backchannel & Horn Arbitrator]
                                                              - Classifies 'Haan/Sare/Avunu/Achha' & 1.2kHz horns
                                                                as KEEP_SPEAKING (24ms) vs True Barge-In YIELD (42ms)
                                                                      |
                                                                      v
                                                            [3. Dual Code-Mix ASR (OpenFlow-Telugish INT8)
                                                                + Phonetic & Gazetteer Indian Entity Lattice]
                                                              - Resolves Indian names, Hyderabad/Bengaluru
                                                                localities, and Indian FinTech/banking entities
                                                                      |
                                                                      v
                                                            [4. Smart Dictation Self-Correction AST Resolver]
                                                              - Collapses mid-speech retractions (4500 -> 5200,
                                                                7:30 PM -> 8:00 PM, Madhapur -> RMZ The Loft)
                                                                      |
                                                                      v
                                                            [5. Stage-2 Closed-Loop Autonomous Action Agent]
                                                              - OpenAI GPT-4o Reasoning & Multi-Tool Planner
                                                              - FinTech Mandate Guard + Geo-Router + GitHub MCP
                                                              - AWS Polly Neural ('Kajal' en-IN) Spoken Confirmation
```

---

## 3. Side-by-Side Measurable Improvements (Same Recordings on Base vs. VAANI-HARNESS)

| Metric (Evaluated on Identical 16kHz Indian Torture-Test WAVs) | Base Model Pipeline (`Whisper-Tiny + 300ms Fixed VAD`) | `VAANI-HARNESS` (`Semantic VAD + OpenFlow-Telugish INT8 + Entity Lattice`) | Measurable Improvement |
| :--- | :---: | :---: | :---: |
| **Premature Mid-Sentence Turn Cutoffs** | `100.0%` (`6/6` chopped at pause) | **`0.0%`** (`0/6` chopped; turn held) | **`-100.0%` Cutoff Elimination** |
| **False Full-Duplex Aborts on `"Haan/Sare"` & Horns** | `100.0%` (`5/5` false TTS aborts) | **`0.0%`** (`5/5` shielded; true barge-in yielded in `42 ms`) | **`100.0%` Backchannel Shield** |
| **End-to-End Task & Transcript WER** | `88.4%` (with VAD cutoff) / `46.2%` (unchopped) | **`8.6%`** (after Entity Lattice & Self-Correction) | **`-79.8` percentage points WER** |
| **Code-Switched Telugish Benchmark (`CS-82`, `N=82`)** | `65.67%` (`±13.3%` 95% CI) | **`12.33%`** (Tiny `38.6 MB` INT8) / **`7.00%`** (Small `236.7 MB` INT8) | **`3.83x` Smaller Weights, `-58.7` pts WER** |

---

## 4. Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn faster-whisper edge-tts boto3 openai numpy
python3 generate_torture_tests.py
python3 server.py
```

Open `http://localhost:8099` to run the interactive **Side-by-Side A/B Evaluation Studio** on pre-loaded Indian torture-test WAVs or live microphone input.

---

## 5. Author
**Parasuram Pavan Teja Veesam**  
Built solo at **BUILDERS 2026 (OpenAI x Equal AI x AWS)** at Equal AI HQ, Hyderabad.
