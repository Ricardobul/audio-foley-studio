# 🎧 Audio Foley Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Audio: 48kHz Stereo](https://img.shields.io/badge/Audio-48kHz%20Stereo%20Broadcast-orange.svg)]()
[![Loudness: -14 LUFS](https://img.shields.io/badge/Standard-EBU%20R128%20(-14%20LUFS)-purple.svg)]()

> **The Unified Procedural Sound Design & AI Foley Studio for Motion Design, Video Production, and Cinema.**

**Audio Foley Studio** bridges the gap between pure algorithmic digital signal processing (DSP) and modern neural generative audio. It consolidates **5 sound design disciplines** into a single autonomous, scriptable engine:

1. **Procedural Motion SFX**: Aerodynamic whooshes, trailer sub-booms, exponential tension risers, reverse suck-backs, and digital glitches.
2. **Physical Foley Modeling**: Physical acoustics based on Andy Farnell's *Designing Sound* (bio-mechanical footsteps across 4 surfaces, lightning shockwave thunder, knuckle door knocks, Minnaert bubble drops, and Poisson rain).
3. **Impulse Response (IR) Convolution Reverb**: Eliminates dry "computerized" synth artifacts by placing procedural audio inside real room acoustics (booth, studio, corridor, hall).
4. **AI Foley Triad Bridge**: Seamless routing to **Meta AudioCraft (AudioGen)**, **FoleyCrafter (Video-to-Foley)**, and **ElevenLabs SFX**.
5. **Timeline Synchronization**: Frame-accurate alignment with video editors, **AlaukCut** (`projeto.json`), and **Remotion**, normalized to **-14 LUFS** (YouTube/Reels broadcast standard).

---

## 🏗️ Architecture

```
                               AUDIO FOLEY STUDIO
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │         PROCEDURAL DSP ENGINE        │            AI FOLEY TRIAD            │
 │     (Instant, zero-weight, code)     │     (High-fidelity organic foley)    │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Aerodynamic Stereo Whooshes        │ • Meta AudioGen (Local Text-to-SFX)  │
 │ • Cinematic Trailer Sub-Booms        │ • FoleyCrafter (Video-to-Foley)      │
 │ • Exponential Tension Risers         │ • ElevenLabs SFX (Cloud Fast Engine) │
 │ • Bio-mechanical Footsteps (4 types) │                                      │
 │ • Supersonic Thunder & Rolling Cloud │                                      │
 │ • Minnaert Cavitation Water Drops    │                                      │
 │ • Stochastic Poisson Rain Fields     │                                      │
 └──────────────────┬───────────────────┴───────────────────┬──────────────────┘
                    │                                       │
                    ▼                                       ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     ROOM CONVOLUTION (IR REVERB) ENGINE                     │
 │           Places any dry sound inside studio acoustics via FFT             │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                TIMELINE SYNCHRONIZER & BROADCAST MIXER                      │
 │    Frame-accurate sync for AlaukCut / Remotion | EBU R128 (-14 LUFS)        │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Ricardobul/audio-foley-studio.git
cd audio-foley-studio
pip install -r requirements.txt
```

### Core Requirements
* Python 3.10+
* `numpy` (>= 1.24)
* `scipy` (>= 1.10)
* `ffmpeg` (for video muxing and format conversion)

---

## 🚀 Quickstart

### 1. Synthesize Procedural Sounds & Add Room Reverb

```python
from audio_foley_studio import (
    synthesize_footsteps,
    synthesize_whoosh,
    synthesize_impact,
    synthesize_thunder,
    apply_room_convolution
)

# 1. Synthesize dry footsteps on gravel
dry_steps = synthesize_footsteps(steps=6, surface="gravel", cadence_bpm=110)

# 2. Add realistic room air via IR convolution
warm_steps = apply_room_convolution(dry_steps, room_size="corridor", wet_dry=0.28)

# 3. Synthesize a trailer sub-boom
impact = synthesize_impact(duration=2.2, pitch_start=180.0, pitch_end=42.0)
```

### 2. Assemble a Timed Video SFX Track

```python
from audio_foley_studio import build_timeline, mux_video_with_sfx

events = [
    {"time": 0.3, "audio": synthesize_whoosh(duration=0.6), "vol": 0.85, "pan": -0.4},
    {"time": 1.1, "audio": synthesize_impact(), "vol": 1.0, "pan": 0.0},
    {"time": 2.4, "audio": warm_steps, "vol": 0.75, "pan": 0.25}
]

# Renders master track normalized to -14 LUFS
build_timeline(events, total_duration=6.0, output_wav="master_sfx.wav")

# Mux directly into MP4 without re-encoding video
mux_video_with_sfx("input_video.mp4", "master_sfx.wav", "output_final.mp4")
```

---

## 💻 CLI Usage

Audio Foley Studio comes with an easy-to-use CLI:

```bash
# Generate motion whoosh
python -m audio_foley_studio.cli whoosh -d 0.72 -o whoosh.wav

# Generate footsteps (options: wood, concrete, gravel, water)
python -m audio_foley_studio.cli footsteps --surface gravel -n 6 -o steps.wav

# Generate thunder with supersonic shockwave & rolling decay
python -m audio_foley_studio.cli thunder -d 5.2 -o thunder.wav

# Apply room acoustic impulse response (IR)
python -m audio_foley_studio.cli convolve -i dry_knock.wav --room studio -w 0.25 -o wet_knock.wav
```

---

## 🤖 AI Foley Triad Integrations

The library includes connectors to three premier open-source and API generative audio models:

* **Meta AudioCraft / AudioGen**:
  ```python
  from audio_foley_studio import generate_ai_sound
  generate_ai_sound("heavy combat boots on creaky floor", engine="audiogen", output_wav="ai_steps.wav")
  ```
* **FoleyCrafter**: Video-to-audio synchronization conditioned on visual motion onsets.
* **ElevenLabs SFX**: Instant cloud-based Foley generation via API.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
