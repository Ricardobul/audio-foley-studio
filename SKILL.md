---
name: audio-foley-studio
description: Unified Procedural Audio & AI Foley Studio. Combines physical acoustic modeling from Andy Farnell's Designing Sound, modal synthesis, aerodynamic motion sound design, impulse response room convolution, multi-track timeline synchronization (-14 LUFS) for AlaukCut (projeto.json) and Remotion, plus integrations with Meta AudioCraft, FoleyCrafter, and ElevenLabs.
---

# Audio Foley Studio

Unified Procedural Sound Design & AI Foley Studio for Motion Design, Video Editing, and Cinema.

This skill consolidates **5 audio design disciplines** into a single autonomous engine:
1. **Procedural Motion SFX**: Aerodynamic stereo whooshes, exponential tension risers, trailer sub-booms, reverse suck-backs, and sci-fi digital glitches.
2. **Physical Foley Modeling**: Acoustic modeling based on Andy Farnell's *Designing Sound* (bio-mechanical footsteps across 4 surfaces, supersonic lightning thunder with atmospheric dispersion, knuckle-on-wood door knocks, Minnaert cavitation drops, and stochastic Poisson rain).
3. **Acoustic Room Convolution**: Impulse Response (IR) FFT convolution engine that places dry procedural audio inside real room acoustics (booth, studio, corridor, hall), eliminating synthetic artifacts.
4. **AI Foley Triad Bridge**:
   - **Meta AudioCraft / AudioGen**: Local offline Text-to-SFX neural synthesis.
   - **FoleyCrafter**: Video-to-Foley neural model that watches video frames and generates synchronized audio onsets.
   - **ElevenLabs SFX API**: Studio-grade cloud foley generation in 1 second.
5. **Timeline Sync & Broadcast Loudness**:
   - Frame-accurate alignment with AlaukCut (`projeto.json`) and Remotion timelines.
   - EBU R128 (-14 LUFS) broadcast normalization for YouTube, Reels, and TikTok.
   - Lossless audio muxing into video via FFmpeg.

---

## 1. Quick Usage in Python

```python
from audio_foley_studio import (
    synthesize_whoosh,
    synthesize_impact,
    synthesize_footsteps,
    synthesize_thunder,
    apply_room_convolution,
    build_timeline,
    mux_video_with_sfx
)

# 1. Synthesize dry sound
dry_steps = synthesize_footsteps(steps=4, surface="wood", cadence_bpm=108)

# 2. Add realistic room acoustic air (IR convolution)
warm_steps = apply_room_convolution(dry_steps, room_size="corridor", wet_dry=0.3)

# 3. Assemble full video timeline
events = [
    {"time": 0.4, "audio": synthesize_whoosh(duration=0.6), "vol": 0.8, "pan": -0.3},
    {"time": 1.0, "audio": synthesize_impact(), "vol": 1.0, "pan": 0.0},
    {"time": 2.5, "audio": warm_steps, "vol": 0.7, "pan": 0.2}
]
master_track = build_timeline(events, total_duration=5.0, output_wav="master_sfx.wav")

# 4. Mux directly with video
mux_video_with_sfx("silent_cut.mp4", "master_sfx.wav", "final_video_with_sound.mp4")
```

---

## 2. CLI Commands

```bash
# Generate motion whoosh
python -m audio_foley_studio.cli whoosh -d 0.65 -o whoosh.wav

# Generate bio-mechanical footsteps (wood, gravel, concrete, water)
python -m audio_foley_studio.cli footsteps --surface gravel -n 6 -o steps.wav

# Generate supersonic thunderclap with cloud rumble
python -m audio_foley_studio.cli thunder -d 5.0 -o thunder.wav

# Apply room acoustic convolution
python -m audio_foley_studio.cli convolve -i dry_knock.wav --room studio -w 0.25 -o warm_knock.wav
```

---

## 3. Direct Timeline Integration with AlaukCut (`projeto.json`)

When working with AlaukCut, the audio events are injected directly into the `t3` (Sound Effects) track of `projeto.json`:
- Calculates start and end times in frames and seconds.
- Normalizes peaks and integrates with background music (`t2`) and speech voiceover (`t1`).
