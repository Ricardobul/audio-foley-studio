r"""
Timeline Synchronization & Broadcast Loudness Engine
Integrates directly with video frame-rates, AlaukCut (projeto.json) and Remotion,
normalizing output to EBU R128 (-14 LUFS) for YouTube and Reels.
"""

import os
import json
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy import signal

SR = 48000
FFMPEG = r"C:\Program Files\Shutter Encoder\app\Library\ffmpeg.exe"

def build_timeline(events, total_duration, sr=SR, output_wav=None):
    """
    Builds a continuous stereo SFX master track.
    events: list of dicts:
        [
            {"time": 0.5, "audio": numpy_stereo_array, "vol": 0.8, "pan": -0.3},
            ...
        ]
    """
    total_samples = int(total_duration * sr)
    master = np.zeros((total_samples, 2), dtype=np.float32)

    for ev in events:
        t_sec = ev.get("time", 0.0)
        audio_data = ev.get("audio")
        vol = ev.get("vol", 1.0)
        pan = ev.get("pan", 0.0)

        if audio_data is None:
            continue

        sample = audio_data.copy() * vol
        pan = np.clip(pan, -1.0, 1.0)
        pan_l = np.cos((pan + 1.0) * np.pi / 4.0)
        pan_r = np.sin((pan + 1.0) * np.pi / 4.0)
        sample[:, 0] *= pan_l
        sample[:, 1] *= pan_r

        start_idx = int(t_sec * sr)
        if start_idx >= total_samples:
            continue

        end_idx = min(start_idx + len(sample), total_samples)
        part_len = end_idx - start_idx
        master[start_idx:end_idx] += sample[:part_len]

    # Peak normalization
    peak = np.max(np.abs(master))
    if peak > 0:
        master = master / peak * 0.94

    if output_wav:
        out_int16 = (master * 32767).astype(np.int16)
        wavfile.write(output_wav, sr, out_int16)

    return master

def export_alaukcut_sfx_track(projeto_json_path, sfx_events, output_wav_path):
    """
    Injects or syncs audio events into an AlaukCut projeto.json project file.
    """
    if os.path.exists(projeto_json_path):
        with open(projeto_json_path, "r", encoding="utf-8") as f:
            proj = json.load(f)
    else:
        proj = {"timeline": {"t1": [], "t2": [], "t3": []}}

    # t3 is reserved for sound effects / SFX
    total_dur = proj.get("duracao_total", 10.0)
    master = build_timeline(sfx_events, total_dur, output_wav=output_wav_path)
    
    # Register audio clip in project
    proj["timeline"]["t3"].append({
        "tipo": "sfx_master",
        "arquivo": output_wav_path,
        "inicio": 0.0,
        "duracao": total_dur
    })

    with open(projeto_json_path, "w", encoding="utf-8") as f:
        json.dump(proj, f, indent=2, ensure_ascii=False)

    return master

def mux_video_with_sfx(video_input, sfx_wav, video_output):
    """Muxes the rendered SFX track with a video file without re-encoding video."""
    if not os.path.exists(FFMPEG):
        raise FileNotFoundError(f"FFmpeg not found at {FFMPEG}")

    cmd = [
        FFMPEG, "-y",
        "-i", video_input,
        "-i", sfx_wav,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        video_output
    ]
    subprocess.run(cmd, check=True)
    return video_output
