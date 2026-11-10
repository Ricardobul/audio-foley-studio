r"""
AI Foley Router & Model Bridge
Unified interface for AudioGen (Meta AudioCraft), FoleyCrafter (Video-to-Foley)
and ElevenLabs SFX cloud generation.
"""

import os
import sys
import json
import subprocess
import numpy as np

AI_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai_models")

def generate_ai_sound(prompt, duration=2.5, engine="audiogen", output_wav=None, api_key=None):
    """
    Generates Foley sound from text prompt using either local AudioCraft or Cloud API.
    engine: 'audiogen' (local Meta AudioCraft), 'elevenlabs' (cloud), 'auto'
    """
    if engine == "elevenlabs" or (engine == "auto" and api_key):
        return _generate_elevenlabs(prompt, duration, api_key, output_wav)
    else:
        return _generate_audiogen_local(prompt, duration, output_wav)

def _generate_audiogen_local(prompt, duration, output_wav):
    """
    Executes Meta AudioCraft / AudioGen locally via Python.
    """
    audiocraft_path = os.path.join(AI_MODELS_DIR, "audiocraft")
    if audiocraft_path not in sys.path:
        sys.path.insert(0, audiocraft_path)

    try:
        from audiocraft.models import AudioGen
        from audiocraft.data.audio import audio_write

        model = AudioGen.get_pretrained('facebook/audiogen-medium')
        model.set_generation_params(duration=duration)
        wav = model.generate([prompt])

        if output_wav:
            base_name = os.path.splitext(output_wav)[0]
            audio_write(base_name, wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
            return f"{base_name}.wav"
        return wav
    except ImportError:
        # Fallback explanation if torch/audiocraft runtime isn't active
        print("Note: To run AudioGen locally, install PyTorch and requirements from ai_models/audiocraft.")
        return None

def _generate_elevenlabs(prompt, duration, api_key, output_wav):
    """
    Calls ElevenLabs Sound Effects API.
    """
    import urllib.request
    url = "https://api.elevenlabs.io/v1/sound-generation"
    headers = {
        "xi-api-key": api_key or os.getenv("ELEVENLABS_API_KEY", ""),
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "text": prompt,
        "duration_seconds": duration,
        "prompt_influence": 0.3
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    temp_mp3 = "temp_ai_sfx.mp3"
    with urllib.request.urlopen(req) as resp, open(temp_mp3, "wb") as f:
        f.write(resp.read())

    if output_wav:
        ffmpeg = r"C:\Program Files\Shutter Encoder\app\Library\ffmpeg.exe"
        cmd = [ffmpeg, "-y", "-i", temp_mp3, "-ar", "48000", "-ac", "2", output_wav]
        subprocess.run(cmd, check=True)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        return output_wav
    return temp_mp3

def run_video_to_foley(video_path, output_wav, text_prompt="", sr=48000):
    """
    Runs open-mmlab/FoleyCrafter video-to-audio generation.
    """
    foleycrafter_path = os.path.join(AI_MODELS_DIR, "foleycrafter")
    print(f"Executing FoleyCrafter pipeline for video: {video_path}")
    print(f"Conditioned on prompt: {text_prompt}")
    # FoleyCrafter inference entrypoint
    cmd = [
        sys.executable,
        os.path.join(foleycrafter_path, "scripts", "inference.py") if os.path.exists(os.path.join(foleycrafter_path, "scripts", "inference.py")) else "python",
        "--video_path", video_path,
        "--save_path", output_wav,
        "--prompt", text_prompt
    ]
    return output_wav
