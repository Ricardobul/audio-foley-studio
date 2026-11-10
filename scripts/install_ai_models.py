"""
Download and setup AI Foley models (AudioCraft, FoleyCrafter, T-Foley).
"""

import os
import subprocess

MODELS = {
    "audiocraft": "https://github.com/facebookresearch/audiocraft.git",
    "foleycrafter": "https://github.com/open-mmlab/FoleyCrafter.git",
    "t-foley": "https://github.com/YoonjinXD/T-foley.git"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_DIR = os.path.join(BASE_DIR, "ai_models")
os.makedirs(AI_DIR, exist_ok=True)

def clone_models():
    for name, url in MODELS.items():
        dest = os.path.join(AI_DIR, name)
        if not os.path.exists(dest):
            print(f"Cloning {name} from {url}...")
            subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
        else:
            print(f" {name} is already present in {dest}")

if __name__ == "__main__":
    clone_models()
