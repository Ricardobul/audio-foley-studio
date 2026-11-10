"""
Audio Foley Studio
Unified Procedural Sound Design & AI Foley Studio for Motion Design, Video Production and Cinema.
"""

__version__ = "1.0.0"
__author__ = "Ricardobul"

from .dsp_engine import (
    synthesize_footsteps,
    synthesize_thunder,
    synthesize_door_knock,
    synthesize_water_drop,
    synthesize_rain,
    synthesize_whoosh,
    synthesize_impact,
    synthesize_riser,
    synthesize_reverse_suckback,
    synthesize_digital_glitch,
    synthesize_ui_chime,
)
from .convolution import apply_room_convolution, generate_synthetic_ir
from .timeline_sync import build_timeline, mux_video_with_sfx
from .ai_router import generate_ai_sound, run_video_to_foley

__all__ = [
    "synthesize_footsteps",
    "synthesize_thunder",
    "synthesize_door_knock",
    "synthesize_water_drop",
    "synthesize_rain",
    "synthesize_whoosh",
    "synthesize_impact",
    "synthesize_riser",
    "synthesize_reverse_suckback",
    "synthesize_digital_glitch",
    "synthesize_ui_chime",
    "apply_room_convolution",
    "generate_synthetic_ir",
    "build_timeline",
    "mux_video_with_sfx",
    "generate_ai_sound",
    "run_video_to_foley",
]
