r"""
Command Line Interface (CLI) for Audio Foley Studio
"""

import argparse
import sys
import os
from scipy.io import wavfile

try:
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
        SR
    )
    from .convolution import apply_room_convolution
except ImportError:
    from dsp_engine import (
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
        SR
    )
    from convolution import apply_room_convolution

def save_output(data, output_path, sr=SR):
    int16_data = (data * 32767).astype("int16")
    wavfile.write(output_path, sr, int16_data)
    print(f" Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Audio Foley Studio CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Whoosh
    p_whoosh = subparsers.add_parser("whoosh", help="Synthesize motion whoosh")
    p_whoosh.add_argument("-d", "--duration", type=float, default=0.72)
    p_whoosh.add_argument("-o", "--output", default="whoosh.wav")

    # Footsteps
    p_steps = subparsers.add_parser("footsteps", help="Synthesize footsteps")
    p_steps.add_argument("-s", "--surface", default="wood", choices=["wood", "concrete", "gravel", "water"])
    p_steps.add_argument("-n", "--steps", type=int, default=4)
    p_steps.add_argument("-o", "--output", default="footsteps.wav")

    # Thunder
    p_thunder = subparsers.add_parser("thunder", help="Synthesize thunder & atmospheric rumble")
    p_thunder.add_argument("-d", "--duration", type=float, default=5.2)
    p_thunder.add_argument("-o", "--output", default="thunder.wav")

    # Door Knock
    p_door = subparsers.add_parser("door", help="Synthesize door knock")
    p_door.add_argument("-o", "--output", default="door_knock.wav")

    # Impact
    p_impact = subparsers.add_parser("impact", help="Synthesize cinematic impact sub-boom")
    p_impact.add_argument("-d", "--duration", type=float, default=2.2)
    p_impact.add_argument("-o", "--output", default="impact.wav")

    # Riser
    p_riser = subparsers.add_parser("riser", help="Synthesize tension riser")
    p_riser.add_argument("-d", "--duration", type=float, default=2.0)
    p_riser.add_argument("-o", "--output", default="riser.wav")

    # Convolve
    p_conv = subparsers.add_parser("convolve", help="Apply acoustic room impulse response")
    p_conv.add_argument("-i", "--input", required=True)
    p_conv.add_argument("-r", "--room", default="studio", choices=["booth", "studio", "corridor", "hall"])
    p_conv.add_argument("-w", "--wet", type=float, default=0.25)
    p_conv.add_argument("-o", "--output", default="convolved.wav")

    args = parser.parse_args()

    if args.command == "whoosh":
        audio = synthesize_whoosh(duration=args.duration)
        save_output(audio, args.output)
    elif args.command == "footsteps":
        audio = synthesize_footsteps(steps=args.steps, surface=args.surface)
        save_output(audio, args.output)
    elif args.command == "thunder":
        audio = synthesize_thunder(duration=args.duration)
        save_output(audio, args.output)
    elif args.command == "door":
        audio = synthesize_door_knock()
        save_output(audio, args.output)
    elif args.command == "impact":
        audio = synthesize_impact(duration=args.duration)
        save_output(audio, args.output)
    elif args.command == "riser":
        audio = synthesize_riser(duration=args.duration)
        save_output(audio, args.output)
    elif args.command == "convolve":
        fs, data = wavfile.read(args.input)
        if data.ndim == 1:
            data = data.astype("float32") / 32768.0
            data = np.column_stack((data, data))
        else:
            data = data.astype("float32") / 32768.0
        audio = apply_room_convolution(data, room_size=args.room, wet_dry=args.wet, sr=fs)
        save_output(audio, args.output, sr=fs)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
