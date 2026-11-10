r"""
Acoustic Room & Impulse Response (IR) Convolution Engine
Removes dry 'computerized' sound artifacts by placing procedural audio
inside a realistic physical acoustic space using FFT convolution.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000

def generate_synthetic_ir(room_size="studio", rt60=0.35, sr=SR):
    """
    Generates a realistic early-reflection + diffuse late decay Impulse Response (IR).
    room_size: 'booth' (0.15s), 'studio' (0.35s), 'hall' (1.2s), 'corridor' (0.6s)
    """
    preset_rt60 = {
        "booth": 0.12,
        "studio": 0.32,
        "corridor": 0.65,
        "hall": 1.4,
    }
    target_rt60 = preset_rt60.get(room_size.lower(), rt60)
    ir_len = int(target_rt60 * 1.2 * sr)
    t = np.linspace(0, target_rt60 * 1.2, ir_len, endpoint=False)

    # 1. Early reflections (discrete rays bouncing off walls)
    ir_l = np.zeros(ir_len)
    ir_r = np.zeros(ir_len)

    # Direct sound
    ir_l[0] = 1.0
    ir_r[0] = 1.0

    # Early bounce taps (simulated room geometry in ms)
    taps = [
        (0.007, 0.45, -0.4),
        (0.013, 0.38, 0.5),
        (0.019, 0.28, -0.2),
        (0.027, 0.22, 0.3),
        (0.038, 0.18, -0.5),
    ]
    for delay_s, gain, pan in taps:
        idx = int(delay_s * sr)
        if idx < ir_len:
            gl = np.cos((pan + 1) * np.pi / 4) * gain
            gr = np.sin((pan + 1) * np.pi / 4) * gain
            ir_l[idx] += gl
            ir_r[idx] += gr

    # 2. Late diffuse reverberation (exponentially decaying Gaussian noise with high-frequency absorption)
    decay_rate = 6.91 / target_rt60  # -60 dB decay
    env = np.exp(-t * decay_rate)
    noise_l = np.random.normal(0, 1.0, ir_len) * env
    noise_r = np.random.normal(0, 1.0, ir_len) * env

    # Air absorption filter (rolls off high frequencies in reverb tail)
    sos = signal.butter(2, 4500, btype="lowpass", fs=sr, output="sos")
    tail_l = signal.sosfilt(sos, noise_l) * 0.35
    tail_r = signal.sosfilt(sos, noise_r) * 0.35

    ir_l += tail_l
    ir_r += tail_r

    # Normalize IR energy
    norm_factor = np.sqrt(np.sum(ir_l ** 2) + np.sum(ir_r ** 2))
    if norm_factor > 0:
        ir_l /= norm_factor
        ir_r /= norm_factor

    return ir_l, ir_r

def apply_room_convolution(audio_stereo, ir=None, room_size="studio", wet_dry=0.25, sr=SR):
    """
    Convolves dry stereo audio with an acoustic room impulse response.
    wet_dry: 0.0 (completely dry) to 1.0 (completely wet). Default 0.25 gives natural warmth.
    """
    if ir is None:
        ir_l, ir_r = generate_synthetic_ir(room_size=room_size, sr=sr)
    elif isinstance(ir, str):
        fs_ir, ir_data = wavfile.read(ir)
        if ir_data.ndim == 1:
            ir_l = ir_r = ir_data.astype(np.float32) / 32768.0
        else:
            ir_l = ir_data[:, 0].astype(np.float32) / 32768.0
            ir_r = ir_data[:, 1].astype(np.float32) / 32768.0
    else:
        ir_l, ir_r = ir

    dry_l = audio_stereo[:, 0]
    dry_r = audio_stereo[:, 1]

    # Fast FFT convolution
    wet_l = signal.fftconvolve(dry_l, ir_l)[:len(dry_l)]
    wet_r = signal.fftconvolve(dry_r, ir_r)[:len(dry_r)]

    out_l = dry_l * (1.0 - wet_dry) + wet_l * wet_dry
    out_r = dry_r * (1.0 - wet_dry) + wet_r * wet_dry

    peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)))
    if peak > 0:
        out_l = out_l / peak * 0.94
        out_r = out_r / peak * 0.94

    return np.column_stack((out_l, out_r))
