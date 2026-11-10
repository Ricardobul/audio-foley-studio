r"""
Unified DSP & Physical Modeling Audio Engine
Implements physical acoustics from Andy Farnell's "Designing Sound",
modal synthesis, aerodynamic turbulence, and motion design procedural audio.
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal

SR = 48000  # Broadcast standard (48 kHz)

def normalize_audio(left, right=None, target_peak=0.94):
    """Removes DC offset and normalizes stereo audio to target peak (-0.5 dB default)."""
    if right is None:
        right = left
    left = left - np.mean(left)
    right = right - np.mean(right)
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 0:
        left = left / peak * target_peak
        right = right / peak * target_peak
    return np.column_stack((left, right))

# -------------------------------------------------------------
# 1. PHYSICAL FOLEY: FOOTSTEPS
# -------------------------------------------------------------
def synthesize_footsteps(steps=4, cadence_bpm=108, surface="wood", sr=SR):
    """
    Bio-mechanical footstep model:
    - Heel strike: Low-frequency damped mechanical impulse
    - Sole roll: Forward friction contact delayed by 20-30 ms
    - Surface texture: Filtered resonance according to surface material
    """
    interval = 60.0 / cadence_bpm
    dur = steps * interval + 0.6
    num_samples = int(dur * sr)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    out_l = np.zeros(num_samples)
    out_r = np.zeros(num_samples)

    surface_filters = {
        "wood": ([55, 220], [350, 1200]),
        "concrete": ([70, 300], [500, 2200]),
        "gravel": ([45, 180], [1200, 4800]),
        "water": ([80, 250], [280, 1600]),
    }
    heel_bp, sole_bp = surface_filters.get(surface.lower(), surface_filters["wood"])

    sos_heel = signal.butter(2, heel_bp, btype="bandpass", fs=sr, output="sos")
    sos_sole = signal.butter(2, sole_bp, btype="bandpass", fs=sr, output="sos")

    for i in range(steps):
        t_step = 0.2 + i * interval
        mask = t >= t_step
        ts = t[mask] - t_step
        ls = len(ts)
        if ls == 0:
            continue

        # Alternating left and right feet with stereo position
        is_left = (i % 2 == 0)
        pan = -0.35 if is_left else 0.35

        # 1. Heel strike impulse
        heel_noise = np.random.normal(0, 1.0, ls) * np.exp(-ts * 85.0)
        heel = signal.sosfilt(sos_heel, heel_noise) * 1.6

        # 2. Sole roll friction (delayed by 20ms)
        sole_delay = int(sr * 0.02)
        sole = np.zeros(ls)
        if ls > sole_delay:
            ts_sole = ts[sole_delay:]
            sole_noise = np.random.normal(0, 0.8, len(ts_sole)) * np.exp(-ts_sole * 115.0)
            sole[sole_delay:] = signal.sosfilt(sos_sole, sole_noise)

        # 3. Gravel crunch or puddle splash if applicable
        extra = np.zeros(ls)
        if surface.lower() == "gravel":
            crunch = np.random.uniform(-1, 1, ls) * np.exp(-ts * 140.0)
            sos_crunch = signal.butter(2, [1800, 6000], btype="bandpass", fs=sr, output="sos")
            extra = signal.sosfilt(sos_crunch, crunch) * 0.4
        elif surface.lower() == "water":
            splash = np.random.normal(0, 0.6, ls) * np.exp(-ts * 100.0)
            sos_splash = signal.butter(2, [600, 2400], btype="bandpass", fs=sr, output="sos")
            extra = signal.sosfilt(sos_splash, splash) * 0.5

        step_sound = heel * 0.8 + sole * 0.5 + extra
        gain_l = np.cos((pan + 1.0) * np.pi / 4)
        gain_r = np.sin((pan + 1.0) * np.pi / 4)

        out_l[mask] += step_sound * gain_l
        out_r[mask] += step_sound * gain_r

    return normalize_audio(out_l, out_r)

# -------------------------------------------------------------
# 2. PHYSICAL FOLEY: THUNDER & ATMOSPHERIC SHOCKWAVE
# -------------------------------------------------------------
def synthesize_thunder(duration=5.2, strike_intensity=1.0, sr=SR):
    """
    Thunderstorm acoustics:
    - Supersonic shockwave crack (initial lightning channel discharge)
    - Non-linear atmospheric dispersion (higher frequencies attenuate rapidly)
    - Multiple cloud rebound echos and continuous rolling sub-bass rumble
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # 1. Supersonic crack
    t_strike = 0.12
    mask_crack = (t >= t_strike) & (t < t_strike + 0.08)
    crack_len = np.sum(mask_crack)
    crack = np.zeros(num_samples)
    if crack_len > 0:
        tc = np.linspace(0, 0.08, crack_len)
        crack_raw = np.random.normal(0, 2.5, crack_len) * np.exp(-tc * 70.0)
        crack[mask_crack] = np.clip(crack_raw, -1.0, 1.0) * strike_intensity

    # 2. Multi-stage atmospheric cloud rebounds
    bursts = [
        (0.14, 1.0, 1.8),
        (0.48, 0.82, 1.4),
        (1.05, 0.65, 1.1),
        (1.85, 0.50, 0.8),
        (2.90, 0.35, 0.6),
    ]

    rumble_l = np.zeros(num_samples)
    rumble_r = np.zeros(num_samples)
    sos_sub = signal.butter(2, [28, 90], btype="bandpass", fs=sr, output="sos")
    sos_mid = signal.butter(2, [90, 280], btype="bandpass", fs=sr, output="sos")

    for tb, amp, decay_rate in bursts:
        m = t >= tb
        if not np.any(m):
            continue
        dt = t[m] - tb
        raw = np.random.normal(0, 1.0, len(dt)) * np.exp(-dt * decay_rate) * amp
        sub_wave = signal.sosfilt(sos_sub, raw) * 1.6
        mid_wave = signal.sosfilt(sos_mid, raw) * 0.7

        pan_shift = np.sin(dt * 0.8) * 0.35
        rumble_l[m] += (sub_wave + mid_wave) * (1.0 - pan_shift)
        rumble_r[m] += (sub_wave + mid_wave) * (1.0 + pan_shift)

    mix_l = np.tanh(crack * 0.7 + rumble_l * 1.5)
    mix_r = np.tanh(crack * 0.7 + rumble_r * 1.5)

    return normalize_audio(mix_l, mix_r)

# -------------------------------------------------------------
# 3. PHYSICAL FOLEY: DOOR KNOCK & WOOD PANEL RESONANCE
# -------------------------------------------------------------
def synthesize_door_knock(double_knock=True, material="solid_wood", sr=SR):
    """
    Acoustic impulse of knuckles impacting a 2D wood membrane:
    - Broadband mechanical transient of skin/bone
    - Wood panel modal resonances
    - Early room reflections (hallway/room acoustic)
    """
    dur = 1.0
    num_samples = int(dur * sr)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    out_l = np.zeros(num_samples)
    out_r = np.zeros(num_samples)

    def single_knock(t_start, intensity=1.0):
        mask = t >= t_start
        tk = t[mask] - t_start
        lk = len(tk)

        raw_impulse = np.random.normal(0, 1.0, lk) * np.exp(-tk * 280.0)
        sos_impact = signal.butter(2, [120, 1400], btype="bandpass", fs=sr, output="sos")
        impact_transient = signal.sosfilt(sos_impact, raw_impulse)

        thud = (
            0.7 * np.sin(2 * np.pi * 92.0 * tk) * np.exp(-tk * 75.0) +
            0.4 * np.sin(2 * np.pi * 215.0 * tk) * np.exp(-tk * 95.0) +
            0.2 * np.sin(2 * np.pi * 380.0 * tk) * np.exp(-tk * 130.0)
        )
        knock = (impact_transient * 0.9 + thud * 1.2) * intensity

        d1 = int(sr * 0.014)
        d2 = int(sr * 0.028)
        refl_l = np.zeros(lk)
        refl_r = np.zeros(lk)
        if lk > d1:
            refl_l[d1:] += knock[:-d1] * 0.28
        if lk > d2:
            refl_r[d2:] += knock[:-d2] * 0.22

        return mask, knock + refl_l, knock + refl_r

    m1, l1, r1 = single_knock(0.12, 1.0)
    out_l[m1] += l1
    out_r[m1] += r1

    if double_knock:
        m2, l2, r2 = single_knock(0.35, 0.84)
        out_l[m2] += l2
        out_r[m2] += r2

    return normalize_audio(out_l, out_r)

# -------------------------------------------------------------
# 4. PHYSICAL FOLEY: WATER DROP (MINNAERT RESONANCE)
# -------------------------------------------------------------
def synthesize_water_drop(base_freq=440.0, chirp_range=340.0, sr=SR):
    """
    Minnaert bubble cavity resonance:
    Chirps rapidly upward within 25 ms, coupled with fluid cavitation slap.
    Avoids high bird-like frequencies.
    """
    dur = 0.45
    num_samples = int(dur * sr)
    t = np.linspace(0, dur, num_samples, endpoint=False)

    f_t = base_freq + chirp_range * (1.0 - np.exp(-t * 90.0))
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    env = np.exp(-t * 32.0) * (1.0 - np.exp(-t * 600.0))
    bubble = np.sin(phase) * env

    splash = np.random.normal(0, 0.3, num_samples) * np.exp(-t * 150.0)
    sos = signal.butter(2, [300, 1600], btype="bandpass", fs=sr, output="sos")
    splash = signal.sosfilt(sos, splash)

    audio = bubble * 0.9 + splash * 0.35
    return normalize_audio(audio, audio)

# -------------------------------------------------------------
# 5. PHYSICAL FOLEY: RAIN (STOCHASTIC POISSON FIELD)
# -------------------------------------------------------------
def synthesize_rain(duration=4.5, density_drops_sec=140, sr=SR):
    """
    Organic continuous rain:
    - Bandpass shaped noise bed with gentle wind-gust modulations
    - Stochastic Poisson droplet patter spatialized across stereo field
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    white_l = np.random.normal(0, 1.0, num_samples)
    white_r = np.random.normal(0, 1.0, num_samples)

    sos_rain = signal.butter(3, [250, 6800], btype="bandpass", fs=sr, output="sos")
    bed_l = signal.sosfilt(sos_rain, white_l)
    bed_r = signal.sosfilt(sos_rain, white_r)

    mod_wind = 0.75 + 0.25 * np.sin(2 * np.pi * 0.35 * t + 0.5)
    bed_l *= mod_wind * 0.4
    bed_r *= mod_wind * 0.4

    drops_l = np.zeros(num_samples)
    drops_r = np.zeros(num_samples)
    num_patter = int(duration * density_drops_sec)
    indices = np.random.randint(0, num_samples - int(sr * 0.04), num_patter)

    for idx in indices:
        p_dur = np.random.uniform(0.008, 0.025)
        p_len = int(sr * p_dur)
        pt = np.linspace(0, p_dur, p_len, endpoint=False)
        p_freq = np.random.uniform(450, 1400)
        p_env = np.exp(-pt * np.random.uniform(160, 320))
        p_wave = (np.sin(2 * np.pi * p_freq * pt) + 0.3 * np.random.normal(0, 1, p_len)) * p_env
        gain = np.random.uniform(0.08, 0.35)

        p_pan = np.random.uniform(0.0, 1.0)
        drops_l[idx:idx + p_len] += p_wave * gain * np.cos(p_pan * np.pi / 2)
        drops_r[idx:idx + p_len] += p_wave * gain * np.sin(p_pan * np.pi / 2)

    out_l = bed_l + drops_l * 0.6
    out_r = bed_r + drops_r * 0.6
    return normalize_audio(out_l, out_r)

# -------------------------------------------------------------
# 6. MOTION DESIGN: AERODYNAMIC WHOOSH / SWISH
# -------------------------------------------------------------
def synthesize_whoosh(duration=0.72, sweep_center_max=2800.0, sr=SR):
    """
    Cinematic motion transition whoosh:
    - Time-varying resonant bandpass filter tracking raised-cosine curve
    - Stereo fly-by spatialization (left to right)
    - Sub-bass air displacement weight (65-100 Hz)
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    white_l = np.random.normal(0, 1.0, num_samples)
    white_r = np.random.normal(0, 1.0, num_samples)

    env = (np.sin(np.pi * (t / duration) ** 0.8)) ** 3.2
    filtered_l = np.zeros(num_samples)
    filtered_r = np.zeros(num_samples)

    chunk = 256
    for i in range(0, num_samples, chunk):
        end = min(i + chunk, num_samples)
        tc = t[i] / duration
        fc = 160.0 + sweep_center_max * (np.sin(np.pi * tc) ** 2.8)
        fc = np.clip(fc, 120, 11000)

        sos = signal.butter(2, [max(fc * 0.45, 60), min(fc * 1.7, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
        filtered_l[i:end] = signal.sosfilt(sos, white_l[i:end])
        filtered_r[i:end] = signal.sosfilt(sos, white_r[i:end])

    pan = (t / duration) * 1.6 - 0.8
    gain_l = np.cos((pan + 1.0) * np.pi / 4)
    gain_r = np.sin((pan + 1.0) * np.pi / 4)

    sub_phase = 2 * np.pi * np.cumsum(65.0 + 35.0 * np.sin(np.pi * t / duration)) / sr
    sub_weight = np.sin(sub_phase) * (env ** 2) * 0.5

    out_l = (filtered_l * env * gain_l) + sub_weight
    out_r = (filtered_r * env * gain_r) + sub_weight
    return normalize_audio(out_l, out_r)

# -------------------------------------------------------------
# 7. MOTION DESIGN: CINEMATIC IMPACT & SUB-BOOM
# -------------------------------------------------------------
def synthesize_impact(duration=2.2, pitch_start=180.0, pitch_end=42.0, sr=SR):
    """
    Cinematic trailer hit / sub-boom:
    - Pitch-drop oscillator (180 Hz to 42 Hz) with exponential envelope
    - Transient attack punch at 1.4 kHz - 5.2 kHz
    - Analog tape hyperbolic tangent saturation
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    f_t = pitch_end + (pitch_start - pitch_end) * np.exp(-t * 28.0)
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    sub_punch = np.sin(phase) * np.exp(-t * 2.1)

    snap = np.random.normal(0, 1.0, num_samples) * np.exp(-t * 130.0)
    sos_snap = signal.butter(2, [1400, 5200], btype="bandpass", fs=sr, output="sos")
    snap = signal.sosfilt(sos_snap, snap)

    impact = np.tanh((sub_punch * 1.6 + snap * 0.7) * 2.4)
    return normalize_audio(impact, impact)

# -------------------------------------------------------------
# 8. MOTION DESIGN: TENSION RISER
# -------------------------------------------------------------
def synthesize_riser(duration=2.0, start_freq=70.0, end_freq=1150.0, sr=SR):
    """
    Exponential frequency tension riser with accelerating pulse LFO.
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    f_t = start_freq * np.exp(np.log(end_freq / start_freq) * (t / duration))
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    lfo_freq = 4.0 + 22.0 * ((t / duration) ** 2)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(lfo_freq) / sr)
    env = (t / duration) ** 1.85

    audio = (np.sin(phase) + 0.28 * signal.sawtooth(phase)) * lfo * env
    return normalize_audio(audio, audio)

# -------------------------------------------------------------
# 9. MOTION DESIGN: REVERSE SUCK-BACK
# -------------------------------------------------------------
def synthesize_reverse_suckback(duration=1.1, sr=SR):
    """
    Reverse suction whoosh that cuts cleanly before an impact.
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    white = np.random.normal(0, 1.0, num_samples)
    sos = signal.butter(2, [220, 3600], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, white)
    env = np.exp(4.2 * (t / duration - 1.0))

    f_t = 110.0 + 850.0 * ((t / duration) ** 3)
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    audio = (filtered * 0.75 + np.sin(phase) * 0.25) * env

    pan = np.linspace(-0.9, 0.0, num_samples)
    out_l = audio * np.cos((pan + 1) * np.pi / 4)
    out_r = audio * np.sin((pan + 1) * np.pi / 4)
    return normalize_audio(out_l, out_r)

# -------------------------------------------------------------
# 10. MOTION DESIGN: DIGITAL GLITCH
# -------------------------------------------------------------
def synthesize_digital_glitch(duration=0.45, density=10, sr=SR):
    """
    Fragmented digital audio corruption / frame glitch.
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    audio = np.zeros(num_samples)

    for _ in range(density):
        start = np.random.randint(0, num_samples - int(sr * 0.04))
        g_len = np.random.randint(int(sr * 0.006), int(sr * 0.035))
        gt = np.linspace(0, 1, g_len)
        freq = np.random.choice([140, 280, 520, 1100, 2200])
        wave = signal.square(2 * np.pi * freq * gt) if np.random.rand() > 0.4 else np.random.uniform(-1, 1, g_len)
        audio[start:start+g_len] += wave * np.random.uniform(0.4, 0.85)

    audio = np.clip(audio, -0.9, 0.9)
    return normalize_audio(audio, audio)

# -------------------------------------------------------------
# 11. UI TECH: CHIME & HOLOGRAM
# -------------------------------------------------------------
def synthesize_ui_chime(chord="lydian", duration=0.38, sr=SR):
    """
    Crystalline futuristic confirmation chime (E6 + B6 + F#7).
    """
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    f1, f2, f3 = 1318.5, 1975.5, 2959.9
    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 16.0 * t)

    wave = (
        0.5 * np.sin(2 * np.pi * f1 * vibrato * t) +
        0.35 * np.sin(2 * np.pi * f2 * vibrato * t) +
        0.15 * np.sin(2 * np.pi * f3 * vibrato * t)
    )
    env = np.exp(-t * 15.0) * (1.0 - np.exp(-t * 350.0))
    audio = wave * env
    return normalize_audio(audio, audio)
