"""
overnight.py -- Headless overnight brain development run.

Pattern per cycle:
  music, 5s silence, voice, 5s silence, music, 5s silence, voice, ...
  10 music segments + 10 voice clips, then 10 minutes sleep.

Loads classical music from media/classical/ (if available) or generates
synthetic tonal patterns. Voice clips from media/womb_phase/.

All growth features enabled: neurogenesis, synaptogenesis, pruning.
Snapshots every 30 minutes. Brain saves every cycle.

Usage:
    py overnight.py
    py overnight.py --brain brains/regional_cortex_heavy_s42.db --cycles 15
    py overnight.py --no-growth --cycles 5

Output:
    overnight_logs/<timestamp>/  -- CSV, snapshots, brain states
"""
import os, sys, time, argparse, glob, functools
import numpy as np

print = functools.partial(print, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from engine.loader import load
from engine.runner import Brain


# ================================================================
# Audio loading + encoding
# ================================================================

def load_audio_files(audio_dir, target_sr=1000):
    """Load MP3/WAV/OGG files, resample to target_sr. Returns [(name, samples)]."""
    import soundfile as sf

    files = sorted(glob.glob(os.path.join(audio_dir, '*.mp3')) +
                   glob.glob(os.path.join(audio_dir, '*.wav')) +
                   glob.glob(os.path.join(audio_dir, '*.ogg')))
    audio = []
    for fpath in files:
        data, sr = sf.read(fpath, dtype='float64')
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            ratio = sr / target_sr
            indices = np.arange(0, len(data), ratio).astype(int)
            indices = indices[indices < len(data)]
            data = data[indices]
        audio.append((os.path.basename(fpath), data))
    return audio


def generate_synthetic_music(n_songs=10, duration_s=25, sr=1000, rng=None):
    """Generate synthetic tonal patterns that simulate musical content.

    Multi-harmonic signals with slow melody modulation. Each "song" has
    different fundamental frequency, timbre, and rhythm -- providing
    spectrally rich, temporally varying input distinct from voice.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    songs = []
    # Musical fundamentals (Hz) -- spans piano range below Nyquist/2
    fundamentals = [110, 131, 147, 165, 196, 220, 247, 262, 294, 330,
                    175, 233, 277, 311, 349]

    for i in range(n_songs):
        n_samples = duration_s * sr
        t = np.arange(n_samples) / sr

        base_freq = fundamentals[i % len(fundamentals)]
        signal = np.zeros(n_samples, dtype=np.float64)

        # 3-5 harmonics (instrument timbre)
        n_harmonics = rng.randint(3, 6)
        for h in range(1, n_harmonics + 1):
            freq = base_freq * h
            if freq > sr / 2:
                break
            amplitude = 0.4 / h
            phase_offset = rng.rand() * 2 * np.pi
            signal += amplitude * np.sin(2 * np.pi * freq * t + phase_offset)

        # Slow amplitude modulation (breathing/phrasing)
        breath_freq = 0.15 + rng.rand() * 0.25
        signal *= 0.6 + 0.4 * np.sin(2 * np.pi * breath_freq * t)

        # Slow frequency modulation (vibrato/melody)
        vibrato = 0.02 * np.sin(2 * np.pi * (4.0 + rng.rand() * 2) * t)
        melody = 0.08 * np.sin(2 * np.pi * (0.1 + rng.rand() * 0.2) * t)
        mod_signal = np.zeros_like(signal)
        for h in range(1, n_harmonics + 1):
            freq = base_freq * h * (1 + vibrato + melody)
            if np.max(freq) > sr / 2:
                break
            mod_signal += (0.4 / h) * np.sin(2 * np.pi * np.cumsum(freq) / sr)
        signal = 0.5 * signal + 0.5 * mod_signal

        # Normalize to similar amplitude as voice files (~0.25 peak)
        peak = np.max(np.abs(signal))
        if peak > 1e-8:
            signal = signal / peak * 0.25

        songs.append((f'synth_music_{i+1:02d}', signal))

    return songs


def audio_to_fft_bands(samples, n_bands=16, window=64):
    """Convert audio samples to (n_frames, n_bands) FFT energy array."""
    n_frames = len(samples) // window
    if n_frames == 0:
        return np.zeros((1, n_bands))

    trimmed = samples[:n_frames * window].reshape(n_frames, window)
    spectra = np.abs(np.fft.rfft(trimmed, axis=1))

    n_fft = spectra.shape[1]
    bins_per_band = max(1, n_fft // n_bands)
    bands = np.zeros((n_frames, n_bands))
    for b in range(n_bands):
        start = b * bins_per_band
        end = min(start + bins_per_band, n_fft)
        if start < n_fft:
            bands[:, b] = spectra[:, start:end].mean(axis=1)

    return bands


def womb_filter(bands, sr=1000, window=64, cutoff=500, n_bands=16):
    """Apply amniotic fluid low-pass to FFT bands.

    In the womb, sound passes through fluid and tissue. Frequencies above
    ~500 Hz are heavily attenuated. Mother's voice comes through as pitch
    and rhythm, not consonants. Music is bass and midrange only.

    Applies a smooth roll-off (not a brick wall) matching measured
    intra-uterine acoustic transfer functions.
    """
    nyquist = sr / 2.0  # 500 Hz at sr=1000
    band_width = nyquist / n_bands  # ~31.25 Hz per band

    # Build attenuation curve: gentle roll-off starting at cutoff/3
    rolloff_start = cutoff / 3.0  # ~167 Hz: start attenuating
    atten = np.ones(n_bands, dtype=np.float64)
    for b in range(n_bands):
        center_freq = (b + 0.5) * band_width
        if center_freq > rolloff_start:
            # -6 dB/octave roll-off (tissue + fluid absorption)
            octaves_above = np.log2(center_freq / rolloff_start)
            atten[b] = max(0.02, 10 ** (-6 * octaves_above / 20))

    # Apply to all frames
    if bands.ndim == 2:
        return bands * atten[np.newaxis, :]
    return bands * atten


# ================================================================
# Regional neuron mapping + brainstem drive
# ================================================================

# Region definitions matching grow_regional.py
# Region short codes -> full names (and reverse)
REGION_CODES = {
    'brainstem': 'BS', 'sensory': 'SN', 'thalamus': 'TH',
    'amygdala': 'AM', 'hippocampus': 'HP', 'basal_ganglia': 'BG',
    'cortex': 'CX', 'somatosensory': 'SM',
}
CODE_TO_REGION = {v: k for k, v in REGION_CODES.items()}

# Spatial definitions (used for backfill of old brains + wiring)
REGION_DEFS = {
    'brainstem':      {'center': (250, 150, 100), 'radius': 80,  'code': 'BS'},
    'sensory':        {'center': (250, 100, 350), 'radius': 85,  'code': 'SN'},
    'somatosensory':  {'center': (350, 150, 350), 'radius': 70,  'code': 'SM'},
    'thalamus':       {'center': (250, 250, 250), 'radius': 60,  'code': 'TH'},
    'amygdala':       {'center': (350, 400, 200), 'radius': 60,  'code': 'AM'},
    'hippocampus':    {'center': (150, 400, 250), 'radius': 80,  'code': 'HP'},
    'basal_ganglia':  {'center': (250, 300, 250), 'radius': 70,  'code': 'BG'},
    'cortex':         {'center': (250, 250, 400), 'radius': 160, 'code': 'CX'},
}

# Regional excitability: membrane bias per region.
# Brainstem is LOW -- the heartbeat provides its rhythmic drive.
REGION_EXCITABILITY = {
    'BS': 2.0,   # brainstem: subthreshold baseline, heartbeat drives it
    'TH': 2.5,   # thalamus: relay, near threshold
    'AM': 2.5,   # amygdala: fast emotional, responsive
    'SN': 2.0,   # sensory: quiet at rest, lights up with input
    'SM': 2.0,   # somatosensory: quiet at rest, touch drives it
    'HP': 1.8,   # hippocampus: needs cortical/thalamic drive
    'BG': 1.5,   # basal_ganglia: inhibitory gating
    'CX': 1.5,   # cortex: needs thalamic relay to activate
}


class Heartbeat:
    """Rhythmic brainstem driver -- the mother's heartbeat.

    Replaces static brainstem excitability with a periodic current pulse.
    Fetal heart rate ~120 BPM (2 Hz). Short half-sine burst each beat.
    Brainstem fires on each beat, thalamus relays upward, cortex rides on top.

    The heartbeat is THE fundamental alive signal. It goes from conception
    to death. Every other rhythm in the brain sits on top of this one.
    """

    def __init__(self, bs_indices, n_total, bpm=120, amplitude=2.5,
                 pulse_width=60, variability=0.05, seed=42):
        """
        Args:
            bs_indices: numpy array of brainstem neuron indices
            n_total: total neuron count (for current array size)
            bpm: beats per minute (fetal range: 108-138, default 120)
            amplitude: peak current in mA during beat (added to baseline)
            pulse_width: duration of each beat pulse in ticks/ms
            variability: beat-to-beat jitter (fraction of period)
            seed: RNG seed for jitter reproducibility
        """
        self.bs_indices = bs_indices
        self.n_total = n_total
        self.bpm = bpm
        self.period = int(60000 / bpm)  # ms per beat (500 at 120 BPM)
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        self.variability = variability
        self.phase = 0
        self.current_period = self.period
        self.beat_count = 0
        self.rng = np.random.RandomState(seed)

    def current(self):
        """Return current array for this tick. Call once per brain.tick().

        Returns zeros between beats, half-sine pulse during beats.
        """
        I = np.zeros(self.n_total, dtype=np.float64)

        if self.phase < self.pulse_width and len(self.bs_indices) > 0:
            # Half-sine pulse: smooth rise and fall, peaks mid-beat
            t_norm = self.phase / self.pulse_width
            pulse = self.amplitude * np.sin(np.pi * t_norm)
            I[self.bs_indices] = pulse

        self.phase += 1
        if self.phase >= self.current_period:
            self.phase = 0
            self.beat_count += 1
            # Slight beat-to-beat variability (real hearts aren't metronomes)
            jitter = 1.0 + self.rng.uniform(-self.variability, self.variability)
            self.current_period = max(200, int(self.period * jitter))

        return I

    def update_indices(self, bs_indices, n_total):
        """Update after neurogenesis changes brain size."""
        self.bs_indices = bs_indices
        self.n_total = n_total


class WombTouch:
    """Fetal somatosensory input -- what a baby feels in the womb.

    Touch and proprioception develop BEFORE hearing (~8 weeks vs ~20 weeks).
    A fetus in the womb feels:

    1. Amniotic pressure: constant, enveloping, low-level. The baseline of
       being held. Uniform across all touch neurons.

    2. Heartbeat vibration: mother's heartbeat transmitted through fluid as
       a pressure wave felt on skin. Rhythmic, synchronized with brainstem
       heartbeat but experienced as TOUCH, not neural drive.

    3. Breathing rhythm: slow pressure oscillation (~15 breaths/min) as the
       uterus gently compresses/releases with diaphragm movement.

    4. Maternal movement: episodic vestibular/pressure changes. Walking is
       rhythmic (~2 Hz), turning/bending is irregular. Random, gentle.

    5. Self-movement: fetal kicks, stretches, hiccups. Brief, sharp,
       localized. The first experience of agency -- "I moved and felt it."

    The key property: most of this input is PREDICTABLE and SOOTHING.
    The arousal trigger should see small deltas (familiar patterns),
    letting oxytocin build. Novel self-movements create small surprises.
    """

    def __init__(self, sm_indices, n_total, heartbeat_bpm=120,
                 breath_rate=15, seed=42):
        self.sm_indices = sm_indices
        self.n_total = n_total
        self.n_sm = len(sm_indices)
        self.rng = np.random.RandomState(seed)

        # Heartbeat vibration (synced to brainstem heartbeat)
        self.hb_period = int(60000 / heartbeat_bpm)  # ms per beat
        self.hb_phase = 0
        self.hb_amplitude = 1.5   # felt pressure, gentler than brainstem drive

        # Breathing rhythm
        self.breath_period = int(60000 / breath_rate)  # ~4000ms at 15/min
        self.breath_phase = 0
        self.breath_amplitude = 0.8

        # Amniotic pressure (constant baseline)
        self.ambient_pressure = 0.3

        # Maternal movement state
        self.maternal_active = False
        self.maternal_timer = 0
        self.maternal_amplitude = 0.0

        # Self-movement state (fetal kicks)
        self.kick_timer = 0
        self.kick_target = None  # which SM neurons get the kick

        # Assign touch channels to neuron subgroups
        # Split SM neurons into functional zones
        if self.n_sm > 0:
            n_per = max(1, self.n_sm // 5)
            self.zone_pressure = sm_indices[:n_per]           # ambient
            self.zone_heartbeat = sm_indices[n_per:2*n_per]   # cardiac vibration
            self.zone_breathing = sm_indices[2*n_per:3*n_per] # respiratory
            self.zone_vestibular = sm_indices[3*n_per:4*n_per] # maternal movement
            self.zone_proprio = sm_indices[4*n_per:]           # self-movement

    def current(self):
        """Generate per-neuron touch current for this tick."""
        I = np.zeros(self.n_total, dtype=np.float64)
        if self.n_sm == 0:
            return I

        # 1. Ambient amniotic pressure (constant, all touch neurons)
        I[self.sm_indices] += self.ambient_pressure

        # 2. Heartbeat vibration (half-sine pulse, like Heartbeat but as touch)
        pulse_width = 80  # slightly wider than brainstem pulse (felt, not driven)
        if self.hb_phase < pulse_width:
            t_norm = self.hb_phase / pulse_width
            pulse = self.hb_amplitude * np.sin(np.pi * t_norm)
            I[self.zone_heartbeat] += pulse
            # Heartbeat is felt everywhere faintly
            I[self.sm_indices] += pulse * 0.2

        self.hb_phase += 1
        if self.hb_phase >= self.hb_period:
            self.hb_phase = 0

        # 3. Breathing rhythm (slow sine wave)
        breath_t = self.breath_phase / self.breath_period
        breath_signal = self.breath_amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * breath_t))
        I[self.zone_breathing] += breath_signal
        # Breathing is felt as gentle global pressure change
        I[self.sm_indices] += breath_signal * 0.15

        self.breath_phase = (self.breath_phase + 1) % self.breath_period

        # 4. Maternal movement (episodic, random onset/offset)
        if not self.maternal_active:
            # Random chance of starting movement (~once per 30s on average)
            if self.rng.random() < 0.00003:  # ~1 per 33K ticks
                self.maternal_active = True
                self.maternal_timer = self.rng.randint(2000, 8000)  # 2-8s duration
                self.maternal_amplitude = self.rng.uniform(0.5, 2.0)
        else:
            # Walking-like oscillation at ~2 Hz during movement
            walk_signal = self.maternal_amplitude * np.sin(2 * np.pi * 2.0 * self.maternal_timer / 1000)
            I[self.zone_vestibular] += abs(walk_signal)
            self.maternal_timer -= 1
            if self.maternal_timer <= 0:
                self.maternal_active = False

        # 5. Self-movement / fetal kick (brief, sharp, localized)
        if self.kick_timer <= 0:
            # Random kicks (~once per 20s on average)
            if self.rng.random() < 0.00005:  # ~1 per 20K ticks
                self.kick_timer = self.rng.randint(50, 200)  # 50-200ms kick
                # Pick a random subset of proprioceptive neurons
                n_kick = max(1, len(self.zone_proprio) // 3)
                self.kick_target = self.rng.choice(
                    self.zone_proprio, size=n_kick, replace=False)
        else:
            # Active kick -- sharp input to localized neurons
            kick_strength = 3.0 * np.sin(np.pi * self.kick_timer / 100)
            I[self.kick_target] += max(0, kick_strength)
            self.kick_timer -= 1

        return I

    def update_indices(self, sm_indices, n_total):
        """Update after neurogenesis changes brain size."""
        self.sm_indices = sm_indices
        self.n_total = n_total
        self.n_sm = len(sm_indices)
        if self.n_sm > 0:
            n_per = max(1, self.n_sm // 5)
            self.zone_pressure = sm_indices[:n_per]
            self.zone_heartbeat = sm_indices[n_per:2*n_per]
            self.zone_breathing = sm_indices[2*n_per:3*n_per]
            self.zone_vestibular = sm_indices[3*n_per:4*n_per]
            self.zone_proprio = sm_indices[4*n_per:]


def backfill_region_tags(brain):
    """Assign region codes to neurons that don't have them (old brains).

    Uses 3D position to find nearest region. Only runs once at load.
    """
    neurons = brain.data['neurons']
    tagged = sum(1 for n in neurons if n.get('region', ''))
    if tagged == len(neurons):
        return  # all tagged already

    pos = np.array([[n.get('pos_x', 0), n.get('pos_y', 0), n.get('pos_z', 0)]
                     for n in neurons], dtype=np.float64)
    centers = np.array([REGION_DEFS[r]['center'] for r in REGION_DEFS])
    radii = np.array([REGION_DEFS[r]['radius'] for r in REGION_DEFS])
    rnames = list(REGION_DEFS.keys())

    count = 0
    for i, n in enumerate(neurons):
        if n.get('region', ''):
            continue
        dists = np.linalg.norm(pos[i] - centers, axis=1)
        # Find closest region within radius
        best = None
        best_d = float('inf')
        for j, rname in enumerate(rnames):
            if dists[j] < radii[j] and dists[j] < best_d:
                best_d = dists[j]
                best = rname
        n['region'] = REGION_CODES.get(best, '') if best else ''
        count += 1

    if count > 0:
        print(f"  Backfilled region tags for {count} neurons")


def get_region_indices(brain, code):
    """Get neuron indices for a region code. O(n) scan but simple."""
    return np.array([i for i, n in enumerate(brain.data['neurons'])
                     if n.get('region', '') == code], dtype=np.intp)


def get_region_masks(brain):
    """Build boolean masks for all regions. Call once at setup."""
    masks = {}
    for i, n in enumerate(brain.data['neurons']):
        code = n.get('region', '')
        if code not in masks:
            masks[code] = []
        masks[code].append(i)
    return {code: np.array(indices, dtype=np.intp) for code, indices in masks.items()}


def find_sensory_neurons(brain):
    """Find neurons in the sensory region by tag."""
    return get_region_indices(brain, 'SN')


def bootstrap_regional_drive(brain):
    """Replace flat tonic with biologically grounded regional excitability.

    Uses region tags (not spatial lookup). Backfills tags if missing.
    """
    backfill_region_tags(brain)
    masks = get_region_masks(brain)

    regions_found = {}
    total_assigned = 0

    for code, indices in masks.items():
        if code in REGION_EXCITABILITY:
            excit = REGION_EXCITABILITY[code]
            brain.excitability[indices] = excit
            rname = CODE_TO_REGION.get(code, code)
            regions_found[rname] = (len(indices), excit)
            total_assigned += len(indices)

    # Untagged neurons get low excitability
    unassigned = brain.n - total_assigned
    if unassigned > 0:
        untagged = masks.get('', np.array([], dtype=np.intp))
        if len(untagged) > 0:
            brain.excitability[untagged] = 1.0
            regions_found['(unassigned)'] = (len(untagged), 1.0)

    print(f"  Regional drive (replaces tonic):")
    for rname, (count, excit) in sorted(regions_found.items(), key=lambda x: -x[1][1]):
        print(f"    {rname:18s}: {count:5d}N, excitability={excit:.1f}")

    return regions_found


def bands_to_current(bands, sensory_indices, n_total, n_bands=16, audio_gain=6.0):
    """Convert FFT bands to per-neuron current via Gaussian population encoding.

    Distributes n_bands channels across ALL sensory region neurons.
    Each band gets ~N/n_bands neurons with Gaussian tuning curves.
    """
    if bands.ndim == 2:
        bands = bands[0]

    I = np.zeros(n_total, dtype=np.float64)
    n_sensory = len(sensory_indices)
    if n_sensory == 0:
        return I

    band_max = np.max(np.abs(bands))
    if band_max > 1e-8:
        bands = bands / band_max

    pop_size = max(1, n_sensory // n_bands)
    for ch in range(min(n_bands, n_sensory)):
        energy = float(bands[ch]) * audio_gain
        start = ch * pop_size
        end = min(start + pop_size, n_sensory)
        for j in range(start, end):
            preferred = (j - start) / max(end - start - 1, 1)
            response = energy * np.exp(-4.0 * (preferred - 0.5) ** 2)
            I[sensory_indices[j]] = response

    return I


# ================================================================
# Brain save helper
# ================================================================

def save_brain_to(brain_data, path):
    """Full save: create fresh DB with schema, INSERT all neurons + synapses.

    The engine's save() only does UPDATEs, which fails on a fresh DB and
    misses neurons born during growth. This does a complete dump.
    """
    import sqlite3, json
    from schema import create_brain_db

    # Remove existing file to start clean
    if os.path.exists(path):
        os.remove(path)
    create_brain_db(path)
    conn = sqlite3.connect(path)

    # INSERT all neurons
    for n in brain_data['neurons']:
        conn.execute(
            "INSERT INTO neurons (id, neuron_type, a, b, c, d, v, u, last_spike, "
            "pos_x, pos_y, pos_z, dopamine_sens, excitability, activity_trace, region) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n['id'], n.get('type', 'RS'), n['a'], n['b'], n['c'], n['d'],
             n['v'], n['u'], n['last_spike'],
             n.get('pos_x', 0.0), n.get('pos_y', 0.0), n.get('pos_z', 0.0),
             n.get('dopamine_sens', 0.0), n.get('excitability', 0.0),
             n.get('activity_trace', 0.0), n.get('region', ''))
        )

    # INSERT all synapses
    for s in brain_data['synapses'] + brain_data.get('gap_junctions', []):
        module = s.get('module')
        state = {}
        if module and hasattr(module, 'INITIAL_STATE'):
            for key in module.INITIAL_STATE:
                if key in s:
                    state[key] = s[key]
        params = {}
        if module and hasattr(module, 'DEFAULTS'):
            for key in module.DEFAULTS:
                if key in s:
                    params[key] = s[key]
        conn.execute(
            "INSERT INTO synapses (id, source, target, weight, delay, synapse_type, params, state) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (s['id'], s['source_db_id'], s['target_db_id'], s['weight'], s.get('delay', 1),
             s['type'], json.dumps(params), json.dumps(state))
        )

    conn.commit()
    conn.close()


# ================================================================
# Snapshot -- the "pictures of the journey"
# ================================================================

def take_snapshot(brain, sensory_indices, rng, total_ticks, cycle,
                  run_dir, snapshot_num, heartbeat=None):
    """Take a development snapshot: probe cascade + statistics."""
    snap_path = os.path.join(run_dir, f'snapshot_{snapshot_num:03d}_cycle{cycle:03d}.txt')

    # Probe: stimulate 10 random sensory + 10 random other neurons
    n_stim_sensory = min(10, len(sensory_indices))
    n_stim_other = min(10, brain.n - len(sensory_indices))
    stim_sensory = rng.choice(sensory_indices, size=n_stim_sensory, replace=False)
    other_mask = np.ones(brain.n, dtype=bool)
    other_mask[sensory_indices] = False
    other_neurons = np.flatnonzero(other_mask)
    stim_other = rng.choice(other_neurons, size=n_stim_other, replace=False) if len(other_neurons) > 0 else np.array([])

    zeros = np.zeros(brain.n, dtype=np.float64)

    # Run sensory probe (heartbeat continues during probe)
    I_stim = zeros.copy()
    I_stim[stim_sensory] += 15.0
    pre_spikes = len(brain.recorder.spikes)
    sensory_cascade = []
    for t in range(200):
        I_base = heartbeat.current() if heartbeat else zeros.copy()
        if t < 5:
            I_base[stim_sensory] += 15.0
        fired = brain.tick(I_base)
        sensory_cascade.append(len(fired))
    sensory_total = len(brain.recorder.spikes) - pre_spikes

    # Run internal probe
    pre_spikes2 = len(brain.recorder.spikes)
    internal_cascade = []
    for t in range(200):
        I_base = heartbeat.current() if heartbeat else zeros.copy()
        if t < 5 and len(stim_other) > 0:
            I_base[stim_other.astype(int)] += 15.0
        fired = brain.tick(I_base)
        internal_cascade.append(len(fired))
    internal_total = len(brain.recorder.spikes) - pre_spikes2

    # Compute region activity from activity_trace (using region tags)
    region_activity = {}
    masks = get_region_masks(brain)
    for code, indices in masks.items():
        if not code:
            continue
        label = CODE_TO_REGION.get(code, code)
        trace = brain.activity_trace[indices]
        region_activity[label] = (len(indices), float(np.mean(trace)),
                                   float(np.std(trace)))

    # Write snapshot
    with open(snap_path, 'w') as f:
        f.write(f"DEVELOPMENT SNAPSHOT #{snapshot_num}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Cycle: {cycle}, Tick: {total_ticks}\n")
        f.write(f"Wall time: {time.strftime('%H:%M:%S')}\n")
        f.write(f"Brain: {brain.n} neurons, {len(brain.synapses)} synapses\n\n")

        f.write(f"Signals:\n")
        f.write(f"  D (surprise):      {getattr(brain, 'surprise', 0.0):.6f}\n")
        f.write(f"  D (ema):           {brain.sensory_ema:.6f}\n")
        f.write(f"  A (arousal):       {brain.arousal:.6f}\n")
        f.write(f"  S (learning rate): {brain.learning_rate_scale:.4f}\n")
        f.write(f"  C (cortisol):      {getattr(brain, 'cortisol', 0.0):.6f}\n")
        f.write(f"  O (oxytocin):      {getattr(brain, 'oxytocin', 0.0):.6f}\n\n")

        f.write(f"Sensory Probe (stim {n_stim_sensory} sensory neurons):\n")
        f.write(f"  Cascade: {sensory_total} spikes in 200 ticks\n")
        f.write(f"  Peak: {max(sensory_cascade)} spikes/tick at t={sensory_cascade.index(max(sensory_cascade))}\n")
        f.write(f"  Tail (last 50): {sum(sensory_cascade[-50:])} spikes\n\n")

        f.write(f"Internal Probe (stim {n_stim_other} random neurons):\n")
        f.write(f"  Cascade: {internal_total} spikes in 200 ticks\n")
        f.write(f"  Peak: {max(internal_cascade)} spikes/tick at t={internal_cascade.index(max(internal_cascade))}\n")
        f.write(f"  Tail (last 50): {sum(internal_cascade[-50:])} spikes\n\n")

        f.write(f"Region Activity (mean +/- std of activity_trace):\n")
        for label, (count, mean, std) in sorted(region_activity.items()):
            f.write(f"  {label:15s}: {count:4d}N, trace={mean:.6f} +/- {std:.6f}\n")

        # Weight distribution
        if brain.plastic_w_arr is not None and len(brain.plastic_w_arr) > 0:
            f.write(f"\nPlastic Weight Distribution:\n")
            f.write(f"  Count: {len(brain.plastic_w_arr)}\n")
            f.write(f"  Mean:  {np.mean(brain.plastic_w_arr):.4f}\n")
            f.write(f"  Std:   {np.std(brain.plastic_w_arr):.4f}\n")
            f.write(f"  Range: [{np.min(brain.plastic_w_arr):.4f}, {np.max(brain.plastic_w_arr):.4f}]\n")

        if hasattr(brain, 'reward_w_arr') and len(brain.reward_w_arr) > 0:
            f.write(f"\nReward Weight Distribution:\n")
            f.write(f"  Count: {len(brain.reward_w_arr)}\n")
            f.write(f"  Mean:  {np.mean(brain.reward_w_arr):.4f}\n")
            f.write(f"  Std:   {np.std(brain.reward_w_arr):.4f}\n")
            f.write(f"  Range: [{np.min(brain.reward_w_arr):.4f}, {np.max(brain.reward_w_arr):.4f}]\n")

        f.write(f"\nExcitability Distribution:\n")
        f.write(f"  Mean:  {np.mean(brain.excitability):.4f}\n")
        f.write(f"  Std:   {np.std(brain.excitability):.4f}\n")
        f.write(f"  Range: [{np.min(brain.excitability):.4f}, {np.max(brain.excitability):.4f}]\n")

        f.write(f"\nExcitability by Region:\n")
        for code, ridx in masks.items():
            if not code:
                continue
            label = CODE_TO_REGION.get(code, code)
            ex = brain.excitability[ridx]
            f.write(f"  {label:18s}: {len(ridx):5d}N, mean={np.mean(ex):.3f}, "
                    f"range=[{np.min(ex):.3f}, {np.max(ex):.3f}]\n")

    print(f"  Snapshot #{snapshot_num}: sensory={sensory_total}, internal={internal_total} cascade spikes")
    return snap_path


# ================================================================
# Main overnight loop
# ================================================================

def run_overnight(args):
    """Main overnight loop. Steven's pattern: music, silence, voice, silence."""

    # Setup output
    log_dir = os.path.join(BASE, 'overnight_logs')
    os.makedirs(log_dir, exist_ok=True)
    run_id = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(log_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Load brain
    brain_path = os.path.join(BASE, args.brain)
    if not os.path.exists(brain_path):
        print(f"Brain not found: {brain_path}")
        print(f"Generate one first: py grow_regional.py --config cortex_heavy")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  OVERNIGHT BRAIN DEVELOPMENT RUN")
    print(f"  Brain: {args.brain}")
    print(f"  Cycles: {args.cycles}")
    print(f"  Growth: {'ON' if args.growth else 'OFF'}")
    print(f"  Womb filter: {'ON' if args.womb else 'OFF'}")
    print(f"  Sleep ticks: {args.sleep_ticks} ({args.sleep_ticks/100:.0f}s wall @ 100t/s)")
    print(f"  Output: {run_dir}")
    print(f"{'='*60}")

    data = load(brain_path)
    brain = Brain(data, learn=True)
    print(f"  Loaded: {brain.n} neurons, {len(brain.synapses)} synapses")

    # Find sensory neurons + set up DAS mask
    sensory_indices = find_sensory_neurons(brain)
    brain.sensory_mask = np.zeros(brain.n, dtype=bool)
    brain.sensory_mask[sensory_indices] = True
    print(f"  Sensory region: {len(sensory_indices)} neurons")

    # Bootstrap: replace flat tonic with regional excitability
    # Brainstem is low baseline -- heartbeat provides its rhythmic drive.
    bootstrap_regional_drive(brain)

    # Heartbeat: the mother's heart drives the brainstem rhythmically.
    # 120 BPM = 2 Hz, half-sine pulse, 60ms wide, 2.5 mA peak.
    bs_indices = get_region_indices(brain, 'BS')
    heartbeat = Heartbeat(bs_indices, brain.n, bpm=args.heartbeat_bpm,
                          amplitude=2.5, pulse_width=60, variability=0.05)
    print(f"  Heartbeat: {heartbeat.bpm} BPM, {len(bs_indices)} brainstem neurons, "
          f"{heartbeat.amplitude} mA peak")

    # Somatosensory: what the fetus FEELS in the womb.
    # Touch develops before hearing. Always on -- the baby always feels.
    sm_indices = get_region_indices(brain, 'SM')
    if len(sm_indices) > 0:
        womb_touch = WombTouch(sm_indices, brain.n, heartbeat_bpm=args.heartbeat_bpm)
        print(f"  Somatosensory: {len(sm_indices)} neurons (touch/pressure/proprioception)")
    else:
        womb_touch = None
        print(f"  Somatosensory: no SM neurons (old brain, touch disabled)")

    # Load voice clips
    voice_dir = os.path.join(BASE, 'media', 'womb_phase')
    voice_files = load_audio_files(voice_dir)
    voice_bands = []
    if voice_files:
        print(f"  Voice clips: {len(voice_files)} files")
        for fname, samples in voice_files:
            bands = audio_to_fft_bands(samples, n_bands=16, window=64)
            if args.womb:
                bands = womb_filter(bands)
            voice_bands.append((fname, bands))
    else:
        print(f"  No voice clips found in {voice_dir}")

    # Load classical music (or generate synthetic)
    classical_dir = os.path.join(BASE, 'media', 'classical')
    music_files = load_audio_files(classical_dir) if os.path.isdir(classical_dir) else []
    music_bands = []
    if music_files:
        print(f"  Classical music: {len(music_files)} files")
        for fname, samples in music_files:
            bands = audio_to_fft_bands(samples, n_bands=16, window=64)
            if args.womb:
                bands = womb_filter(bands)
            music_bands.append((fname, bands))
    else:
        print(f"  No classical music -- generating synthetic tonal patterns")
        rng_music = np.random.RandomState(77)
        synth = generate_synthetic_music(n_songs=10, duration_s=25, rng=rng_music)
        for fname, samples in synth:
            bands = audio_to_fft_bands(samples, n_bands=16, window=64)
            if args.womb:
                bands = womb_filter(bands)
            music_bands.append((fname, bands))
            print(f"    {fname}: {bands.shape[0]} frames")

    # CSV log
    log_path = os.path.join(run_dir, 'development.csv')
    with open(log_path, 'w') as f:
        f.write('cycle,tick,wall_s,neurons,synapses,cycle_spikes,'
                'surprise,surprise_ema,stability,arousal,cortisol,oxytocin,'
                'born,culled,new_synapses,'
                'sleep_replay,sleep_sprouted,sleep_drifted\n')

    # Save initial snapshot
    rng = np.random.RandomState(42)
    total_ticks = 0
    start_time = time.time()
    last_snapshot_time = start_time
    snapshot_num = 0

    print(f"\n  Taking initial snapshot...")
    take_snapshot(brain, sensory_indices, rng, 0, 0, run_dir, 0, heartbeat)

    # Post-birth callback: tag region + assign excitability to newly born neurons
    def assign_regional_excitability(brain, new_indices):
        for ni in new_indices:
            n = brain.neurons[ni]
            pos = np.array([n.get('pos_x', 0), n.get('pos_y', 0), n.get('pos_z', 0)])
            best_region = None
            best_dist = float('inf')
            for rname, rdef in REGION_DEFS.items():
                d = np.linalg.norm(pos - np.array(rdef['center']))
                if d < rdef['radius'] and d < best_dist:
                    best_dist = d
                    best_region = rname
            # Tag the neuron with its region code
            code = REGION_CODES.get(best_region, '') if best_region else ''
            n['region'] = code
            excit = REGION_EXCITABILITY.get(code, 1.0) if code else 1.0
            brain.excitability[ni] = excit

    def base_current():
        """Heartbeat + somatosensory touch. The constant background of being alive."""
        I = heartbeat.current()
        if womb_touch is not None:
            I += womb_touch.current()
        return I

    silence_ticks = int(args.silence * 1000)  # seconds -> ticks (1 tick = 1ms)
    n_music = min(args.tracks, len(music_bands))
    n_voice = min(args.tracks, len(voice_bands))

    def synapse_breakdown():
        """Count synapses by type."""
        counts = {}
        for s in brain.synapses:
            counts[s['type']] = counts.get(s['type'], 0) + 1
        return counts

    print(f"\n  Pattern per cycle: {n_music} music + {n_voice} voice, "
          f"{args.silence}s silence gaps, {args.sleep_ticks} tick sleep")
    print(f"  Starting {'='*40}")

    for cycle in range(1, args.cycles + 1):
        cycle_start = time.time()
        cycle_spikes = 0
        cycle_growth = {'neurons_born': 0, 'synapses_added': 0, 'neurons_culled': 0}

        print(f"\n  --- Cycle {cycle}/{args.cycles} ---")
        sc = synapse_breakdown()
        sc_str = ' + '.join(f'{v} {k}' for k, v in sorted(sc.items()))
        print(f"  Brain: {brain.n}N, {len(brain.synapses)}S ({sc_str})")
        cortisol = getattr(brain, 'cortisol', 0.0)
        oxytocin = getattr(brain, 'oxytocin', 0.0)
        print(f"  DAS: D={getattr(brain, 'surprise', 0.0):.4f} A={brain.arousal:.4f} S={brain.learning_rate_scale:.3f} "
              f"C={cortisol:.4f} O={oxytocin:.4f}")

        # Phase 1: WARM UP (1000 ticks, heartbeat + touch)
        for t in range(1000):
            fired = brain.tick(base_current())
            cycle_spikes += len(fired)
            total_ticks += 1

        # Phase 2: Alternating music + voice with silence gaps
        for track_i in range(max(n_music, n_voice)):
            # --- MUSIC ---
            if track_i < n_music:
                mi = track_i % len(music_bands)
                fname, bands = music_bands[mi]
                file_spikes = 0
                for frame in range(bands.shape[0]):
                    I = base_current()
                    I += bands_to_current(bands[frame:frame+1], sensory_indices,
                                          brain.n, audio_gain=args.audio_gain)
                    fired = brain.tick(I)
                    file_spikes += len(fired)
                    cycle_spikes += len(fired)
                    total_ticks += 1

                if args.growth:
                    # Trim recorder before growth to prevent MemoryError during
                    # synapse rebuild (108M+ spike tuples = ~7 GB per cycle)
                    brain.recorder.trim(5000)
                    stats = brain.dynamic_growth(rng=rng, post_birth_fn=assign_regional_excitability)
                    for k in cycle_growth:
                        cycle_growth[k] += stats.get(k, 0)
                    if stats.get('neurons_born', 0) > 0:
                        heartbeat.update_indices(get_region_indices(brain, 'BS'), brain.n)
                        sensory_indices = find_sensory_neurons(brain)
                        if womb_touch is not None:
                            womb_touch.update_indices(get_region_indices(brain, 'SM'), brain.n)

                g = cycle_growth
                gstr = f", +{g['neurons_born']}N +{g['synapses_added']}S" if g['neurons_born'] else ''
                c_val = getattr(brain, 'cortisol', 0.0)
                o_val = getattr(brain, 'oxytocin', 0.0)
                print(f"    M[{track_i+1:2d}] {fname}: {file_spikes} spk, "
                      f"D={getattr(brain, 'surprise', 0.0):.3f} A={brain.arousal:.3f} "
                      f"C={c_val:.3f} O={o_val:.3f}{gstr}")

            # --- SILENCE (heartbeat + touch) ---
            for t in range(silence_ticks):
                fired = brain.tick(base_current())
                cycle_spikes += len(fired)
                total_ticks += 1

            # --- VOICE ---
            if track_i < n_voice:
                vi = track_i % len(voice_bands)
                fname, bands = voice_bands[vi]
                file_spikes = 0
                for frame in range(bands.shape[0]):
                    I = base_current()
                    I += bands_to_current(bands[frame:frame+1], sensory_indices,
                                          brain.n, audio_gain=args.audio_gain)
                    fired = brain.tick(I)
                    file_spikes += len(fired)
                    cycle_spikes += len(fired)
                    total_ticks += 1

                if args.growth:
                    brain.recorder.trim(5000)
                    stats = brain.dynamic_growth(rng=rng, post_birth_fn=assign_regional_excitability)
                    for k in cycle_growth:
                        cycle_growth[k] += stats.get(k, 0)
                    if stats.get('neurons_born', 0) > 0:
                        heartbeat.update_indices(get_region_indices(brain, 'BS'), brain.n)
                        sensory_indices = find_sensory_neurons(brain)
                        if womb_touch is not None:
                            womb_touch.update_indices(get_region_indices(brain, 'SM'), brain.n)

                born = cycle_growth['neurons_born']
                born_str = f', +{born}N' if born else ''
                c_val = getattr(brain, 'cortisol', 0.0)
                o_val = getattr(brain, 'oxytocin', 0.0)
                print(f"    V[{track_i+1:2d}] {fname}: {file_spikes} spk, "
                      f"D={getattr(brain, 'surprise', 0.0):.3f} A={brain.arousal:.3f} "
                      f"C={c_val:.3f} O={o_val:.3f}{born_str}")

            # --- SILENCE (heartbeat + touch) ---
            for t in range(silence_ticks):
                fired = brain.tick(base_current())
                cycle_spikes += len(fired)
                total_ticks += 1

        # Phase 3: SLEEP
        # Trim recorder before sleep to prevent MemoryError on large brains.
        # Full cycle = 108M+ spike tuples = ~7 GB. Sprout uses sleep replay
        # window; drift needs enough to identify silent neurons.
        brain.recorder.trim(5000)
        print(f"  Sleep phase ({args.sleep_ticks} ticks)...", end='')
        sleep_result = brain.sleep(ticks=args.sleep_ticks, compression=0.85, seed=cycle)
        total_ticks += args.sleep_ticks
        print(f" replay={sleep_result.get('replay_spikes', 0)}, "
              f"compressed={sleep_result.get('compressed', 0)}+{sleep_result.get('plastic_compressed', 0)}, "
              f"sprouted={sleep_result.get('sprouted', 0)}, "
              f"drifted={sleep_result.get('drifted', 0)}")

        # Log to CSV
        elapsed = time.time() - start_time
        with open(log_path, 'a') as f:
            c_end = getattr(brain, 'cortisol', 0.0)
            o_end = getattr(brain, 'oxytocin', 0.0)
            f.write(f"{cycle},{total_ticks},{elapsed:.0f},"
                    f"{brain.n},{len(brain.synapses)},{cycle_spikes},"
                    f"{getattr(brain, 'surprise', 0.0):.6f},{brain.sensory_ema:.6f},"
                    f"{brain.learning_rate_scale:.4f},{brain.arousal:.6f},"
                    f"{c_end:.6f},{o_end:.6f},"
                    f"{cycle_growth['neurons_born']},{cycle_growth['neurons_culled']},"
                    f"{cycle_growth['synapses_added']},"
                    f"{sleep_result.get('replay_spikes', 0)},"
                    f"{sleep_result.get('sprouted', 0)},{sleep_result.get('drifted', 0)}\n")

        cycle_elapsed = time.time() - cycle_start
        print(f"  Cycle {cycle} done: {cycle_elapsed:.0f}s, {cycle_spikes} spk, "
              f"{brain.n}N {len(brain.synapses)}S, {heartbeat.beat_count} heartbeats")

        # Save brain every cycle
        save_path = os.path.join(run_dir, f'brain_cycle_{cycle:03d}.db')
        brain.sync_state()
        save_brain_to(brain.data, save_path)
        print(f"  Saved: brain_cycle_{cycle:03d}.db")

        # Flush recorder to free memory (78M+ spikes/cycle = ~1.2 GB)
        brain.recorder.spikes.clear()
        brain.recorder.intervals.clear()

        # Snapshot at configured interval (0 = every cycle)
        now = time.time()
        if now - last_snapshot_time >= args.snapshot_interval:
            snapshot_num += 1
            take_snapshot(brain, sensory_indices, rng,
                         total_ticks, cycle, run_dir, snapshot_num, heartbeat)
            last_snapshot_time = now

    # Final snapshot + save
    snapshot_num += 1
    take_snapshot(brain, sensory_indices, rng,
                 total_ticks, args.cycles, run_dir, snapshot_num, heartbeat)

    final_path = os.path.join(run_dir, 'brain_final.db')
    brain.sync_state()
    save_brain_to(brain.data, final_path)

    total_time = time.time() - start_time
    hours = total_time / 3600
    print(f"\n{'='*60}")
    print(f"  OVERNIGHT RUN COMPLETE")
    print(f"  Duration: {hours:.1f} hours ({total_time:.0f}s)")
    print(f"  Ticks: {total_ticks} ({total_ticks/total_time:.0f} t/s)")
    print(f"  Brain: {brain.n}N, {len(brain.synapses)}S")
    print(f"  Snapshots: {snapshot_num + 1}")
    print(f"  Final: {final_path}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser(description='Overnight brain development')
    p.add_argument('--brain', default='brains/test_v10.db',
                   help='Brain DB path (relative to project root)')
    p.add_argument('--cycles', type=int, default=15,
                   help='Number of cycles (each ~25-30 min)')
    p.add_argument('--tracks', type=int, default=10,
                   help='Music + voice tracks per cycle')
    p.add_argument('--tonic', type=float, default=0.0,
                   help='Legacy flat tonic (0=off, uses brainstem drive instead)')
    p.add_argument('--audio-gain', type=float, default=6.0,
                   help='Audio-to-current multiplier')
    p.add_argument('--silence', type=float, default=5.0,
                   help='Silence gap in seconds between tracks')
    p.add_argument('--sleep-ticks', type=int, default=60000,
                   help='Sleep phase ticks per cycle (~10 min at 100 t/s)')
    p.add_argument('--growth', action='store_true', default=True,
                   help='Enable runtime neurogenesis (default: on)')
    p.add_argument('--no-growth', dest='growth', action='store_false',
                   help='Disable runtime neurogenesis')
    p.add_argument('--snapshot-interval', type=int, default=1800,
                   help='Snapshot interval in seconds (default: 1800=30min)')
    p.add_argument('--practice', action='store_true',
                   help='Practice mode: 2 cycles, 1 track, short sleep, snapshot every cycle')
    p.add_argument('--womb', action='store_true', default=True,
                   help='Womb mode: low-pass filter on audio (default: on)')
    p.add_argument('--no-womb', dest='womb', action='store_false',
                   help='Disable womb audio filter (full spectrum)')
    p.add_argument('--heartbeat-bpm', type=int, default=120,
                   help='Heartbeat rate in BPM (fetal range: 108-138)')
    args = p.parse_args()
    if args.practice:
        args.cycles = 2
        args.tracks = 1
        args.sleep_ticks = 5000   # ~50s wall instead of 10 min
        args.silence = 2.0        # 2s gaps instead of 5
        args.snapshot_interval = 0  # snapshot every cycle
    run_overnight(args)


if __name__ == '__main__':
    main()
