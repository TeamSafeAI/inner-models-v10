"""
signal.py -- General neuromodulatory signal system.

Each neuromodulator (dopamine, serotonin, norepinephrine, cortisol, oxytocin, ...)
is a Signal instance configured by a dict. No subclasses, no custom code per chemical.

Architecture:
  Signal       -- name, config, runtime level, internal state
  SignalSystem -- ordered collection, shared context, per-tick update
  Triggers     -- registered functions that compute signal activation

Adding a new chemical:
  1. Define its config dict (trigger type, params)
  2. If it needs a new trigger type, write and register it
  3. system.add(name, config)

Trigger types:
  prediction_error  -- EMA tracking, fires on deviation (Dopamine)
  input_delta       -- absolute change detection, spike+decay (Norepinephrine)
  population_stat   -- periodic population variance, dynamic baseline (Serotonin)
  sustained_signal  -- fires when another signal stays elevated (Cortisol)
  calm_familiar     -- fires when input is stable and known (Oxytocin)
"""

import numpy as np


# =====================================================================
# Signal + SignalSystem
# =====================================================================

class Signal:
    """One neuromodulatory signal."""
    __slots__ = ('name', 'config', 'level', 'state')

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.level = config.get('initial_level', 0.0)
        self.state = {}


class SignalSystem:
    """Manages all signals. One instance per brain."""

    def __init__(self):
        self.signals = {}  # insertion-ordered (Python 3.7+)

    def add(self, name, config):
        """Add a signal. Order of addition = update order."""
        self.signals[name] = Signal(name, config)
        return self.signals[name]

    def __getitem__(self, name):
        return self.signals[name]

    def update_all(self, brain, external_I, tick):
        """Update all signals once per tick."""
        context = {
            'sensory_energy': _sensory_energy(brain, external_I),
            'tick': tick,
            'signals': self.signals,
        }
        for sig in self.signals.values():
            _TRIGGERS[sig.config['trigger']['type']](sig, brain, context)


# =====================================================================
# Shared metrics
# =====================================================================

def _sensory_energy(brain, external_I):
    """Variance of external input, masked to sensory neurons if available."""
    if external_I is None:
        return 0.0
    if hasattr(brain, 'sensory_mask') and brain.sensory_mask is not None:
        mask = brain.sensory_mask
        if len(mask) < len(external_I):
            mask = np.concatenate([mask, np.zeros(len(external_I) - len(mask), dtype=bool)])
            brain.sensory_mask = mask
        return float(np.var(external_I[mask]))
    return float(np.var(external_I))


# =====================================================================
# Trigger: prediction_error (Dopamine-like)
#
# Tracks EMA of sensory energy. Fires reward when fractional deviation
# exceeds threshold. "Something unexpected -- update your model."
#
# Config keys:
#   ema_alpha    -- EMA smoothing (0.02 = slow tracking)
#   threshold    -- fractional deviation to trigger (0.15 = 15%)
#   reward_scale -- surprise magnitude -> reward magnitude (capped at 1.0)
# =====================================================================

def _trigger_prediction_error(sig, brain, ctx):
    cfg = sig.config['trigger']
    energy = ctx['sensory_energy']

    # Update EMA
    ema = sig.state.get('ema', 0.0)
    ema = (1 - cfg['ema_alpha']) * ema + cfg['ema_alpha'] * energy
    sig.state['ema'] = ema
    brain.sensory_ema = ema

    # Surprise = fractional deviation from EMA
    if ema > 1e-6:
        surprise = abs(energy - ema) / ema
        sig.level = surprise
        brain.surprise = surprise
        if surprise > cfg['threshold']:
            brain.deliver_reward(min(1.0, cfg['reward_scale'] * surprise))
    else:
        sig.level = 0.0
        brain.surprise = 0.0


# =====================================================================
# Trigger: population_stat (Serotonin-like)
#
# Periodically measures population activity variance. Learns a dynamic
# baseline via EMA. Deviation above baseline reduces signal level
# (= learning rate scale). "Population unstable -- slow learning."
#
# Config keys:
#   eval_interval -- ticks between evaluations (100)
#   ema_alpha     -- baseline adaptation speed (0.05)
#   sensitivity   -- fractional: 0.5 above baseline -> level drops (2.0)
#   min_scale     -- floor for learning rate scale (0.2)
# =====================================================================

def _trigger_population_stat(sig, brain, ctx):
    cfg = sig.config['trigger']
    tick = ctx['tick']

    if tick == 0 or tick % cfg['eval_interval'] != 0:
        return

    pop_var = float(np.var(brain.activity_trace))

    # Dynamic baseline via EMA
    var_ema = sig.state.get('var_ema', None)
    if var_ema is None:
        sig.state['var_ema'] = pop_var
    else:
        sig.state['var_ema'] = ((1 - cfg['ema_alpha']) * var_ema +
                                 cfg['ema_alpha'] * pop_var)

    baseline = sig.state['var_ema']
    if baseline > 1e-8:
        frac_above = max(0.0, (pop_var - baseline) / baseline)
    else:
        frac_above = 0.0

    sig.level = max(cfg['min_scale'], min(1.0,
        1.0 - frac_above * cfg['sensitivity']))

    brain.learning_rate_scale = sig.level
    brain.stability_var_ema = sig.state['var_ema']


# =====================================================================
# Trigger: input_delta (Norepinephrine-like)
#
# Detects UNEXPECTED changes in sensory energy. Tracks an EMA of
# tick-to-tick deltas (adapted expectation). Only spikes when current
# delta exceeds the adapted expectation by delta_threshold factor.
# Predictable input habituates away. Novel input spikes.
# "Something unexpected changed -- wake up."
#
# Config keys:
#   delta_threshold     -- ratio: delta must exceed EMA by this factor (2.0)
#   delta_floor         -- absolute minimum delta to spike (0.05)
#   habituation_alpha   -- delta EMA adaptation rate (0.01)
#   spike_scale         -- novelty ratio -> arousal increment (0.4)
#   decay               -- per-tick multiplicative decay (0.995)
# =====================================================================

def _trigger_input_delta(sig, brain, ctx):
    cfg = sig.config['trigger']
    energy = ctx['sensory_energy']
    prev = sig.state.get('prev_energy', 0.0)

    raw_delta = abs(energy - prev)

    # Habituation: adapt to expected delta magnitude
    hab_alpha = cfg.get('habituation_alpha', 0.01)
    delta_ema = sig.state.get('delta_ema', raw_delta + 1e-6)
    delta_ema = (1.0 - hab_alpha) * delta_ema + hab_alpha * raw_delta
    sig.state['delta_ema'] = delta_ema

    # Spike on UNEXPECTED deltas (exceed adapted expectation)
    floor = cfg.get('delta_floor', 0.05)
    threshold = max(floor, delta_ema * cfg['delta_threshold'])
    if raw_delta > threshold:
        novelty = raw_delta / max(1e-6, delta_ema)
        sig.level = min(1.0, sig.level + cfg['spike_scale'] * (novelty - 1.0))

    sig.level *= cfg['decay']
    sig.state['prev_energy'] = energy

    brain.arousal = sig.level
    brain._prev_sensory_energy = energy


# =====================================================================
# Trigger: allostatic_load (Cortisol-like)
#
# Responds to sustained sensory overload RELATIVE TO adapted baseline.
# The brain maintains slow EMAs of sensory energy mean and variance.
# When input exceeds the adapted range (z-score > threshold), an onset
# counter accumulates. Only sustained overload triggers cortisol rise.
#
# Predictable stimulation adapts away. Only escalating or unpredictable
# input that keeps exceeding the brain's model produces cortisol.
#
# Config keys:
#   ema_alpha       -- sensory energy baseline adaptation rate (0.005)
#   var_alpha       -- energy variance adaptation rate (0.005)
#   z_threshold     -- std devs above mean to count as overload (2.0)
#   onset_ticks     -- sustained overload ticks before rising (200)
#   rise_rate       -- per-tick level increase when active (0.002)
#   decay           -- per-tick multiplicative decay (0.9995)
#   counter_decay   -- counter decay rate when not overloaded (0.95)
#   buffer          -- rise rate multiplier (0.15 fetal, 1.0 postnatal)
# =====================================================================

def _trigger_allostatic_load(sig, brain, ctx):
    cfg = sig.config['trigger']
    energy = ctx['sensory_energy']

    # Slow-adapting baseline of sensory energy
    alpha = cfg['ema_alpha']
    ema = sig.state.get('energy_ema', energy)
    ema = (1.0 - alpha) * ema + alpha * energy
    sig.state['energy_ema'] = ema

    # Slow-adapting variance (second-order -- how variable is input normally?)
    var_alpha = cfg['var_alpha']
    deviation_sq = (energy - ema) ** 2
    ema_var = sig.state.get('energy_var_ema', deviation_sq + 1e-6)
    ema_var = (1.0 - var_alpha) * ema_var + var_alpha * deviation_sq
    sig.state['energy_var_ema'] = ema_var

    # Z-score: how far above adapted normal? (one-sided -- only overstimulation)
    std = max(0.01, ema_var ** 0.5)
    z = max(0.0, (energy - ema) / std)

    # Onset counter: accumulates during sustained overload, decays otherwise
    counter = sig.state.get('onset_counter', 0.0)
    if z > cfg['z_threshold']:
        counter += 1.0
    else:
        counter *= cfg['counter_decay']
    sig.state['onset_counter'] = counter

    # Rise when sustained (scaled by placental buffer)
    if counter >= cfg['onset_ticks']:
        sig.level = min(1.0, sig.level + cfg['rise_rate'] * cfg.get('buffer', 1.0))

    # Always decay
    sig.level *= cfg['decay']

    if sig.level < 1e-6:
        sig.level = 0.0


# =====================================================================
# Trigger: calm_familiar (Oxytocin-like)
#
# Monitors multiple signals. Requires ALL to be below their ceilings for
# a sustained period. Counter does a HARD RESET when any signal exceeds
# its ceiling (safety perception is fragile -- any disruption breaks it).
# "You're safe -- consolidate learning, keep new neurons."
#
# Config keys:
#   watch_signals -- list of (signal_name, ceiling) pairs
#   onset_ticks   -- calm counter must reach this before rising (300)
#   rise_rate     -- per-tick level increase when calm (0.003)
#   decay         -- per-tick multiplicative decay (0.998)
# =====================================================================

def _trigger_calm_familiar(sig, brain, ctx):
    cfg = sig.config['trigger']
    signals = ctx['signals']

    # Check if ALL watched signals are below their ceilings
    all_calm = True
    for watch_name, ceiling in cfg['watch_signals']:
        level = signals[watch_name].level if watch_name in signals else 0.0
        if level > ceiling:
            all_calm = False
            break

    # Hard reset on any disruption, accumulate during calm
    counter = sig.state.get('calm_counter', 0.0)
    if all_calm:
        counter += 1.0
    else:
        counter = 0.0
    sig.state['calm_counter'] = counter

    # Rise when calm long enough
    if counter >= cfg['onset_ticks']:
        sig.level = min(1.0, sig.level + cfg['rise_rate'])

    # Always decay
    sig.level *= cfg['decay']

    if sig.level < 1e-6:
        sig.level = 0.0


# =====================================================================
# Trigger registry -- extend by adding entries here
# =====================================================================

_TRIGGERS = {
    'prediction_error': _trigger_prediction_error,
    'population_stat': _trigger_population_stat,
    'input_delta': _trigger_input_delta,
    'allostatic_load': _trigger_allostatic_load,
    'calm_familiar': _trigger_calm_familiar,
}
