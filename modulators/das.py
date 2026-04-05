"""
Neuromodulatory system -- built on the general Signal architecture.

D = Dopamine (surprise / prediction error)
    "Something unexpected -- update your model."

A = Adrenaline/norepinephrine (arousal / gain)
    "Input changed -- wake up, increase gain."

S = Serotonin (stability / learning rate)
    "Population unstable -- slow learning until things settle."

C = Cortisol (sustained stress / growth suppression)
    "Sustained threat -- stop growing, survive."

O = Oxytocin (safety / bonding / learning boost)
    "You're safe -- consolidate learning, keep new neurons."

All five are Signal instances. Same external API: init_signals, update_signals,
deliver_reward. runner.py reads brain attributes set by signal triggers.
"""
import numpy as np
from modulators.signal import SignalSystem
from modulators.constants import (
    SURPRISE_EMA_ALPHA, SURPRISE_THRESHOLD, SURPRISE_REWARD_SCALE,
    STABILITY_TARGET_VAR, STABILITY_VAR_EMA_ALPHA, STABILITY_SENSITIVITY,
    STABILITY_MIN_SCALE, STABILITY_EVAL_INTERVAL,
    AROUSAL_DECAY, AROUSAL_GAIN, AROUSAL_SPIKE_SCALE, AROUSAL_DELTA_THRESHOLD,
    AROUSAL_DELTA_FLOOR, AROUSAL_HABITUATION_UP, AROUSAL_HABITUATION_DOWN,
    NEUROMOD_GAIN,
    CORTISOL_EMA_ALPHA, CORTISOL_VAR_ALPHA, CORTISOL_Z_THRESHOLD,
    CORTISOL_ONSET_TICKS, CORTISOL_RISE_RATE,
    CORTISOL_DECAY, CORTISOL_COUNTER_DECAY, CORTISOL_PLACENTAL_BUFFER,
    OXYTOCIN_AROUSAL_CEILING, OXYTOCIN_SURPRISE_CEILING, OXYTOCIN_ONSET_TICKS,
    OXYTOCIN_RISE_RATE, OXYTOCIN_DECAY, OXYTOCIN_LR_BOOST,
)


# =====================================================================
# Signal configurations -- biology as data, not code
# =====================================================================

D_CONFIG = {
    'trigger': {
        'type': 'prediction_error',
        'ema_alpha': SURPRISE_EMA_ALPHA,       # 0.02
        'threshold': SURPRISE_THRESHOLD,        # 0.15
        'reward_scale': SURPRISE_REWARD_SCALE,  # 0.8
    },
}

S_CONFIG = {
    'initial_level': 1.0,
    'trigger': {
        'type': 'population_stat',
        'eval_interval': STABILITY_EVAL_INTERVAL,  # 100
        'ema_alpha': STABILITY_VAR_EMA_ALPHA,       # 0.05
        'sensitivity': STABILITY_SENSITIVITY,        # 2.0
        'min_scale': STABILITY_MIN_SCALE,            # 0.2
    },
}

A_CONFIG = {
    'trigger': {
        'type': 'input_delta',
        'delta_threshold': AROUSAL_DELTA_THRESHOLD,      # 2.0 (ratio, not absolute)
        'delta_floor': AROUSAL_DELTA_FLOOR,               # 0.05 (absolute minimum)
        'habituation_up': AROUSAL_HABITUATION_UP,         # 0.01 (fast to new input)
        'habituation_down': AROUSAL_HABITUATION_DOWN,     # 0.001 (slow decay in silence)
        'spike_scale': AROUSAL_SPIKE_SCALE,               # 0.4
        'decay': AROUSAL_DECAY,                            # 0.995
    },
}

CORTISOL_CONFIG = {
    'trigger': {
        'type': 'allostatic_load',
        'ema_alpha': CORTISOL_EMA_ALPHA,               # 0.005 baseline adaptation
        'var_alpha': CORTISOL_VAR_ALPHA,                # 0.005 variance adaptation
        'z_threshold': CORTISOL_Z_THRESHOLD,            # 2.0 std devs
        'onset_ticks': CORTISOL_ONSET_TICKS,            # 200
        'rise_rate': CORTISOL_RISE_RATE,                # 0.002
        'decay': CORTISOL_DECAY,                        # 0.9995
        'counter_decay': CORTISOL_COUNTER_DECAY,        # 0.95
        'buffer': CORTISOL_PLACENTAL_BUFFER,            # 0.15 (fetal)
    },
}

OXYTOCIN_CONFIG = {
    'trigger': {
        'type': 'calm_familiar',
        'watch_signals': [
            ('A', OXYTOCIN_AROUSAL_CEILING),     # A < 0.2
            ('D', OXYTOCIN_SURPRISE_CEILING),    # D < 0.3
        ],
        'onset_ticks': OXYTOCIN_ONSET_TICKS,     # 300
        'rise_rate': OXYTOCIN_RISE_RATE,          # 0.003
        'decay': OXYTOCIN_DECAY,                  # 0.998
    },
}


# =====================================================================
# API (same interface as before -- runner.py reads brain attributes)
# =====================================================================

def init_signals(brain):
    """Initialize all neuromodulatory signals on a Brain instance."""
    system = SignalSystem()
    # Order matters: D, S, A first (original DAS), then cortisol, oxytocin
    # (cortisol reads A; oxytocin reads A and D)
    system.add('D', D_CONFIG)
    system.add('S', S_CONFIG)
    system.add('A', A_CONFIG)
    system.add('cortisol', CORTISOL_CONFIG)
    system.add('oxytocin', OXYTOCIN_CONFIG)
    brain._signal_system = system

    # Brain attributes read by runner.py and external code
    brain.sensory_ema = 0.0
    brain.surprise = 0.0
    brain.surprise_threshold = SURPRISE_THRESHOLD
    brain._prev_sensory_energy = 0.0
    brain.learning_rate_scale = 1.0
    brain.stability_var_ema = None
    brain.stability_target_var = STABILITY_TARGET_VAR
    brain.arousal = 0.0
    brain.arousal_decay = AROUSAL_DECAY
    brain.arousal_gain = AROUSAL_GAIN
    brain.cortisol = 0.0
    brain.oxytocin = 0.0
    brain.oxytocin_lr_boost = 1.0


def _apply_signal_effects(brain):
    """Post-update hook: copy signal levels to brain attributes + compute effects."""
    system = brain._signal_system
    brain.cortisol = system['cortisol'].level
    brain.oxytocin = system['oxytocin'].level
    brain.oxytocin_lr_boost = 1.0 + OXYTOCIN_LR_BOOST * system['oxytocin'].level


def update_signals(brain, external_I, tick):
    """Update all signals from input statistics.

    Called once per tick from Brain.tick(), after spike detection.
    """
    brain._signal_system.update_all(brain, external_I, tick)
    _apply_signal_effects(brain)


def deliver_reward(brain, magnitude):
    """Deliver reward signal to reward_plastic synapses + neuromodulatory current.

    magnitude > 0: positive reward (potentiate eligible)
    magnitude < 0: negative reward (depress eligible)
    magnitude = 0: no effect

    Injects neuromodulatory current per neuron based on dopamine_sens:
    - D1-like (sens > 0): reward excites
    - D2-like (sens < 0): reward inhibits
    - Neutral (sens = 0): unaffected
    """
    if abs(magnitude) < 1e-6:
        return

    # Neuromodulatory current: per-neuron effect based on receptor type
    brain.neuromod_current += brain.dopamine_sens * magnitude * NEUROMOD_GAIN

    if not brain.has_reward:
        return

    # Vectorized reward delivery
    elig = brain.reward_elig
    active = np.abs(elig) > 1e-6
    if not np.any(active):
        return

    w = brain.reward_w_arr[active]
    lr = brain.reward_lr_arr[active] * brain.learning_rate_scale * getattr(brain, 'oxytocin_lr_boost', 1.0)
    wmin = brain.reward_wmin_arr[active]
    wmax = brain.reward_wmax_arr[active]
    rng = wmax - wmin
    e = elig[active]
    is_inh = brain.reward_is_inh[active]

    valid = rng > 1e-6
    if np.any(valid):
        if magnitude > 0:
            dw = np.where(is_inh[valid],
                          -lr[valid] * e[valid] * magnitude * (w[valid] - wmin[valid]) / rng[valid],
                          lr[valid] * e[valid] * magnitude * (wmax[valid] - w[valid]) / rng[valid])
        else:
            dw = np.where(is_inh[valid],
                          -lr[valid] * e[valid] * magnitude * (w[valid] - wmin[valid]) / rng[valid],
                          lr[valid] * e[valid] * magnitude * (w[valid] - wmin[valid]) / rng[valid])

        new_w = np.clip(w[valid] + dw, wmin[valid], wmax[valid])
        active_idx = np.flatnonzero(active)
        valid_idx = active_idx[valid]
        brain.reward_w_arr[valid_idx] = new_w

    # Partially consume eligibility
    brain.reward_elig[active] *= 0.5

    # Synaptic homeostasis (off by default)
    if not brain.reward_homeostasis:
        return
    homeo_rate = 0.1
    for tid, indices in brain.reward_target_groups.items():
        if len(indices) < 2:
            continue
        idx_arr = np.array(indices, dtype=np.intp)
        current_total = np.sum(brain.reward_w_arr[idx_arr])
        if current_total < 1e-6:
            continue
        target_total = brain.reward_target_w0[tid]
        ratio = target_total / current_total
        scale = 1.0 + homeo_rate * (ratio - 1.0)
        if abs(scale - 1.0) < 1e-8:
            continue
        new_w = np.clip(brain.reward_w_arr[idx_arr] * scale,
                        brain.reward_wmin_arr[idx_arr],
                        brain.reward_wmax_arr[idx_arr])
        brain.reward_w_arr[idx_arr] = new_w
