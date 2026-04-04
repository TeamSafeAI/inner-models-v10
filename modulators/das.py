"""
DAS modulators -- the three chemical gradients that gate growth and learning.

D = Dopamine (surprise / prediction error)
    "Something unexpected -- update your model."

A = Adrenaline/norepinephrine (arousal / gain)
    "Input changed -- wake up, increase gain."

S = Serotonin (stability / learning rate)
    "Population unstable -- slow learning until things settle."

All fire on sensory statistics. When input is flat tonic, all signals
stay neutral (backward compatible with pre-DAS brains).
"""
import numpy as np
from modulators.constants import (
    SURPRISE_EMA_ALPHA, SURPRISE_THRESHOLD, SURPRISE_REWARD_SCALE,
    STABILITY_TARGET_VAR, STABILITY_SENSITIVITY, STABILITY_MIN_SCALE,
    STABILITY_EVAL_INTERVAL,
    AROUSAL_DECAY, AROUSAL_GAIN, AROUSAL_SPIKE_SCALE, AROUSAL_DELTA_THRESHOLD,
    NEUROMOD_DECAY, NEUROMOD_GAIN,
)


def init_signals(brain):
    """Initialize DAS signal state on a Brain instance."""
    # Surprise (dopamine)
    brain.sensory_ema = 0.0
    brain.surprise_threshold = SURPRISE_THRESHOLD
    brain._prev_sensory_energy = 0.0

    # Stability (serotonin)
    brain.learning_rate_scale = 1.0
    brain.stability_target_var = STABILITY_TARGET_VAR

    # Arousal (norepinephrine)
    brain.arousal = 0.0
    brain.arousal_decay = AROUSAL_DECAY
    brain.arousal_gain = AROUSAL_GAIN


def update_signals(brain, external_I, tick):
    """Update all three DAS signals from input statistics.

    Called once per tick from Brain.tick(), after spike detection.
    Modifies brain state in place.
    """
    # Compute sensory energy from external input
    if external_I is not None:
        sensory_energy = float(np.var(external_I))
    else:
        sensory_energy = 0.0

    # --- D: Surprise (dopamine) ---
    # Prediction error on input variance
    brain.sensory_ema = (1 - SURPRISE_EMA_ALPHA) * brain.sensory_ema + \
                        SURPRISE_EMA_ALPHA * sensory_energy
    if brain.sensory_ema > 1e-6:
        surprise = abs(sensory_energy - brain.sensory_ema) / brain.sensory_ema
        if surprise > brain.surprise_threshold:
            brain.deliver_reward(min(1.0, SURPRISE_REWARD_SCALE * surprise))

    # --- S: Stability (serotonin) ---
    # Global learning rate modulation based on population variance
    if tick > 0 and tick % STABILITY_EVAL_INTERVAL == 0:
        pop_var = float(np.var(brain.activity_trace))
        brain.learning_rate_scale = max(STABILITY_MIN_SCALE, min(1.0,
            1.0 - (pop_var - brain.stability_target_var) * STABILITY_SENSITIVITY))

    # --- A: Arousal (norepinephrine) ---
    # Sudden input change detection
    delta = abs(sensory_energy - brain._prev_sensory_energy)
    if delta > AROUSAL_DELTA_THRESHOLD:
        brain.arousal = min(1.0, brain.arousal + AROUSAL_SPIKE_SCALE * delta)
    brain.arousal *= brain.arousal_decay
    brain._prev_sensory_energy = sensory_energy


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
    lr = brain.reward_lr_arr[active] * brain.learning_rate_scale
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
