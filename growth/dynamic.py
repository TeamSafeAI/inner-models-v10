"""
dynamic.py -- Emergent growth controller. No rate parameter.

Growth is bursty and self-limiting, matching fetal development:
- Neurogenesis requires BOTH arousal AND surprise (non-linear gate)
- Cortisol suppresses growth under sustained stress
- New neurons get sparse wiring (5-10 connections, weak)
- Survival window modulated by cortisol (harsh) and oxytocin (generous)
- As brain stabilizes on familiar input, growth naturally tapers

Call between frame batches (not every tick).
"""
import math
import numpy as np
from growth.neurogenesis import birth_neurons
from growth.wiring import wire_new_neurons
from modulators.constants import (
    CORTISOL_GROWTH_SUPPRESSION, CORTISOL_SURVIVAL_REDUCTION,
    OXYTOCIN_SURVIVAL_BOOST,
)


def dynamic_growth(brain, rng=None, post_birth_fn=None):
    """Emergent growth. No rate parameter -- signals decide everything.

    post_birth_fn(brain, new_indices): optional callback after birth to set
    regional excitability or other properties on new neurons.

    Returns dict with stats: neurons_born, synapses_added, neurons_culled.
    """
    if rng is None:
        rng = np.random.RandomState()

    stats = {'neurons_born': 0, 'synapses_added': 0, 'neurons_culled': 0}

    # Read neuromodulator state (backward compatible with old brains)
    cortisol = getattr(brain, 'cortisol', 0.0)
    oxytocin = getattr(brain, 'oxytocin', 0.0)

    # Compute surprise level (same metric the signal uses)
    surprise_level = 0.0
    if brain.sensory_ema > 1e-6 and brain._prev_sensory_energy > 0:
        surprise_level = abs(brain._prev_sensory_energy - brain.sensory_ema) / brain.sensory_ema

    # 1. NEUROGENESIS: requires both arousal AND surprise (non-linear)
    #    Product gate: arousal * surprise gives bursty, input-dependent growth.
    #    Cortisol suppresses: sustained stress -> stop growing.
    growth_signal = brain.arousal * surprise_level
    if cortisol > 0.01:
        growth_signal *= (1.0 - CORTISOL_GROWTH_SUPPRESSION * cortisol)
    if growth_signal > 0.05:
        n_birth = max(1, int(3.0 * math.sqrt(growth_signal)))

        # Birth near the most active neurons (sensory/hippocampal stream)
        top = np.argsort(brain.activity_trace)[-min(50, brain.n):]
        center = np.array([
            np.mean([brain.neurons[i].get('pos_x', 0) for i in top]),
            np.mean([brain.neurons[i].get('pos_y', 0) for i in top]),
            np.mean([brain.neurons[i].get('pos_z', 0) for i in top]),
        ])

        new_idx = birth_neurons(brain, n_birth, pos_center=center,
                                pos_spread=20.0, rng=rng)
        stats['neurons_born'] = len(new_idx)

        # Sparse initial wiring: 5-10 weak connections per new neuron
        if new_idx:
            n_syn = wire_new_neurons(
                brain, new_idx, radius=40.0, density=0.008,
                syn_type='plastic', weight=0.3, rng=rng)
            stats['synapses_added'] = n_syn

            # Tag birth tick for survival tracking
            for ni in new_idx:
                brain.neurons[ni]['birth_tick'] = brain.tick_count

            # Let caller assign regional excitability etc.
            if post_birth_fn is not None:
                post_birth_fn(brain, new_idx)

    # 2. SYNAPTOGENESIS on existing neurons: when brain is unstable
    if brain.learning_rate_scale < 0.7 and surprise_level > 0.1:
        result = brain.sprout(max_new=10, window=500, min_cofire=2,
                              weight=0.2, max_distance=12.0)
        stats['synapses_added'] += result.get('sprouted', 0)

    # 3. APOPTOSIS: cull neurons that failed to integrate
    #    Cortisol shortens window (stress kills weak neurons faster)
    #    Oxytocin extends window (safety gives newborns more time)
    survival_window = (500
                       + int(OXYTOCIN_SURVIVAL_BOOST * oxytocin)
                       - int(CORTISOL_SURVIVAL_REDUCTION * cortisol))
    survival_window = max(200, survival_window)
    min_activity = 0.005
    culled = 0
    for i in range(brain.n - 1, -1, -1):
        bt = brain.neurons[i].get('birth_tick', -99999)
        age = brain.tick_count - bt
        if 0 < age < survival_window:
            continue
        if age >= survival_window and age < survival_window + 100:
            if brain.activity_trace[i] < min_activity:
                for syn in brain.synapses:
                    if syn['source'] == i or syn['target'] == i:
                        syn['weight'] = 0.0
                brain.excitability[i] = -brain.ip_clamp
                culled += 1

    if culled > 0:
        stats['neurons_culled'] = culled
        brain.sync_state()
        brain._build_synapse_structures()

    return stats
