"""
neurogenesis.py -- Birth new neurons at runtime.

Extends ALL 14 numpy arrays in the Brain. Neurons appear near a spatial
center with Gaussian jitter. Type follows cortical distribution (62/38 E/I).
"""
import numpy as np

# Izhikevich params per type (from engine/neurons/*.py)
NEURON_PARAMS = {
    'RS':  {'a': 0.02, 'b': 0.2,  'c': -65.0, 'd': 8.0, 'inh': False},
    'FS':  {'a': 0.1,  'b': 0.2,  'c': -65.0, 'd': 2.0, 'inh': True},
    'IB':  {'a': 0.02, 'b': 0.2,  'c': -55.0, 'd': 4.0, 'inh': False},
    'CH':  {'a': 0.02, 'b': 0.2,  'c': -50.0, 'd': 2.0, 'inh': False},
    'LTS': {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.0, 'inh': True},
}

# Cortical type distribution (62% excitatory, 38% inhibitory)
CORTICAL_TYPE_WEIGHTS = ['RS']*55 + ['IB']*10 + ['CH']*5 + ['FS']*20 + ['LTS']*10


def birth_neurons(brain, count, pos_center=None, pos_spread=30.0, rng=None):
    """Birth new neurons. Extends ALL numpy arrays. Returns new indices.

    Neurons appear near pos_center with Gaussian jitter. Type follows
    cortical distribution (62% excitatory, 38% inhibitory).
    """
    if count <= 0:
        return []
    if rng is None:
        rng = np.random.RandomState()

    # Default position: center of existing brain
    if pos_center is None:
        pos_center = np.array([
            np.mean([n.get('pos_x', 0) for n in brain.neurons]),
            np.mean([n.get('pos_y', 0) for n in brain.neurons]),
            np.mean([n.get('pos_z', 0) for n in brain.neurons]),
        ])

    old_n = brain.n
    max_id = max((n['id'] for n in brain.neurons), default=0)
    chosen_types = [CORTICAL_TYPE_WEIGHTS[rng.randint(100)] for _ in range(count)]

    new_a, new_b, new_c, new_d = [], [], [], []
    new_dsens = []

    for i, ntype in enumerate(chosen_types):
        p = NEURON_PARAMS[ntype]
        pos = pos_center + rng.randn(3) * pos_spread
        max_id += 1

        # D1/D2 dopamine receptor assignment
        if p['inh']:
            dsens = rng.choice([-0.5, -0.3, 0.0], p=[0.5, 0.3, 0.2])
        else:
            dsens = rng.choice([0.5, 0.3, -0.3, 0.0], p=[0.3, 0.3, 0.2, 0.2])

        brain.neurons.append({
            'id': max_id, 'type': ntype,
            'v': -65.0, 'u': p['b'] * (-65.0),
            'a': p['a'], 'b': p['b'], 'c': p['c'], 'd': p['d'],
            'last_spike': -1000,
            'dopamine_sens': dsens, 'excitability': 0.0, 'activity_trace': 0.0,
            'pos_x': float(pos[0]), 'pos_y': float(pos[1]), 'pos_z': float(pos[2]),
        })
        new_a.append(p['a']); new_b.append(p['b'])
        new_c.append(p['c']); new_d.append(p['d'])
        new_dsens.append(dsens)

    # --- Extend every per-neuron numpy array ---
    brain.v = np.concatenate([brain.v, np.full(count, -65.0)])
    brain.u = np.concatenate([brain.u, np.array([NEURON_PARAMS[t]['b'] * (-65.0) for t in chosen_types])])
    brain.a = np.concatenate([brain.a, np.array(new_a)])
    brain.b = np.concatenate([brain.b, np.array(new_b)])
    brain.c = np.concatenate([brain.c, np.array(new_c)])
    brain.d = np.concatenate([brain.d, np.array(new_d)])
    brain.last_spike = np.concatenate([brain.last_spike, np.full(count, -1000, dtype=np.int64)])
    brain.dopamine_sens = np.concatenate([brain.dopamine_sens, np.array(new_dsens)])
    brain.excitability = np.concatenate([brain.excitability, np.zeros(count)])
    brain.activity_trace = np.concatenate([brain.activity_trace, np.zeros(count)])
    brain.neuromod_current = np.concatenate([brain.neuromod_current, np.zeros(count)])

    # Spike delay buffer (buf_size x n -> buf_size x new_n)
    brain.spike_buf = np.concatenate([
        brain.spike_buf, np.zeros((brain.buf_size, count))], axis=1)

    # Modulator ring buffer (if gated synapses exist)
    if brain.has_gated:
        brain.mod_ring = np.concatenate([
            brain.mod_ring, np.zeros((brain.mod_window, count), dtype=np.int8)], axis=1)
        brain.mod_counts = np.concatenate([brain.mod_counts, np.zeros(count, dtype=np.int32)])

    brain.n = len(brain.neurons)
    brain.data['neurons'] = brain.neurons
    return list(range(old_n, brain.n))
