"""
wiring.py -- Connect new neurons to nearby existing neurons.

Sparse initial wiring: each new neuron gets outgoing connections
(density chance per neighbor in radius) and ~30% return connections.
After wiring, rebuilds all synapse structures.
"""
import numpy as np


def wire_new_neurons(brain, neuron_indices, radius=50.0, density=0.05,
                     syn_type='plastic', weight=0.5, rng=None):
    """Connect new neurons to nearby existing neurons. Returns synapse count.

    Each new neuron connects outward (density chance per neighbor in radius)
    and gets ~30% return connections. Inhibitory neurons get negative weights.
    After wiring, rebuilds all synapse structures.
    """
    if not neuron_indices:
        return 0
    if rng is None:
        rng = np.random.RandomState()

    from engine.paths import plastic as plastic_module

    pos = np.array([[n.get('pos_x', 0), n.get('pos_y', 0), n.get('pos_z', 0)]
                     for n in brain.neurons])

    existing = set()
    for s in brain.synapses:
        existing.add((s['source'], s['target']))

    max_syn_id = max((s['id'] for s in brain.synapses), default=0)
    new_syns = []

    for ni in neuron_indices:
        ni_pos = pos[ni]
        dists = np.linalg.norm(pos - ni_pos, axis=1)
        nearby = np.flatnonzero((dists < radius) & (dists > 0))

        for tgt in nearby:
            tgt = int(tgt)
            if rng.random() > density:
                continue

            # Outgoing: new neuron -> neighbor
            if (ni, tgt) not in existing:
                max_syn_id += 1
                is_inh = brain.neurons[ni]['type'] in ('FS', 'LTS')
                w = -abs(weight) if is_inh else abs(weight)
                syn = {
                    'id': max_syn_id, 'source': ni, 'target': tgt,
                    'source_db_id': brain.neurons[ni]['id'],
                    'target_db_id': brain.neurons[tgt]['id'],
                    'type': syn_type, 'weight': w, 'delay': 1,
                    'learning_rate': 0.01,
                    'w_min': -10.0 if is_inh else 0.0,
                    'w_max': 0.0 if is_inh else 10.0,
                }
                syn.update(dict(plastic_module.DEFAULTS))
                syn['weight'] = w
                syn['w_min'] = -10.0 if is_inh else 0.0
                syn['w_max'] = 0.0 if is_inh else 10.0
                syn.update(dict(plastic_module.INITIAL_STATE))
                new_syns.append(syn)
                existing.add((ni, tgt))

            # Return: neighbor -> new neuron (30% chance)
            if rng.random() < 0.3 and (tgt, ni) not in existing:
                max_syn_id += 1
                is_inh = brain.neurons[tgt]['type'] in ('FS', 'LTS')
                w = -abs(weight) if is_inh else abs(weight)
                syn = {
                    'id': max_syn_id, 'source': tgt, 'target': ni,
                    'source_db_id': brain.neurons[tgt]['id'],
                    'target_db_id': brain.neurons[ni]['id'],
                    'type': syn_type, 'weight': w, 'delay': 1,
                    'learning_rate': 0.01,
                    'w_min': -10.0 if is_inh else 0.0,
                    'w_max': 0.0 if is_inh else 10.0,
                }
                syn.update(dict(plastic_module.DEFAULTS))
                syn['weight'] = w
                syn['w_min'] = -10.0 if is_inh else 0.0
                syn['w_max'] = 0.0 if is_inh else 10.0
                syn.update(dict(plastic_module.INITIAL_STATE))
                new_syns.append(syn)
                existing.add((tgt, ni))

    if new_syns:
        brain.synapses.extend(new_syns)
        brain.data['synapses'] = brain.synapses
        brain._build_synapse_structures()

    return len(new_syns)
