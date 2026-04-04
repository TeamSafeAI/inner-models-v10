"""
signal_probe.py -- Signal propagation probe. Stimulate neurons, watch the cascade.

Not a test. Not a score. A picture of what the brain does.

Outputs:
  - PNG snapshot showing which neurons fired and when
  - Text cascade report (which regions, how far, how fast)

Usage:
    py probes/signal_probe.py --brain brains/test_v10.db
    py probes/signal_probe.py --brain brains/test_v10.db --stim-region cortex
    py probes/signal_probe.py --brain brains/test_v10.db --compare-before-after
"""
import os, sys, argparse, json
import numpy as np
import functools

print = functools.partial(print, flush=True)

# Project root (parent of probes/)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from engine.loader import load
from engine.runner import Brain


REGION_CENTERS = {
    'cortex':        (250, 250, 400),
    'hippocampus':   (150, 400, 250),
    'amygdala':      (350, 400, 200),
    'basal_ganglia': (250, 300, 250),
    'thalamus':      (250, 250, 250),
    'brainstem':     (250, 150, 100),
    'sensory':       (250, 100, 350),
}


CODE_TO_REGION = {
    'BS': 'brainstem', 'SN': 'sensory', 'TH': 'thalamus',
    'AM': 'amygdala', 'HP': 'hippocampus', 'BG': 'basal_ganglia',
    'CX': 'cortex', 'SM': 'somatosensory',
}

def get_neuron_regions(neurons, n):
    """Get region for each neuron. Uses region tag if available, falls back to spatial."""
    # Try region tag first
    if neurons and neurons[0].get('region', ''):
        regions = {}
        for i in range(n):
            code = neurons[i].get('region', '')
            regions[i] = CODE_TO_REGION.get(code, code or 'unknown')
        return regions

    # Fallback: spatial lookup (old brains without tags)
    centers = np.array(list(REGION_CENTERS.values()), dtype=float)
    names = list(REGION_CENTERS.keys())
    positions = np.array([[nn['pos_x'], nn['pos_y'], nn['pos_z']] for nn in neurons])
    dists = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    regions = {}
    for i in range(n):
        regions[i] = names[nearest[i]]
    return regions


def warm_up(brain, tonic, ticks=500):
    """Run brain to reach steady-state spontaneous activity."""
    I_ext = np.full(brain.n, tonic)
    for _ in range(ticks):
        brain.tick(external_I=I_ext)
    return brain.tick_count


def probe_cascade(brain, stim_neurons, stim_strength, tonic, cascade_ticks,
                  pre_ticks=100):
    """Stimulate neurons and record the firing cascade.

    Uses baseline subtraction: only neurons that were NOT spontaneously
    active during the pre-stimulus period count as "recruited" by the stimulus.

    Returns:
        pre_active: set of neuron IDs that fired during baseline
        cascade: list of (tick_offset, fired_list) for each tick after stimulus
        stim_tick: the tick number when stimulus was applied
        baseline_rate: average spikes/tick during baseline
    """
    n = brain.n
    I_ext = np.full(n, tonic)

    # Record baseline activity (which neurons fire spontaneously?)
    pre_active = set()
    total_pre_spikes = 0
    for t in range(pre_ticks):
        fired = brain.tick(external_I=I_ext)
        for f in fired:
            pre_active.add(int(f))
        total_pre_spikes += len(fired)

    baseline_rate = total_pre_spikes / max(pre_ticks, 1)
    stim_tick = brain.tick_count

    # Apply stimulus
    I_stim = I_ext.copy()
    I_stim[stim_neurons] += stim_strength

    cascade = []
    # Strong stimulus for 5 ticks, then observe
    for t in range(cascade_ticks):
        if t < 5:
            fired = brain.tick(external_I=I_stim)
        else:
            fired = brain.tick(external_I=I_ext)
        fired_list = [int(f) for f in fired]
        cascade.append((t, fired_list))

    return pre_active, cascade, stim_tick, baseline_rate


def cascade_report(neurons, regions, stim_neurons, pre_active, cascade,
                   baseline_rate):
    """Print text report of cascade propagation.

    Only counts neurons NOT in pre_active (stimulus-evoked, not spontaneous).
    """
    n = len(neurons)
    stim_set = set(stim_neurons)

    # First fire time per neuron — ONLY neurons not active during baseline
    first_fire = {}
    for t, fired_list in cascade:
        for nid in fired_list:
            if nid not in first_fire and nid not in stim_set and nid not in pre_active:
                first_fire[nid] = t

    # How many neurons recruited over time
    recruited_by_tick = np.zeros(len(cascade), dtype=int)
    total_recruited = 0
    for t, fired_list in cascade:
        new = sum(1 for nid in fired_list if first_fire.get(nid) == t and nid not in stim_set)
        total_recruited += new
        recruited_by_tick[t] = total_recruited

    # Region breakdown
    region_first_fire = {}
    for nid, t in first_fire.items():
        r = regions.get(nid, 'unknown')
        if r not in region_first_fire or t < region_first_fire[r]:
            region_first_fire[r] = t

    # Fire counts by region over cascade
    region_total = {}
    for t, fired_list in cascade:
        for nid in fired_list:
            r = regions.get(nid, 'unknown')
            region_total[r] = region_total.get(r, 0) + 1

    # Type breakdown
    type_recruited = {}
    for nid, t in first_fire.items():
        nt = neurons[nid]['type']
        type_recruited[nt] = type_recruited.get(nt, 0) + 1

    # Propagation depth (max hops from stim, estimated by first-fire time)
    max_depth = max(first_fire.values()) if first_fire else 0

    print(f"\n  CASCADE REPORT")
    print(f"  {'='*50}")
    print(f"  Stimulus: {len(stim_neurons)} neurons, types: {set(neurons[i]['type'] for i in stim_neurons)}")
    stim_regions = set(regions.get(i, '?') for i in stim_neurons)
    print(f"  Stim regions: {stim_regions}")
    print(f"  Baseline rate: {baseline_rate:.1f} spikes/tick")
    print(f"  Total recruited: {len(first_fire)} / {n} neurons ({len(first_fire)/n*100:.1f}%)")
    print(f"  Max propagation depth: {max_depth} ticks")

    print(f"\n  REGION ARRIVAL (first spike after stimulus):")
    for r, t in sorted(region_first_fire.items(), key=lambda x: x[1]):
        total = region_total.get(r, 0)
        print(f"    {r:>20s}: t+{t:>3d}  ({total} total spikes)")

    print(f"\n  TYPE RECRUITMENT:")
    for nt, count in sorted(type_recruited.items(), key=lambda x: -x[1]):
        print(f"    {nt:>4s}: {count} neurons recruited")

    # Propagation timeline (every 10 ticks)
    print(f"\n  PROPAGATION TIMELINE:")
    print(f"  {'Tick':>6s} | {'Fired':>6s} | {'New':>5s} | {'Total Recruited':>15s}")
    print(f"  {'-'*40}")
    prev = 0
    for t in range(0, len(cascade), max(1, len(cascade) // 20)):
        _, fired = cascade[t]
        new = recruited_by_tick[t] - prev if t > 0 else recruited_by_tick[t]
        prev = recruited_by_tick[t]
        print(f"  {t:>6d} | {len(fired):>6d} | {new:>5d} | {recruited_by_tick[t]:>15d}")

    return {
        'total_recruited': len(first_fire),
        'max_depth': max_depth,
        'baseline_rate': baseline_rate,
        'region_arrival': region_first_fire,
        'first_fire': first_fire,
    }


def save_snapshot_png(neurons, regions, stim_neurons, cascade_data, out_path,
                      title="Signal Propagation"):
    """Save a 2D projection snapshot as PNG.

    Neurons colored by first-fire time (propagation wavefront).
    Stimulus neurons in white. Unfired neurons dim gray.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
    except ImportError:
        print("  matplotlib not available, skipping PNG output")
        return

    first_fire = cascade_data['first_fire']
    max_depth = cascade_data['max_depth']
    stim_set = set(stim_neurons)

    n = len(neurons)
    x = np.array([neurons[i]['pos_x'] for i in range(n)])
    y = np.array([neurons[i]['pos_y'] for i in range(n)])

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    # Layer 1: all neurons (dim)
    ax.scatter(x, y, s=2, c='#1a1a2e', alpha=0.5, zorder=1)

    # Layer 2: fired neurons (colored by arrival time)
    if first_fire:
        fired_ids = list(first_fire.keys())
        fired_x = x[fired_ids]
        fired_y = y[fired_ids]
        fired_times = np.array([first_fire[i] for i in fired_ids])

        norm = Normalize(vmin=0, vmax=max(max_depth, 1))
        colors = cm.plasma(norm(fired_times))
        sizes = np.clip(20 - fired_times * 0.1, 4, 20)

        ax.scatter(fired_x, fired_y, s=sizes, c=colors, alpha=0.8, zorder=2)

    # Layer 3: stim neurons (bright white)
    stim_list = list(stim_set)
    ax.scatter(x[stim_list], y[stim_list], s=40, c='white', marker='*',
               edgecolors='cyan', linewidths=0.5, zorder=3)

    # Colorbar
    if first_fire:
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Ticks after stimulus', color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Region labels
    region_centers = {}
    for i in range(n):
        r = regions.get(i, 'unknown')
        if r not in region_centers:
            region_centers[r] = {'x': [], 'y': []}
        region_centers[r]['x'].append(x[i])
        region_centers[r]['y'].append(y[i])
    for r, coords in region_centers.items():
        cx = np.mean(coords['x'])
        cy = np.mean(coords['y'])
        ax.text(cx, cy, r, color='white', fontsize=7, alpha=0.4,
                ha='center', va='center', zorder=0)

    recruited = cascade_data['total_recruited']
    pct = recruited / n * 100
    ax.set_title(f"{title}\n{recruited}/{n} neurons reached ({pct:.1f}%), "
                 f"depth={max_depth} ticks",
                 color='white', fontsize=12)
    ax.set_xlabel('X', color='gray')
    ax.set_ylabel('Y', color='gray')
    ax.tick_params(colors='gray')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Snapshot saved: {out_path}")


def select_stim_neurons(neurons, regions, stim_region=None, stim_neuron=None,
                        stim_count=10, rng=None):
    """Select neurons to stimulate."""
    n = len(neurons)

    if stim_neuron is not None:
        return [stim_neuron]

    if stim_region:
        candidates = [i for i in range(n) if regions.get(i, '') == stim_region]
        if not candidates:
            print(f"  Warning: no neurons in region '{stim_region}', using random")
            candidates = list(range(n))
    else:
        # Default: highest out-degree neurons (hubs)
        out_deg = np.zeros(n, dtype=int)
        # We don't have synapse data here, so just pick random
        candidates = list(range(n))

    count = min(stim_count, len(candidates))
    return list(rng.choice(candidates, count, replace=False))


def main():
    p = argparse.ArgumentParser(description='Brain signal propagation probe')
    p.add_argument('--brain', required=True, help='Path to brain DB')
    p.add_argument('--tonic', type=float, default=2.8, help='Tonic drive')
    p.add_argument('--warmup', type=int, default=500, help='Warmup ticks')
    p.add_argument('--cascade-ticks', type=int, default=200, help='Ticks to observe after stimulus')
    p.add_argument('--stim-strength', type=float, default=20.0, help='Stimulus current (mA)')
    p.add_argument('--stim-region', default=None, help='Region to stimulate')
    p.add_argument('--stim-neuron', type=int, default=None, help='Specific neuron to stimulate')
    p.add_argument('--stim-count', type=int, default=10, help='Number of neurons to stimulate')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--output', default=None, help='Output PNG path')
    p.add_argument('--compare', action='store_true',
                   help='Run two probes: before and after 10K ticks of development')
    p.add_argument('--develop-ticks', type=int, default=10000,
                   help='Development ticks between compare snapshots')
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)

    print(f"  Loading brain: {args.brain}")
    data = load(args.brain)
    brain = Brain(data, learn=True)
    neurons = data['neurons']
    n = brain.n
    regions = get_neuron_regions(neurons, n)

    region_counts = {}
    for r in regions.values():
        region_counts[r] = region_counts.get(r, 0) + 1
    print(f"  {n} neurons, {len(data['synapses'])} synapses")
    print(f"  Regions: {dict(sorted(region_counts.items(), key=lambda x: -x[1]))}")

    # Select stim targets
    stim = select_stim_neurons(neurons, regions, args.stim_region,
                               args.stim_neuron, args.stim_count, rng)
    stim_regions = set(regions.get(i, '?') for i in stim)
    print(f"  Stimulating {len(stim)} neurons in {stim_regions}")

    # Warmup
    print(f"  Warming up ({args.warmup} ticks)...")
    warm_up(brain, args.tonic, args.warmup)

    # Probe
    print(f"  Probing cascade ({args.cascade_ticks} ticks)...")
    pre_active, cascade, stim_tick, baseline_rate = probe_cascade(
        brain, stim, args.stim_strength, args.tonic, args.cascade_ticks)
    print(f"  Baseline: {len(pre_active)} spontaneously active neurons, {baseline_rate:.1f} spikes/tick")

    # Report (only stimulus-evoked activity, baseline subtracted)
    cascade_data = cascade_report(neurons, regions, stim, pre_active, cascade,
                                  baseline_rate)

    # Save PNG
    out_dir = os.path.join(BASE, 'snapshots')
    os.makedirs(out_dir, exist_ok=True)
    brain_name = os.path.splitext(os.path.basename(args.brain))[0]

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(out_dir, f"probe_{brain_name}_s{args.seed}.png")

    save_snapshot_png(neurons, regions, stim, cascade_data, out_path,
                      title=f"Probe: {brain_name}")

    # Compare mode: probe again after development
    if args.compare:
        print(f"\n  Running {args.develop_ticks} ticks of development...")
        I_ext = np.full(n, args.tonic)
        for t in range(args.develop_ticks):
            brain.tick(external_I=I_ext)
            if t % 2000 == 0 and t > 0:
                print(f"    t={t}, tick={brain.tick_count}")

        print(f"  Probing cascade AFTER development...")
        pre_active2, cascade2, stim_tick2, baseline_rate2 = probe_cascade(
            brain, stim, args.stim_strength, args.tonic, args.cascade_ticks)

        cascade_data2 = cascade_report(neurons, regions, stim, pre_active2, cascade2,
                                       baseline_rate2)

        out_path2 = out_path.replace('.png', '_after.png')
        out_path_before = out_path.replace('.png', '_before.png')
        # Rename first snapshot
        if os.path.exists(out_path):
            os.rename(out_path, out_path_before)
        save_snapshot_png(neurons, regions, stim, cascade_data, out_path_before,
                          title=f"BEFORE: {brain_name}")
        save_snapshot_png(neurons, regions, stim, cascade_data2, out_path2,
                          title=f"AFTER {args.develop_ticks} ticks: {brain_name}")

        # Summary comparison
        r1 = cascade_data['total_recruited']
        r2 = cascade_data2['total_recruited']
        d1 = cascade_data['max_depth']
        d2 = cascade_data2['max_depth']
        print(f"\n  COMPARISON:")
        print(f"    Before: {r1} recruited, depth {d1}")
        print(f"    After:  {r2} recruited, depth {d2}")
        print(f"    Change: {r2-r1:+d} recruited, {d2-d1:+d} depth")


if __name__ == '__main__':
    main()
