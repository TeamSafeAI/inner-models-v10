# inner-models-v10

Growing neural brain with DAS modulators, runtime neurogenesis, and the Harness.

8000+ Izhikevich neurons. 8 synapse types. 3 autonomous chemical signals.
Neurons are born, wired, and culled at runtime based on sensory input --
no rate sliders, no parameter tuning. The brain decides when to grow.

## Quick Start

```bash
# 1. Install dependencies
pip install numpy websockets

# 2. Grow a brain (creates brains/regional_cortex_heavy_s42.db, ~5MB, ~60 seconds)
py grow_regional.py --config cortex_heavy --neurons 8000 --metabolic-cost 0.02 --prune 0.15

# 3. Run the harness
py harness/server.py --brain brains/regional_cortex_heavy_s42.db

# 4. Open browser
#    http://localhost:8890/harness/viewer.html
```

Drag audio files onto the viewer to play them through the brain.
Toggle **Womb** mode for 500Hz low-pass filtering (amniotic fluid simulation).

## Structure

```
engine/               Core simulation (vectorized NumPy, Izhikevich, CSR synapses)
  runner.py           Brain class -- tick(), orchestrates everything
  neurons/            5 types: RS, FS, IB, CH, LTS
  paths/              8 synapse types: fixed, plastic, gated, reward_plastic,
                      facilitating, depressing, developmental, gap_junction
  loader.py           Load/save brain SQLite DBs
  recorder.py         Spike recording
  encoder.py          Signal encoding (audio/tone -> neuron currents)

growth/               Everything that changes brain size at runtime
  neurogenesis.py     birth_neurons() -- extend all 14 numpy arrays
  wiring.py           wire_new_neurons() -- sparse connections to neighbors
  dynamic.py          dynamic_growth() -- emergent controller (DAS-gated)

modulators/           DAS system (three chemical gradients)
  das.py              D = Dopamine/surprise, A = Arousal/norepinephrine,
                      S = Serotonin/stability
  constants.py        Thresholds, decay rates, biological mappings

harness/              The thing you run
  server.py           WebSocket server -- ticks brain, streams state to viewer
  viewer.html         3D brain viewer + audio playback + womb mode + signals
  audio/
    generate_womb.py  Generate TTS voice lines (needs OpenAI API key)

probes/               Observation tools
  signal_probe.py     Stimulate neurons, watch cascade, save PNG snapshots

blocks/               32 building blocks (JSON recipes)
brain_generator.py    Compose blocks into brain DBs
grow_regional.py      Grow 7-region brain with metabolic cost + pruning
analyze_brain.py      Network statistics (degree distribution, rich club, etc.)
schema.py             SQLite schema for brain DBs
media/                Audio files (womb_phase/ voice lines, classical music)
```

## Growing a Brain

Brain DBs are not included (too large for git). Generate one:

```bash
# Default balanced brain (8K neurons)
py grow_regional.py

# Cortex-heavy with metabolic pruning (best for harness)
py grow_regional.py --config cortex_heavy --metabolic-cost 0.02 --prune 0.15

# Options
py grow_regional.py --neurons 5000 --seed 77 --config memory_dense
py grow_regional.py --neurons 15000 --config thalamic_hub --synapse-mode reward
```

Configs: `balanced`, `cortex_heavy`, `memory_dense`, `thalamic_hub`, `spread`, `amygdala_driven`

## The Harness

The harness is the brain's world. It provides input (audio, webcam, touch),
displays output (3D activity, motor readout, EEG traces), and shows the
three DAS modulator levels in real time.

```bash
py harness/server.py --brain brains/YOUR_BRAIN.db
py harness/server.py --brain brains/YOUR_BRAIN.db --tonic 3.0 --growth
```

Flags:
- `--tonic N` -- baseline current (default 2.8, higher = more active)
- `--speed N` -- simulation speed multiplier
- `--growth` -- enable runtime neurogenesis (neurons born/culled during run)
- `--port N` -- WebSocket port (default 8891, HTTP on port-1)

### Audio Input
- Drag MP3/WAV/OGG files onto the viewer
- Click tracks to play, auto-advances through playlist
- 16-band FFT spectrum fed to sensory neurons
- Volume slider: brain gets full signal, speakers attenuated independently

### Womb Mode
Toggle enables 500Hz low-pass filter on all audio (simulates sound heard
through amniotic fluid). For prenatal learning experiments:
1. Generate voice lines: `py harness/audio/generate_womb.py --key YOUR_OPENAI_KEY`
2. Drop the `media/womb_phase/` folder onto the viewer
3. Enable Womb mode
4. Run with `--growth` to watch the brain develop

## DAS Modulators

Three autonomous signals that fire on sensory statistics. When input is
flat tonic, all signals stay neutral (backward compatible).

| Signal | Chemical | What it detects | What it does |
|--------|----------|----------------|-------------|
| **D** Surprise | Dopamine | Prediction error on input variance | Triggers reward delivery, updates model |
| **A** Arousal | Norepinephrine | Sudden input change | Global tonic boost, increases gain |
| **S** Stability | Serotonin | Population firing variance | Scales learning rate down when unstable |

## Runtime Growth

When `--growth` is enabled, `dynamic_growth()` runs between frame batches:

1. **Neurogenesis**: Product gate (arousal * surprise). Both must be elevated.
   New neurons appear near the most active region with sparse wiring.
2. **Synaptogenesis**: When brain is unstable + surprised, sprout connections
   between co-active nearby neurons.
3. **Apoptosis**: Neurons that fail to integrate within 500 ticks are culled
   (weights zeroed, excitability suppressed).

Growth is self-limiting: familiar input causes arousal to decay, which
closes the growth gate. No rate parameter -- the brain decides.

## Dependencies

Required:
- Python 3.10+
- numpy

For the harness:
- websockets (`pip install websockets`)

Optional:
- matplotlib (for probe PNG snapshots)
- openai (for TTS voice generation)

## Background

Part of the LIFE project (TeamSafeAI). V10 builds on:
- **V8** (inner-models-v8): Modular engine, 85+ brains, 32 blocks, 8 synapse types
- **V9** (inner-models-v9): Growth-based wiring, arena testing (concluded: STDP
  can't learn navigation -- credit assignment is the barrier)

V10 drops the arena and scorecard. No tests. No scores. Just build a brain,
give it sensory input, and watch it develop. The viewer IS the experiment.
