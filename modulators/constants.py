"""
Neuromodulator constants -- thresholds, decay rates, biological mappings.

D = Dopamine (surprise / prediction error)
A = Adrenaline/norepinephrine (arousal / gain)
S = Serotonin (stability / mood baseline)
C = Cortisol (sustained stress / growth suppression)
O = Oxytocin (safety / learning boost)
"""

# Surprise (dopamine analog)
SURPRISE_EMA_ALPHA = 0.02        # sensory EMA smoothing (1-alpha = 0.98)
SURPRISE_THRESHOLD = 0.15        # fractional deviation triggers reward
SURPRISE_REWARD_SCALE = 0.8      # maps surprise -> reward magnitude (capped at 1.0)

# Stability (serotonin analog)
# Target variance is now DYNAMIC: the brain learns its own baseline via EMA.
# STABILITY_TARGET_VAR is used as initial seed only (overwritten after first eval).
# Sensitivity measures how sharply S responds to deviation FROM the learned baseline.
# A 50% variance spike above baseline -> S drops to ~0.5 (moderate throttle).
# A 200%+ spike -> S hits floor (0.2). Variance BELOW baseline -> S stays at 1.0.
STABILITY_TARGET_VAR = 0.0004    # initial seed (overwritten by dynamic EMA)
STABILITY_VAR_EMA_ALPHA = 0.05   # how fast the baseline adapts (~20 evals to converge)
STABILITY_SENSITIVITY = 2.0      # fractional: 0.5 = 50% above baseline -> S=0.0 (clamped to 0.2)
STABILITY_MIN_SCALE = 0.2        # minimum learning rate scale
STABILITY_EVAL_INTERVAL = 100    # ticks between evaluations

# Arousal (norepinephrine analog)
# Biology: locus coeruleus HABITUATES to predictable sustained input.
# Arousal tracks an EMA of input deltas (expected variation). Only spikes
# when current delta exceeds the adapted expectation (novelty detection).
# Predictable music -> delta_ema rises -> threshold rises -> arousal decays.
# Novel input -> delta exceeds expectation -> arousal spikes -> habituates again.
AROUSAL_DECAY = 0.995            # per-tick decay (~140 tick half-life)
AROUSAL_GAIN = 2.5               # max mA boost at full arousal (1.0)
AROUSAL_SPIKE_SCALE = 0.4        # novelty ratio -> arousal increment
AROUSAL_DELTA_THRESHOLD = 2.0    # ratio: delta must exceed EMA by this factor
AROUSAL_DELTA_FLOOR = 0.05       # absolute minimum delta to spike (prevents noise)
AROUSAL_HABITUATION_ALPHA = 0.01 # delta EMA adaptation rate (~100 tick tau)

# Neuromodulatory current
NEUROMOD_DECAY = 0.995           # current decay per tick (~200ms tau)
NEUROMOD_GAIN = 3.0              # uA per unit magnitude per unit sensitivity

# Cortisol (sustained stress -- allostatic load model)
# Biology: HPA axis responds to sustained overload RELATIVE TO adapted baseline.
# Predictable stimulation (music) doesn't sustain cortisol. Unpredictable,
# escalating input does. The brain maintains its own model of "normal" and
# reacts when pushed beyond it.
# Fetal: placenta blocks 80-90% of maternal cortisol (buffer < 1.0).
CORTISOL_EMA_ALPHA = 0.005          # sensory baseline adaptation (~200 tick tau)
CORTISOL_VAR_ALPHA = 0.005          # variance baseline adaptation (same timescale)
CORTISOL_Z_THRESHOLD = 2.0          # std devs above adapted mean to count as overload
CORTISOL_ONSET_TICKS = 200          # sustained overload ticks before cortisol rises
CORTISOL_RISE_RATE = 0.002          # per-tick rise (~500 ticks to 1.0)
CORTISOL_DECAY = 0.9995             # per-tick decay (~1400 tick half-life)
CORTISOL_COUNTER_DECAY = 0.95       # onset counter decay when not overloaded
CORTISOL_PLACENTAL_BUFFER = 0.15    # fetal: placenta blocks ~85% (set 1.0 post-birth)
CORTISOL_GROWTH_SUPPRESSION = 0.8   # growth reduced 80% at level=1.0
CORTISOL_SURVIVAL_REDUCTION = 300   # ticks removed from survival window at level=1.0

# Oxytocin (safety / bonding)
# Biology: released during calm, familiar, predictable contexts.
# "You're safe -- consolidate learning, keep new neurons."
OXYTOCIN_AROUSAL_CEILING = 0.2     # A must be below this for calm
OXYTOCIN_SURPRISE_CEILING = 0.3    # D must be below this for calm
OXYTOCIN_ONSET_TICKS = 300         # sustained calm ticks before oxytocin rises
OXYTOCIN_RISE_RATE = 0.003         # per-tick rise (~333 ticks to 1.0)
OXYTOCIN_DECAY = 0.998             # per-tick decay (~350 tick half-life)
OXYTOCIN_LR_BOOST = 0.5            # max additional LR multiplier at level=1.0
OXYTOCIN_SURVIVAL_BOOST = 500      # ticks added to survival window at level=1.0
