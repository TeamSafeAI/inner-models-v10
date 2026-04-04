"""
DAS modulator constants -- thresholds, decay rates, biological mappings.

D = Dopamine (surprise / prediction error)
A = Adrenaline/norepinephrine (arousal / gain)
S = Serotonin (stability / mood baseline)
"""

# Surprise (dopamine analog)
SURPRISE_EMA_ALPHA = 0.02        # sensory EMA smoothing (1-alpha = 0.98)
SURPRISE_THRESHOLD = 0.15        # fractional deviation triggers reward
SURPRISE_REWARD_SCALE = 0.8      # maps surprise -> reward magnitude (capped at 1.0)

# Stability (serotonin analog)
STABILITY_TARGET_VAR = 0.0004    # target population rate variance
STABILITY_SENSITIVITY = 500      # how fast scale responds to variance deviation
STABILITY_MIN_SCALE = 0.2        # minimum learning rate scale
STABILITY_EVAL_INTERVAL = 100    # ticks between evaluations

# Arousal (norepinephrine analog)
AROUSAL_DECAY = 0.995            # per-tick decay (~140 tick half-life)
AROUSAL_GAIN = 2.5               # max mA boost at full arousal (1.0)
AROUSAL_SPIKE_SCALE = 0.4        # input delta -> arousal increment
AROUSAL_DELTA_THRESHOLD = 0.1    # minimum input change to trigger arousal

# Neuromodulatory current
NEUROMOD_DECAY = 0.995           # current decay per tick (~200ms tau)
NEUROMOD_GAIN = 3.0              # uA per unit magnitude per unit sensitivity
