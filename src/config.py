from constants import FP, DM, EDIT, REP, RST, COR, PRO, SLR

HIGH_PROB = 0.30
MID_PROB = 0.20
LOW_PROB = 0.10
VERY_LOW_PROB = 0.05
ZERO_PROB = 0.0

DISFLUENCY_PROBABILITIES: dict[str, dict[str, float]] = {
    "INFORM_INTENT": {
        "PLANNING": HIGH_PROB,
        "ARTICULATION": LOW_PROB,
        "REPAIR": LOW_PROB,
    },
    "INFORM": {
        "PLANNING": LOW_PROB,
        "ARTICULATION": HIGH_PROB,
        "REPAIR": LOW_PROB,
    },
    "SELECT": {
        "PLANNING": MID_PROB,
        "ARTICULATION": LOW_PROB,
        "REPAIR": HIGH_PROB,
    },
    "NEGATE": {
        "PLANNING": VERY_LOW_PROB,
        "ARTICULATION": VERY_LOW_PROB,
        "REPAIR": HIGH_PROB,
    },
    "NEGATE_INTENT": {
        "PLANNING": VERY_LOW_PROB,
        "ARTICULATION": VERY_LOW_PROB,
        "REPAIR": HIGH_PROB,
    },
    "REQUEST": {
        "PLANNING": MID_PROB,
        "ARTICULATION": MID_PROB,
        "REPAIR": LOW_PROB,
    },
    "REQUEST_ALTS": {
        "PLANNING": MID_PROB,
        "ARTICULATION": MID_PROB,
        "REPAIR": LOW_PROB,
    },
    "AFFIRM": {
        "PLANNING": VERY_LOW_PROB,
        "ARTICULATION": VERY_LOW_PROB,
        "REPAIR": VERY_LOW_PROB,
    },
    "AFFIRM_INTENT": {
        "PLANNING": VERY_LOW_PROB,
        "ARTICULATION": VERY_LOW_PROB,
        "REPAIR": VERY_LOW_PROB,
    },
    "THANK_YOU": {
        "PLANNING": VERY_LOW_PROB,
        "ARTICULATION": VERY_LOW_PROB,
        "REPAIR": ZERO_PROB,
    },
    "GOODBYE": {
        "PLANNING": VERY_LOW_PROB,
        "ARTICULATION": VERY_LOW_PROB,
        "REPAIR": ZERO_PROB,
    },
}

# Filler types that are mutually exclusive (only one can be selected per position)
FILLER_TYPES: list[str] = [FP, DM, EDIT]

PHASE_TO_TYPES: dict[str, list[str]] = {
    "PLANNING": ["FP", "DM", "EDIT", "PRO"],  # Type names as strings (not tags)
    "ARTICULATION": ["REP", "SLR"],
    "REPAIR": ["COR", "RST"],
}

# Injection order: individual disfluency types
INJECTION_ORDER: list[str] = ["RST", "COR", "SLR", "FP", "DM", "EDIT", "PRO", "REP"]

# LLM-based types (structural changes requiring slot recovery)
LLM_BASED_TYPES: list[str] = [RST, COR]

# Rule-based types (surface noise with offset tracking)
RULE_BASED_TYPES: list[str] = [FP, DM, EDIT, PRO, REP, SLR]

# =============================================================================
# POSITION WEIGHT PARAMETERS
# =============================================================================

WEIGHT_BEFORE_SLOT = 3.0  # Position directly before slot (PLANING phase)
WEIGHT_AT_SLOT = 2.5  # Slot value positions (RETRIEVAL/REPAIR)
WEIGHT_AFTER_SLOT = 2.0  # Position directly after slot (monitoring)
WEIGHT_NEAR_SLOT = 1.5  # Near-slot positions (±2 tokens)
WEIGHT_DEFAULT = 1.0  # Default weight for other positions

# =============================================================================
# TTS DISFLUENCY TAGS
# - [FP]: Filled Pause (uh, um)
# - [DM]: Discourse Marker (like, you know, well)
# - [REP]: Repetition
# - [RST]: Restart
# - [COR]: Correction
# - PRO (Prolongation) and SLR (Slurring) are NOT tagged in output
# =============================================================================

DISFLUENCY_TAGS: dict[str, str] = {
    "FP": FP,
    "DM": DM,
    "EDIT": EDIT,
    "REP": REP,
    "RST": RST,
    "COR": COR,
    "PRO": PRO,
    "SLR": SLR,
}

FP_VOCAB = [
    "um",
    "uh",
    "ah",
    "er",
    "hmm",
    "huh",
    "eh",
    "oops",
    "oh",
]
DM_VOCAB = [
    "like",
    "you know",
    "well",
    "so",
    "right",
    "listen",
    "basically",
    "just",
    "actually",
    "now",
    "see",
    "sure",
]
EDIT_VOCAB = [
    "I mean",
    "sorry",
    "wait",
    "excuse me",
    "let me see",
    "no wait",
]

# Rule-based filler vocabulary mapping
FILLERS: dict[str, list[str]] = {
    "filled_pause": FP_VOCAB,
    "discourse_marker": DM_VOCAB,
    "editing_term": EDIT_VOCAB,
}
