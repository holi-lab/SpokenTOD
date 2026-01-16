"""Position weighting for slot-aware disfluency placement."""

import re

import numpy as np

from data.augmentation.config import (
    WEIGHT_AFTER_SLOT,
    WEIGHT_AT_SLOT,
    WEIGHT_BEFORE_SLOT,
    WEIGHT_DEFAULT,
    WEIGHT_NEAR_SLOT,
)
from data.augmentation.types import SlotPositions


def tokenize_with_positions(utterance: str) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Whitespace tokenization with character positions.

    Args:
        utterance: Text to tokenize

    Returns:
        tokens: List of token strings
        positions: List of (start_char, end_char) tuples
    """
    tokens: list[str] = []
    positions: list[tuple[int, int]] = []

    for match in re.finditer(r"\S+", utterance):
        tokens.append(match.group())
        positions.append((match.start(), match.end()))

    return tokens, positions


def char_to_token_positions(
    slot_positions: SlotPositions,
    token_char_positions: list[tuple[int, int]],
) -> SlotPositions:
    """
    Convert character-level slot positions to token indices.

    Args:
        slot_positions: Slot info with char start/end
        token_char_positions: List of (start, end) char positions for each token

    Returns:
        Updated slot_positions with token_start and token_end fields
    """
    for slot_name, slot_info in slot_positions.items():
        char_start = slot_info.get("start")
        char_end = slot_info.get("end")

        if char_start is None or char_end is None:
            continue

        token_start = None
        token_end = None

        for idx, (t_start, t_end) in enumerate(token_char_positions):
            # Token contains start of slot
            if t_start <= char_start < t_end:
                token_start = idx
            # Token contains end of slot
            if t_start < char_end <= t_end:
                token_end = idx

        slot_info["token_start"] = token_start
        slot_info["token_end"] = token_end if token_end is not None else token_start

    return slot_positions


def compute_position_weights(
    num_tokens: int,
    slot_positions: SlotPositions,
) -> np.ndarray:
    """
    Compute slot-weighted disfluency probability for each token position.

    Weights reflect cognitive load at different positions relative to slots:
    - Before slot: planning phase (highest weight)
    - At slot: articulation/repair phase
    - After slot: monitoring phase
    - Near slot: moderate weight
    - Default: baseline weight

    Args:
        num_tokens: Number of tokens in utterance
        slot_positions: Slot info with token_start and token_end

    Returns:
        Array of weights, one per token position
    """
    weights = np.full(num_tokens, WEIGHT_DEFAULT)

    for slot_info in slot_positions.values():
        token_start = slot_info.get("token_start")
        token_end = slot_info.get("token_end")

        if token_start is None:
            continue

        if token_end is None:
            token_end = token_start

        # Directly before slot (planning phase) - highest weight
        if token_start > 0:
            weights[token_start - 1] = max(weights[token_start - 1], WEIGHT_BEFORE_SLOT)

        # Slot value tokens (articulation/repair phase)
        for i in range(token_start, min(token_end + 1, num_tokens)):
            weights[i] = max(weights[i], WEIGHT_AT_SLOT)

        # Directly after slot (monitoring phase)
        if token_end + 1 < num_tokens:
            weights[token_end + 1] = max(weights[token_end + 1], WEIGHT_AFTER_SLOT)

        # Near-slot (±2 tokens)
        for i in range(max(0, token_start - 2), min(num_tokens, token_end + 3)):
            if weights[i] == WEIGHT_DEFAULT:
                weights[i] = WEIGHT_NEAR_SLOT

    return weights


def select_weighted_position(
    weights: np.ndarray,
    used_positions: set | None = None,
) -> int:
    """
    Select a position based on weights, avoiding used positions.

    Args:
        weights: Position weight array
        used_positions: Set of positions to exclude

    Returns:
        Selected position index
    """
    adjusted_weights = weights.copy()

    if used_positions:
        for pos in used_positions:
            if 0 <= pos < len(adjusted_weights):
                adjusted_weights[pos] = 0

    # Normalize to probability distribution
    total = adjusted_weights.sum()
    if total > 0:
        probs = adjusted_weights / total
        return int(np.random.choice(len(weights), p=probs))
    else:
        # Fallback: random position
        import random

        return random.randint(0, len(weights) - 1)
