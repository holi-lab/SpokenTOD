"""Probability engine for sampling disfluencies based on dialogue acts."""

import random
from typing import Any

import numpy as np

from config import DISFLUENCY_PROBABILITIES, INJECTION_ORDER, PHASE_TO_TYPES
from definitions import SlotPositions


def sample_disfluencies_for_turn(
    dialogue_acts: list[str],
) -> list[dict[str, Any]]:
    """
    Sample which disfluency types to apply based on dialogue acts.

    Uses max probability across all dialogue acts for each phase,
    then samples independently for each phase.

    Args:
        dialogue_acts: List of dialogue act names from this turn

    Returns:
        List of disfluency specifications sorted by INJECTION_ORDER:
        [
            {"type": "RST", "phase": "repair"},
            {"type": "FP", "phase": "planning"},
            ...
        ]
    """
    # Aggregate probabilities across all dialogue acts
    # Use max probability for each phase (most demanding act dominates)
    phase_probs = {"PLANNING": 0.0, "ARTICULATION": 0.0, "REPAIR": 0.0}

    for act in dialogue_acts:
        if act in DISFLUENCY_PROBABILITIES:
            act_probs = DISFLUENCY_PROBABILITIES[act]
            for phase in phase_probs:
                phase_probs[phase] = max(phase_probs[phase], act_probs[phase])

    # If no known acts, use INFORM as default
    if all(p == 0.0 for p in phase_probs.values()):
        phase_probs = DISFLUENCY_PROBABILITIES.get(
            "INFORM", {"PLANNING": 0.10, "ARTICULATION": 0.10, "REPAIR": 0.05}
        )

    # Sample for each phase independently
    sampled: list[dict[str, Any]] = []

    for phase, prob in phase_probs.items():
        if random.random() < prob:
            # Select one type from this phase
            types_for_phase = PHASE_TO_TYPES.get(phase, [])
            if types_for_phase:
                disfluency_type = random.choice(types_for_phase)
                sampled.append(
                    {
                        "type": disfluency_type,
                        "phase": phase,
                    }
                )

    # Sort by injection order (RST, COR, SLR, FP, DM, EDIT, PRO, REP)
    type_order = {t: i for i, t in enumerate(INJECTION_ORDER)}
    sampled.sort(key=lambda x: type_order.get(x["type"], 99))

    # Ensure at least one disfluency is sampled
    if not sampled:
        best_phase = max(phase_probs, key=phase_probs.get)
        types_for_phase = PHASE_TO_TYPES.get(best_phase, [])
        if types_for_phase:
            disfluency_type = random.choice(types_for_phase)
            sampled.append(
                {
                    "type": disfluency_type,
                    "phase": best_phase,
                }
            )

    return sampled


def select_positions(
    sampled_disfluencies: list[dict[str, Any]],
    position_weights: np.ndarray,
    slot_positions: SlotPositions,
    tokens: list[str],
) -> list[dict[str, Any]]:
    """
    Assign token positions to each sampled disfluency based on weights.

    Args:
        sampled_disfluencies: List from sample_disfluencies_for_turn
        position_weights: Weight array from compute_position_weights
        slot_positions: Slot info with token positions
        tokens: List of tokens

    Returns:
        Updated disfluencies with position and slot_related fields
    """
    num_tokens = len(position_weights)
    used_positions: set = set()

    for disf in sampled_disfluencies:
        disf_type = disf["type"]

        # COR, RST, REP, and SLR must be at slot positions
        if disf_type in ("COR", "RST", "REP", "SLR"):
            slot_names = list(slot_positions.keys())
            if slot_names:
                chosen_slot = random.choice(slot_names)
                slot_info = slot_positions[chosen_slot]
                token_start = slot_info.get("token_start", 0)
                token_end = slot_info.get("token_end", token_start)
                if token_start is None:
                    token_start = 0
                if token_end is None:
                    token_end = token_start
                if token_end < token_start:
                    token_end = token_start
                if disf_type in ("REP", "SLR"):
                    disf["token_position"] = random.randint(token_start, token_end)
                else:
                    disf["token_position"] = token_start
                disf["slot_related"] = chosen_slot
            else:
                # No slots, assign random position
                disf["token_position"] = random.randint(0, max(0, num_tokens - 1))
            continue

        # For rule-based types, use weighted random selection
        adjusted_weights = position_weights.copy()

        # Zero out already-used positions
        for pos in used_positions:
            if 0 <= pos < len(adjusted_weights):
                adjusted_weights[pos] = 0

        # Normalize to probability distribution
        total = adjusted_weights.sum()
        if total > 0:
            probs = adjusted_weights / total
            chosen_pos = int(np.random.choice(num_tokens, p=probs))
        else:
            chosen_pos = random.randint(0, max(0, num_tokens - 1))

        disf["token_position"] = chosen_pos
        used_positions.add(chosen_pos)

        # Find related slot (if near a slot)
        for slot_name, slot_info in slot_positions.items():
            t_start = slot_info.get("token_start", -10)
            t_end = slot_info.get("token_end", -10)
            if t_end is None:
                t_end = t_start
            if t_start is not None and t_start - 2 <= chosen_pos <= t_end + 2:
                disf["slot_related"] = slot_name
                break

    return sampled_disfluencies
