"""Slot position extraction from SGD frames."""

import re
from typing import Any

from augmentation.disfluency.definitions import SlotPositions


def extract_slot_positions(
    utterance: str,
    frames: list[dict[str, Any]],
) -> SlotPositions:
    """
    Find character positions of slot values in utterance.

    Args:
        utterance: The user utterance text
        frames: List of frames from SGD turn

    Returns:
        Dictionary mapping slot names to position info:
        {
            "slot_name": {
                "value": str,
                "start": int,  # character index
                "end": int,    # exclusive end
            }
        }
    """
    slot_positions: SlotPositions = {}

    for frame in frames:
        # Get slot values from state
        state = frame.get("state", {})
        slot_values = state.get("slot_values", {})

        for slot in frame.get("slots", []):
            slot_name = slot["slot"]

            # SGD provides start/exclusive_end for slot spans in utterance
            if "start" in slot and "exclusive_end" in slot:
                slot_positions[slot_name] = {
                    "value": utterance[slot["start"] : slot["exclusive_end"]],
                    "start": slot["start"],
                    "end": slot["exclusive_end"],
                }
            else:
                # Fallback: search for value in utterance
                # Get value from slot_values in state
                values = slot_values.get(slot_name, [])
                if values:
                    value = values[0]  # Take first value
                    match = re.search(re.escape(value), utterance, re.IGNORECASE)
                    if match:
                        slot_positions[slot_name] = {
                            "value": value,
                            "start": match.start(),
                            "end": match.end(),
                        }

    return slot_positions


def extract_dialogue_acts(frames: list[dict[str, Any]]) -> list[str]:
    """
    Extract dialogue act names from frames.

    Args:
        frames: List of frames from SGD turn

    Returns:
        List of unique dialogue act names
    """
    acts: list[str] = []

    for frame in frames:
        for action in frame.get("actions", []):
            act = action.get("act", "")
            if act and act not in acts:
                acts.append(act)

    return acts
