"""Output formatter for generating final JSON."""

from typing import Any

from definitions import SlotPositions


def format_output(
    dialogue_id: str,
    turn_id: int,
    original_utterance: str,
    disfluent_utterance: str,
    tagged_utterance: str,
    slots: SlotPositions,
    injections: list[dict[str, Any]],
    dialogue_acts: list[str],
) -> dict[str, Any]:
    """
    Format a single turn output in the specified JSON format.

    Args:
        dialogue_id: SGD dialogue ID
        turn_id: Turn index
        original_utterance: Original clean utterance
        disfluent_utterance: Utterance with disfluencies (no TTS tags)
        tagged_utterance: Utterance with TTS tags
        slots: Slot positions with updated indices
        injections: List of injection annotations
        dialogue_acts: List of dialogue act names

    Returns:
        Formatted output dictionary
    """
    # Format slots as list
    slots_list = []
    for slot_name, slot_info in slots.items():
        slot_entry = {
            "slot": slot_name,
            "value": slot_info.get("value", ""),
        }
        if slot_info.get("start") is not None:
            slot_entry["start"] = slot_info["start"]
        if slot_info.get("end") is not None:
            slot_entry["end"] = slot_info["end"]
        slots_list.append(slot_entry)

    # Format injections with trigger_act
    injections_list = []
    for inj in injections:
        inj_entry = {
            "type": inj.get("type", ""),
            "position": inj.get("position", inj.get("insert_position", 0)),
        }
        if "text" in inj:
            inj_entry["text"] = inj["text"]
        if "original" in inj:
            inj_entry["original"] = inj["original"]
        if "prolonged" in inj:
            inj_entry["prolonged"] = inj["prolonged"]
        if "repeated_unit" in inj:
            inj_entry["repeated_unit"] = inj["repeated_unit"]
        if "replacements" in inj:
            inj_entry["replacements"] = inj["replacements"]
        if "slot_name" in inj:
            inj_entry["slot_name"] = inj["slot_name"]
        if "correct_value" in inj:
            inj_entry["correct_value"] = inj["correct_value"]
        if "modified_utterance" in inj:
            inj_entry["modified_utterance"] = inj["modified_utterance"]

        # Add trigger act (first dialogue act)
        if dialogue_acts:
            inj_entry["trigger_act"] = dialogue_acts[0]

        injections_list.append(inj_entry)

    return {
        "dialogue_id": dialogue_id,
        "turn_id": turn_id,
        "original_utterance": original_utterance,
        "disfluent_utterance": disfluent_utterance,
        "tagged_utterance": tagged_utterance,
        "slots": slots_list,
        "injections": injections_list,
    }


def compute_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute statistics about the augmentation.

    Args:
        results: List of formatted output dictionaries

    Returns:
        Statistics dictionary
    """
    total = len(results)

    # Count disfluent turns
    disfluent = sum(1 for r in results if r.get("injections"))

    # Count by type
    type_counts: dict[str, int] = {}
    for r in results:
        for inj in r.get("injections", []):
            t = inj.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

    # Count by dialogue act
    act_counts: dict[str, int] = {}
    for r in results:
        for inj in r.get("injections", []):
            act = inj.get("trigger_act", "unknown")
            act_counts[act] = act_counts.get(act, 0) + 1

    return {
        "total_turns": total,
        "disfluent_turns": disfluent,
        "disfluent_ratio": disfluent / total if total > 0 else 0,
        "disfluency_type_distribution": type_counts,
        "dialogue_act_distribution": act_counts,
        "avg_injections_per_disfluent_turn": (
            sum(len(r.get("injections", [])) for r in results) / disfluent if disfluent > 0 else 0
        ),
    }
