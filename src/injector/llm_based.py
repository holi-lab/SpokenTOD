"""LLM-based disfluency injectors: RST (Restart), COR (Correction).

These injectors use LLM to generate natural structural changes to utterances.
"""

import re

from common.llm import async_completion
from common.utils import load_prompt

from config import DISFLUENCY_TAGS
from definitions import SlotPositions
from .base import InjectionResult

# Prompt templates
CORRECTION_PROMPT = load_prompt("correction.txt")
RESTART_PROMPT = load_prompt("restart.txt")


def recover_slots(
    new_text: str,
    original_slots: SlotPositions,
    slurred_values: dict[str, str] | None = None,
) -> SlotPositions:
    """
    Re-find slot values in LLM-rewritten text via fuzzy matching.

    After LLM rewrites the utterance (COR, RST, or SLR), slot positions
    are invalidated. This function finds the new positions.

    Args:
        new_text: The rewritten utterance from LLM
        original_slots: Original slot position dictionary
        slurred_values: Optional dict of {slot_name: slurred_value} for SLR

    Returns:
        Updated slot positions with new start/end indices
    """
    recovered = {}

    for slot_name, slot_info in original_slots.items():
        original_value = slot_info.get("value", "")
        if not original_value:
            recovered[slot_name] = slot_info.copy()
            continue

        # For slurred slots, search for the slurred value instead
        search_values = [original_value]
        if slurred_values and slot_name in slurred_values:
            slurred_value = slurred_values[slot_name]
            search_values = [slurred_value, original_value]

        found = False
        for value in search_values:
            # Try exact match first
            match = re.search(re.escape(value), new_text, re.IGNORECASE)

            if match:
                recovered[slot_name] = {
                    "slot": slot_name,
                    "value": value,  # Use the matched value (may be slurred)
                    "original_value": original_value,
                    "start": match.start(),
                    "end": match.end(),
                }
                found = True
                break

        if not found:
            # Fuzzy: try partial match (for corrections where value might be split)
            patterns = [
                rf"(?:no|actually|wait|I mean)[,\s]+{re.escape(original_value)}",
                rf"{re.escape(original_value)}",
            ]

            for pattern in patterns:
                match = re.search(pattern, new_text, re.IGNORECASE)
                if match:
                    value_match = re.search(re.escape(original_value), match.group(), re.IGNORECASE)
                    if value_match:
                        actual_start = match.start() + value_match.start()
                        actual_end = match.start() + value_match.end()
                        recovered[slot_name] = {
                            "slot": slot_name,
                            "value": original_value,
                            "start": actual_start,
                            "end": actual_end,
                        }
                        found = True
                        break

            if not found:
                # Keep original info but mark as unrecovered
                recovered[slot_name] = {
                    "slot": slot_name,
                    "value": original_value,
                    "start": None,
                    "end": None,
                    "unrecovered": True,
                }

    return recovered


async def inject_correction(
    utterance: str,
    slot_name: str,
    slot_value: str,
    slot_positions: SlotPositions,
    dialogue_act: str,
    model: str,
    provider: str = "openrouter",
    base_url: str = "http://localhost:8000/v1",
    include_usage: bool | None = None,
) -> tuple[InjectionResult, SlotPositions]:
    """
    Use LLM to generate a self-correction for a slot value.

    Args:
        utterance: Current utterance text
        slot_name: Name of slot to correct
        slot_value: Correct value of the slot
        slot_positions: Current slot positions
        dialogue_act: Dialogue act for context
        llm_call_fn: Async function to call LLM

    Returns:
        Tuple of (InjectionResult, updated_slot_positions)
    """
    prompt = CORRECTION_PROMPT.format(
        dialogue_act=dialogue_act,
        utterance=utterance,
        slot_name=slot_name,
        slot_value=slot_value,
    )

    messages = [{"role": "user", "content": prompt}]
    response = await async_completion(
        messages=messages,
        model=model,
        provider=provider,
        base_url=base_url,
        include_usage=include_usage,
    )
    modified_utterance = response.strip().strip('"')

    # Recover slot positions in new text
    updated_slots = recover_slots(modified_utterance, slot_positions)

    # Build tagged version following Switchboard format
    # Pattern: "<reparandum> [COR] <repair>"
    tag = DISFLUENCY_TAGS["COR"]

    # Find the correction pattern and insert tag AFTER reparandum
    # Match patterns with or without editing terms
    # Use \S+ to match the word immediately before dash (non-whitespace)
    correction_patterns = [
        # With editing terms
        (
            r"(\S+)\s*[-–]\s*(no[,\s]+)",
            r"\1 {tag} \2",
        ),  # "lunch- no," -> "lunch [COR] no,"
        (
            r"(\S+)\s*[-–]\s*(wait[,\s]+)",
            r"\1 {tag} \2",
        ),  # "X- wait," -> "X [COR] wait,"
        (
            r"(\S+)\s*[-–]\s*(actually[,\s]+)",
            r"\1 {tag} \2",
        ),  # "X- actually," -> "X [COR] actually,"
        (
            r"(\S+)\s*[-–]\s*(I mean[,\s]+)",
            r"\1 {tag} \2",
        ),  # "X- I mean," -> "X [COR] I mean,"
        (r"(\S+)\s*\.\.\.\s*", r"\1 {tag} "),  # "X... " -> "X [COR] "
        # Without editing terms (direct correction)
        (r"(\S+)\s*[-–]\s+", r"\1 {tag} "),  # "a- the" -> "a [COR] the"
    ]

    tagged_utterance = modified_utterance
    for pattern, replacement in correction_patterns:
        replacement_with_tag = replacement.format(tag=tag)
        tagged_utterance = re.sub(
            pattern,
            replacement_with_tag,
            modified_utterance,
            count=1,
            flags=re.IGNORECASE,
        )
        if tagged_utterance != modified_utterance:
            break

    annotation = {
        "type": "COR",
        "slot_name": slot_name,
        "correct_value": slot_value,
        "original_utterance": utterance,
        "modified_utterance": modified_utterance,
    }

    result = InjectionResult(
        raw_text=modified_utterance,
        tagged_text=tagged_utterance,
        char_offset=len(modified_utterance) - len(utterance),
        insert_position=0,
        annotation=annotation,
    )

    return result, updated_slots


async def inject_restart(
    utterance: str,
    tokens: list[str],
    token_position: int,
    slot_positions: SlotPositions,
    dialogue_act: str,
    model: str,
    provider: str = "openrouter",
    base_url: str = "http://localhost:8000/v1",
    include_usage: bool | None = None,
) -> tuple[InjectionResult, SlotPositions]:
    """
    Use LLM to generate a sentence restart.

    Args:
        utterance: Current utterance text
        tokens: List of tokens
        token_position: Token position near which to restart
        slot_positions: Current slot positions
        dialogue_act: Dialogue act for context
        llm_call_fn: Async function to call LLM

    Returns:
        Tuple of (InjectionResult, updated_slot_positions)
    """
    word_at_position = tokens[token_position] if token_position < len(tokens) else tokens[-1]

    prompt = RESTART_PROMPT.format(
        dialogue_act=dialogue_act,
        utterance=utterance,
        position=token_position,
        word_at_position=word_at_position,
    )

    messages = [{"role": "user", "content": prompt}]
    response = await async_completion(
        messages=messages,
        model=model,
        provider=provider,
        base_url=base_url,
        include_usage=include_usage,
    )
    modified_utterance = response.strip().strip('"')

    # Recover slot positions in new text
    updated_slots = recover_slots(modified_utterance, slot_positions)

    # Build tagged version following Switchboard format
    # Pattern: "<reparandum> [RST]" (tag comes AFTER the incomplete phrase)
    tag = DISFLUENCY_TAGS["RST"]

    # Find restart patterns and insert tag AFTER reparandum
    restart_patterns = [
        (r"(\w+\s+\w+)\s*[-–]\s+", r"\1 {tag} "),  # "I want to- " -> "I want to [RST] "
        (r"(\w+)\s*[-–]\s+", r"\1 {tag} "),  # "The- " -> "The [RST] "
        (r"([^.]+?)\s*\.\.\.\s+", r"\1 {tag} "),  # "I need... " -> "I need [RST] "
    ]

    tagged_utterance = modified_utterance
    for pattern, replacement in restart_patterns:
        replacement_with_tag = replacement.format(tag=tag)
        tagged_utterance = re.sub(pattern, replacement_with_tag, modified_utterance, count=1)
        if tagged_utterance != modified_utterance:
            break

    annotation = {
        "type": "RST",
        "restart_near": token_position,
        "original_utterance": utterance,
        "modified_utterance": modified_utterance,
    }

    result = InjectionResult(
        raw_text=modified_utterance,
        tagged_text=tagged_utterance,
        char_offset=len(modified_utterance) - len(utterance),
        insert_position=0,
        annotation=annotation,
    )

    return result, updated_slots
