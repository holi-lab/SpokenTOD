"""Multi-turn dialogue generator for cross-turn slot speaking."""

import random
from dataclasses import dataclass

from .rules import segment_slot, inject_error_correction
from augmentation.constants import SEGMENTABLE_SLOTS, ERROR_CORRECTION_PROB


@dataclass
class SegmentedTurn:
    """A turn in a segmented slot dialogue."""
    role: str  # "user" or "assistant"
    text: str
    segment: dict | None = None  # {"slot", "value", "idx", "total", "is_correction"}


# Templates for user utterances
USER_FIRST_TEMPLATES = [
    "My {slot_name} is {value}",
    "The {slot_name} is {value}",
    "It's {value}",
]

USER_CONTINUE_TEMPLATES = [
    "And then {value}",
    "Then {value}",
    "Next is {value}",
    "Followed by {value}",
]

USER_CONFIRM_CONTINUE_TEMPLATES = [
    "Yes. And then {value}",
    "That's right. Then {value}",
    "Correct. Next is {value}",
]

USER_CORRECTION_TEMPLATES = [
    "I'm sorry, {value}",
    "Wait, no. {value}",
    "Actually, {value}",
    "Sorry, I meant {value}",
]

USER_FINAL_TEMPLATES = [
    "And then {value}",
    "And finally {value}",
    "Last part is {value}",
]

USER_DOUBLE_TEMPLATES = [
    "And then double {value}",
    "Then double {value}",
]

# Templates for assistant confirmation
ASST_CONFIRM_TEMPLATES = [
    "So it's {value}.",
    "{value}.",
    "Got it, {value}.",
    "Okay, {value}.",
]

ASST_CORRECTION_CONFIRM_TEMPLATES = [
    "Yes. Okay, so it's {value}.",
    "Got it. {value}.",
    "Understood. {value}.",
]

ASST_FINAL_TEMPLATES = [
    "Got it. The full {slot_name} is {full_value}.",
    "Alright. I have {full_value} as your {slot_name}.",
    "Thank you. Your {slot_name} is {full_value}.",
]


def _naturalize_slot_name(slot_name: str) -> str:
    """Convert slot name to natural spoken form.
    
    Example: "phone_number" -> "phone number"
    """
    return slot_name.replace("_", " ").replace(".", " ")


def _has_double_chars(segment: str) -> tuple[bool, str]:
    """Check if segment ends with double characters.
    
    Example: "9 9 0 3" -> (True, "9 0 3") for "double 9"
    """
    parts = segment.split()
    if len(parts) >= 2 and parts[-2] == parts[-1]:
        return True, " ".join(parts[:-1])
    return False, segment


def generate_crossturn_dialogue(
    slot_name: str,
    slot_value: str,
    slot_type: str,
    error_prob: float = ERROR_CORRECTION_PROB,
) -> list[SegmentedTurn]:
    """Generate SpokenWOZ-style cross-turn dialogue for a slot value.
    
    Args:
        slot_name: Name of the slot (e.g., "phone_number")
        slot_value: Full value of the slot (e.g., "5258576375249903")
        slot_type: Type for segmentation (e.g., "phone", "email")
        error_prob: Probability of error correction insertion
    
    Returns:
        List of SegmentedTurn objects forming the dialogue
    """
    # Segment the value
    segments = segment_slot(slot_value, slot_type)
    
    if len(segments) == 0:
        return []
    
    # Inject error corrections
    segments_with_errors = inject_error_correction(segments, error_prob)
    
    natural_name = _naturalize_slot_name(slot_name)
    turns = []
    total_segments = len([s for s, is_corr in segments_with_errors if not is_corr])
    seg_idx = 0
    
    for i, (segment, is_correction) in enumerate(segments_with_errors):
        is_first = i == 0
        is_last = i == len(segments_with_errors) - 1
        
        # Check for double pattern in final segment
        has_double, modified_seg = _has_double_chars(segment)
        
        # User turn
        if is_correction:
            template = random.choice(USER_CORRECTION_TEMPLATES)
            user_text = template.format(value=segment)
        elif is_first:
            template = random.choice(USER_FIRST_TEMPLATES)
            user_text = template.format(slot_name=natural_name, value=segment)
        elif is_last and has_double:
            template = random.choice(USER_DOUBLE_TEMPLATES)
            user_text = template.format(value=modified_seg)
        elif is_last:
            template = random.choice(USER_FINAL_TEMPLATES)
            user_text = template.format(value=segment)
        else:
            # Check if previous was correction
            prev_was_correction = i > 0 and segments_with_errors[i - 1][1]
            if prev_was_correction:
                template = random.choice(USER_CONFIRM_CONTINUE_TEMPLATES)
            else:
                template = random.choice(USER_CONTINUE_TEMPLATES)
            user_text = template.format(value=segment)
        
        user_turn = SegmentedTurn(
            role="user",
            text=user_text,
            segment={
                "slot": slot_name,
                "value": segment,
                "idx": seg_idx if not is_correction else seg_idx - 1,
                "total": total_segments,
                "is_correction": is_correction,
            },
        )
        turns.append(user_turn)
        
        if not is_correction:
            seg_idx += 1
        
        # Assistant turn
        if is_correction:
            template = random.choice(ASST_CORRECTION_CONFIRM_TEMPLATES)
            asst_text = template.format(value=segment)
        elif is_last:
            template = random.choice(ASST_FINAL_TEMPLATES)
            asst_text = template.format(
                slot_name=natural_name,
                full_value=slot_value,
            )
        else:
            template = random.choice(ASST_CONFIRM_TEMPLATES)
            asst_text = template.format(value=segment)
        
        asst_turn = SegmentedTurn(role="assistant", text=asst_text)
        turns.append(asst_turn)
    
    return turns


def find_segmentable_slots(
    state: dict,
    dataset: str,
) -> list[tuple[str, str, str]]:
    """Find slots that can be segmented in a dialogue state.
    
    Args:
        state: Dialogue state dict {domain: {slot: value}}
        dataset: Dataset name ("sgd", "abcd", "emowoz", "tm2")
    
    Returns:
        List of (slot_name, slot_value, slot_type) tuples
    """
    segmentable = SEGMENTABLE_SLOTS.get(dataset, {})
    result = []
    
    for domain, slots in state.items():
        if not isinstance(slots, dict):
            continue
        for slot_name, slot_value in slots.items():
            if not slot_value or not isinstance(slot_value, str):
                continue
            
            # Check if slot is segmentable
            slot_type = segmentable.get(slot_name)
            if slot_type:
                result.append((slot_name, slot_value, slot_type))
            
            # Also check domain.slot format
            full_name = f"{domain}.{slot_name}"
            slot_type = segmentable.get(full_name)
            if slot_type:
                result.append((full_name, slot_value, slot_type))
    
    return result
