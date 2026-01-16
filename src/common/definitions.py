"""Common data model definitions for SGD-style dialogues."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    act: str = Field(
        description="Dialogue act: INFORM_INTENT, INFORM, REQUEST, AFFIRM, NEGATE, SELECT, REQUEST_ALTS, THANK_YOU, GOODBYE"
    )
    canonical_values: List[str] = Field(
        default_factory=list, description="Canonical form of values"
    )
    slot: str = Field(description="Slot name, empty string if not applicable")
    values: List[str] = Field(default_factory=list, description="Slot values mentioned")


class Slot(BaseModel):
    slot: str = Field(description="Slot name (e.g., 'city', 'cuisine')")
    start: int = Field(description="Start character index")
    exclusive_end: int = Field(description="End character index (exclusive)")


class DialogueState(BaseModel):
    """Current dialogue state"""

    active_intent: str = Field(description="Current active intent name")
    requested_slots: List[str] = Field(default_factory=list, description="Slots requested by user")
    slot_values: Dict[str, List[str]] = Field(
        default_factory=dict, description="Accumulated slot-value pairs"
    )


class UserFrame(BaseModel):
    """Single frame annotation for a turn"""

    actions: List[Action] = Field(description="List of dialogue actions in this turn")
    service: str = Field(description="Service name (e.g., 'Restaurants_1')")
    slots: List[Slot] = Field(default_factory=list, description="Slots in utterance")
    state: Optional[DialogueState] = Field(default=None, description="Dialogue state")


class Turn(BaseModel):
    """Single dialogue turn in SGD format"""

    speaker: Literal["USER", "SYSTEM"] = Field(description="Speaker role")
    utterance: str = Field(description="The actual text spoken")
    frames: List[UserFrame] = Field(description="Frame annotations")


class SGDDialogue(BaseModel):
    """Complete SGD-formatted dialogue"""

    dialogue_id: str = Field(description="Unique dialogue identifier")
    services: List[str] = Field(description="List of services used in dialogue")
    turns: List[Turn] = Field(description="All dialogue turns")
