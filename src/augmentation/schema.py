"""Data structures for augmentation pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """Single dialogue turn."""
    role: str  # "user" or "assistant"
    text: str
    tagged: str | None = None
    emotion: dict | None = None  # {"label": int, "token": str}
    disfluency: list[dict] | None = None
    segment: dict | None = None  # cross-turn segment info


@dataclass
class StructuredGoal:
    """Structured user goal with intents and slots."""
    domains: list[str]
    intents: list[dict]  # [{"domain", "intent", "slots", "requests"}]


@dataclass
class Goal:
    """User goal with text and structured form."""
    text: str
    structured: StructuredGoal


@dataclass
class Dialogue:
    """Raw dialogue from source dataset."""
    id: str
    source: str
    turns: list[dict]
    goal: Goal | None = None
    state: dict = field(default_factory=dict)
    emotion_labels: list[int] | None = None  # For EmoWOZ
    metadata: dict = field(default_factory=dict)


@dataclass
class AugmentedDialogue:
    """Augmented dialogue with all features applied."""
    id: str
    source: str
    goal: dict  # {"text": str, "structured": dict}
    turns: list[Turn]
    state: dict
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "dialogue_id": self.id,
            "source": self.source,
            "goal": self.goal,
            "turns": [
                {k: v for k, v in {
                    "role": t.role,
                    "text": t.text,
                    "tagged": t.tagged,
                    "emotion": t.emotion,
                    "disfluency": t.disfluency,
                    "segment": t.segment,
                }.items() if v is not None}
                for t in self.turns
            ],
            "state": self.state,
        }
