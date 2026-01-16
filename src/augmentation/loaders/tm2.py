"""TaskMaster-2 dataset loader with user goal from segments."""

import json
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue, Goal, StructuredGoal
from .base import BaseLoader


# Domain file mapping
TM2_DOMAINS = {
    "restaurants": "restaurant-search.json",
    "food_ordering": "food-ordering.json",
    "hotels": "hotels.json",
    "flights": "flights.json",
    "movies": "movies.json",
    "music": "music.json",
    "sports": "sports.json",
}


class TM2Loader(BaseLoader):
    """Load TaskMaster-2 dialogues with goals from segments."""

    def __init__(self, data_dir: Path, split: str = "train", domains: list[str] | None = None):
        super().__init__(data_dir, split)
        self.domains = domains or list(TM2_DOMAINS.keys())

    @property
    def name(self) -> str:
        return "tm2"

    def load(self) -> Iterator[Dialogue]:
        """Yield TM-2 dialogues."""
        # TM-2 doesn't have split files, we'll create our own split
        for domain in self.domains:
            file_name = TM2_DOMAINS.get(domain)
            if not file_name:
                continue

            data_path = self.data_dir / "data" / file_name
            if not data_path.exists():
                continue

            with open(data_path, "r") as f:
                dialogues = json.load(f)

            # Simple split: 80% train, 10% valid, 10% test
            total = len(dialogues)
            if self.split == "train":
                dialogues = dialogues[:int(total * 0.8)]
            elif self.split in ("valid", "validation", "dev"):
                dialogues = dialogues[int(total * 0.8):int(total * 0.9)]
            else:  # test
                dialogues = dialogues[int(total * 0.9):]

            for dlg in dialogues:
                dlg_id = dlg.get("conversation_id", "")
                instruction_id = dlg.get("instruction_id", "")

                # Extract goal from segments
                goal = self._extract_goal(dlg, domain, instruction_id)

                # Extract turns
                turns = []
                for utt in dlg.get("utterances", []):
                    role = "user" if utt.get("speaker") == "USER" else "assistant"
                    text = utt.get("text", "")
                    turns.append({"role": role, "text": text})

                # Extract state from segments
                state = self._extract_state(dlg, domain)

                yield Dialogue(
                    id=dlg_id,
                    source=self.name,
                    turns=turns,
                    goal=goal,
                    state=state,
                    metadata={"domain": domain, "instruction_id": instruction_id},
                )

    def _extract_goal(self, dlg: dict, domain: str, instruction_id: str) -> Goal:
        """Extract user goal from segment annotations."""
        slots = {}

        for utt in dlg.get("utterances", []):
            if utt.get("speaker") != "USER":
                continue
            for seg in utt.get("segments", []):
                for ann in seg.get("annotations", []):
                    slot_name = ann.get("name", "")
                    slot_value = seg.get("text", "")
                    if slot_name and slot_value:
                        slots[slot_name] = slot_value

        # Build text goal
        text = self._build_goal_text(domain, instruction_id, slots)

        # Build structured goal
        structured = self._build_structured_goal(domain, instruction_id, slots)

        return Goal(text=text, structured=structured)

    def _build_goal_text(self, domain: str, instruction_id: str, slots: dict) -> str:
        """Build natural language goal."""
        parts = []

        # Domain-specific intro
        domain_intros = {
            "restaurants": "You want to find a restaurant.",
            "food_ordering": "You want to order food.",
            "hotels": "You are looking for a hotel.",
            "flights": "You need to book a flight.",
            "movies": "You want to find a movie.",
            "music": "You are looking for music.",
            "sports": "You want sports information.",
        }
        
        intro = domain_intros.get(domain, f"You need help with {domain.replace('_', ' ')}.")
        parts.append(intro)

        if slots:
            # Convert slots to natural descriptions
            slot_phrases = []
            for slot, value in list(slots.items())[:5]:
                # Extract the actual slot name (e.g., "restaurant.name" -> "name")
                slot_key = slot.split(".")[-1].lower().replace("_", " ")
                
                if "name" in slot_key:
                    slot_phrases.append(f"called {value}")
                elif "location" in slot_key or "city" in slot_key:
                    slot_phrases.append(f"in {value}")
                elif "date" in slot_key:
                    slot_phrases.append(f"on {value}")
                elif "time" in slot_key:
                    slot_phrases.append(f"at {value}")
                elif "party" in slot_key or "people" in slot_key or "guests" in slot_key:
                    slot_phrases.append(f"for {value} people")
                elif "cuisine" in slot_key or "food" in slot_key:
                    slot_phrases.append(f"serving {value}")
                else:
                    slot_phrases.append(f"with {slot_key} {value}")
            
            if slot_phrases:
                parts.append("You are looking for something " + ", ".join(slot_phrases) + ".")

        return " ".join(parts)

    def _build_structured_goal(
        self,
        domain: str,
        instruction_id: str,
        slots: dict,
    ) -> StructuredGoal:
        """Build structured goal."""
        # Infer intent from instruction_id (e.g., "flight-6" -> "flight")
        intent = instruction_id.split("-")[0] if instruction_id else domain

        return StructuredGoal(
            domains=[domain],
            intents=[{
                "domain": domain,
                "intent": intent,
                "slots": slots,
                "requests": [],
            }],
        )

    def _extract_state(self, dlg: dict, domain: str) -> dict:
        """Extract dialogue state from segments."""
        state = {domain: {}}

        for utt in dlg.get("utterances", []):
            if utt.get("speaker") != "USER":
                continue
            for seg in utt.get("segments", []):
                for ann in seg.get("annotations", []):
                    slot_name = ann.get("name", "")
                    slot_value = seg.get("text", "")
                    if slot_name and slot_value:
                        state[domain][slot_name] = slot_value

        return state
