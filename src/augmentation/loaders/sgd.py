"""SGD dataset loader with user goal from slot_values."""

import json
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue, Goal, StructuredGoal
from .base import BaseLoader


class SGDLoader(BaseLoader):
    """Load SGD dialogues with goals reconstructed from slot_values."""

    def __init__(self, data_dir: Path, split: str = "train"):
        super().__init__(data_dir, split)

    @property
    def name(self) -> str:
        return "sgd"

    def load(self) -> Iterator[Dialogue]:
        """Yield SGD dialogues."""
        split_dir = self.data_dir / self.split

        # SGD has multiple dialogue files per split
        for dlg_path in sorted(split_dir.glob("dialogues_*.json")):
            with open(dlg_path, "r") as f:
                dialogues = json.load(f)

            for dlg in dialogues:
                dlg_id = dlg["dialogue_id"]
                services = dlg.get("services", [])
                turns_data = dlg.get("turns", [])

                # Extract turns
                turns = []
                for turn in turns_data:
                    role = "user" if turn["speaker"] == "USER" else "assistant"
                    text = turn["utterance"]
                    turns.append({"role": role, "text": text})

                # Extract goal from last USER turn's slot_values
                goal = self._extract_goal(turns_data, services)

                # Extract final dialogue state
                state = self._extract_state(turns_data)

                yield Dialogue(
                    id=dlg_id,
                    source=self.name,
                    turns=turns,
                    goal=goal,
                    state=state,
                    metadata={"services": services},
                )

    def _extract_goal(self, turns: list[dict], services: list[str]) -> Goal:
        """Extract user goal from accumulated slot_values."""
        # Collect all slot values from USER turns
        slot_values = {}
        active_intents = {}

        for turn in turns:
            if turn["speaker"] != "USER":
                continue
            for frame in turn.get("frames", []):
                service = frame.get("service", "")
                state = frame.get("state", {})

                intent = state.get("active_intent", "")
                if intent and intent != "NONE":
                    active_intents[service] = intent

                for slot, values in state.get("slot_values", {}).items():
                    key = f"{service}.{slot}"
                    slot_values[key] = values[0] if values else ""

        # Build text goal
        text = self._build_goal_text(slot_values, active_intents, services)

        # Build structured goal
        structured = self._build_structured_goal(slot_values, active_intents, services)

        return Goal(text=text, structured=structured)

    def _build_goal_text(
        self,
        slot_values: dict,
        active_intents: dict,
        services: list[str],
    ) -> str:
        """Build natural language goal from slot values."""
        parts = []

        for service in services:
            intent = active_intents.get(service, "")
            domain = service.split("_")[0].lower()

            # Collect slots for this service
            svc_slots = {
                k.split(".", 1)[1]: v
                for k, v in slot_values.items()
                if k.startswith(f"{service}.")
            }

            if not svc_slots and not intent:
                continue

            # Build intent sentence
            if intent:
                intent_phrase = self._intent_to_phrase(intent, domain)
                parts.append(intent_phrase)

            # Build natural language constraints
            if svc_slots:
                constraint_text = self._slots_to_natural(svc_slots, domain)
                if constraint_text:
                    parts.append(constraint_text)

        return " ".join(parts) if parts else "You have a general inquiry."

    def _intent_to_phrase(self, intent: str, domain: str) -> str:
        """Convert intent to natural language phrase."""
        intent_lower = intent.lower()
        
        if "find" in intent_lower and "reserve" in intent_lower:
            return f"You want to find and book a {domain}."
        elif "find" in intent_lower:
            return f"You are looking for a {domain}."
        elif "reserve" in intent_lower or "book" in intent_lower:
            return f"You want to make a reservation at a {domain}."
        elif "search" in intent_lower:
            return f"You want to search for a {domain}."
        elif "get" in intent_lower:
            return f"You want to get information about a {domain}."
        else:
            # CamelCase to words: FindRestaurants -> find restaurants
            words = []
            for i, c in enumerate(intent):
                if c.isupper() and i > 0:
                    words.append(" ")
                words.append(c.lower())
            return f"You want to {''.join(words)}."

    def _slots_to_natural(self, slots: dict, domain: str) -> str:
        """Convert slots to natural language."""
        phrases = []
        
        # Slot name to natural phrasing
        slot_templates = {
            "city": "in {value}",
            "location": "in {value}",
            "area": "in the {value} area",
            "cuisine": "serving {value} food",
            "food": "serving {value}",
            "restaurant_name": "at {value}",
            "hotel_name": "at {value}",
            "name": "called {value}",
            "date": "on {value}",
            "time": "at {value}",
            "party_size": "for {value} people",
            "number_of_seats": "for {value} people",
            "people": "for {value} people",
            "price_range": "with {value} price",
            "pricerange": "with {value} price",
            "destination": "to {value}",
            "origin": "from {value}",
            "departure_date": "departing on {value}",
            "return_date": "returning on {value}",
            "category": "in the {value} category",
            "type": "of type {value}",
        }
        
        for slot, value in slots.items():
            if not value:
                continue
            
            slot_lower = slot.lower()
            template = slot_templates.get(slot_lower)
            
            if template:
                phrases.append(template.format(value=value))
            else:
                # Generic: convert slot_name to "slot name: value"
                readable = slot.replace("_", " ")
                phrases.append(f"with {readable} {value}")
        
        if not phrases:
            return ""
        
        return "You are looking for something " + ", ".join(phrases) + "."

    def _build_structured_goal(
        self,
        slot_values: dict,
        active_intents: dict,
        services: list[str],
    ) -> StructuredGoal:
        """Build structured goal from slot values."""
        domains = []
        intents = []

        for service in services:
            domain = service.split("_")[0]
            domains.append(domain)

            intent_name = active_intents.get(service, "inform")

            svc_slots = {
                k.split(".", 1)[1]: v
                for k, v in slot_values.items()
                if k.startswith(f"{service}.")
            }

            intents.append({
                "domain": domain,
                "intent": intent_name,
                "slots": svc_slots,
                "requests": [],
            })

        return StructuredGoal(domains=list(set(domains)), intents=intents)

    def _extract_state(self, turns: list[dict]) -> dict:
        """Extract final dialogue state from last USER turn."""
        state = {}

        for turn in reversed(turns):
            if turn["speaker"] != "USER":
                continue
            for frame in turn.get("frames", []):
                service = frame.get("service", "")
                turn_state = frame.get("state", {})

                if service not in state:
                    state[service] = {}

                for slot, values in turn_state.get("slot_values", {}).items():
                    state[service][slot] = values[0] if values else ""
            break

        return state
