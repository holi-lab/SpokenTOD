"""SpokenWOZ dataset loader with user goal from MultiWOZ."""

import json
import re
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue, Goal, StructuredGoal
from .base import BaseLoader


class SpokenWOZLoader(BaseLoader):
    """Load SpokenWOZ dialogues with goals from MultiWOZ."""

    def __init__(
        self,
        data_dir: Path,
        multiwoz_dir: Path,
        split: str = "train",
    ):
        super().__init__(data_dir, split)
        self.multiwoz_dir = Path(multiwoz_dir)
        self._multiwoz_cache: dict | None = None

    @property
    def name(self) -> str:
        return "spokenwoz"

    def _load_multiwoz(self) -> dict:
        """Load and cache MultiWOZ data."""
        if self._multiwoz_cache is None:
            mwoz_path = self.multiwoz_dir / "data.json"
            with open(mwoz_path, "r") as f:
                self._multiwoz_cache = json.load(f)
        return self._multiwoz_cache

    def load(self) -> Iterator[Dialogue]:
        """Yield SpokenWOZ dialogues."""
        # SpokenWOZ uses train.json / test.json
        # These files have concatenated JSON objects like {id1: {...}}{id2: {...}}
        file_name = f"{self.split}.json"
        data_path = self.data_dir / file_name

        if not data_path.exists():
            return

        # Parse concatenated JSON using regex
        data = self._parse_concatenated_json(data_path)

        multiwoz = self._load_multiwoz()

        for dlg_id, dlg_data in data.items():
            # Extract goal from SpokenWOZ's own goal field or MultiWOZ fallback
            goal = self._extract_goal_from_data(dlg_id, dlg_data, multiwoz)

            # Extract turns
            turns = []
            log = dlg_data.get("log", [])

            for i, turn_data in enumerate(log):
                role = "user" if i % 2 == 0 else "assistant"
                text = turn_data.get("text", "")
                turns.append({"role": role, "text": text})

            # Extract dialogue state from last turn
            state = {}
            if log:
                last_turn = log[-1]
                meta = last_turn.get("metadata", {})
                for domain, domain_state in meta.items():
                    state[domain] = {}
                    if "semi" in domain_state:
                        state[domain].update(domain_state["semi"])
                    if "book" in domain_state:
                        state[domain].update({f"book_{k}": v for k, v in domain_state["book"].items()})

            yield Dialogue(
                id=dlg_id,
                source=self.name,
                turns=turns,
                goal=goal,
                state=state,
                metadata={"has_native_crossturn": True},
            )

    def _parse_concatenated_json(self, path: Path) -> dict:
        """Parse SpokenWOZ's concatenated JSON format.
        
        The format is: {"id1": {...}}{"id2": {...}} with no newlines between.
        """
        with open(path, "r") as f:
            content = f.read()
        
        # SpokenWOZ format starts with {"dialogue_id": {data}}
        # Split by pattern: }{"MUL or }{SNG
        result = {}
        
        # Use regex to find all dialogue IDs and extract their content
        # Pattern: "DIALOGUE_ID": { ... entire object ... }
        pattern = r'"(MUL\d+|SNG\d+)"\s*:\s*'
        
        parts = re.split(pattern, content)
        # parts will be: ['', 'MUL0001', '{...}', 'MUL0002', '{...}', ...]
        
        i = 1
        while i < len(parts) - 1:
            dlg_id = parts[i]
            json_str = parts[i + 1]
            
            # Find the matching closing brace for this dialogue
            # Simple approach: take until we see the pattern again or EOF
            try:
                # Parse just the value part (the {...})
                obj = json.loads(json_str.rstrip('}') + '}')
                result[dlg_id] = obj
            except json.JSONDecodeError:
                # Try to extract valid JSON
                brace_count = 0
                end_idx = 0
                for j, c in enumerate(json_str):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = j + 1
                            break
                
                if end_idx > 0:
                    try:
                        obj = json.loads(json_str[:end_idx])
                        result[dlg_id] = obj
                    except json.JSONDecodeError:
                        pass
            
            i += 2
        
        return result

    def _extract_goal_from_data(
        self, dlg_id: str, dlg_data: dict, multiwoz: dict
    ) -> Goal:
        """Extract user goal from SpokenWOZ data or MultiWOZ fallback."""
        # SpokenWOZ has its own goal field
        goal_data = dlg_data.get("goal", {})
        
        if goal_data:
            # Use SpokenWOZ's embedded goal
            messages = goal_data.get("message", [])
            if not messages:
                # Build from domain goals
                messages = self._build_messages_from_goal(goal_data)
            
            text = self._messages_to_text(messages)
            structured = self._goal_to_structured(goal_data)
            return Goal(text=text, structured=structured)
        
        # Fallback to MultiWOZ
        return self._extract_goal(dlg_id, multiwoz)

    def _build_messages_from_goal(self, goal_data: dict) -> list[str]:
        """Build goal messages from domain-specific goals."""
        messages = []
        for domain in ["restaurant", "hotel", "attraction", "taxi", "train"]:
            if domain not in goal_data or not goal_data[domain]:
                continue
            d_goal = goal_data[domain]
            if "info" in d_goal:
                info = d_goal["info"]
                slots = ", ".join(f"{k}: {v}" for k, v in info.items() if v)
                if slots:
                    messages.append(f"Find a {domain} with {slots}.")
            if "book" in d_goal:
                book = d_goal["book"]
                slots = ", ".join(f"{k}: {v}" for k, v in book.items() if v)
                if slots:
                    messages.append(f"Book with {slots}.")
        return messages

    def _extract_goal(self, dlg_id: str, multiwoz: dict) -> Goal:
        """Extract user goal from MultiWOZ by dialogue ID."""
        mwoz_data = multiwoz.get(dlg_id, {})
        goal_data = mwoz_data.get("goal", {})
        messages = goal_data.get("message", [])

        text = self._messages_to_text(messages)
        structured = self._goal_to_structured(goal_data)

        return Goal(text=text, structured=structured)

    def _messages_to_text(self, messages: list[str]) -> str:
        """Convert MultiWOZ messages to natural language goal."""
        clean_msgs = []
        for msg in messages:
            clean = re.sub(r"<[^>]+>", "", msg).strip()
            # Add period if not already ending with punctuation
            if clean and clean[-1] not in ".!?":
                clean += "."
            clean_msgs.append(clean)
        return " ".join(clean_msgs)

    def _goal_to_structured(self, goal_data: dict) -> StructuredGoal:
        """Convert MultiWOZ goal to structured format."""
        domains = []
        intents = []

        for domain in ["restaurant", "hotel", "attraction", "taxi", "train", "hospital", "police", "profile"]:
            if domain not in goal_data or not goal_data[domain]:
                continue

            domains.append(domain)
            d_goal = goal_data[domain]

            intent = {
                "domain": domain,
                "intent": self._infer_intent(d_goal),
                "slots": {},
                "requests": [],
            }

            if "info" in d_goal:
                intent["slots"].update(d_goal["info"])
            if "book" in d_goal:
                intent["slots"].update({f"book_{k}": v for k, v in d_goal["book"].items()})
            if "reqt" in d_goal:
                intent["requests"] = d_goal["reqt"]

            intents.append(intent)

        return StructuredGoal(domains=domains, intents=intents)

    def _infer_intent(self, domain_goal: dict) -> str:
        """Infer intent from domain goal."""
        has_book = "book" in domain_goal and domain_goal["book"]
        has_info = "info" in domain_goal and domain_goal["info"]

        if has_book and has_info:
            return "find_and_book"
        elif has_book:
            return "book"
        elif has_info:
            return "find"
        else:
            return "inform"
