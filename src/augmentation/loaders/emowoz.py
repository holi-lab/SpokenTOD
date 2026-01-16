"""EmoWOZ dataset loader with user goal from MultiWOZ."""

import json
import re
from pathlib import Path
from typing import Iterator

from augmentation.schema import Dialogue, Goal, StructuredGoal, Turn
from .base import BaseLoader


class EmoWOZLoader(BaseLoader):
    """Load EmoWOZ dialogues with emotion labels and goals from MultiWOZ."""

    def __init__(
        self,
        data_dir: Path,
        multiwoz_dir: Path,
        split: str = "train",
    ):
        super().__init__(data_dir, split)
        self.multiwoz_dir = Path(multiwoz_dir)
        self._multiwoz_cache: dict | None = None
        self._split_ids: set | None = None

    @property
    def name(self) -> str:
        return "emowoz"

    def _load_multiwoz(self) -> dict:
        """Load and cache MultiWOZ data."""
        if self._multiwoz_cache is None:
            mwoz_path = self.multiwoz_dir / "data.json"
            with open(mwoz_path, "r") as f:
                self._multiwoz_cache = json.load(f)
        return self._multiwoz_cache

    def load(self) -> Iterator[Dialogue]:
        """Yield EmoWOZ dialogues."""
        # Load EmoWOZ data (MultiWOZ subset)
        emowoz_path = self.data_dir / "emowoz-multiwoz.json"
        if not emowoz_path.exists():
            return
            
        with open(emowoz_path, "r") as f:
            data = json.load(f)

        multiwoz = self._load_multiwoz()
        
        # Get all dialogue IDs and split by ratio (80/10/10)
        all_ids = sorted(data.keys())
        total = len(all_ids)
        
        if self.split == "train":
            split_ids = all_ids[:int(total * 0.8)]
        elif self.split in ("valid", "validation", "dev"):
            split_ids = all_ids[int(total * 0.8):int(total * 0.9)]
        else:  # test
            split_ids = all_ids[int(total * 0.9):]

        for dlg_id in split_ids:
            dlg_data = data[dlg_id]

            # Get goal from MultiWOZ
            goal = self._extract_goal(dlg_id, multiwoz)

            # Extract turns with emotion labels
            # log is a list of dicts with keys: text, emotion, dialog_act, span_info
            turns = []
            emotion_labels = []
            log = dlg_data.get("log", [])

            for i, turn_data in enumerate(log):
                role = "user" if i % 2 == 0 else "assistant"
                text = turn_data.get("text", "")
                
                # Emotion is a list of annotations, extract the final emotion value
                emotion_raw = turn_data.get("emotion", [])
                if role == "assistant" or not emotion_raw:
                    emotion = -1  # System turn or no annotation
                else:
                    # Get emotion from last item in list (usually has 'emotion' key)
                    emotion = self._extract_emotion_value(emotion_raw)
                
                turns.append({"role": role, "text": text})
                emotion_labels.append(emotion)

            yield Dialogue(
                id=dlg_id,
                source=self.name,
                turns=turns,
                goal=goal,
                emotion_labels=emotion_labels,
            )

    def _extract_emotion_value(self, emotion_raw: list) -> int:
        """Extract emotion value from EmoWOZ annotation list.
        
        Format: [{'annotator': '...', 'annotation': 0}, ..., {'emotion': 0, 'sentiment': 0}]
        The last item typically has the final 'emotion' value.
        """
        if not emotion_raw:
            return 0
        
        # Try to find 'emotion' key in last item
        last_item = emotion_raw[-1]
        if isinstance(last_item, dict):
            if 'emotion' in last_item:
                return int(last_item['emotion'])
            if 'annotation' in last_item:
                return int(last_item['annotation'])
        
        # If first item has annotation
        first_item = emotion_raw[0]
        if isinstance(first_item, dict) and 'annotation' in first_item:
            return int(first_item['annotation'])
        
        return 0

    def _extract_goal(self, dlg_id: str, multiwoz: dict) -> Goal:
        """Extract user goal from MultiWOZ by dialogue ID."""
        mwoz_data = multiwoz.get(dlg_id, {})
        goal_data = mwoz_data.get("goal", {})
        messages = goal_data.get("message", [])

        # Build natural language goal text
        text = self._messages_to_text(messages)

        # Build structured goal
        structured = self._goal_to_structured(goal_data)

        return Goal(text=text, structured=structured)

    def _messages_to_text(self, messages: list[str]) -> str:
        """Convert MultiWOZ messages to natural language goal."""
        # Clean HTML spans and ensure proper sentence endings
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

        for domain in ["restaurant", "hotel", "attraction", "taxi", "train", "hospital", "police"]:
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

            # Info slots (constraints)
            if "info" in d_goal:
                intent["slots"].update(d_goal["info"])

            # Book slots
            if "book" in d_goal:
                intent["slots"].update({f"book_{k}": v for k, v in d_goal["book"].items()})

            # Requested slots
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
