"""Unit tests for dataset loaders."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from augmentation.loaders.base import BaseLoader
from augmentation.loaders.emowoz import EmoWOZLoader
from augmentation.loaders.spokenwoz import SpokenWOZLoader
from augmentation.loaders.sgd import SGDLoader
from augmentation.loaders.abcd import ABCDLoader
from augmentation.loaders.tm2 import TM2Loader
from augmentation.schema import Goal, StructuredGoal


# Sample test data
SAMPLE_MULTIWOZ = {
    "MUL0001.json": {
        "goal": {
            "message": [
                "You are looking for info in Cambridge",
                "Find a <span class='emphasis'>restaurant</span> in the <span class='emphasis'>centre</span>",
            ],
            "restaurant": {
                "info": {"food": "chinese", "area": "centre"},
                "book": {"people": "4", "day": "friday"},
                "reqt": ["phone"],
            },
        }
    }
}

SAMPLE_EMOWOZ = {
    "MUL0001.json": {
        "log": {
            "text": ["I'm looking for a restaurant", "Sure, what type of food?"],
            "emotion": [0, -1],
        }
    }
}

SAMPLE_SGD = [
    {
        "dialogue_id": "1_00000",
        "services": ["Restaurants_1"],
        "turns": [
            {
                "speaker": "USER",
                "utterance": "Find me a restaurant",
                "frames": [{
                    "service": "Restaurants_1",
                    "state": {
                        "active_intent": "FindRestaurants",
                        "slot_values": {"city": ["San Jose"], "cuisine": ["Mexican"]},
                    },
                }],
            },
            {
                "speaker": "SYSTEM",
                "utterance": "I found some options",
                "frames": [],
            },
        ],
    }
]

SAMPLE_ABCD = {
    "train": [
        {
            "convo_id": "abcd_001",
            "scenario": {
                "flow": "order_issue",
                "subflow": "missing_item",
                "personal": {"customer_name": "John Smith", "phone_number": "555-1234"},
                "order": {"order_id": "ORD123", "product_names": ["Blue Jacket"]},
                "product": {},
            },
            "delexed": [
                {"speaker": "customer", "text": "My order is missing an item"},
                {"speaker": "agent", "text": "I can help with that"},
            ],
        }
    ]
}

SAMPLE_TM2 = [
    {
        "conversation_id": "dlg-001",
        "instruction_id": "flight-6",
        "utterances": [
            {
                "index": 0,
                "speaker": "USER",
                "text": "I need a flight to New York",
                "segments": [{
                    "text": "New York",
                    "annotations": [{"name": "flight.destination"}],
                }],
            },
            {
                "index": 1,
                "speaker": "ASSISTANT",
                "text": "When would you like to travel?",
                "segments": [],
            },
        ],
    }
]


class TestEmoWOZLoader:
    """Tests for EmoWOZ loader."""

    def test_messages_to_text(self):
        """Verify HTML span removal from messages."""
        loader = EmoWOZLoader(Path("."), Path("."))
        messages = [
            "Find a <span class='emphasis'>restaurant</span>",
            "in the <span>centre</span>",
        ]
        result = loader._messages_to_text(messages)
        assert "restaurant" in result
        assert "centre" in result
        assert "<span" not in result

    def test_goal_to_structured(self):
        """Verify structured goal extraction."""
        loader = EmoWOZLoader(Path("."), Path("."))
        goal_data = {
            "restaurant": {
                "info": {"food": "chinese"},
                "book": {"people": "4"},
                "reqt": ["phone"],
            }
        }
        result = loader._goal_to_structured(goal_data)
        assert "restaurant" in result.domains
        assert len(result.intents) == 1
        assert result.intents[0]["slots"]["food"] == "chinese"
        assert result.intents[0]["requests"] == ["phone"]

    def test_infer_intent(self):
        """Verify intent inference logic."""
        loader = EmoWOZLoader(Path("."), Path("."))

        # Book + info -> find_and_book
        assert loader._infer_intent({"info": {"a": 1}, "book": {"b": 2}}) == "find_and_book"
        # Only book -> book
        assert loader._infer_intent({"book": {"b": 2}}) == "book"
        # Only info -> find
        assert loader._infer_intent({"info": {"a": 1}}) == "find"


class TestSGDLoader:
    """Tests for SGD loader."""

    def test_build_goal_text(self):
        """Verify goal text generation."""
        loader = SGDLoader(Path("."))
        slot_values = {
            "Restaurants_1.city": "San Jose",
            "Restaurants_1.cuisine": "Mexican",
        }
        active_intents = {"Restaurants_1": "FindRestaurants"}
        services = ["Restaurants_1"]

        result = loader._build_goal_text(slot_values, active_intents, services)
        # Check for natural language output
        assert "looking for" in result.lower() or "restaurant" in result.lower()
        assert "San Jose" in result

    def test_build_structured_goal(self):
        """Verify structured goal generation."""
        loader = SGDLoader(Path("."))
        slot_values = {"Restaurants_1.city": "San Jose"}
        active_intents = {"Restaurants_1": "FindRestaurants"}
        services = ["Restaurants_1"]

        result = loader._build_structured_goal(slot_values, active_intents, services)
        assert "Restaurants" in result.domains
        assert len(result.intents) == 1
        assert result.intents[0]["slots"]["city"] == "San Jose"


class TestABCDLoader:
    """Tests for ABCD loader."""

    def test_build_goal_text(self):
        """Verify goal text generation."""
        loader = ABCDLoader(Path("."))
        result = loader._build_goal_text(
            flow="order_issue",
            subflow="missing_item",
            personal={"customer_name": "John"},
            order={"product_names": ["Jacket"]},
            product={},
        )
        # Check for natural language - "order issue" or "issue with your order"
        assert "order" in result.lower() and "issue" in result.lower()
        # Check for subflow mention
        assert "missing" in result.lower()
        assert "John" in result

    def test_build_structured_goal(self):
        """Verify structured goal generation."""
        loader = ABCDLoader(Path("."))
        result = loader._build_structured_goal(
            flow="order_issue",
            subflow="missing_item",
            personal={"customer_name": "John"},
            order={},
            product={},
        )
        assert result.domains == ["customer_service"]
        assert result.intents[0]["intent"] == "order_issue.missing_item"
        assert result.intents[0]["slots"]["customer_name"] == "John"


class TestTM2Loader:
    """Tests for TM-2 loader."""

    def test_build_goal_text(self):
        """Verify goal text generation."""
        loader = TM2Loader(Path("."))
        result = loader._build_goal_text(
            domain="flights",
            instruction_id="flight-6",
            slots={"flight.destination": "New York"},
        )
        # Check for natural language - "flight" in some form
        assert "flight" in result.lower()
        assert "New York" in result

    def test_build_structured_goal(self):
        """Verify structured goal generation."""
        loader = TM2Loader(Path("."))
        result = loader._build_structured_goal(
            domain="flights",
            instruction_id="flight-6",
            slots={"flight.destination": "New York"},
        )
        assert result.domains == ["flights"]
        assert result.intents[0]["intent"] == "flight"
        assert result.intents[0]["slots"]["flight.destination"] == "New York"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
