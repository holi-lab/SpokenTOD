"""Unit tests for emotion tagging module."""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from augmentation.emotion.prompts import (
    build_emotion_prompt,
    build_batch_prompts,
    parse_emotion_response,
    FEWSHOT_EXAMPLES,
)
from augmentation.emotion.tagger import (
    EmotionTagger,
    tag_dialogue_emotions,
)
from augmentation.batch.client import MockBatchClient


class TestEmotionPrompts:
    """Tests for emotion prompts."""

    def test_few_shot_examples_coverage(self):
        """Verify all emotion labels have examples."""
        labels_with_examples = set(label for _, label in FEWSHOT_EXAMPLES)
        
        # Should have examples for all 7 labels (0-6)
        for label in range(7):
            assert label in labels_with_examples, f"Missing example for label {label}"

    def test_build_emotion_prompt(self):
        """Test prompt construction."""
        prompt = build_emotion_prompt("I need a restaurant")
        
        # Should contain instructions
        assert "emotion classifier" in prompt.lower()
        assert "0-6" in prompt or "0:" in prompt
        
        # Should contain the utterance
        assert "I need a restaurant" in prompt
        
        # Should contain examples
        assert "Examples" in prompt

    def test_build_batch_prompts(self):
        """Test batch prompt construction."""
        utterances = ["Hello", "Thank you", "This is frustrating"]
        prompts = build_batch_prompts(utterances)
        
        assert len(prompts) == 3
        
        for i, prompt in enumerate(prompts):
            assert prompt["custom_id"] == f"emotion-{i}"
            assert prompt["method"] == "POST"
            assert prompt["url"] == "/v1/chat/completions"
            assert prompt["body"]["model"] == "gpt-4.1-mini"
            assert len(prompt["body"]["messages"]) == 1


class TestParseEmotionResponse:
    """Tests for response parsing."""

    def test_parse_single_digit(self):
        """Parse simple digit."""
        assert parse_emotion_response("0") == 0
        assert parse_emotion_response("6") == 6
        assert parse_emotion_response("3") == 3

    def test_parse_with_text(self):
        """Parse digit with surrounding text."""
        assert parse_emotion_response("The emotion is 2 (dissatisfied)") == 2
        assert parse_emotion_response("Label: 5") == 5

    def test_parse_invalid(self):
        """Invalid responses should return 0."""
        assert parse_emotion_response("") == 0
        assert parse_emotion_response("invalid") == 0
        assert parse_emotion_response("99") == 0  # First digit 9 is out of range


class TestMockBatchClient:
    """Tests for mock batch client."""

    def test_create_batch(self):
        """Test batch creation."""
        client = MockBatchClient(default_emotion=2)
        
        requests = [
            {"custom_id": "emotion-0", "method": "POST", "url": "/v1/chat/completions", "body": {}},
            {"custom_id": "emotion-1", "method": "POST", "url": "/v1/chat/completions", "body": {}},
        ]
        
        batch_id = client.create_batch(requests)
        assert batch_id.startswith("mock_batch_")

    def test_get_results(self):
        """Test getting results."""
        client = MockBatchClient(default_emotion=4)
        
        requests = [{"custom_id": "emotion-0"}, {"custom_id": "emotion-1"}]
        batch_id = client.create_batch(requests)
        
        results = client.get_results(batch_id)
        assert len(results) == 2
        
        parsed = client.parse_results(results)
        assert parsed["emotion-0"] == "4"
        assert parsed["emotion-1"] == "4"


class TestEmotionTagger:
    """Tests for EmotionTagger class."""

    def test_tag_utterances(self):
        """Test utterance tagging."""
        tagger = EmotionTagger(use_mock=True)
        
        utterances = ["Hello", "Thank you"]
        emotions = tagger.tag_utterances(utterances)
        
        assert len(emotions) == 2
        for emo in emotions:
            assert "label" in emo
            assert "token" in emo
            assert emo["label"] >= 0
            assert emo["token"].startswith("[")

    def test_should_tag(self):
        """Test dataset exclusion logic."""
        tagger = EmotionTagger(use_mock=True)
        
        # EmoWOZ should be excluded
        assert not tagger.should_tag("emowoz")
        
        # Others should be included
        assert tagger.should_tag("sgd")
        assert tagger.should_tag("abcd")
        assert tagger.should_tag("spokenwoz")
        assert tagger.should_tag("tm2")


class TestTagDialogueEmotions:
    """Tests for dialogue emotion tagging."""

    def test_with_existing_labels(self):
        """Test with pre-existing labels (EmoWOZ)."""
        turns = [
            {"role": "user", "text": "Hello"},
            {"role": "assistant", "text": "Hi there"},
            {"role": "user", "text": "Thanks"},
        ]
        existing_labels = [0, -1, 6]  # -1 for assistant
        
        result = tag_dialogue_emotions(turns, "emowoz", existing_labels=existing_labels)
        
        assert result[0]["emotion"]["label"] == 0
        assert result[0]["emotion"]["token"] == "[neutral]"
        assert "emotion" not in result[1]  # Assistant turn
        assert result[2]["emotion"]["label"] == 6
        assert result[2]["emotion"]["token"] == "[happy]"

    def test_with_mock_tagger(self):
        """Test with mock tagger."""
        turns = [
            {"role": "user", "text": "I need help"},
            {"role": "assistant", "text": "Sure"},
            {"role": "user", "text": "Thank you"},
        ]
        
        tagger = EmotionTagger(use_mock=True)
        result = tag_dialogue_emotions(turns, "sgd", tagger=tagger)
        
        # User turns should have emotions
        assert "emotion" in result[0]
        assert "emotion" in result[2]
        # Assistant turn should not
        assert "emotion" not in result[1]

    def test_excluded_dataset(self):
        """Test excluded dataset (EmoWOZ without labels)."""
        turns = [
            {"role": "user", "text": "Hello"},
        ]
        
        result = tag_dialogue_emotions(turns, "emowoz")
        
        # Should return unchanged since EmoWOZ is excluded
        assert result == turns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
