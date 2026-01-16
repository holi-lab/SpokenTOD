"""Data loader for SGD dataset."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_sgd_dialogues(
    data_dir: str,
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Load dialogues from SGD dataset.

    Args:
        data_dir: Path to SGD dataset root directory
        split: Dataset split ("train", "dev", "test")
        limit: Maximum number of dialogues to load (None for all)

    Yields:
        Dialogue dictionaries
    """
    split_path = Path(data_dir) / split
    dialogue_count = 0

    for file_path in sorted(split_path.glob("dialogues_*.json")):
        with open(file_path, encoding="utf-8") as f:
            dialogues = json.load(f)

            for dialogue in dialogues:
                if limit is not None and dialogue_count >= limit:
                    return
                yield dialogue
                dialogue_count += 1


def extract_user_turns(dialogue: dict[str, Any]) -> Iterator[dict[str, Any]]:
    dialogue_id = dialogue["dialogue_id"]

    for turn_idx, turn in enumerate(dialogue["turns"]):
        if turn["speaker"] == "USER":
            yield {
                "dialogue_id": dialogue_id,
                "turn_id": turn_idx,
                "utterance": turn["utterance"],
                "frames": turn.get("frames", []),
            }


def count_user_turns(data_dir: str, split: str = "train") -> int:
    """Count total user turns in a split for progress tracking."""
    count = 0
    for dialogue in load_sgd_dialogues(data_dir, split):
        for _ in extract_user_turns(dialogue):
            count += 1
    return count
