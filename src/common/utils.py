"""Common utilities for dataset loading and processing."""

import json
from pathlib import Path


def load_dialogues(
    data_dir: str, split: str, limit: int | None = None
) -> list[tuple[Path, list[dict]]]:
    """Load dialogue files as (path, dialogues) pairs, preserving file boundaries.

    Args:
        data_dir: Base directory containing split subdirectories
        split: Dataset split (train/dev/test)
        limit: Maximum number of dialogues to load

    Returns:
        List of (file_path, dialogues_list) tuples
    """
    dialogue_dir = Path(data_dir) / split
    results: list[tuple[Path, list[dict]]] = []
    loaded = 0

    for dialogue_file in sorted(dialogue_dir.glob("dialogues_*.json")):
        with open(dialogue_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            if limit is None:
                dialogues = data
            else:
                remaining = limit - loaded
                if remaining <= 0:
                    break
                dialogues = data[:remaining]
        else:
            if limit is not None and loaded >= limit:
                break
            dialogues = [data]

        results.append((dialogue_file, dialogues))
        loaded += len(dialogues)

        if limit is not None and loaded >= limit:
            break

    return results


def load_prompt(name: str) -> str:
    """Load a prompt template from src/prompts directory.
    
    Args:
        name: Filename of the prompt (e.g., "correction.txt")
    
    Returns:
        Prompt text content
    """
    prompt_dir = Path(__file__).parent.parent / "prompts"
    prompt_path = prompt_dir / name
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    
    with open(prompt_path, "r") as f:
        return f.read()

