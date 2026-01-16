"""Main augmentation pipeline orchestrator."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from augmentation.constants import (
    DATASETS,
    SEGMENTABLE_SLOTS,
    CROSSTURN_EXCLUDED,
    EMOTION_EXCLUDED,
)
from augmentation.schema import Dialogue, AugmentedDialogue, Turn
from augmentation.loaders.emowoz import EmoWOZLoader
from augmentation.loaders.spokenwoz import SpokenWOZLoader
from augmentation.loaders.sgd import SGDLoader
from augmentation.loaders.abcd import ABCDLoader
from augmentation.loaders.tm2 import TM2Loader
from augmentation.segmentation.generator import (
    generate_crossturn_dialogue,
    find_segmentable_slots,
)
from augmentation.emotion.tagger import tag_dialogue_emotions, EmotionTagger
from augmentation.disfluency import inject_disfluency_dialogue, DisfluencyConfig


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    model: str = "gpt-4.1-mini"
    datasets: list[str] = field(default_factory=lambda: DATASETS)
    data_dir: Path = field(default_factory=lambda: Path("datasets"))
    output_dir: Path = field(default_factory=lambda: Path("data"))
    multiwoz_dir: Path | None = None
    batch_size: int = 100
    disfluency_config: DisfluencyConfig = field(default_factory=DisfluencyConfig)
    sample_size: int | None = None  # For testing: limit samples per dataset


class AugmentationPipeline:
    """Main pipeline for voice dataset augmentation."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.emotion_tagger = EmotionTagger(model=config.model)
        
        # Set default MultiWOZ dir if not provided
        if config.multiwoz_dir is None:
            config.multiwoz_dir = config.data_dir / "MultiWOZ_2.1"

    def _get_loader(self, dataset: str, split: str):
        """Get appropriate loader for dataset."""
        cfg = self.config
        
        if dataset == "emowoz":
            return EmoWOZLoader(
                cfg.data_dir / "EmoWOZ",
                cfg.multiwoz_dir,
                split=split,
            )
        elif dataset == "spokenwoz":
            return SpokenWOZLoader(
                cfg.data_dir / "SpokenWOZ",
                cfg.multiwoz_dir,
                split=split,
            )
        elif dataset == "sgd":
            return SGDLoader(cfg.data_dir / "dstc8-schema-guided-dialogue", split=split)
        elif dataset == "abcd":
            return ABCDLoader(cfg.data_dir / "abcd", split=split)
        elif dataset == "tm2":
            return TM2Loader(cfg.data_dir / "TM-2-2020", split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def process_dialogue(self, dialogue: Dialogue) -> AugmentedDialogue:
        """Process a single dialogue through the augmentation pipeline.
        
        Steps:
        1. Apply cross-turn slot segmentation (if applicable)
        2. Apply emotion tagging (if applicable)
        3. Apply disfluency injection
        """
        turns = list(dialogue.turns)
        dataset = dialogue.source
        
        # 1. Cross-turn slot segmentation (skip SpokenWOZ - has native)
        if dataset not in CROSSTURN_EXCLUDED:
            turns = self._apply_crossturn(turns, dialogue.state, dataset)
        
        # 2. Emotion tagging
        if dataset not in EMOTION_EXCLUDED:
            turns = tag_dialogue_emotions(
                turns, dataset, self.emotion_tagger
            )
        elif dialogue.emotion_labels:
            # Use existing labels for EmoWOZ
            turns = tag_dialogue_emotions(
                turns, dataset, existing_labels=dialogue.emotion_labels
            )
        
        # 3. Disfluency injection
        turns = inject_disfluency_dialogue(turns, self.config.disfluency_config)
        
        # Build augmented dialogue
        goal_dict = {
            "text": dialogue.goal.text if dialogue.goal else "",
            "structured": {
                "domains": dialogue.goal.structured.domains if dialogue.goal else [],
                "intents": dialogue.goal.structured.intents if dialogue.goal else [],
            } if dialogue.goal else {},
        }
        
        augmented_turns = []
        for t in turns:
            turn = Turn(
                role=t.get("role", "user"),
                text=t.get("text", ""),
                tagged=t.get("tagged"),
                emotion=t.get("emotion"),
                disfluency=t.get("disfluency"),
                segment=t.get("segment"),
            )
            augmented_turns.append(turn)
        
        return AugmentedDialogue(
            id=dialogue.id,
            source=dataset,
            goal=goal_dict,
            turns=augmented_turns,
            state=dialogue.state,
            metadata=dialogue.metadata,
        )

    def _apply_crossturn(
        self,
        turns: list[dict],
        state: dict,
        dataset: str,
    ) -> list[dict]:
        """Apply cross-turn slot segmentation."""
        # Find segmentable slots
        slots = find_segmentable_slots(state, dataset)
        
        if not slots:
            return turns
        
        # For simplicity, just process one slot per dialogue
        slot_name, slot_value, slot_type = slots[0]
        
        # Generate cross-turn dialogue
        crossturn_turns = generate_crossturn_dialogue(
            slot_name, slot_value, slot_type
        )
        
        # Insert cross-turn turns into dialogue
        # Find a good insertion point (after first few turns)
        insert_idx = min(2, len(turns))
        
        new_turns = turns[:insert_idx]
        for ct in crossturn_turns:
            new_turns.append({
                "role": ct.role,
                "text": ct.text,
                "segment": ct.segment,
            })
        new_turns.extend(turns[insert_idx:])
        
        return new_turns

    def process_dataset(
        self,
        dataset: str,
        split: str = "train",
    ) -> Iterator[AugmentedDialogue]:
        """Process all dialogues in a dataset split."""
        loader = self._get_loader(dataset, split)
        
        count = 0
        for dialogue in loader.load():
            yield self.process_dialogue(dialogue)
            
            count += 1
            if self.config.sample_size and count >= self.config.sample_size:
                break

    def run(self, splits: list[str] | None = None) -> dict[str, int]:
        """Run pipeline on all configured datasets.
        
        Output structure: data/train.jsonl, data/valid.jsonl, data/test.jsonl
        Each dialogue gets a prefixed ID: {source}_{original_id}
        
        Args:
            splits: List of splits to process (default: train, valid, test)
        
        Returns:
            Dict mapping dataset to processed count
        """
        if splits is None:
            splits = ["train", "valid", "test"]
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {}
        
        # Open one file per split
        split_files = {}
        for split in splits:
            output_path = self.config.output_dir / f"{split}.jsonl"
            split_files[split] = open(output_path, "w")
        
        try:
            for dataset in self.config.datasets:
                print(f"Processing dataset: {dataset}")
                total = 0
                
                for split in splits:
                    f = split_files.get(split)
                    if not f:
                        continue
                    
                    try:
                        for augmented in self.process_dataset(dataset, split):
                            # Prefix dialogue ID with source to avoid collisions
                            augmented_dict = augmented.to_dict()
                            original_id = augmented_dict.get("dialogue_id", "")
                            augmented_dict["dialogue_id"] = f"{dataset}_{original_id}"
                            
                            f.write(json.dumps(augmented_dict) + "\n")
                            total += 1
                    except FileNotFoundError:
                        # Skip if split doesn't exist for this dataset
                        continue
                
                stats[dataset] = total
        
        finally:
            # Close all files
            for f in split_files.values():
                f.close()
        
        return stats

