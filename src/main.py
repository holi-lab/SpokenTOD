import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from config import DISFLUENCY_TAGS
from data import (
    extract_dialogue_acts,
    extract_slot_positions,
)
from engine import (
    AlignmentHandler,
    char_to_token_positions,
    compute_position_weights,
    sample_disfluencies_for_turn,
    select_positions,
    tokenize_with_positions,
)
from injector import (
    inject_correction,
    inject_discourse_marker,
    inject_editing_term,
    inject_filled_pause,
    inject_prolongation,
    inject_repetition,
    inject_restart,
    inject_slurring,
)
from common.llm import get_usage_totals
from common.utils import load_dialogues


def _adjust_annotation_positions(
    annotations: list[dict[str, Any]],
    insert_position: int,
    char_offset: int,
) -> None:
    if not char_offset:
        return
    for annotation in annotations:
        position = annotation.get("position")
        if position is not None and position >= insert_position:
            annotation["position"] = position + char_offset


def _find_correction_position(text: str) -> int | None:
    correction_patterns = [
        r"(\w+)[-–]\s*no[,\s]+",
        r"(\w+)[-–]\s*wait[,\s]+",
        r"(\w+)[-–]\s*actually[,\s]+",
        r"(\w+)[-–]\s*I mean[,\s]+",
        r"(\w+)\.\.\.\s*",
    ]
    for pattern in correction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start()
    return None


def _find_restart_position(text: str) -> int | None:
    restart_patterns = [
        r"(\w+\s+\w+)[-–]\s+",
        r"(\w+)[-–]\s+",
        r"([^.]+)\.\.\.\s*",
    ]
    for pattern in restart_patterns:
        match = re.search(pattern, text)
        if match:
            return match.start()
    return None


def _find_slurring_position(
    text: str,
    annotation: dict[str, Any],
    slot_positions: dict[str, Any],
) -> int | None:
    slot_name = annotation.get("slot_name")
    if slot_name and slot_name in slot_positions:
        slot_start = slot_positions[slot_name].get("start")
        if slot_start is not None:
            return slot_start
    for key in ("slurred_value", "original_value"):
        value = annotation.get(key)
        if value:
            match = re.search(re.escape(value), text, re.IGNORECASE)
            if match:
                return match.start()
    return None


def _update_llm_positions(
    annotations: list[dict[str, Any]],
    text: str,
    slot_positions: dict[str, Any],
) -> None:
    for annotation in annotations:
        ann_type = annotation.get("type")
        if ann_type == "COR":
            position = _find_correction_position(text)
        elif ann_type == "RST":
            position = _find_restart_position(text)
        elif ann_type == "SLR":
            position = _find_slurring_position(text, annotation, slot_positions)
        else:
            continue
        if position is not None:
            annotation["position"] = position
        else:
            annotation.pop("position", None)


def build_tagged_utterance(
    raw_text: str,
    annotations: list[dict[str, Any]],
) -> str:
    insertions: list[tuple[int, int, str]] = []
    for idx, annotation in enumerate(annotations):
        ann_type = annotation.get("type")
        tag = annotation.get("tag") or DISFLUENCY_TAGS.get(ann_type)
        position = annotation.get("position")
        if tag is None or position is None:
            continue
        if not isinstance(position, int):
            continue
        if position < 0 or position > len(raw_text):
            continue
        insertions.append((position, idx, tag))

    insertions.sort(key=lambda item: (item[0], item[1]), reverse=True)
    tagged_text = raw_text
    for position, _, tag in insertions:
        tagged_text = tagged_text[:position] + f"{tag} " + tagged_text[position:]

    return tagged_text


async def process_turn(
    turn: dict[str, Any],
    use_llm: bool = True,
    model: str | None = None,
    provider: str = "openrouter",
    base_url: str = "http://localhost:8000/v1",
    tracing: bool = False,
) -> dict[str, Any]:
    """
    Process a single turn to inject disfluencies and generate follow-up turns if needed.
    Returns:
        augmented_turn: The modified user turn with new fields.
    """
    utterance = turn["utterance"]
    frames = turn["frames"]

    # Keep original fields
    augmented_turn = turn.copy()

    # Default values if no augmentation
    augmented_turn["disfluent_utterance"] = utterance
    augmented_turn["tagged_utterance"] = utterance
    augmented_turn["injections"] = []

    # Step 1: Extract metadata
    dialogue_acts = extract_dialogue_acts(frames)
    slot_positions_map = extract_slot_positions(utterance, frames)

    # Step 2: Tokenize and compute position weights
    tokens, token_char_positions = tokenize_with_positions(utterance)
    slot_positions = char_to_token_positions(slot_positions_map, token_char_positions)
    position_weights = compute_position_weights(len(tokens), slot_positions)

    # Step 3: Sample disfluencies
    sampled = sample_disfluencies_for_turn(dialogue_acts)

    if not sampled:
        return augmented_turn, []

    # Step 4: Assign positions to disfluencies
    sampled = select_positions(sampled, position_weights, slot_positions, tokens)

    # Initialize trackers
    current_raw_text = utterance
    current_slots = slot_positions.copy()
    alignment_handler = AlignmentHandler()
    annotations: list[dict[str, Any]] = []

    # Process injections
    for disf in sampled:
        disf_type = disf["type"]
        token_pos = disf.get("token_position", 0)

        # Re-tokenize after each modification
        tokens, token_char_positions = tokenize_with_positions(current_raw_text)

        if token_pos < len(token_char_positions):
            char_pos = token_char_positions[token_pos][0]
        else:
            char_pos = len(current_raw_text)

        # ===== PHASE 1: LLM-based (structural changes) =====
        if disf_type == "RST":
            if use_llm and model:
                try:
                    if not dialogue_acts:
                        continue
                    result, current_slots = await inject_restart(
                        utterance=current_raw_text,
                        tokens=tokens,
                        token_position=token_pos,
                        slot_positions=current_slots,
                        dialogue_act=dialogue_acts[0],
                        model=model,
                        provider=provider,
                        base_url=base_url,
                        include_usage=tracing,
                    )
                    current_raw_text = result.raw_text
                    annotations.append(result.annotation)
                    _update_llm_positions(annotations, current_raw_text, current_slots)
                    alignment_handler.reset()
                except Exception as e:
                    logger.error(f"RST injection failed: {e}")
                    continue

        elif disf_type == "COR":
            slot_name = disf.get("slot_related")
            if use_llm and model and slot_name and slot_name in current_slots:
                try:
                    slot_value = current_slots[slot_name].get("value", "")
                    result, current_slots = await inject_correction(
                        utterance=current_raw_text,
                        slot_name=slot_name,
                        slot_value=slot_value,
                        slot_positions=current_slots,
                        dialogue_act=dialogue_acts[0] if dialogue_acts else "INFORM",
                        model=model,
                        provider=provider,
                        base_url=base_url,
                        include_usage=tracing,
                    )
                    current_raw_text = result.raw_text
                    annotations.append(result.annotation)
                    _update_llm_positions(annotations, current_raw_text, current_slots)
                    alignment_handler.reset()
                except Exception as e:
                    logger.error(f"COR injection failed: {e}")
                    continue

        elif disf_type == "SLR":
            result = inject_slurring(
                text=current_raw_text,
                tokens=tokens,
                token_char_positions=token_char_positions,
                token_index=token_pos,
            )
            _adjust_annotation_positions(
                annotations,
                result.insert_position,
                result.char_offset,
            )
            current_raw_text = result.raw_text
            alignment_handler.record_insertion(char_pos, result.char_offset)
            annotations.append(result.annotation)

        # ===== PHASE 2: Rule-based (surface changes) =====
        elif disf_type in ("FP", "DM", "EDIT"):
            if disf_type == "DM":
                result = inject_discourse_marker(
                    text=current_raw_text,
                    char_position=char_pos,
                )
            elif disf_type == "EDIT":
                result = inject_editing_term(
                    text=current_raw_text,
                    char_position=char_pos,
                )
            else:
                result = inject_filled_pause(
                    text=current_raw_text,
                    char_position=char_pos,
                )
            _adjust_annotation_positions(
                annotations,
                result.insert_position,
                result.char_offset,
            )
            current_raw_text = result.raw_text
            alignment_handler.record_insertion(char_pos, result.char_offset)
            annotations.append(result.annotation)

        elif disf_type == "PRO":
            result = inject_prolongation(
                text=current_raw_text,
                tokens=tokens,
                token_char_positions=token_char_positions,
                token_index=token_pos,
            )
            _adjust_annotation_positions(
                annotations,
                result.insert_position,
                result.char_offset,
            )
            current_raw_text = result.raw_text
            alignment_handler.record_insertion(char_pos, result.char_offset)
            annotations.append(result.annotation)

        elif disf_type == "REP":
            level = disf.get("meta", {}).get("level", "word")
            # Randomly decide phoneme level if not specified but config allows?
            # For now stick to strict sampling.
            # Use logic to call correct injector or pass level to inject_repetition

            # Note: sample_disfluencies might pick REP without meta level sometimes?
            # Default to word if not set
            if random.random() < 0.2:  # 20% chance of phoneme rep if generic REP
                level = "phoneme"

            result = inject_repetition(
                text=current_raw_text,
                tokens=tokens,
                token_char_positions=token_char_positions,
                token_index=token_pos,
                level=level,
            )
            _adjust_annotation_positions(
                annotations,
                result.insert_position,
                result.char_offset,
            )
            current_raw_text = result.raw_text
            current_slots = alignment_handler.update_slot_positions(
                current_slots
            )  # Repetition changes indices?
            # Actually inject_repetition returns offset.
            alignment_handler.record_insertion(char_pos, result.char_offset)
            annotations.append(result.annotation)

    augmented_turn["disfluent_utterance"] = current_raw_text
    augmented_turn["tagged_utterance"] = build_tagged_utterance(current_raw_text, annotations)
    augmented_turn["injections"] = annotations
    return augmented_turn


async def main(
    data_dir: str,
    output_dir: str,
    limit: int | None = None,
    use_llm: bool = True,
    split: str = "train",
    model: str | None = None,
    provider: str = "openrouter",
    base_url: str = "http://localhost:8000/v1",
    tracing: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_dialogues: list[dict[str, Any]] = []
    batch_idx = 0

    dialogue_files = load_dialogues(data_dir, split, limit)
    total_dialogues = sum(len(d) for _, d in dialogue_files)
    logger.info(f"Processing {total_dialogues} dialogues...")

    with tqdm(total=total_dialogues) as pbar:
        for dialogue_file, dialogues in dialogue_files:
            processed_dialogues_in_file = []

            # Step 1: Collect tasks for all USER turns across all dialogues in this file
            semaphore = asyncio.Semaphore(200)

            async def sem_process_turn(turn_to_proc):
                async with semaphore:
                    return await process_turn(
                        turn_to_proc,
                        use_llm=use_llm,
                        model=model,
                        provider=provider,
                        base_url=base_url,
                        tracing=tracing,
                    )

            tasks = []
            task_indices = []  # Stores (dialogue_idx, turn_idx)

            for d_idx, dialogue in enumerate(dialogues):
                for t_idx, turn in enumerate(dialogue["turns"]):
                    if turn["speaker"] == "USER":
                        tasks.append(sem_process_turn(turn))
                        task_indices.append((d_idx, t_idx))

            # Step 2: Execute all tasks for this file in parallel (but throttled by semaphore)
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Step 3: Map results back to the original structure
                for result, (d_idx, t_idx) in zip(results, task_indices):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error processing turn in {dialogues[d_idx].get('dialogue_id', 'unknown')}: {result}"
                        )
                        # Keep original if failed
                    else:
                        dialogues[d_idx]["turns"][t_idx] = result

            # Step 4: Save processed dialogues
            processed_dialogues_in_file = dialogues
            pbar.update(len(dialogues))

            # Save batch
            stem = dialogue_file.stem
            suffix = stem.split("_")[-1] if "_" in stem else f"{batch_idx + 1:03d}"
            batch_file = output_path / f"dialogues_{suffix}.json"

            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump(processed_dialogues_in_file, f, indent=2, ensure_ascii=False)

            all_dialogues.extend(processed_dialogues_in_file)
            batch_idx += 1

    usage_path = output_path / "usage.json"
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(get_usage_totals(), f, indent=2)
    logger.info(f"Usage totals saved to {usage_path}")

    logger.success(f"Done! Processed {len(all_dialogues)} dialogues.")
    logger.info(f"Output saved to {output_path}")
