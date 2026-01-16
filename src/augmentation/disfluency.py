"""Standalone disfluency injection for augmentation pipeline.

Self-contained implementation to avoid dependency issues with legacy injector module.
"""

import random
import re
from dataclasses import dataclass


@dataclass
class DisfluencyConfig:
    """Configuration for disfluency injection."""
    fp_prob: float = 0.15      # Filled pause probability
    dm_prob: float = 0.10      # Discourse marker probability
    edit_prob: float = 0.08    # Editing term probability
    pro_prob: float = 0.05     # Prolongation probability
    rep_prob: float = 0.12     # Repetition probability
    slr_prob: float = 0.05     # Slurring probability
    max_per_utt: int = 2       # Max disfluencies per utterance


# Filler vocabularies
FILLERS = {
    "FP": ["um", "uh", "er", "ah"],
    "DM": ["like", "you know", "well", "so", "I mean"],
    "EDIT": ["wait", "no wait", "I mean", "sorry"],
}

# TTS Tags
TAGS = {
    "FP": "[FP]",
    "DM": "[DM]",
    "EDIT": "[EDIT]",
    "REP": "[REP]",
    "PRO": "[PRO]",
    "SLR": "[SLR]",
}


def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Simple tokenizer returning tokens and positions."""
    tokens = []
    positions = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group())
        positions.append((match.start(), match.end()))
    return tokens, positions


def inject_filled_pause(text: str, pos: int) -> tuple[str, str]:
    """Inject filled pause (um, uh, etc.)."""
    filler = random.choice(FILLERS["FP"])
    raw = text[:pos] + filler + ", " + text[pos:]
    tagged = text[:pos] + f"{TAGS['FP']} {filler}, " + text[pos:]
    return raw, tagged


def inject_discourse_marker(text: str, pos: int) -> tuple[str, str]:
    """Inject discourse marker (like, you know, etc.)."""
    filler = random.choice(FILLERS["DM"])
    raw = text[:pos] + filler + ", " + text[pos:]
    tagged = text[:pos] + f"{TAGS['DM']} {filler}, " + text[pos:]
    return raw, tagged


def inject_editing_term(text: str, pos: int) -> tuple[str, str]:
    """Inject editing term (wait, I mean, etc.)."""
    filler = random.choice(FILLERS["EDIT"])
    raw = text[:pos] + filler + ", " + text[pos:]
    tagged = text[:pos] + f"{TAGS['EDIT']} {filler}, " + text[pos:]
    return raw, tagged


def inject_repetition(text: str, tokens: list, positions: list, idx: int) -> tuple[str, str]:
    """Inject word repetition."""
    if idx >= len(tokens):
        return text, text
    
    word = tokens[idx]
    start, end = positions[idx]
    
    # Repeat the word
    repeated = f"{word}, {word}"
    raw = text[:start] + repeated + text[end:]
    tagged = text[:start] + f"{TAGS['REP']} {repeated}" + text[end:]
    return raw, tagged


def inject_prolongation(text: str, tokens: list, positions: list, idx: int) -> tuple[str, str]:
    """Inject vowel prolongation (e.g., 'goooood')."""
    if idx >= len(tokens):
        return text, text
    
    word = tokens[idx]
    start, end = positions[idx]
    
    # Prolong a vowel in the word
    vowels = "aeiouAEIOU"
    prolonged = word
    for i, c in enumerate(word):
        if c in vowels:
            prolonged = word[:i] + c * 3 + word[i+1:]
            break
    
    raw = text[:start] + prolonged + text[end:]
    tagged = text[:start] + f"{TAGS['PRO']} {prolonged}" + text[end:]
    return raw, tagged


def inject_slurring(text: str, tokens: list, positions: list, idx: int) -> tuple[str, str]:
    """Inject slurred speech (merge two words)."""
    if idx >= len(tokens) - 1:
        return text, text
    
    word1 = tokens[idx]
    word2 = tokens[idx + 1]
    start1, _ = positions[idx]
    _, end2 = positions[idx + 1]
    
    # Simple slur: merge words
    slurred = word1 + word2.lower()
    raw = text[:start1] + slurred + text[end2:]
    tagged = text[:start1] + f"{TAGS['SLR']} {slurred}" + text[end2:]
    return raw, tagged


def inject_disfluency(
    text: str,
    config: DisfluencyConfig | None = None,
) -> dict:
    """Inject disfluencies into a user utterance.
    
    Returns:
        Dict with "text", "tagged", and "disfluency" annotations
    """
    if config is None:
        config = DisfluencyConfig()
    
    if not text.strip():
        return {"text": text, "tagged": text, "disfluency": []}
    
    tokens, positions = tokenize(text)
    if not tokens:
        return {"text": text, "tagged": text, "disfluency": []}
    
    # Collect applicable disfluencies
    applicable = []
    if random.random() < config.fp_prob:
        applicable.append("FP")
    if random.random() < config.dm_prob:
        applicable.append("DM")
    if random.random() < config.edit_prob:
        applicable.append("EDIT")
    if random.random() < config.rep_prob:
        applicable.append("REP")
    if random.random() < config.pro_prob:
        applicable.append("PRO")
    if random.random() < config.slr_prob and len(tokens) > 1:
        applicable.append("SLR")
    
    # Limit
    if len(applicable) > config.max_per_utt:
        applicable = random.sample(applicable, config.max_per_utt)
    
    result_text = text
    result_tagged = text
    injections = []
    
    for dtype in applicable:
        # Re-tokenize after each modification
        tokens, positions = tokenize(result_text)
        if not tokens:
            break
        
        # Pick random position
        idx = random.randint(0, len(tokens) - 1)
        pos = positions[idx][0]
        
        try:
            if dtype == "FP":
                result_text, result_tagged = inject_filled_pause(result_text, pos)
                injections.append({"type": "FP", "pos": pos})
            elif dtype == "DM":
                result_text, result_tagged = inject_discourse_marker(result_text, pos)
                injections.append({"type": "DM", "pos": pos})
            elif dtype == "EDIT":
                result_text, result_tagged = inject_editing_term(result_text, pos)
                injections.append({"type": "EDIT", "pos": pos})
            elif dtype == "REP":
                result_text, result_tagged = inject_repetition(result_text, tokens, positions, idx)
                injections.append({"type": "REP", "pos": pos})
            elif dtype == "PRO":
                result_text, result_tagged = inject_prolongation(result_text, tokens, positions, idx)
                injections.append({"type": "PRO", "pos": pos})
            elif dtype == "SLR":
                result_text, result_tagged = inject_slurring(result_text, tokens, positions, idx)
                injections.append({"type": "SLR", "pos": pos})
        except Exception:
            continue
    
    return {
        "text": result_text,
        "tagged": result_tagged,
        "disfluency": injections,
    }


def inject_disfluency_dialogue(
    turns: list[dict],
    config: DisfluencyConfig | None = None,
) -> list[dict]:
    """Inject disfluencies into all user turns."""
    result = []
    
    for turn in turns:
        new_turn = dict(turn)
        
        if turn["role"] == "user":
            injection = inject_disfluency(turn["text"], config)
            new_turn["text"] = injection["text"]
            new_turn["tagged"] = injection["tagged"]
            new_turn["disfluency"] = injection["disfluency"]
        
        result.append(new_turn)
    
    return result
