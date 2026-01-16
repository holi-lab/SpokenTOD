"""Rule-based disfluency injectors: FP, DM, EDIT, PRO, REP, SLR.

All injectors are simple rule-based insertions with tag markers.
"""

import random

from data.augmentation.config import DISFLUENCY_TAGS, FILLERS
from data.augmentation.injector.base import InjectionResult
from data.augmentation.types import SlotPositions


def _choose_filler(category: str, fallback: str) -> str:
    options = FILLERS.get(category) or FILLERS.get(fallback)
    if not options:
        options = ["um"]
    return random.choice(options)


def _inject_filler_with_tag(
    text: str,
    char_position: int,
    filler: str,
    tag: str,
    injection_type: str,
    filler_category: str,
) -> InjectionResult:
    """
    Insert a tagged filler phrase before the specified character position.
    """
    # Include comma for natural pause (no [PAUSE] token needed)
    filler_with_punctuation = f"{filler}, "

    # Build raw text (without tag)
    raw_text = text[:char_position] + filler_with_punctuation + text[char_position:]

    # Build tagged text (with TTS tag before filler)
    tagged_text = text[:char_position] + f"{tag} {filler_with_punctuation}" + text[char_position:]

    annotation = {
        "type": injection_type,
        "text": filler_with_punctuation,
        "position": char_position,
        "filler_category": filler_category,
    }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=len(filler_with_punctuation),
        insert_position=char_position,
        annotation=annotation,
    )


def inject_filled_pause(
    text: str,
    char_position: int,
) -> InjectionResult:
    """
    Insert a filled pause (e.g., um, uh) before the specified character position.
    """
    filler = _choose_filler("filled_pause", "common")
    tag = DISFLUENCY_TAGS["FP"]
    return _inject_filler_with_tag(
        text=text,
        char_position=char_position,
        filler=filler,
        tag=tag,
        injection_type="FP",
        filler_category="filled_pause",
    )


def inject_discourse_marker(
    text: str,
    char_position: int,
) -> InjectionResult:
    """
    Insert a discourse marker (e.g., well, you know) before the specified character position.
    """
    filler = _choose_filler("discourse_marker", "thinking")
    tag = DISFLUENCY_TAGS["DM"]
    return _inject_filler_with_tag(
        text=text,
        char_position=char_position,
        filler=filler,
        tag=tag,
        injection_type="DM",
        filler_category="discourse_marker",
    )


def inject_editing_term(
    text: str,
    char_position: int,
) -> InjectionResult:
    """
    Insert an explicit editing term (e.g., I mean, sorry) before the specified character position.
    """
    filler = _choose_filler("editing_term", "hedge")
    tag = DISFLUENCY_TAGS["EDIT"]
    return _inject_filler_with_tag(
        text=text,
        char_position=char_position,
        filler=filler,
        tag=tag,
        injection_type="EDIT",
        filler_category="editing_term",
    )


def inject_prolongation(
    text: str,
    tokens: list[str],
    token_char_positions: list[tuple[int, int]],
    token_index: int,
) -> InjectionResult:
    """
    Mark a word for prolongation without modifying the text itself.
    Tag format: "[PRO] word"

    Args:
        text: Current utterance text
        char_position: Character start of token
        tokens: List of tokens
        token_char_positions: List of (start, end) for each token
        token_index: Index of token to modify

    Returns:
        InjectionResult with PRO tag before the word
    """
    if token_index >= len(tokens):
        token_index = len(tokens) - 1

    word = tokens[token_index]
    t_start, t_end = token_char_positions[token_index]

    # Raw text stays the same (no modification)
    raw_text = text

    # Tagged text: add [PRO] tag before the word
    tag = DISFLUENCY_TAGS["PRO"]
    tagged_text = text[:t_start] + f"{tag} " + text[t_start:]

    annotation = {
        "type": "PRO",
        "word": word,
        "position": t_start,
    }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=0,  # No text modification, only tag insertion
        insert_position=t_start,
        annotation=annotation,
    )


def inject_repetition(
    text: str,
    tokens: list[str],
    token_char_positions: list[tuple[int, int]],
    token_index: int,
    level: str = "word",
) -> InjectionResult:
    """
    Repeat word(s) at specified token position.

    Args:
        text: Current utterance text
        tokens: List of tokens
        token_char_positions: List of (start, end) for each token
        token_index: Index of token to repeat
        level: "word" or "phrase"

    Returns:
        InjectionResult with repeated text
    """
    if token_index >= len(tokens):
        token_index = len(tokens) - 1

    if level == "phoneme":
        return inject_phoneme_rep(text, tokens, token_char_positions, token_index)

    if level == "word":
        word = tokens[token_index]
        t_start, t_end = token_char_positions[token_index]

        # Repeat with comma for pause
        repeated = f"{word}, {word}"
        char_offset = len(repeated) - len(word)

        raw_text = text[:t_start] + repeated + text[t_end:]

        tag = DISFLUENCY_TAGS["REP"]
        tagged_text = text[:t_start] + f"{tag} {repeated}" + text[t_end:]

        annotation = {
            "type": "REP",
            "repeated_unit": word,
            "level": "word",
            "position": t_start,
        }

    else:  # phrase level (2-3 words)
        end_idx = min(token_index + 2, len(tokens))
        phrase_tokens = tokens[token_index:end_idx]
        phrase = " ".join(phrase_tokens)

        t_start = token_char_positions[token_index][0]
        t_end = token_char_positions[end_idx - 1][1]

        repeated = f"{phrase}, {phrase}"
        char_offset = len(repeated) - (t_end - t_start)

        raw_text = text[:t_start] + repeated + text[t_end:]

        tag = DISFLUENCY_TAGS["REP"]
        tagged_text = text[:t_start] + f"{tag} {repeated}" + text[t_end:]

        annotation = {
            "type": "REP",
            "repeated_unit": phrase,
            "level": "phrase",
            "position": t_start,
        }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=char_offset,
        insert_position=t_start,
        annotation=annotation,
    )


def inject_phoneme_rep(
    text: str,
    tokens: list[str],
    token_char_positions: list[tuple[int, int]],
    token_index: int,
) -> InjectionResult:
    """
    Repeat a random number of phonemes/characters from the beginning of a word.
    Examples: "London" -> "L- London" or "Lon- London"
              "restaurant" -> "r- restaurant" or "rest- restaurant"

    Args:
        text: Current utterance text
        tokens: List of tokens
        token_char_positions: List of (start, end) for each token
        token_index: Index of token to repeat

    Returns:
        InjectionResult with phoneme repetition
    """
    if token_index >= len(tokens):
        token_index = len(tokens) - 1

    word = tokens[token_index]
    t_start, t_end = token_char_positions[token_index]

    # Check if first char is a letter
    if not word or not word[0].isalpha():
        # Fallback to word repetition if not applicable
        return inject_repetition(text, tokens, token_char_positions, token_index, level="word")

    # Randomly choose how many characters to repeat (1 to half of word length, max 4)
    max_chars = min(len(word) // 2 + 1, 4)
    num_chars = random.randint(1, max_chars)

    phoneme_part = word[:num_chars]
    full_insertion = f"{phoneme_part}- {word}"

    char_offset = len(full_insertion) - len(word)

    raw_text = text[:t_start] + full_insertion + text[t_end:]

    # Tagged text
    tag = DISFLUENCY_TAGS["REP"]
    tagged_text = text[:t_start] + f"{tag} {full_insertion}" + text[t_end:]

    annotation = {
        "type": "REP",
        "repeated_unit": phoneme_part,
        "level": "phoneme",
        "position": t_start,
    }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=char_offset,
        insert_position=t_start,
        annotation=annotation,
    )


def inject_slurring(
    text: str,
    tokens: list[str],
    token_char_positions: list[tuple[int, int]],
    token_index: int,
) -> InjectionResult:
    """
    Mark a word for slurring without modifying the text itself.
    Tag format: "[SLR] word"

    Args:
        text: Current utterance text
        char_position: Character start of token
        tokens: List of tokens
        token_char_positions: List of (start, end) for each token
        token_index: Index of token to mark for slurring

    Returns:
        InjectionResult with SLR tag before the word
    """
    if token_index >= len(tokens):
        token_index = len(tokens) - 1

    word = tokens[token_index]
    t_start, t_end = token_char_positions[token_index]

    # Raw text stays the same (no modification)
    raw_text = text

    # Tagged text: add [SLR] tag before the word
    tag = DISFLUENCY_TAGS["SLR"]
    tagged_text = text[:t_start] + f"{tag} " + text[t_start:]

    annotation = {
        "type": "SLR",
        "word": word,
        "position": t_start,
    }

    return InjectionResult(
        raw_text=raw_text,
        tagged_text=tagged_text,
        char_offset=0,  # No text modification, only tag insertion
        insert_position=t_start,
        annotation=annotation,
    )
