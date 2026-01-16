"""Utility functions for tokenization and text manipulation."""

import re


def tokenize_with_positions(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Whitespace tokenization preserving character positions.

    Args:
        text: Text to tokenize

    Returns:
        tokens: List of token strings
        positions: List of (start_char, end_char) tuples
    """
    tokens: list[str] = []
    positions: list[tuple[int, int]] = []

    for match in re.finditer(r"\S+", text):
        tokens.append(match.group())
        positions.append((match.start(), match.end()))

    return tokens, positions


def get_char_position_for_token(
    token_index: int,
    token_char_positions: list[tuple[int, int]],
) -> int:
    """
    Get character position for a token index.

    Args:
        token_index: Index of token
        token_char_positions: List of (start, end) char positions

    Returns:
        Character position (start of token)
    """
    if token_index < 0:
        return 0
    if token_index >= len(token_char_positions):
        if token_char_positions:
            return token_char_positions[-1][0]
        return 0
    return token_char_positions[token_index][0]


def find_word_boundary_before(text: str, position: int) -> int:
    """
    Find the start of a word at or before the given position.

    Args:
        text: The text
        position: Character position

    Returns:
        Character position of word start
    """
    if position <= 0:
        return 0

    # Go backwards to find word boundary
    i = min(position, len(text) - 1)
    while i > 0 and text[i - 1] not in " \t\n":
        i -= 1
    return i


def clean_utterance(text: str) -> str:
    """
    Clean utterance text for processing.

    Args:
        text: Raw utterance

    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
