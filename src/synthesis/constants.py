from pathlib import Path

from synthesis.utils import read_reference_transcript

NEUTRAL = 0
FEARFUL = 1
DISSATISFIED = 2
APOLOGETIC = 3
ABUSIVE = 4
EXCITED = 5
SATISFIED = 6

# 대분류 이름 매핑
EMOTION_CATEGORIES = {
    NEUTRAL: "neutral",
    FEARFUL: "fearful",
    DISSATISFIED: "dissatisfied",
    APOLOGETIC: "apologetic",
    ABUSIVE: "abusive",
    EXCITED: "excited",
    SATISFIED: "satisfied",
}

# 하위 감정/특성 매핑
EMOTION_SUBCATEGORIES = {
    NEUTRAL: ["calm", "indifferent", "patient", "relaxed"],
    FEARFUL: [
        "fearful",
        "shocked",
        "surprised",
    ],
    DISSATISFIED: [
        "angry",
        "contempt",
        "disgusted",
        "defiant",
    ],
    APOLOGETIC: ["compassionate", "selfless", "humble"],
    ABUSIVE: [
        "commanding",
        "authoritative",
        "merciless",
        "loud",
        "vengeful",
    ],
    EXCITED: [
        "adventurous",
        "energetic",
        "passionate",
        "curious",
        "creative",
        "joyful",
    ],
    SATISFIED: [
        "proud",
        "hopeful",
        "happy",
        "cheerful",
    ],
}

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LANGUAGE = "English"
MAX_NEW_TOKENS = 4608
REFERENCE_TRANSCRIPT = read_reference_transcript(
    Path(__file__).resolve().parents[2] / "datasets/SpeechAccentArchive/reading-passage.txt"
)