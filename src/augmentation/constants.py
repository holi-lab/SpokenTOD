"""Constants for voice dataset augmentation pipeline."""

# Segmentable slots per dataset (slot_name -> slot_type)
SEGMENTABLE_SLOTS = {
    "sgd": {
        "phone_number": "phone",
        "email": "email",
        "account_number": "id_number",
        "confirmation_number": "id_number",
        "flight_number": "complex_id",
        "event_id": "id_number",
        "passenger_name": "user_name",
        "name": "user_name",
    },
    "abcd": {
        "phone_number": "phone",
        "email": "email",
        "account_id": "id_number",
        "order_id": "id_number",
        "membership_number": "id_number",
        "customer_name": "user_name",
        "full_name": "user_name",
    },
    "emowoz": {
        "phone": "phone",
        "reference": "id_number",
        "trainID": "complex_id",
    },
    "tm2": {
        "reservation.phone": "phone",
        "delivery.phone": "phone",
        "booking.phone": "phone",
        "reservation.name": "user_name",
        "booking.name": "user_name",
        "passenger.name": "user_name",
        "customer.name": "user_name",
        "booking.confirmation": "id_number",
    },
}

# CosyVoice3 emotion token mapping (EmoWOZ label -> token)
EMOTION_TOKENS = {
    0: "[neutral]",
    1: "[sad]",       # fearful/sad
    2: "[angry]",     # dissatisfied
    3: "[sad]",       # apologetic
    4: "[angry]",     # abusive
    5: "[happy]",     # excited
    6: "[happy]",     # satisfied
}

# EmoWOZ emotion label names
EMOTION_LABELS = {
    0: "neutral",
    1: "fearful",
    2: "dissatisfied",
    3: "apologetic",
    4: "abusive",
    5: "excited",
    6: "satisfied",
}

# Datasets configuration
DATASETS = ["emowoz", "sgd", "abcd", "spokenwoz", "tm2"]

# Cross-turn slot excluded datasets (already has native cross-turn)
CROSSTURN_EXCLUDED = {"spokenwoz"}

# Emotion tagging excluded datasets (already has emotion labels)
EMOTION_EXCLUDED = {"emowoz"}

# Error correction probability for cross-turn segmentation
ERROR_CORRECTION_PROB = 0.20

# OpenAI batch settings
OPENAI_MODEL = "gpt-4.1-mini"
BATCH_SIZE = 100
