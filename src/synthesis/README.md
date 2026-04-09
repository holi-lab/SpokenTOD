# Speech Synthesis

This directory contains the Python-package integration for Qwen3-TTS.

## Backend status

- Supported now: `Qwen3-TTS` via the `qwen-tts>=0.1.1` Python package
- Planned later: `Qwen3-TTS` via a `vllm-omni` server

## Install

```bash
uv sync --extra synthesis
```

The `synthesis` extra installs `qwen-tts` and related audio dependencies from
[pyproject.toml](../../pyproject.toml).

## Core API

The core implementation lives in [qwen.py](qwen.py).

```python
from synthesis.qwen import (
    QwenTTSSynthesizer,
    build_voice_clone_prompt,
    load_qwen_model,
    save_wav,
    synthesize_voice_clone,
)
```

Available building blocks:

- `load_qwen_model(...)`: load `Qwen3TTSModel.from_pretrained(...)`
- `build_voice_clone_prompt(...)`: build a voice-clone prompt from one reference audio
- `synthesize_voice_clone(...)`: synthesize one utterance with one voice prompt
- `save_wav(...)`: write the generated waveform to a WAV file
- `QwenTTSSynthesizer`: thin convenience wrapper around the same steps

## Minimal example

```python
from synthesis.qwen import (
    build_voice_clone_prompt,
    load_qwen_model,
    save_wav,
    synthesize_voice_clone,
)

model = load_qwen_model(
    model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
)

voice_prompt = build_voice_clone_prompt(
    model,
    ref_audio="datasets/SpeechAccentArchive/example.wav",
)

wav, sample_rate = synthesize_voice_clone(
    model=model,
    text="Hello, this is a synthesized dialogue turn.",
    voice_clone_prompt=voice_prompt,
    emotion_name="calm",
)

save_wav("outputs/qwen_sample.wav", wav, sample_rate)
```

## Convenience wrapper

```python
from synthesis.qwen import QwenTTSSynthesizer

synthesizer = QwenTTSSynthesizer.from_pretrained(
    model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype="bfloat16",
)

output_path = synthesizer.synthesize_to_file(
    text="Your reservation is confirmed.",
    ref_audio="datasets/SpeechAccentArchive/example.wav",
    output_path="outputs/reservation.wav",
    emotion_name="cheerful",
)

print(output_path)
```

## CLI example

For a single-utterance smoke test:

```bash
uv run python src/synthesize.py \
  --text "Hello, this is a Qwen3-TTS smoke test." \
  --emotion calm \
  --ref-audio datasets/SpeechAccentArchive/example.wav \
  --output-path outputs/qwen_smoke_test.wav
```
