# SpokenTOD

SpokenTOD is a large-scale spoken task-oriented dialogue (TOD) dataset built by
augmenting existing TOD benchmarks with realistic spoken behaviors and emotion
labels, then optionally synthesizing speech. This repository implements the
SpokenTOD construction pipeline described in our paper.

## What this repository provides

- **Source datasets**: SpokenWOZ, EmoWOZ, SGD, TaskMaster-2, ABCD
- **Augmentations**
  - Cross-turn slots (multi-turn slot construction and self-correction)
  - Barge-in (error recovery, clarification, efficiency)
  - Disfluency (FP, DM, EDIT, REP, RST, COR)
- **Emotion annotation** using the EmoWOZ label set
- **Speaker demographic sampling** using Speech Accent Archive (SAA)
- **Optional speech synthesis** via Qwen3-TTS conditioned on emotion

## Repository layout

- `src/augmentation/` - text augmentation pipeline
- `src/augment.py` - CLI for text augmentation
- `src/synthesize.py` - CLI for single-utterance speech synthesis (Qwen3-TTS)
- `scripts/download_dataset.sh` - dataset downloader

## Setup

```bash
# Install uv once if needed
# https://docs.astral.sh/uv/getting-started/installation/

# Create/update the project virtualenv (.venv) and install the package
uv sync

# Optional extras
uv sync --extra synthesis

# Activate the environment only if you want an interactive shell inside it
source .venv/bin/activate
```

To see available Makefile targets:

```bash
make help
```

For speech synthesis you will also need:
- `Qwen3-TTS` served by python package (qwen-tts>=0.1.1) or vllm-omni server.
- a PyTorch / torchaudio environment compatible with your machine

For more detail on speech synthesis, please see [src/synthesis/README.md](src/synthesis/README.md).

## Download datasets

Use the Makefile target (recommended):

```bash
make download DATASETS=emowoz,sgd,abcd,tm2,spokenwoz,saa
```

Direct script usage (same logic):

```bash
bash scripts/download_dataset.sh emowoz,sgd,abcd,tm2,spokenwoz,saa
```

Notes:
- The script will fail if a target directory exists unless `FORCE=1`.
- Datasets are large; plan for storage and download time.

## Text augmentation

```bash
# Full run (Makefile defaults)
make augment

# Quick smoke test
make augment-sample
```

Common overrides:

```bash
make augment DATASETS=emowoz,sgd,abcd SPLITS=train,valid AUGMENT_MODEL=Qwen/Qwen3-32B CHUNK_SIZE=100
make augment-sample SAMPLE_SIZE=5 AUGMENT_MODEL=gpt-4.1-mini
make augment-full   # background run with nohup
```

### Model configuration

The pipeline uses LLM calls for emotion tagging, barge-in generation, and some
LLM-based disfluency types through LiteLLM.

- `AUGMENT_MODEL` is passed to LiteLLM exactly as provided. The project does
  not rewrite, normalize, or alias model names.
- Configure provider credentials with the environment variables LiteLLM expects
  for that provider, such as `OPENAI_API_KEY`.
- For OpenAI-compatible local endpoints, set `LITELLM_API_BASE` to the base URL
  to use for augmentation requests.

## Output format

Text augmentation writes JSONL files to `datasets/SpokenTOD/`:

- `train.jsonl`, `valid.jsonl`, `test.jsonl`
- `train_demographics.json`, `valid_demographics.json`, `test_demographics.json`
- `overall_demographics.json`

Each dialogue record contains:

```json
{
  "dialogue_id": "...",
  "source": "emowoz|sgd|abcd|tm2|spokenwoz",
  "goal": {"text": "...", "structured": {...}},
  "turns": [
    {
      "role": "user|assistant",
      "text": "...",
      "slots": [{"slot": "...", "value": "...", "start": 0, "end": 4}],
      "tagged": "...",          // disfluency tags (optional)
      "emotion": {"label": 0, "name": "neutral"},
      "disfluency": [...],       // structured disfluency annotations
      "segment": {...},           // cross-turn segment info
      "state": {...},
      "bargein": {"type": "...", "subtype": "..."},
      "audio_path": "..."        // for datasets with native audio
    }
  ],
  "speaker": {...},
  "assistant_speaker": {...},
  "metadata": {...}
}
```

## Speech synthesis (Qwen3-TTS)

```bash
uv sync --extra synthesis
```

```bash
uv run python src/synthesize.py \
  --text "Hello, this is a Qwen3-TTS smoke test." \
  --emotion calm \
  --ref-audio datasets/SpeechAccentArchive/example.wav \
  --output-path outputs/qwen_smoke_test.wav
```

If you modify dependencies, refresh the lockfile with:

```bash
uv lock
```

## Behavior taxonomy (summary)

- **Emotion**: neutral, fearful, dissatisfied, apologetic, abusive, excited, satisfied
- **Cross-turn slots**: phone numbers, emails, reservation IDs, etc.
- **Barge-in**: error recovery, clarification, efficiency
- **Disfluency tags**:
  - FP (filled pause), DM (discourse marker), EDIT (editing term)
  - REP (repetition), RST (restart), COR (correction)

## Citation

If you use this pipeline or dataset, please cite our paper and the original source datasets (SpokenWOZ, EmoWOZ, SGD, TaskMaster-2, ABCD, SpeechAccentArchive) and Qwen3-TTS.

```
@article{spokentod,
  title   = {SpokenUS: A Large-Scale Spoken Task-Oriented Dialogue Dataset},
  author  = {TBD},
  journal = {TBD},
  year    = {2025}
}
```

## License

MIT License.
