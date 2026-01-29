# Embodied Chain-of-Thought (ECoT) Data Generation

This document describes the first-pass pipeline for generating ECoT annotations on top of
the processed episodes created by `scripts/convert_recordings.py`.

## Inputs

Each processed episode directory is expected to contain:

- `frames/` with JPEG images (`frame_000000.jpg`, ...).
- `actions.jsonl` with per-frame actions (as emitted by `data_pipeline.convert`).
- Optional: `ecot_meta.json` to supply episode-specific metadata (for example
  `{"sign_color": "red"}`).

## Outputs

The generator writes JSONL examples with:

- `frame_path`: path to the image.
- `instruction`: the task string.
- `state`: current ECoT state (sign found/color, target found).
- `visible_objects`: list of detected objects (from the vision model).
- `reasoning`: chain-of-thought annotation (from the reasoning model).
- `action`: the original action record.

## OpenAI Model Calls

Two calls are made per frame:

1. **Vision pass** to detect sign color and objects (bounding boxes).
2. **Reasoning pass** to generate the reasoning annotation using the task, state,
   visible objects, and action description.

Configure access with environment variables:

- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (optional, for compatible endpoints)

## Example Usage

Generate ECoT annotations for processed episodes:

```bash
python scripts/generate_ecot.py \
  --input data/processed \
  --output data/ecot/ecot_annotations.jsonl \
  --vision-model gpt-4o-mini \
  --reasoning-model o1-mini
```

Dry-run mode skips OpenAI calls and emits placeholder annotations:

```bash
python scripts/generate_ecot.py \
  --input data/processed \
  --output data/ecot/ecot_annotations.jsonl \
  --dry-run
```
