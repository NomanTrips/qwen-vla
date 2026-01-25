# Data Pipeline: Raw Capture Schema (Episode 0001)

This document captures the observed raw recording schema for `raw_screen_capture/ep_0001` and the alignment rules needed to convert the recording into per-frame action labels.

## Episode Layout

Each episode directory contains:

- `video.mp4`: Screen recording at 384×384 canvas, 10 FPS.
- `events.jsonl`: Line-delimited JSON input events with QPC timestamps.
- `frame_ts.bin`: Little-endian `int64` QPC timestamps per video frame (duplicates are present).
- `meta.json`: Recording metadata used to decode timestamps.

## `meta.json`

Observed fields:

```json
{
  "bitrate": 6000,
  "canvas_h": 384,
  "canvas_w": 384,
  "codec": "h264_nvenc",
  "fps": 10,
  "offset_ms": 0,
  "qpc_freq": 10000000,
  "timestamps": "frame_ts.bin"
}
```

- `qpc_freq` is used to convert QPC ticks to milliseconds: `ms = (ts / qpc_freq) * 1000`.
- `offset_ms` is applied after conversion (currently 0 in this episode).

## `events.jsonl`

Each line is a JSON object with `ts` (QPC timestamp) and `kind`. Observed event kinds and fields:

### Mouse Events

- `mousemove`
  - `x`, `y`: Pixel coordinates on the canvas.
- `mouseclick`
  - `x`, `y`: Pixel coordinates on the canvas.
  - `buttons` (optional): Button bitmask (observed value `1` for left click).

### Keyboard Events

- `keydown`
  - `vk`: Virtual key code.
- `keyup`
  - `vk`: Virtual key code.

### Text Input Events

- `textInput`
  - `vk`: Virtual key code.
  - `text`: Text payload (e.g., `"w"`, `"\r"`).

Notes:
- `mouseclick` is emitted twice for a single click (one line with `buttons`, one without).
- Both `keydown`/`keyup` and `textInput` appear for keys that generate text.

## `frame_ts.bin`

- Contents are little-endian `int64` QPC timestamps per frame.
- Duplicate timestamps are present (the same QPC value repeated for multiple frames).
- Use the index in this file as the frame index when aligning events.

## Timestamp Alignment

To align events to frames:

1. Convert event timestamps from QPC → ms using `qpc_freq` in `meta.json`.
2. Add `offset_ms` to the converted timestamp.
3. Convert frame timestamps in `frame_ts.bin` using the same conversion.
4. For each frame interval `[ts_i, ts_{i+1})`, associate all events whose timestamps fall in the interval.

## Action Space Definition (Derived from Raw Data)

- Mouse position: raw pixel coordinates (`x`, `y`) on a 384×384 canvas.
- Mouse buttons: derived from `buttons` bitmask or button-specific channels.
- Keyboard state: maintain a pressed-key set using `keydown` and `keyup`.
- Text input: retain `text` as raw input if needed for debugging or instruction reconstruction.

## Conversion Output (Processed Dataset)

The converter emits one directory per episode with:

- `frames/`: JPEG frames named `frame_000000.jpg`.
- `actions.jsonl`: Line-delimited per-frame action records.
  - First line is a header with `key_vocab` (sorted list of observed `vk` codes).
  - Each subsequent line stores normalized mouse position/delta, button state, key state vector, and the raw events seen in that frame interval.
- `meta.json`: Summary metadata (canvas size, FPS, key vocab, frame counts).

To run the converter:

```bash
python scripts/convert_recordings.py \\
  --input raw_screen_capture \\
  --output data/processed \\
  --drop-duplicate-frames
```
