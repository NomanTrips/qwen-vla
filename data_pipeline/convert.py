from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import imageio.v3 as iio
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "imageio is required for video decoding. Install with `pip install imageio[ffmpeg]`."
    ) from exc


@dataclass
class EpisodeMeta:
    canvas_w: int
    canvas_h: int
    fps: float
    qpc_freq: float
    offset_ms: float


@dataclass
class FrameAction:
    frame_index: int
    timestamp_ms: float
    mouse_x: float
    mouse_y: float
    mouse_dx: float
    mouse_dy: float
    buttons: Dict[str, int]
    key_state: List[int]
    raw_events: List[dict]


BUTTON_MASKS = {
    "left": 1,
    "right": 2,
    "middle": 4,
}


def load_meta(path: str) -> EpisodeMeta:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return EpisodeMeta(
        canvas_w=int(payload["canvas_w"]),
        canvas_h=int(payload["canvas_h"]),
        fps=float(payload["fps"]),
        qpc_freq=float(payload["qpc_freq"]),
        offset_ms=float(payload.get("offset_ms", 0.0)),
    )


def read_frame_timestamps(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        data = handle.read()
    return np.frombuffer(data, dtype="<i8")


def qpc_to_ms(ts: np.ndarray, qpc_freq: float, offset_ms: float) -> np.ndarray:
    ts_ms = (ts.astype(np.float64) / qpc_freq) * 1000.0
    if offset_ms:
        ts_ms += offset_ms
    return ts_ms


def load_events(path: str, qpc_freq: float, offset_ms: float) -> List[dict]:
    events: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            ts_ms = qpc_to_ms(np.array([event["ts"]], dtype=np.int64), qpc_freq, offset_ms)[0]
            event["timestamp_ms"] = float(ts_ms)
            events.append(event)
    return events


def align_events_to_frames(
    frame_ts_ms: np.ndarray, events: Sequence[dict]
) -> List[List[dict]]:
    frame_ends = np.concatenate([frame_ts_ms[1:], np.array([math.inf])])
    aligned: List[List[dict]] = [[] for _ in range(len(frame_ts_ms))]
    for event in events:
        idx = int(np.searchsorted(frame_ends, event["timestamp_ms"], side="right"))
        if idx >= len(aligned):
            idx = len(aligned) - 1
        aligned[idx].append(event)
    return aligned


def build_key_vocab(events: Iterable[dict]) -> List[int]:
    vocab = sorted({event.get("vk") for event in events if "vk" in event})
    return [vk for vk in vocab if vk is not None]


def normalize_position(value: float, max_value: int) -> float:
    if max_value <= 1:
        return 0.0
    return (value / (max_value - 1)) * 2.0 - 1.0


def normalize_delta(delta: float, max_value: int) -> float:
    if max_value <= 1:
        return 0.0
    normalized = delta / (max_value - 1)
    return float(np.clip(normalized, -1.0, 1.0))


def decode_button_mask(mask: int) -> Dict[str, int]:
    return {name: int(bool(mask & bit)) for name, bit in BUTTON_MASKS.items()}


def derive_frame_actions(
    frame_ts_ms: np.ndarray,
    events_per_frame: Sequence[Sequence[dict]],
    meta: EpisodeMeta,
    key_vocab: Sequence[int],
) -> List[FrameAction]:
    key_state = {vk: 0 for vk in key_vocab}
    mouse_x = meta.canvas_w / 2.0
    mouse_y = meta.canvas_h / 2.0
    prev_mouse_x = mouse_x
    prev_mouse_y = mouse_y
    button_state = {name: 0 for name in BUTTON_MASKS}

    frame_actions: List[FrameAction] = []
    for idx, (timestamp_ms, frame_events) in enumerate(zip(frame_ts_ms, events_per_frame)):
        for event in frame_events:
            kind = event.get("kind")
            if kind in {"mousemove", "mouseclick"}:
                if "x" in event and "y" in event:
                    mouse_x = float(event["x"])
                    mouse_y = float(event["y"])
                if kind == "mouseclick" and "buttons" in event:
                    button_state.update(decode_button_mask(int(event["buttons"])))
            elif kind == "keydown" and "vk" in event:
                key_state[int(event["vk"])] = 1
            elif kind == "keyup" and "vk" in event:
                key_state[int(event["vk"])] = 0

        mouse_dx = mouse_x - prev_mouse_x
        mouse_dy = mouse_y - prev_mouse_y
        prev_mouse_x = mouse_x
        prev_mouse_y = mouse_y

        action = FrameAction(
            frame_index=idx,
            timestamp_ms=float(timestamp_ms),
            mouse_x=normalize_position(mouse_x, meta.canvas_w),
            mouse_y=normalize_position(mouse_y, meta.canvas_h),
            mouse_dx=normalize_delta(mouse_dx, meta.canvas_w),
            mouse_dy=normalize_delta(mouse_dy, meta.canvas_h),
            buttons={name: int(button_state[name]) for name in BUTTON_MASKS},
            key_state=[key_state[vk] for vk in key_vocab],
            raw_events=list(frame_events),
        )
        frame_actions.append(action)
    return frame_actions


def write_frame_actions(path: str, actions: Sequence[FrameAction], key_vocab: Sequence[int]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        header = {"key_vocab": list(key_vocab)}
        handle.write(json.dumps({"type": "header", **header}) + "\n")
        for action in actions:
            payload = {
                "type": "frame",
                "frame_index": action.frame_index,
                "timestamp_ms": action.timestamp_ms,
                "mouse": {
                    "x": action.mouse_x,
                    "y": action.mouse_y,
                    "dx": action.mouse_dx,
                    "dy": action.mouse_dy,
                },
                "buttons": action.buttons,
                "key_state": action.key_state,
                "raw_events": action.raw_events,
            }
            handle.write(json.dumps(payload) + "\n")


def iter_video_frames(video_path: str) -> Iterable[np.ndarray]:
    yield from iio.imiter(video_path)


def save_frames(
    video_path: str,
    output_dir: str,
    keep_indices: Sequence[int] | None = None,
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    keep_set = set(keep_indices) if keep_indices is not None else None
    for idx, frame in enumerate(iter_video_frames(video_path)):
        if keep_set is not None and idx not in keep_set:
            continue
        frame_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
        iio.imwrite(frame_path, frame)
        count += 1
    return count


def drop_duplicate_timestamps(frame_ts_ms: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    unique_ts = []
    keep_indices = []
    last_ts = None
    for idx, ts in enumerate(frame_ts_ms):
        if last_ts is None or ts != last_ts:
            unique_ts.append(ts)
            keep_indices.append(idx)
            last_ts = ts
    return np.array(unique_ts, dtype=np.float64), keep_indices


def process_episode(
    episode_dir: str,
    output_dir: str,
    drop_duplicates: bool = False,
) -> None:
    meta_path = os.path.join(episode_dir, "meta.json")
    events_path = os.path.join(episode_dir, "events.jsonl")
    frame_ts_path = os.path.join(episode_dir, "frame_ts.bin")
    video_path = os.path.join(episode_dir, "video.mp4")

    meta = load_meta(meta_path)
    raw_frame_ts = read_frame_timestamps(frame_ts_path)
    frame_ts_ms = qpc_to_ms(raw_frame_ts, meta.qpc_freq, meta.offset_ms)

    keep_indices: List[int] | None = None
    if drop_duplicates:
        frame_ts_ms, keep_indices = drop_duplicate_timestamps(frame_ts_ms)

    events = load_events(events_path, meta.qpc_freq, meta.offset_ms)
    events_per_frame = align_events_to_frames(frame_ts_ms, events)
    key_vocab = build_key_vocab(events)
    frame_actions = derive_frame_actions(frame_ts_ms, events_per_frame, meta, key_vocab)

    os.makedirs(output_dir, exist_ok=True)
    frame_dir = os.path.join(output_dir, "frames")
    saved_count = save_frames(video_path, frame_dir, keep_indices)

    actions_path = os.path.join(output_dir, "actions.jsonl")
    write_frame_actions(actions_path, frame_actions, key_vocab)

    output_meta = {
        "canvas_w": meta.canvas_w,
        "canvas_h": meta.canvas_h,
        "fps": meta.fps,
        "qpc_freq": meta.qpc_freq,
        "offset_ms": meta.offset_ms,
        "num_frames": len(frame_actions),
        "saved_frames": saved_count,
        "key_vocab": key_vocab,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(output_meta, handle, indent=2)
