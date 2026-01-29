from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

DEFAULT_TASK = (
    "Navigate to find the sign showing a color, then touch the object matching that color."
)


@dataclass
class LLMConfig:
    vision_model: str
    reasoning_model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"


def load_actions(path: str) -> Tuple[List[dict], List[int]]:
    actions: List[dict] = []
    key_vocab: List[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("type") == "header":
                key_vocab = list(payload.get("key_vocab", []))
                continue
            if payload.get("type") == "frame":
                actions.append(payload)
    return actions, key_vocab


def encode_image_to_data_url(path: str) -> str:
    with open(path, "rb") as handle:
        data = handle.read()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def get_openai_client(config: LLMConfig) -> OpenAI:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in env var {config.api_key_env}")
    base_url = os.environ.get(config.base_url_env)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def parse_json_payload(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def vision_annotate_frame(
    client: OpenAI,
    model: str,
    frame_path: str,
    prompt: str,
) -> dict:
    image_url = encode_image_to_data_url(frame_path)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )
    return parse_json_payload(response.output_text)


def reasoning_annotate_frame(
    client: OpenAI,
    model: str,
    prompt: str,
) -> str:
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    return response.output_text.strip()


def detect_sign_and_objects(
    client: Optional[OpenAI],
    config: LLMConfig,
    frame_path: str,
    dry_run: bool,
) -> dict:
    if dry_run:
        return {"sign_color": None, "objects": []}
    prompt = (
        "You are labeling a robot dataset. Identify the sign color if visible and list "
        "relevant colored objects with bounding boxes. Return JSON only with keys:\n"
        "sign_color: string or null\n"
        "objects: list of {label, color, box} where box=[x1,y1,x2,y2] normalized 0-1."
    )
    return vision_annotate_frame(client, config.vision_model, frame_path, prompt)


def update_state(
    state: Dict[str, Any],
    vision_payload: dict,
) -> Dict[str, Any]:
    sign_color = vision_payload.get("sign_color")
    if sign_color:
        state["sign_found"] = True
        state["sign_color"] = sign_color
    objects = vision_payload.get("objects") or []
    if state.get("sign_color"):
        target_color = str(state["sign_color"]).lower()
        if any(str(obj.get("color", "")).lower() == target_color for obj in objects):
            state["target_found"] = True
    return state


KEY_NAME_MAP = {
    87: "W",
    83: "S",
    65: "A",
    68: "D",
    69: "E",
    70: "F",
    32: "SPACE",
}


def describe_action(action: dict, key_vocab: List[int]) -> str:
    parts: List[str] = []
    buttons = action.get("buttons", {})
    if buttons.get("left"):
        parts.append("left click")
    if buttons.get("right"):
        parts.append("right click")

    key_state = action.get("key_state", [])
    pressed = [
        KEY_NAME_MAP.get(vk, f"VK_{vk}")
        for vk, state in zip(key_vocab, key_state)
        if state
    ]
    if pressed:
        parts.append(f"keys pressed: {', '.join(pressed)}")

    mouse = action.get("mouse", {})
    dx = float(mouse.get("dx", 0.0))
    dy = float(mouse.get("dy", 0.0))
    if abs(dx) > 0.05:
        parts.append("turning right" if dx > 0 else "turning left")
    if abs(dy) > 0.05:
        parts.append("looking down" if dy > 0 else "looking up")

    return ", ".join(parts) if parts else "stationary"


def build_reasoning_prompt(
    task: str,
    state: Dict[str, Any],
    visible_objects: list,
    action_desc: str,
) -> str:
    return (
        "You are annotating robot training data. Given the current state and observation, "
        "generate concise reasoning explaining the agent's behavior.\n\n"
        f"TASK: {task}\n\n"
        "CURRENT STATE:\n"
        f"- Sign found: {state.get('sign_found')}\n"
        f"- Sign color: {state.get('sign_color') or 'unknown'}\n"
        f"- Target found: {state.get('target_found')}\n\n"
        f"VISIBLE OBJECTS: {visible_objects}\n\n"
        f"ACTION BEING TAKEN: {action_desc}\n\n"
        "Respond only in this format:\n"
        "SUBTASK: ...\n"
        "OBSERVATION: ...\n"
        "REASONING: ...\n"
        "MOVE: ...\n"
    )


def generate_ecot_training_data(
    episode_dir: str,
    output_path: str,
    task: str,
    llm_config: LLMConfig,
    dry_run: bool = False,
    max_frames: Optional[int] = None,
    metadata_filename: str = "ecot_meta.json",
) -> int:
    actions_path = os.path.join(episode_dir, "actions.jsonl")
    frames_dir = os.path.join(episode_dir, "frames")
    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"Missing actions.jsonl in {episode_dir}")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Missing frames directory in {episode_dir}")

    actions, key_vocab = load_actions(actions_path)
    if max_frames is not None:
        actions = actions[: max_frames]

    metadata_path = os.path.join(episode_dir, metadata_filename)
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    state = {
        "sign_found": False,
        "sign_color": metadata.get("sign_color"),
        "target_found": False,
    }

    client = None
    if not dry_run:
        client = get_openai_client(llm_config)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    examples_written = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for action in actions:
            frame_index = int(action["frame_index"])
            frame_path = os.path.join(frames_dir, f"frame_{frame_index:06d}.jpg")
            if not os.path.exists(frame_path):
                continue

            vision_payload = detect_sign_and_objects(
                client=client,
                config=llm_config,
                frame_path=frame_path,
                dry_run=dry_run,
            )
            state = update_state(state, vision_payload)

            action_desc = describe_action(action, key_vocab)
            visible_objects = vision_payload.get("objects", [])
            prompt = build_reasoning_prompt(task, state, visible_objects, action_desc)
            if dry_run:
                reasoning = (
                    "SUBTASK: ...\nOBSERVATION: ...\nREASONING: ...\nMOVE: ..."
                )
            else:
                reasoning = reasoning_annotate_frame(client, llm_config.reasoning_model, prompt)

            example = {
                "episode": os.path.basename(episode_dir.rstrip(os.sep)),
                "frame_index": frame_index,
                "frame_path": frame_path,
                "instruction": task,
                "state": dict(state),
                "visible_objects": visible_objects,
                "reasoning": reasoning,
                "action": action,
            }
            handle.write(json.dumps(example) + "\n")
            examples_written += 1

    return examples_written


def iter_episode_dirs(root: str) -> Iterable[str]:
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "actions.jsonl")):
            yield path
