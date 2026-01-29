#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from data_pipeline.ecot import DEFAULT_TASK, LLMConfig, generate_ecot_training_data, iter_episode_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Embodied Chain-of-Thought (ECoT) training data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing processed episodes (with frames/actions.jsonl).",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task instruction string.")
    parser.add_argument(
        "--vision-model",
        default="gpt-4o-mini",
        help="OpenAI model for vision annotation.",
    )
    parser.add_argument(
        "--reasoning-model",
        default="o1-mini",
        help="OpenAI model for reasoning annotation.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key.",
    )
    parser.add_argument(
        "--base-url-env",
        default="OPENAI_BASE_URL",
        help="Environment variable containing the OpenAI base URL (optional).",
    )
    parser.add_argument(
        "--metadata-filename",
        default="ecot_meta.json",
        help="Optional per-episode metadata filename (for sign_color, etc).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames per episode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI calls and emit placeholder annotations.",
    )
    args = parser.parse_args()

    llm_config = LLMConfig(
        vision_model=args.vision_model,
        reasoning_model=args.reasoning_model,
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
    )

    episodes = list(iter_episode_dirs(args.input))
    if not episodes:
        raise SystemExit(f"No processed episodes found in {args.input}")

    total = 0
    for episode_dir in episodes:
        episode_name = os.path.basename(episode_dir.rstrip(os.sep))
        episode_output = args.output
        if len(episodes) > 1:
            root, ext = os.path.splitext(args.output)
            episode_output = f"{root}_{episode_name}{ext or '.jsonl'}"
        count = generate_ecot_training_data(
            episode_dir=episode_dir,
            output_path=episode_output,
            task=args.task,
            llm_config=llm_config,
            dry_run=args.dry_run,
            max_frames=args.max_frames,
            metadata_filename=args.metadata_filename,
        )
        total += count
        print(f"Wrote {count} ECoT examples for {episode_name} -> {episode_output}")

    print(f"Total examples written: {total}")


if __name__ == "__main__":
    main()
