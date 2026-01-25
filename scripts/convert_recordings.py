#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

from data_pipeline.convert import process_episode


def find_episode_dirs(root: str) -> List[str]:
    episodes = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "meta.json")):
            episodes.append(path)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw screen capture recordings.")
    parser.add_argument("--input", required=True, help="Input directory containing episodes.")
    parser.add_argument("--output", required=True, help="Output directory for processed data.")
    parser.add_argument(
        "--drop-duplicate-frames",
        action="store_true",
        help="Drop duplicate frame timestamps and skip corresponding frames in the video.",
    )
    args = parser.parse_args()

    episodes = find_episode_dirs(args.input)
    if not episodes:
        raise SystemExit(f"No episodes found in {args.input}")

    for episode_dir in episodes:
        episode_name = os.path.basename(episode_dir.rstrip(os.sep))
        output_dir = os.path.join(args.output, episode_name)
        process_episode(
            episode_dir=episode_dir,
            output_dir=output_dir,
            drop_duplicates=args.drop_duplicate_frames,
        )
        print(f"Processed {episode_name} -> {output_dir}")


if __name__ == "__main__":
    main()
