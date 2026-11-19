"""Timestamp metadata for videos represented by pre-extracted JPEG frames."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence


@lru_cache(maxsize=256)
def _load_manifest(path: str) -> tuple[list[float], int]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    timestamps = [float(value) for value in manifest["timestamps_seconds"]]
    if len(timestamps) != 128:
        raise ValueError(f"Expected 128 timestamps in {path}, got {len(timestamps)}")
    return timestamps, int(manifest["total_decoded_frames"])


def load_qwen3vl_video_metadata(
    sampled_frames: Sequence[str | Path],
) -> tuple[dict[str, object], dict[str, object]] | None:
    if not sampled_frames:
        return None
    manifest_path = Path(sampled_frames[0]).parent / "manifest.json"
    if not manifest_path.is_file():
        return None

    timestamps, total_frames = _load_manifest(str(manifest_path))
    duration = timestamps[-1]
    source_fps = (total_frames - 1) / duration if total_frames > 1 and duration > 0 else 1.0
    frame_indices = [timestamp * source_fps for timestamp in timestamps]
    processor_kwargs = {
        "fps": source_fps,
        "do_sample_frames": False,
        "total_num_frames": total_frames,
    }
    video_metadata = {
        "fps": source_fps,
        "frames_indices": frame_indices,
        "total_num_frames": total_frames,
        "duration": duration,
        "video_backend": "opencv",
    }
    return processor_kwargs, video_metadata

